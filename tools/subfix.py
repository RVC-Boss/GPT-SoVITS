import datetime
import os
import threading
import traceback
from dataclasses import dataclass
from functools import partial
from typing import List

import click
import gradio as gr
import librosa
import numpy as np
import soundfile
from gradio.components.audio import WaveformOptions

from tools.i18n.i18n import I18nAuto

PARTIAL_EXIT = partial(os._exit, 0)

LANGUAGE_MAP: dict = {
    "ZH": "ZH",
    "zh": "ZH",
    "JP": "JA",
    "jp": "JA",
    "JA": "JA",
    "ja": "JA",
    "EN": "EN",
    "en": "EN",
    "KO": "KO",
    "ko": "KO",
    "yue": "YUE",
    "YUE": "YUE",
}

LOCK = threading.Lock()

IS_CLI = True


@dataclass
class SubfixErr:
    error: Exception
    tracebacks: str


class Subfix:
    batch_size: int = 2
    cur_idx: int = 0
    list_path: str
    textboxes: List[gr.Textbox] = []
    audios: List[gr.Audio] = []
    languages: List[gr.Dropdown] = []
    selections: List[gr.Checkbox] = []
    transcriptions_list: List[List[str]] = []

    merge_audio_button: gr.Button
    delete_audio_button: gr.Button
    previous_index_button1: gr.Button
    next_index_button1: gr.Button
    previous_index_button2: gr.Button
    next_index_button2: gr.Button
    index_slider: gr.Slider
    batch_size_slider: gr.Slider
    close_button: gr.Button

    def __init__(self, i18n: I18nAuto):
        self.i18n = i18n
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=160):
                self.index_slider = gr.Slider(minimum=0, maximum=1, step=1, label=i18n("音频索引"))
            with gr.Column(scale=1, min_width=160):
                self.previous_index_button1 = gr.Button(value=i18n("上一页"), elem_id="btn_previous")
            with gr.Column(scale=1, min_width=160):
                self.next_index_button1 = gr.Button(value=i18n("下一页"), elem_id="btn_next")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=160):
                self.batch_size_slider = gr.Slider(
                    minimum=4, maximum=20, step=2, value=self.batch_size, label=i18n("每页音频条数")
                )
            with gr.Column(scale=1, min_width=160):
                self.merge_audio_button = gr.Button(value=i18n("合并选中音频"))
            with gr.Column(scale=1, min_width=160):
                self.delete_audio_button = gr.Button(value=i18n("删除选中音频"))
        gr.render(
            inputs=[self.index_slider, self.batch_size_slider],
            triggers=[self.batch_size_slider.change],
        )(self._render_text_area)

    @property
    def max_index(self):
        return len(self.transcriptions_list) - 1

    def load_list(self, list_path: str):
        with open(list_path, mode="r", encoding="utf-8") as f:
            list_data = f.readlines()
        for idx, transcriptions in enumerate(list_data):
            data = transcriptions.split("|")
            if len(data) != 4:
                print(f"Error Line {idx + 1}: {'|'.join(data)}")
                continue
            audio_name, audio_folder, text_language, text = data
            self.transcriptions_list.append(
                [
                    audio_name,
                    audio_folder,
                    LANGUAGE_MAP.get(text_language.upper(), text_language.upper()),
                    text.strip("\n").strip(),
                ]
            )
            self.list_path = list_path

    def save_list(self):
        data = []
        for transcriptions in self.transcriptions_list:
            data.append("|".join(transcriptions))
        try:
            with open(self.list_path, mode="w", encoding="utf-8") as f:
                f.write("\n".join(data))
        except Exception as e:
            return SubfixErr(e, traceback.format_exc())

    def change_index(self, index: int):
        audios = []
        texts = []
        languages = []
        checkboxs = []
        with LOCK:
            for i in range(index, index + self.batch_size):
                if i <= self.max_index:
                    audios.append(gr.Audio(value=self.transcriptions_list[i][0]))
                    texts.append(gr.Textbox(value=self.transcriptions_list[i][3], label=self.i18n("Text") + f" {i}"))
                    languages.append(gr.Dropdown(value=self.transcriptions_list[i][2]))
                else:
                    audios.append(gr.Audio(value=None, interactive=False))
                    texts.append(gr.Textbox(value=None, label=self.i18n("Text") + f" {i}", interactive=False))
                    languages.append(gr.Dropdown(value=None, interactive=False))
            checkboxs = [gr.Checkbox(False) for i in range(self.batch_size)]
        self.cur_idx = index
        return *audios, *texts, *languages, *checkboxs

    def next_page(self, index: int):
        batch_size = self.batch_size
        max_index = max(self.max_index - batch_size + 1, 0)
        index = min(index + batch_size, max_index)
        return gr.Slider(value=index), *self.change_index(index)

    def previous_page(self, index: int):
        batch_size = self.batch_size
        index = max(index - batch_size, 0)
        return gr.Slider(value=index), *self.change_index(index)

    def delete_audio(self, index, *selected):
        delete_index = [i + index for i, _ in enumerate(selected) if _]
        delete_index = [i for i in delete_index if i < self.max_index]
        for idx in delete_index[::-1]:
            self.transcriptions_list.pop(idx)
        self.save_list()
        return gr.Slider(value=index, maximum=self.max_index), *self.change_index(index)

    def submit(self, *input):
        with LOCK:
            index = self.cur_idx
            batch_size = self.batch_size
            texts = input[: len(input) // 2]
            languages = input[len(input) // 2 :]
            if texts is None or languages is None:
                raise ValueError()
            print(index, min(index + batch_size, self.max_index))
            for idx in range(index, min(index + batch_size, self.max_index + 1)):
                self.transcriptions_list[idx][3] = texts[idx - index].strip().strip("\n")
                self.transcriptions_list[idx][2] = languages[idx - index]
            result = self.save_list()
            if isinstance(result, SubfixErr):
                gr.Warning(str(result.error))
                print(result.tracebacks)

    def merge_audio(self, index, *selected):
        batch_size = self.batch_size
        merge_index = [i + index for i, _ in enumerate(selected) if _]
        merge_index = [i for i in merge_index if i < self.max_index]
        if len(merge_index) < 2:
            return *(gr.skip() for _ in range(batch_size * 3 + 1)), *(gr.Checkbox(False) for _ in range(batch_size))
        else:
            merge_texts = []
            merge_audios = []
            first_itm_index = merge_index[0]
            first_itm_path = f"{os.path.splitext(self.transcriptions_list[first_itm_index][0])[0]}_{str(datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S'))}.wav"
            final_audio_list = []
            for idx in merge_index:
                merge_texts.append(self.transcriptions_list[idx][3])
                merge_audios.append(self.transcriptions_list[idx][0])
            for idx in merge_index[:0:-1]:
                self.transcriptions_list.pop(idx)
            for audio_path in merge_audios:
                final_audio_list.append(librosa.load(audio_path, sr=32000, mono=True)[0])
                final_audio_list.append(np.zeros(int(32000 * 0.3)))
            final_audio_list.pop()
            final_audio = np.concatenate(final_audio_list)
            soundfile.write(first_itm_path, final_audio, 32000)
            self.transcriptions_list[first_itm_index][0] = first_itm_path
            self.transcriptions_list[first_itm_index][3] = ",".join(merge_texts)
            return gr.Slider(maximum=self.max_index), *self.change_index(index)

    def _render_text_area(self, index, batch_size):
        i18n = self.i18n
        self.textboxes = []
        self.audios = []
        self.languages = []
        self.selections = []
        self.batch_size = batch_size
        for i in range(index, index + batch_size):
            with gr.Row(equal_height=True):
                if i <= self.max_index:
                    with gr.Column(scale=2, min_width=160):
                        textbox_tmp = gr.Textbox(
                            value=self.transcriptions_list[i][3],
                            label=i18n("Text") + f" {i}",
                            lines=2,
                            max_lines=3,
                            interactive=True,
                        )
                    with gr.Column(scale=1, min_width=160):
                        audio_tmp = gr.Audio(
                            value=self.transcriptions_list[i][0],
                            show_label=False,
                            show_download_button=False,
                            editable=False,
                            waveform_options={"show_recording_waveform": False, "show_controls": False},
                        )
                    with gr.Column(scale=1, min_width=160):
                        with gr.Group():
                            with gr.Row():
                                language_tmp = gr.Dropdown(
                                    choices=["ZH", "EN", "JA", "KO", "YUE"],
                                    value=self.transcriptions_list[i][2],
                                    allow_custom_value=True,
                                    label=i18n("文本语言"),
                                    interactive=True,
                                )
                            with gr.Row():
                                selection_tmp = gr.Checkbox(
                                    label=i18n("选择音频"),
                                )
                else:
                    with gr.Column(scale=2, min_width=160):
                        textbox_tmp = gr.Textbox(
                            label=i18n("Text") + f" {i}",
                            lines=2,
                            max_lines=3,
                            elem_id="subfix_textbox",
                            interactive=False,
                        )
                    with gr.Column(scale=1, min_width=160):
                        audio_tmp = gr.Audio(
                            streaming=True,
                            show_label=False,
                            show_download_button=False,
                            interactive=False,
                            waveform_options=WaveformOptions(show_recording_waveform=False, show_controls=False),
                        )
                    with gr.Column(scale=1, min_width=160):
                        with gr.Group():
                            with gr.Row():
                                language_tmp = gr.Dropdown(
                                    choices=["ZH", "EN", "JA", "KO", "YUE"],
                                    value=None,
                                    allow_custom_value=True,
                                    label=i18n("文本语言"),
                                    interactive=False,
                                )
                            with gr.Row():
                                selection_tmp = gr.Checkbox(
                                    label=i18n("选择音频"),
                                    interactive=False,
                                )

            self.textboxes.append(textbox_tmp)
            self.audios.append(audio_tmp)
            self.languages.append(language_tmp)
            self.selections.append(selection_tmp)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=160):
                self.close_button = gr.Button(value=i18n("保存并关闭打标WebUI"), variant="stop")
            with gr.Column(scale=1, min_width=160):
                self.previous_index_button2 = gr.Button(value=i18n("上一页"))
            with gr.Column(scale=1, min_width=160):
                self.next_index_button2 = gr.Button(value=i18n("下一页"))

        # Event Trigger Binding

        self.index_slider.release(  # Change Index Button
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
        ).success(
            fn=self.change_index,
            inputs=[
                self.index_slider,
            ],
            outputs=[
                *self.audios,
                *self.textboxes,
                *self.languages,
                *self.selections,
            ],
            max_batch_size=1,
            trigger_mode="once",
        )

        self.next_index_button1.click(  # Next Page Button on the Top
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
        ).success(
            fn=self.next_page,
            inputs=[
                self.index_slider,
            ],
            outputs=[
                self.index_slider,
                *self.audios,
                *self.textboxes,
                *self.languages,
                *self.selections,
            ],
            scroll_to_output=True,
            trigger_mode="once",
        )

        self.next_index_button2.click(  # Next Page Button on the Bottom, Binding to Next Page Button on the Top
            lambda: None,
            [],
            [],
            js="""
            () => {
            document.getElementById("btn_next").click();
            }""",
            trigger_mode="once",
        )

        self.previous_index_button1.click(  # Previous Page Button on the Top
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
        ).success(
            fn=self.previous_page,
            inputs=[
                self.index_slider,
            ],
            outputs=[
                self.index_slider,
                *self.audios,
                *self.textboxes,
                *self.languages,
                *self.selections,
            ],
            scroll_to_output=True,
            trigger_mode="once",
        )

        self.previous_index_button2.click(  # Previous Page Button on the Bottom, Binding to Previous Page Button on the Top
            lambda: None,
            [],
            [],
            js="""
            () => {
            document.getElementById("btn_previous").click();
            }""",
            trigger_mode="once",
        )

        self.delete_audio_button.click(  # Delete the Audio in the Transcription File
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
        ).success(
            fn=self.delete_audio,
            inputs=[
                self.index_slider,
                *self.selections,
            ],
            outputs=[
                self.index_slider,
                *self.audios,
                *self.textboxes,
                *self.languages,
                *self.selections,
            ],
            scroll_to_output=True,
        ).success(
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
            show_progress="hidden",
        )

        self.merge_audio_button.click(  # Delete the Audio in the Transcription File
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
        ).success(
            fn=self.merge_audio,
            inputs=[
                self.index_slider,
                *self.selections,
            ],
            outputs=[
                self.index_slider,
                *self.audios,
                *self.textboxes,
                *self.languages,
                *self.selections,
            ],
            scroll_to_output=True,
        ).success(
            fn=self.submit,
            inputs=[
                *self.textboxes,
                *self.languages,
            ],
            outputs=[],
            show_progress="hidden",
        )
        if not IS_CLI:
            self.close_button.click(  # Close the Subfix Tab, Binding to Close Button on Audio Processing Tab
                fn=lambda: None,
                inputs=[],
                outputs=[],
                js="""
                () => {
                document.getElementById("btn_close").click();
                }""",
                trigger_mode="once",
            )
        else:
            self.close_button.click(  # Close the Subfix Tab, Binding to Close Button on Audio Processing Tab
                fn=self.submit,
                inputs=[
                    *self.textboxes,
                    *self.languages,
                ],
                outputs=[],
                trigger_mode="once",
            ).then(
                fn=PARTIAL_EXIT,
                inputs=[],
                outputs=[],
            )

    def render(self, list_path: str, batch_size: int = 10):
        self.batch_size = batch_size
        self.transcriptions_list = []
        self.load_list(list_path=list_path)


@click.command(name="subfix")
@click.argument(
    "list-path",
    metavar="<Path>",
    type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
    required=True,
)
@click.option(
    "--i18n-lang",
    type=str,
    default="Auto",
    help="Languages for internationalisation",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default="9871",
    show_default=True,
)
@click.option(
    "--share",
    type=bool,
    default=False,
    show_default=True,
)
def main(list_path: str = "", i18n_lang="Auto", port=9871, share=False):
    """Web-Based audio subtitle editing and multilingual annotation Tool

    Accept a transcription list path to launch a Gradio WebUI for text editing
    """

    with gr.Blocks(analytics_enabled=False) as app:
        subfix = Subfix(I18nAuto(i18n_lang))
        subfix.render(list_path=list_path)
        if subfix.max_index >= 0:
            timer = gr.Timer(0.1)

            timer.tick(
                fn=lambda: (
                    gr.Slider(value=0, maximum=subfix.max_index, step=1),
                    gr.Slider(value=10),
                    gr.Timer(active=False),
                ),
                inputs=[],
                outputs=[
                    subfix.index_slider,
                    subfix.batch_size_slider,
                    timer,
                ],
            )
        else:
            timer = gr.Timer(2)

            timer.tick(
                fn=lambda x: (_ for _ in ()).throw(gr.Error("Invalid List")) if x is None else None,
                inputs=[],
                outputs=[],
            )
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=share,
        server_port=port,
        quiet=False,
    )


if __name__ == "__main__":
    main()
