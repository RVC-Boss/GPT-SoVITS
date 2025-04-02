import argparse
import os
import copy
import json
import uuid

try:
    import gradio.analytics as analytics

    analytics.version_check = lambda: None
except:
    ...

import librosa
import gradio as gr
import numpy as np
import soundfile

g_json_key_text = ""
g_json_key_path = ""
g_load_file = ""
g_load_format = ""

g_max_json_index = 0
g_index = 0
g_batch = 10
g_text_list = []
g_audio_list = []
g_checkbox_list = []
g_data_json = []


def reload_data(index, batch):
    global g_index
    g_index = index
    global g_batch
    g_batch = batch
    datas = g_data_json[index : index + batch]
    output = []
    for d in datas:
        output.append({g_json_key_text: d[g_json_key_text], g_json_key_path: d[g_json_key_path]})
    return output


def b_change_index(index, batch):
    global g_index, g_batch
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)
    output = []
    for i, _ in enumerate(datas):
        output.append(
            # gr.Textbox(
            #     label=f"Text {i+index}",
            #     value=_[g_json_key_text]#text
            # )
            {"__type__": "update", "label": f"Text {i + index}", "value": _[g_json_key_text]}
        )
    for _ in range(g_batch - len(datas)):
        output.append(
            # gr.Textbox(
            #     label=f"Text",
            #     value=""
            # )
            {"__type__": "update", "label": "Text", "value": ""}
        )
    for _ in datas:
        output.append(_[g_json_key_path])
    for _ in range(g_batch - len(datas)):
        output.append(None)
    for _ in range(g_batch):
        output.append(False)
    return output


def b_next_index(index, batch):
    b_save_file()
    if (index + batch) <= g_max_json_index:
        return index + batch, *b_change_index(index + batch, batch)
    else:
        return index, *b_change_index(index, batch)


def b_previous_index(index, batch):
    b_save_file()
    if (index - batch) >= 0:
        return index - batch, *b_change_index(index - batch, batch)
    else:
        return 0, *b_change_index(0, batch)


def b_submit_change(*text_list):
    global g_data_json
    change = False
    for i, new_text in enumerate(text_list):
        if g_index + i <= g_max_json_index:
            new_text = new_text.strip() + " "
            if g_data_json[g_index + i][g_json_key_text] != new_text:
                g_data_json[g_index + i][g_json_key_text] = new_text
                change = True
    if change:
        b_save_file()
    return g_index, *b_change_index(g_index, g_batch)


def b_delete_audio(*checkbox_list):
    global g_data_json, g_index, g_max_json_index
    b_save_file()
    change = False
    for i, checkbox in reversed(list(enumerate(checkbox_list))):
        if g_index + i < len(g_data_json):
            if checkbox == True:
                g_data_json.pop(g_index + i)
                change = True

    g_max_json_index = len(g_data_json) - 1
    if g_index > g_max_json_index:
        g_index = g_max_json_index
        g_index = g_index if g_index >= 0 else 0
    if change:
        b_save_file()
    # return gr.Slider(value=g_index, maximum=(g_max_json_index if g_max_json_index>=0 else 0)), *b_change_index(g_index, g_batch)
    return {
        "value": g_index,
        "__type__": "update",
        "maximum": (g_max_json_index if g_max_json_index >= 0 else 0),
    }, *b_change_index(g_index, g_batch)


def b_invert_selection(*checkbox_list):
    new_list = [not item if item is True else True for item in checkbox_list]
    return new_list


def get_next_path(filename):
    base_dir = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(100):
        new_path = os.path.join(base_dir, f"{base_name}_{str(i).zfill(2)}.wav")
        if not os.path.exists(new_path):
            return new_path
    return os.path.join(base_dir, f"{str(uuid.uuid4())}.wav")


def b_audio_split(audio_breakpoint, *checkbox_list):
    global g_data_json, g_max_json_index
    checked_index = []
    for i, checkbox in enumerate(checkbox_list):
        if checkbox == True and g_index + i < len(g_data_json):
            checked_index.append(g_index + i)
    if len(checked_index) == 1:
        index = checked_index[0]
        audio_json = copy.deepcopy(g_data_json[index])
        path = audio_json[g_json_key_path]
        data, sample_rate = librosa.load(path, sr=None, mono=True)
        audio_maxframe = len(data)
        break_frame = int(audio_breakpoint * sample_rate)

        if break_frame >= 1 and break_frame < audio_maxframe:
            audio_first = data[0:break_frame]
            audio_second = data[break_frame:]
            nextpath = get_next_path(path)
            soundfile.write(nextpath, audio_second, sample_rate)
            soundfile.write(path, audio_first, sample_rate)
            g_data_json.insert(index + 1, audio_json)
            g_data_json[index + 1][g_json_key_path] = nextpath
            b_save_file()

    g_max_json_index = len(g_data_json) - 1
    # return gr.Slider(value=g_index, maximum=g_max_json_index), *b_change_index(g_index, g_batch)
    return {"value": g_index, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(g_index, g_batch)


def b_merge_audio(interval_r, *checkbox_list):
    global g_data_json, g_max_json_index
    b_save_file()
    checked_index = []
    audios_path = []
    audios_text = []
    for i, checkbox in enumerate(checkbox_list):
        if checkbox == True and g_index + i < len(g_data_json):
            checked_index.append(g_index + i)

    if len(checked_index) > 1:
        for i in checked_index:
            audios_path.append(g_data_json[i][g_json_key_path])
            audios_text.append(g_data_json[i][g_json_key_text])
        for i in reversed(checked_index[1:]):
            g_data_json.pop(i)

        base_index = checked_index[0]
        base_path = audios_path[0]
        g_data_json[base_index][g_json_key_text] = "".join(audios_text)

        audio_list = []
        l_sample_rate = None
        for i, path in enumerate(audios_path):
            data, sample_rate = librosa.load(path, sr=l_sample_rate, mono=True)
            l_sample_rate = sample_rate
            if i > 0:
                silence = np.zeros(int(l_sample_rate * interval_r))
                audio_list.append(silence)

            audio_list.append(data)

        audio_concat = np.concatenate(audio_list)

        soundfile.write(base_path, audio_concat, l_sample_rate)

        b_save_file()

    g_max_json_index = len(g_data_json) - 1

    # return gr.Slider(value=g_index, maximum=g_max_json_index), *b_change_index(g_index, g_batch)
    return {"value": g_index, "maximum": g_max_json_index, "__type__": "update"}, *b_change_index(g_index, g_batch)


def b_save_json():
    with open(g_load_file, "w", encoding="utf-8") as file:
        for data in g_data_json:
            file.write(f"{json.dumps(data, ensure_ascii=False)}\n")


def b_save_list():
    with open(g_load_file, "w", encoding="utf-8") as file:
        for data in g_data_json:
            wav_path = data["wav_path"]
            speaker_name = data["speaker_name"]
            language = data["language"]
            text = data["text"]
            file.write(f"{wav_path}|{speaker_name}|{language}|{text}".strip() + "\n")


def b_load_json():
    global g_data_json, g_max_json_index
    with open(g_load_file, "r", encoding="utf-8") as file:
        g_data_json = file.readlines()
        g_data_json = [json.loads(line) for line in g_data_json]
        g_max_json_index = len(g_data_json) - 1


def b_load_list():
    global g_data_json, g_max_json_index
    with open(g_load_file, "r", encoding="utf-8") as source:
        data_list = source.readlines()
        for _ in data_list:
            data = _.split("|")
            if len(data) == 4:
                wav_path, speaker_name, language, text = data
                g_data_json.append(
                    {"wav_path": wav_path, "speaker_name": speaker_name, "language": language, "text": text.strip()}
                )
            else:
                print("error line:", data)
        g_max_json_index = len(g_data_json) - 1


def b_save_file():
    if g_load_format == "json":
        b_save_json()
    elif g_load_format == "list":
        b_save_list()


def b_load_file():
    if g_load_format == "json":
        b_load_json()
    elif g_load_format == "list":
        b_load_list()


def set_global(load_json, load_list, json_key_text, json_key_path, batch):
    global g_json_key_text, g_json_key_path, g_load_file, g_load_format, g_batch

    g_batch = int(batch)

    if load_json != "None":
        g_load_format = "json"
        g_load_file = load_json
    elif load_list != "None":
        g_load_format = "list"
        g_load_file = load_list
    else:
        g_load_format = "list"
        g_load_file = "demo.list"

    g_json_key_text = json_key_text
    g_json_key_path = json_key_path

    b_load_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--load_json", default="None", help="source file, like demo.json")
    parser.add_argument("--is_share", default="False", help="whether webui is_share=True")
    parser.add_argument("--load_list", default="None", help="source file, like demo.list")
    parser.add_argument("--webui_port_subfix", default=9871, help="source file, like demo.list")
    parser.add_argument("--json_key_text", default="text", help="the text key name in json, Default: text")
    parser.add_argument("--json_key_path", default="wav_path", help="the path key name in json, Default: wav_path")
    parser.add_argument("--g_batch", default=10, help="max number g_batch wav to display, Default: 10")

    args = parser.parse_args()

    set_global(args.load_json, args.load_list, args.json_key_text, args.json_key_path, args.g_batch)

    with gr.Blocks() as demo:
        with gr.Row():
            btn_change_index = gr.Button("Change Index")
            btn_submit_change = gr.Button("Submit Text")
            btn_merge_audio = gr.Button("Merge Audio")
            btn_delete_audio = gr.Button("Delete Audio")
            btn_previous_index = gr.Button("Previous Index")
            btn_next_index = gr.Button("Next Index")

        with gr.Row():
            index_slider = gr.Slider(minimum=0, maximum=g_max_json_index, value=g_index, step=1, label="Index", scale=3)
            splitpoint_slider = gr.Slider(
                minimum=0, maximum=120.0, value=0, step=0.1, label="Audio Split Point(s)", scale=3
            )
            btn_audio_split = gr.Button("Split Audio", scale=1)
            btn_save_json = gr.Button("Save File", visible=True, scale=1)
            btn_invert_selection = gr.Button("Invert Selection", scale=1)

        with gr.Row():
            with gr.Column():
                for _ in range(0, g_batch):
                    with gr.Row():
                        text = gr.Textbox(label="Text", visible=True, scale=5)
                        audio_output = gr.Audio(label="Output Audio", visible=True, scale=5)
                        audio_check = gr.Checkbox(label="Yes", show_label=True, info="Choose Audio", scale=1)
                        g_text_list.append(text)
                        g_audio_list.append(audio_output)
                        g_checkbox_list.append(audio_check)

        with gr.Row():
            batchsize_slider = gr.Slider(
                minimum=1, maximum=g_batch, value=g_batch, step=1, label="Batch Size", scale=3, interactive=False
            )
            interval_slider = gr.Slider(minimum=0, maximum=2, value=0, step=0.01, label="Interval", scale=3)
            btn_theme_dark = gr.Button("Light Theme", link="?__theme=light", scale=1)
            btn_theme_light = gr.Button("Dark Theme", link="?__theme=dark", scale=1)

        btn_change_index.click(
            b_change_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[*g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_submit_change.click(
            b_submit_change,
            inputs=[
                *g_text_list,
            ],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_previous_index.click(
            b_previous_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_next_index.click(
            b_next_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_delete_audio.click(
            b_delete_audio,
            inputs=[*g_checkbox_list],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_merge_audio.click(
            b_merge_audio,
            inputs=[interval_slider, *g_checkbox_list],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_audio_split.click(
            b_audio_split,
            inputs=[splitpoint_slider, *g_checkbox_list],
            outputs=[index_slider, *g_text_list, *g_audio_list, *g_checkbox_list],
        )

        btn_invert_selection.click(b_invert_selection, inputs=[*g_checkbox_list], outputs=[*g_checkbox_list])

        btn_save_json.click(b_save_file)

        demo.load(
            b_change_index,
            inputs=[
                index_slider,
                batchsize_slider,
            ],
            outputs=[*g_text_list, *g_audio_list, *g_checkbox_list],
        )

    demo.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        quiet=True,
        share=eval(args.is_share),
        server_port=int(args.webui_port_subfix),
    )
