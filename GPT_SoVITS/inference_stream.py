import os
import tempfile, io, wave
import gradio as gr
import uvicorn
import argparse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import (
    get_weights_names,
    custom_sort_key,
    change_choices,
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)

api_app = FastAPI()
i18n = I18nAuto()

# API mode Usage: python GPT_SoVITS/inference_stream.py --api
parser = argparse.ArgumentParser(description="GPT-SoVITS Streaming API")
parser.add_argument(
    "-api",
    "--api",
    action="store_true",
    default=False,
    help="是否开启API模式(不开启则是WebUI模式)",
)
parser.add_argument(
    "-s",
    "--sovits_path",
    type=str,
    default="GPT_SoVITS/pretrained_models/s2G488k.pth",
    help="SoVITS模型路径",
)
parser.add_argument(
    "-g",
    "--gpt_path",
    type=str,
    default="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    help="GPT模型路径",
)
parser.add_argument(
    "-rw",
    "--ref_wav",
    type=str,
    # default="./example/archive_ruanmei_8.wav",
    help="参考音频路径",
)
parser.add_argument(
    "-rt",
    "--prompt_text",
    type=str,
    # default="我听不惯现代乐，听戏却极易入迷，琴弦拨动，时间便流往过去。",
    help="参考音频文本",
)
parser.add_argument(
    "-rl",
    "--prompt_language",
    type=str,
    default="中文",
    help="参考音频语种",
)

args = parser.parse_args()

sovits_path = args.sovits_path
gpt_path = args.gpt_path
SoVITS_names, GPT_names = get_weights_names()

EXAMPLES = [
    [
        "中文",
        "根据过年的传说，远古时代有一隻凶残年兽，每到岁末就会从海底跑出来吃人。"
        + "人们为了不被年兽吃掉，家家户户都会祭拜祖先祈求平安，也会聚在一起吃一顿丰盛的晚餐。"
        + "后来人们发现年兽害怕红色、噪音与火光，便开始在当天穿上红衣、门上贴上红纸、燃烧爆竹声，藉此把年兽赶走。"
        + "而这些之后也成为过年吃团圆饭、穿红衣、放鞭炮、贴春联的过年习俗。",
    ],
    [
        "中文",
        "神霄折戟录其二"
        +"「嗯，好吃。」被附体的未央变得温柔了许多，也冷淡了很多。"
        + "她拿起弥耳做的馅饼，小口小口吃了起来。第一口被烫到了,还很可爱地吐着舌头吸气。"
        + "「我一下子有点接受不了, 需要消化消化。」用一只眼睛作为代价维持降灵的弥耳自己也拿了一个馅饼，「你再说一 遍?」"
        + "「当年所谓的陨铁其实是神戟。它被凡人折断，铸成魔剑九柄。这一把是雾海魔剑。 加上他们之前已经收集了两柄。「然后你是?」"
        + "「我是曾经的天帝之女，名字已经忘了。我司掌审判与断罪，用你们的话说，就是刑律。」"
        + "因为光禄寺执掌祭祀典礼的事情，所以仪式、祝词什么的，弥耳被老爹逼得倒是能倒背如流。同时因为尽是接触怪力乱神，弥耳也是知道一些小]道的。 神明要是被知道了真正的秘密名讳，就只能任人驱使了。眼前这位未必是忘了。"
        + "「所以朝廷是想重铸神霄之戟吗?」弥耳说服自己接受了这个设定，追问道。"
        + "「我不知道。这具身体的主人并不知道别的事。她只是很愤怒,想要证明自己。」未央把手放在了胸口上。"
        +"「那接下来,我是应该弄个什么送神仪式把你送走吗?」弥耳摸了摸绷带下已经失去功能的眼睛，「然后我的眼睛也会回来?」"
    ],
]


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def get_streaming_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut,
    top_k,
    top_p,
    temperature,
    ref_free,
    byte_stream=True,
):
    chunks = get_tts_wav(
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language=text_language,
        how_to_cut=how_to_cut,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        ref_free=ref_free,
        stream=True,
    )

    if byte_stream:
        yield wave_header_chunk()
        for chunk in chunks:
            yield chunk
    else:
        # Send chunk files
        i = 0
        format = "wav"
        for chunk in chunks:
            i += 1
            file = f"{tempfile.gettempdir()}/{i}.{format}"
            segment = AudioSegment(chunk, frame_rate=32000, sample_width=2, channels=1)
            segment.export(file, format=format)
            yield file


def webui():
    with gr.Blocks(title="GPT-SoVITS Streaming Demo") as app:
        gr.Markdown(
            value=i18n(
                "流式输出演示，分句推理后推送到组件中。由于目前bytes模式的限制，采用<a href='https://github.com/gradio-app/gradio/blob/gradio%404.17.0/demo/stream_audio_out/run.py'>stream_audio_out</a>中临时文件的方案输出分句。这种方式相比bytes，会增加wav文件解析的延迟。"
            ),
        )

        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=sorted(GPT_names, key=custom_sort_key),
                value=gpt_path,
                interactive=True,
            )
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=sorted(SoVITS_names, key=custom_sort_key),
                value=sovits_path,
                interactive=True,
            )
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(
                fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown]
            )
            SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
            GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

        gr.Markdown(value=i18n("*请上传并填写参考信息"))
        with gr.Row():
            inp_ref = gr.Audio(
                label=i18n("请上传3~10秒内参考音频，超过会报错！"), value=args.ref_wav, type="filepath"
            )
            with gr.Column():
                ref_text_free = gr.Checkbox(
                    label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"),
                    value=False,
                    interactive=True,
                    show_label=True,
                )
                gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT"))
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value=args.prompt_text)
            prompt_language = gr.Dropdown(
                label=i18n("参考音频的语种"),
                choices=[
                    i18n("中文"),
                    i18n("英文"),
                    i18n("日文"),
                    i18n("中英混合"),
                    i18n("日英混合"),
                    i18n("多语种混合"),
                ],
                value=args.prompt_language,
            )

            def load_text(file):
                with open(file.name, "r", encoding="utf-8") as file:
                    return file.read()

            load_button = gr.UploadButton(i18n("加载参考文本"), variant="secondary")
            load_button.upload(load_text, load_button, prompt_text)

        gr.Markdown(
            value=i18n(
                "*请填写需要合成的目标文本。中英混合选中文，日英混合选日文，中日混合暂不支持，非目标语言文本自动遗弃。"
            )
        )
        with gr.Row():
            text = gr.Textbox(
                label=i18n("需要合成的文本"), value="", lines=5, interactive=True
            )
            text_language = gr.Dropdown(
                label=i18n("需要合成的语种"),
                choices=[
                    i18n("中文"),
                    i18n("英文"),
                    i18n("日文"),
                    i18n("中英混合"),
                    i18n("日英混合"),
                    i18n("多语种混合"),
                ],
                value=i18n("中文"),
            )
            how_to_cut = gr.Radio(
                label=i18n("怎么切"),
                choices=[
                    i18n("不切"),
                    i18n("凑四句一切"),
                    i18n("凑50字一切"),
                    i18n("按中文句号。切"),
                    i18n("按英文句号.切"),
                    i18n("按标点符号切"),
                ],
                value=i18n("按标点符号切"),
                interactive=True,
            )

        gr.Markdown(value=i18n("* 参数设置"))
        with gr.Row():
            with gr.Column():
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    label=i18n("top_k"),
                    value=5,
                    interactive=True,
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label=i18n("top_p"),
                    value=1,
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label=i18n("temperature"),
                    value=1,
                    interactive=True,
                )
            inference_button = gr.Button(i18n("合成语音"), variant="primary")

        gr.Markdown(value=i18n("* 结果输出(等待第2句推理结束后会自动播放)"))
        with gr.Row():
            audio_file = gr.Audio(
                value=None,
                label=i18n("输出的语音"),
                streaming=True,
                autoplay=True,
                interactive=False,
                show_label=True,
            )

        inference_button.click(
            get_streaming_tts_wav,
            [
                inp_ref,
                prompt_text,
                prompt_language,
                text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
                ref_text_free,
            ],
            [audio_file],
        ).then(lambda: gr.update(interactive=True), None, [text], queue=False)

        with gr.Row():
            gr.Examples(
                EXAMPLES,
                [text_language, text],
                cache_examples=False,
                run_on_click=False,  # Will not work , user should submit it
            )

    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=8080,
        quiet=True,
    )


@api_app.get("/")
async def tts(
    text: str,  # 必选参数
    language: str = i18n("中文"),
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
):
    ref_wav_path = args.ref_wav
    prompt_text = args.prompt_text
    prompt_language = args.prompt_language
    how_to_cut = i18n("按标点符号切")

    return StreamingResponse(
        get_streaming_tts_wav(
            ref_wav_path=ref_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=language,
            how_to_cut=how_to_cut,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            ref_free=False,
            byte_stream=True,
        ),
        media_type="audio/x-wav",
    )


def api():
    uvicorn.run(
        app="inference_stream:api_app", host="127.0.0.1", port=8080, reload=True
    )


if __name__ == "__main__":
    # 模式选择，默认是webui模式
    if not args.api:
        webui()
    else:
        api()
