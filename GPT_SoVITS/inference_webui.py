import os
from tools.i18n.i18n import I18nAuto
from inference_base import sovits_path, gpt_path, change_choices, GPT_names, custom_sort_key, SoVITS_names, change_sovits_weights, change_gpt_weights, get_tts_wav, cut1, cut2, cut3, cut4, cut5
import gradio as gr

i18n = I18nAuto()
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)

with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    with gr.Group():
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True)
            SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path, interactive=True)
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
            SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
            GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])
        gr.Markdown(value=i18n("*请上传并填写参考信息"))
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath")
            with gr.Column():
                ref_text_free = gr.Checkbox(label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"), value=False, interactive=True, show_label=True)
                gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT"))
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="")
            prompt_language = gr.Dropdown(
                label=i18n("参考音频的语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
            )
        gr.Markdown(value=i18n("*请填写需要合成的目标文本。中英混合选中文，日英混合选日文，中日混合暂不支持，非目标语言文本自动遗弃。"))
        with gr.Row():
            text = gr.Textbox(label=i18n("需要合成的文本"), value="")
            text_language = gr.Dropdown(
                label=i18n("需要合成的语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
            )
            how_to_cut = gr.Radio(
                label=i18n("怎么切"),
                choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                value=i18n("凑四句一切"),
                interactive=True,
            )
            with gr.Row():
                gr.Markdown("gpt采样参数(无参考文本时不要太低)：")
                top_k = gr.Slider(minimum=1,maximum=100,step=1,label=i18n("top_k"),value=5,interactive=True)
                top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("top_p"),value=1,interactive=True)
                temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("temperature"),value=1,interactive=True)
            inference_button = gr.Button(i18n("合成语音"), variant="primary")
            output = gr.Audio(label=i18n("输出的语音"))

        inference_button.click(
            get_tts_wav,
            [inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_text_free],
            [output],
        )

        gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="")
            button1 = gr.Button(i18n("凑四句一切"), variant="primary")
            button2 = gr.Button(i18n("凑50字一切"), variant="primary")
            button3 = gr.Button(i18n("按中文句号。切"), variant="primary")
            button4 = gr.Button(i18n("按英文句号.切"), variant="primary")
            button5 = gr.Button(i18n("按标点符号切"), variant="primary")
            text_opt = gr.Textbox(label=i18n("切分后文本"), value="")
            button1.click(cut1, [text_inp], [text_opt])
            button2.click(cut2, [text_inp], [text_opt])
            button3.click(cut3, [text_inp], [text_opt])
            button4.click(cut4, [text_inp], [text_opt])
            button5.click(cut5, [text_inp], [text_opt])
        gr.Markdown(value=i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

app.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=is_share,
    server_port=infer_ttswebui,
    quiet=True,
)
