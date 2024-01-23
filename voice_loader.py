import gradio as gr
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

SUPPORT_LANGUAGE = [("中文","ZH"),("英文","EN"),("日文","JP")]

with gr.Blocks() as demo:
    with gr.Accordion(label="模型"):
        with gr.Row():
            gpt_dropdown = gr.Dropdown()
            sovits_dropdown = gr.Dropdown()
            with gr.Row():
                model_load_button = gr.Button("加载模型",variant="primary")
                model_refresh_button = gr.Button("刷新模型路径" ,variant="secondary")
    with gr.Accordion(label="参考"):
        with gr.Group():
            with gr.Row():
                with gr.Row():
                    ref_wav_path = gr.Audio(label="参考音频", type="filepath", scale=3)
                    ref_language = gr.Dropdown(choices=SUPPORT_LANGUAGE,value="ZH",label="参考语种",interactive=True,min_width=50, scale=1)
                ref_text = gr.TextArea(label="参考文本",scale=1)
    with gr.Row():
        output_language = gr.Dropdown(choices=SUPPORT_LANGUAGE,value="ZH",label="合成语种",interactive=True, scale=2)
        preprocess_output_text_button = gr.Button("合成文本预处理",variant="primary",scale=3)
    output_text = gr.TextArea(label="合成文本",interactive=True)
demo.launch(server_port=2777)

