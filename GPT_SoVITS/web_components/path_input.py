"""路径输入组件：文本框 + 「选择文件/文件夹」按钮（按钮在文本框下方且宽度自适应文字），复用 path_picker 的弹窗逻辑。"""
import gradio as gr

from GPT_SoVITS.web_components.path_picker import create_path_picker


def create_path_input(
    i18n=None,
    label=None,
    value="",
    placeholder=None,
    btn_label=None,
    mode="both",
    initial_dir=None,
    root_dir=None,
):
    """创建「文本框 + 选择文件/文件夹按钮」组件：文本框在上，按钮在下且宽度自适应文字，选中后自动填入文本框。

    Args:
        i18n: 翻译函数（如 webui 的 i18n），用于翻译按钮和对话框文案；不传则使用原文。
        label / value / placeholder: 文本框的标签、默认值、占位提示。
        btn_label: 按钮文案，默认经 i18n 翻译「选择文件/文件夹」。
        mode: 允许选择的类型，'both' 文件或文件夹、'file' 仅文件、'folder' 仅文件夹。
        initial_dir: 弹窗的初始目录；不传则用当前文本框值所在目录。
        root_dir: 项目根目录；选中路径在该目录内时填入相对路径，不传则用 path_picker 的 PROJECT_ROOT。

    Returns:
        gr.Column: 布局容器组件，子组件通过 .textbox / .button 访问。
    """
    if label is None:
        label = "路径"
        if i18n is not None:
            label = i18n(label)
    with gr.Column() as col:
        textbox = gr.Textbox(label=label, value=value, placeholder=placeholder)
        with gr.Row():
            button = create_path_picker(textbox, i18n=i18n, btn_label=btn_label, mode=mode, initial_dir=initial_dir, root_dir=root_dir)
    col.textbox = textbox
    col.button = button
    return col
