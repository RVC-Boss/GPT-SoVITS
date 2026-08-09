"""路径选择组件：给文本框加一个「选择文件/文件夹」按钮，点击后弹出原生对话框选择文件或文件夹，选中后自动填入文本框。

注意：原生对话框会显示在运行 webui 的那台机器的屏幕上（本机运行时即用户屏幕）。
"""
import os
import tkinter

import gradio as gr

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))


def create_path_picker(
    textbox,
    i18n=None,
    btn_label=None,
    dialog_title=None,
    file_title=None,
    dir_title=None,
    mode="both",
    initial_dir=None,
    root_dir=None,
    scale=0,
    min_width=170,
):
    """给现有文本框加一个「选择文件/文件夹」按钮，选中后自动填入文本框。

    Args:
        textbox: 要填入路径的 gr.Textbox 组件。
        i18n: 翻译函数（如 webui 的 i18n），用于翻译按钮和对话框文案；不传则使用原文。
        btn_label: 按钮文案，默认经 i18n 翻译「选择文件/文件夹」。
        dialog_title / file_title / dir_title: 原生对话框文案，默认经 i18n 翻译。
        mode: 允许选择的类型，'both' 文件或文件夹、'file' 仅文件、'folder' 仅文件夹。
        initial_dir: 弹窗的初始目录；不传则用当前文本框值所在目录。
        root_dir: 项目根目录；选中路径在该目录内时填入相对路径，不传则用 PROJECT_ROOT。
        scale / min_width: 按钮布局参数。

    Returns:
        gr.Button: 创建的选择按钮。
    """
    if i18n is None:
        i18n = lambda x: x
    if mode not in ("both", "file", "folder"):
        mode = "both"
    if btn_label is None:
        btn_label = i18n({"both": "选择文件/文件夹", "file": "选择文件", "folder": "选择文件夹"}[mode])
    if dialog_title is None:
        dialog_title = i18n("选择文件/文件夹")
    if file_title is None:
        file_title = i18n("选择文件")
    if dir_title is None:
        dir_title = i18n("选择文件夹")

    def _pick(current_value):
        """弹出原生对话框选择文件或文件夹，返回所选路径；未选择/取消时返回原值。"""
        try:
            from tkinter import filedialog
        except ImportError:
            return current_value

        # 初始目录：优先用组件指定的 initial_dir，否则用当前文本框值所在目录
        start_dir = initial_dir
        if not start_dir and current_value:
            start_dir = current_value if os.path.isdir(current_value) else os.path.dirname(current_value)

        result = {}

        def _pick_file():
            result["path"] = filedialog.askopenfilename(title=file_title, initialdir=start_dir)
            root.destroy()

        def _pick_dir():
            result["path"] = filedialog.askdirectory(title=dir_title, initialdir=start_dir)
            root.destroy()

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if mode == "file":
            _pick_file()
        elif mode == "folder":
            _pick_dir()
        else:
            chooser = tkinter.Toplevel(root)
            chooser.title(dialog_title)
            chooser.attributes("-topmost", True)
            chooser.resizable(False, False)
            chooser.protocol("WM_DELETE_WINDOW", root.destroy)
            tkinter.Button(chooser, text=file_title, width=22, command=_pick_file).pack(padx=24, pady=(14, 6))
            tkinter.Button(chooser, text=dir_title, width=22, command=_pick_dir).pack(padx=24, pady=(0, 14))

            chooser.update_idletasks()
            w, h = chooser.winfo_width(), chooser.winfo_height()
            x = (chooser.winfo_screenwidth() - w) // 2
            y = (chooser.winfo_screenheight() - h) // 3
            chooser.geometry("+%d+%d" % (x, y))

            root.mainloop()

        try:
            root.destroy()
        except tkinter.TclError:
            pass
        # 取消时 askopenfilename/askdirectory 返回空串，保留文本框原值不清空
        selected = result.get("path")
        if not selected:
            return current_value
        # 选中路径在项目根目录内则填相对路径，否则保留绝对路径
        base = os.path.normpath(root_dir or PROJECT_ROOT)
        try:
            if os.path.commonpath([selected, base]) == base:
                selected = os.path.relpath(selected, base)
        except ValueError:
            pass  # 不同盘符等无法计算相对路径，保留绝对路径
        return selected

    button = gr.Button(value=btn_label, scale=scale, min_width=min_width)
    button.click(_pick, [textbox], [textbox], show_progress=False)
    return button
