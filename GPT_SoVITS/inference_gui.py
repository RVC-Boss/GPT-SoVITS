import os
import sys
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QWidget, QFileDialog, QStatusBar, QComboBox
import soundfile as sf

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

from inference_webui import gpt_path, sovits_path, change_gpt_weights, change_sovits_weights, get_tts_wav


class GPTSoVITSGUI(QMainWindow):
    GPT_Path = gpt_path
    SoVITS_Path = sovits_path

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GPT-SoVITS GUI")
        self.setGeometry(800, 450, 950, 850)

        self.setStyleSheet("""
            QWidget {
                background-color: #a3d3b1; 
            }

            QTabWidget::pane {
                background-color: #a3d3b1;  
            }

            QTabWidget::tab-bar {
                alignment: left;
            }

            QTabBar::tab {
                background: #8da4bf; 
                color: #ffffff;  
                padding: 8px;
            }

            QTabBar::tab:selected {
                background: #2a3f54; 
            }

            QLabel {
                color: #000000;  
            }

            QPushButton {
                background-color: #4CAF50; 
                color: white;  
                padding: 8px;
                border: 1px solid #4CAF50;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #45a049;  
                border: 1px solid #45a049;
                box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.1);
            }
        """)

        license_text = (
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. "
            "如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE."
        )
        license_label = QLabel(license_text)
        license_label.setWordWrap(True)

        self.GPT_model_label = QLabel("选择GPT模型:")
        self.GPT_model_input = QLineEdit()
        self.GPT_model_input.setPlaceholderText("拖拽或选择文件")
        self.GPT_model_input.setText(self.GPT_Path)
        self.GPT_model_input.setReadOnly(True)
        self.GPT_model_button = QPushButton("选择GPT模型文件")
        self.GPT_model_button.clicked.connect(self.select_GPT_model)

        self.SoVITS_model_label = QLabel("选择SoVITS模型:")
        self.SoVITS_model_input = QLineEdit()
        self.SoVITS_model_input.setPlaceholderText("拖拽或选择文件")
        self.SoVITS_model_input.setText(self.SoVITS_Path)
        self.SoVITS_model_input.setReadOnly(True)
        self.SoVITS_model_button = QPushButton("选择SoVITS模型文件")
        self.SoVITS_model_button.clicked.connect(self.select_SoVITS_model)

        self.ref_audio_label = QLabel("上传参考音频:")
        self.ref_audio_input = QLineEdit()
        self.ref_audio_input.setPlaceholderText("拖拽或选择文件")
        self.ref_audio_input.setReadOnly(True)
        self.ref_audio_button = QPushButton("选择音频文件")
        self.ref_audio_button.clicked.connect(self.select_ref_audio)

        self.ref_text_label = QLabel("参考音频文本:")
        self.ref_text_input = QLineEdit()
        self.ref_text_input.setPlaceholderText("直接输入文字或上传文本")
        self.ref_text_button = QPushButton("上传文本")
        self.ref_text_button.clicked.connect(self.upload_ref_text)

        self.ref_language_label = QLabel("参考音频语言:")
        self.ref_language_combobox = QComboBox()
        self.ref_language_combobox.addItems(["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"])
        self.ref_language_combobox.setCurrentText("多语种混合")

        self.target_text_label = QLabel("合成目标文本:")
        self.target_text_input = QLineEdit()
        self.target_text_input.setPlaceholderText("直接输入文字或上传文本")
        self.target_text_button = QPushButton("上传文本")
        self.target_text_button.clicked.connect(self.upload_target_text)

        self.target_language_label = QLabel("合成音频语言:")
        self.target_language_combobox = QComboBox()
        self.target_language_combobox.addItems(["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"])
        self.target_language_combobox.setCurrentText("多语种混合")

        self.output_label = QLabel("输出音频路径:")
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("拖拽或选择文件")
        self.output_input.setReadOnly(True)
        self.output_button = QPushButton("选择文件夹")
        self.output_button.clicked.connect(self.select_output_path)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        self.add_drag_drop_events(
            [
                self.GPT_model_input,
                self.SoVITS_model_input,
                self.ref_audio_input,
                self.ref_text_input,
                self.target_text_input,
                self.output_input,
            ]
        )

        self.synthesize_button = QPushButton("合成")
        self.synthesize_button.clicked.connect(self.synthesize)

        self.clear_output_button = QPushButton("清空输出")
        self.clear_output_button.clicked.connect(self.clear_output)

        self.status_bar = QStatusBar()

        main_layout = QVBoxLayout()

        input_layout = QGridLayout(self)
        input_layout.setSpacing(10)

        input_layout.addWidget(license_label, 0, 0, 1, 3)

        input_layout.addWidget(self.GPT_model_label, 1, 0)
        input_layout.addWidget(self.GPT_model_input, 2, 0, 1, 2)
        input_layout.addWidget(self.GPT_model_button, 2, 2)

        input_layout.addWidget(self.SoVITS_model_label, 3, 0)
        input_layout.addWidget(self.SoVITS_model_input, 4, 0, 1, 2)
        input_layout.addWidget(self.SoVITS_model_button, 4, 2)

        input_layout.addWidget(self.ref_audio_label, 5, 0)
        input_layout.addWidget(self.ref_audio_input, 6, 0, 1, 2)
        input_layout.addWidget(self.ref_audio_button, 6, 2)

        input_layout.addWidget(self.ref_language_label, 7, 0)
        input_layout.addWidget(self.ref_language_combobox, 8, 0, 1, 1)
        input_layout.addWidget(self.ref_text_label, 9, 0)
        input_layout.addWidget(self.ref_text_input, 10, 0, 1, 2)
        input_layout.addWidget(self.ref_text_button, 10, 2)

        input_layout.addWidget(self.target_language_label, 11, 0)
        input_layout.addWidget(self.target_language_combobox, 12, 0, 1, 1)
        input_layout.addWidget(self.target_text_label, 13, 0)
        input_layout.addWidget(self.target_text_input, 14, 0, 1, 2)
        input_layout.addWidget(self.target_text_button, 14, 2)

        input_layout.addWidget(self.output_label, 15, 0)
        input_layout.addWidget(self.output_input, 16, 0, 1, 2)
        input_layout.addWidget(self.output_button, 16, 2)

        main_layout.addLayout(input_layout)

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_text)
        main_layout.addLayout(output_layout)

        main_layout.addWidget(self.synthesize_button)

        main_layout.addWidget(self.clear_output_button)

        main_layout.addWidget(self.status_bar)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            if len(file_paths) == 1:
                self.update_ref_audio(file_paths[0])
            else:
                self.update_ref_audio(", ".join(file_paths))

    def add_drag_drop_events(self, widgets):
        for widget in widgets:
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.DragEnter, QEvent.Drop):
            mime_data = event.mimeData()
            if mime_data.hasUrls():
                event.acceptProposedAction()

        return super().eventFilter(obj, event)

    def select_GPT_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择GPT模型文件", "", "GPT Files (*.ckpt)")
        if file_path:
            self.GPT_model_input.setText(file_path)

    def select_SoVITS_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择SoVITS模型文件", "", "SoVITS Files (*.pth)")
        if file_path:
            self.SoVITS_model_input.setText(file_path)

    def select_ref_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择参考音频文件", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.update_ref_audio(file_path)

    def upload_ref_text(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                self.ref_text_input.setText(content)

    def upload_target_text(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                self.target_text_input.setText(content)

    def select_output_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly

        folder_dialog = QFileDialog()
        folder_dialog.setOptions(options)
        folder_dialog.setFileMode(QFileDialog.Directory)

        if folder_dialog.exec_():
            folder_path = folder_dialog.selectedFiles()[0]
            self.output_input.setText(folder_path)

    def update_ref_audio(self, file_path):
        self.ref_audio_input.setText(file_path)

    def clear_output(self):
        self.output_text.clear()

    def synthesize(self):
        GPT_model_path = self.GPT_model_input.text()
        SoVITS_model_path = self.SoVITS_model_input.text()
        ref_audio_path = self.ref_audio_input.text()
        language_combobox = self.ref_language_combobox.currentText()
        language_combobox = i18n(language_combobox)
        ref_text = self.ref_text_input.text()
        target_language_combobox = self.target_language_combobox.currentText()
        target_language_combobox = i18n(target_language_combobox)
        target_text = self.target_text_input.text()
        output_path = self.output_input.text()

        if GPT_model_path != self.GPT_Path:
            change_gpt_weights(gpt_path=GPT_model_path)
            self.GPT_Path = GPT_model_path
        if SoVITS_model_path != self.SoVITS_Path:
            change_sovits_weights(sovits_path=SoVITS_model_path)
            self.SoVITS_Path = SoVITS_model_path

        synthesis_result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=language_combobox,
            text=target_text,
            text_language=target_language_combobox,
        )

        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = os.path.join(output_path, "output.wav")
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)

            result = "Audio saved to " + output_wav_path

        self.status_bar.showMessage("合成完成！输出路径：" + output_wav_path, 5000)
        self.output_text.append("处理结果：\n" + result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = GPTSoVITSGUI()
    mainWin.show()
    sys.exit(app.exec_())
