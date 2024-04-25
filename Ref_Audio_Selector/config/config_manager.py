import configparser
import re


class ConfigManager:
    def __init__(self):
        self.config_path = 'Ref_Audio_Selector/config.ini'
        self.comments = []
        self.config = None
        self.read_with_comments()

    def read_with_comments(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            self.comments = []
            for i, line in enumerate(lines):
                if line.startswith(';') or line.startswith('#'):
                    self.comments.append((i, line))

            self.config = configparser.ConfigParser()
            self.config.read_string(''.join(lines))

    def write_with_comments(self):
        output_lines = []

        # 先写入配置项
        config_str = self.config.write()
        output_lines.extend(config_str.splitlines(True))  # 保持换行

        # 然后插入原有注释
        for index, comment in sorted(self.comments, reverse=True):  # 从后往前插入，避免行号错乱
            while len(output_lines) < index + 1:
                output_lines.append('\n')  # 补充空行
            output_lines.insert(index, comment)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

    def get_base(self, key):
        return self.config.get('Base', key)

    def set_base(self, key, value):
        self.config.set('Base', key, value)
        self.write_with_comments()

    def get_audio_sample(self, key):
        return self.config.get('AudioSample', key)

    def get_inference(self, key):
        return self.config.get('Inference', key)

    def get_result_check(self, key):
        return self.config.get('ResultCheck', key)

    def get_audio_config(self, key):
        return self.config.get('AudioConfig', key)

    def get_other(self, key):
        return self.config.get('Other', key)


_config = ConfigManager()


def get_config():
    return _config
