import configparser
import re


class ConfigManager:
    def __init__(self):
        self.config_path = 'Ref_Audio_Selector/config.ini'
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path, encoding='utf-8')

    def get_base(self, key):
        return self.config.get('Base', key)

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

    def print(self):
        # 打印所有配置
        for section in self.config.sections():
            print('[{}]'.format(section))
            for key in self.config[section]:
                print('{} = {}'.format(key, self.config[section][key]))
            print()


_config = ConfigManager()


def get_config():
    return _config


if __name__ == '__main__':
    print(_config.print())
