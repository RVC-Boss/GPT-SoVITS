import configparser
import Ref_Audio_Selector.common.common as common


class ParamReadWriteManager:
    def __init__(self):
        self.work_dir_path = 'Ref_Audio_Selector/file/base_info/work_dir.txt'
        self.role_path = 'Ref_Audio_Selector/file/base_info/role.txt'

    def read_work_dir(self):
        content = common.read_file(self.work_dir_path)
        return content.strip()

    def read_role(self):
        content = common.read_file(self.role_path)
        return content.strip()

    def write_work_dir(self, work_dir_content):
        clean_content = work_dir_content.strip()
        common.write_text_to_file(clean_content, self.work_dir_path)

    def write_role(self, role_content):
        clean_content = role_content.strip()
        common.write_text_to_file(clean_content, self.role_path)


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
_param_read_write_manager = ParamReadWriteManager()


def get_config():
    return _config


def get_rw_param():
    return _param_read_write_manager


if __name__ == '__main__':
    print(_config.print())
