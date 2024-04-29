import configparser
import os
import Ref_Audio_Selector.common.common as common


class ParamReadWriteManager:
    def __init__(self):
        self.base_dir = 'Ref_Audio_Selector/file/base_info'
        os.makedirs(self.base_dir, exist_ok=True)
        # 基础信息
        self.work_dir = 'work_dir'
        self.role = 'role'
        # 第一步
        self.subsection_num = 'subsection_num'
        self.sample_num = 'sample_num'
        # 第二步
        self.api_set_model_base_url = 'api_set_model_base_url'
        self.api_gpt_param = 'api_gpt_param'
        self.api_sovits_param = 'api_sovits_param'

        self.api_v2_set_gpt_model_base_url = 'api_v2_set_gpt_model_base_url'
        self.api_v2_gpt_model_param = 'api_v2_gpt_model_param'
        self.api_v2_set_sovits_model_base_url = 'api_v2_set_sovits_model_base_url'
        self.api_v2_sovits_model_param = 'api_v2_sovits_model_param'

        self.text_url = 'text_url'
        self.text_param = 'text_param'
        self.refer_type_param = 'refer_type_param'
        self.ref_path_param = 'ref_path_param'
        self.ref_text_param = 'ref_text_param'
        self.emotion_param = 'emotion_param'

        self.test_content_path = 'test_content_path'
        self.request_concurrency_num = 'request_concurrency_num'

        # 第三步
        self.text_similarity_amplification_boundary = 'text_similarity_amplification_boundary'
        # 第四步
        # 第五步
        self.text_template = 'text_template'

    def read(self, key):
        file_path = os.path.join(self.base_dir, key + '.txt')
        if os.path.exists(file_path):
            content = common.read_file(file_path)
            return content.strip()
        else:
            return ''

    def write(self, key, content):
        file_path = os.path.join(self.base_dir, key + '.txt')

        # 确保内容是字符串类型，如果不是，转换为字符串
        if not isinstance(content, str):
            clean_content = str(content).strip()  # 转换为字符串并移除首尾空白
        else:
            clean_content = content.strip()

        common.write_text_to_file(clean_content, file_path)


class ConfigManager:
    def __init__(self):
        self.config_path = 'Ref_Audio_Selector/config.ini'
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path, encoding='utf-8')

    def get_base(self, key):
        return self.config.get('Base', key)

    def get_log(self, key):
        return self.config.get('Log', key)

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
