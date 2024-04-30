import os
import multiprocessing
import Ref_Audio_Selector.config_param.config_params as params
import Ref_Audio_Selector.tool.audio_inference as audio_inference
import Ref_Audio_Selector.common.common as common

rw_param = params.config_manager.get_rw_param()
# -------------------基本信息---------------------------

# 角色所在工作目录
base_dir_default = None
# 工作目录
text_work_space_dir_default = None
# 角色名称
text_role_default = None
# 参考音频所在目录
text_refer_audio_file_dir_default = None
# 推理音频所在目录
text_inference_audio_file_dir_default = None

# -------------------第一步------------------------------

# 参考音频抽样目录
text_sample_dir_default = None
# 分段数
slider_subsection_num_default = None
# 每段随机抽样个数
slider_sample_num_default = None

# -------------------第二步------------------------------

# api服务模型切换接口地址
text_api_set_model_base_url_default = None
# GPT模型参数名
text_api_gpt_param_default = None
# SoVITS模型参数名
text_api_sovits_param_default = None
# api服务GPT模型切换接口地址
text_api_v2_set_gpt_model_base_url_default = None
# GPT模型参数名
text_api_v2_gpt_model_param_default = None
# api服务SoVITS模型切换接口地址
text_api_v2_set_sovits_model_base_url_default = None
# SoVITS模型参数名
text_api_v2_sovits_model_param_default = None
# 推理服务请求地址与参数
text_url_default = None
# 推理服务请求完整地址
text_whole_url_default = None
# 文本参数名
text_text_default = None
# 参考参数类型
dropdown_refer_type_param_default = None
# 参考音频路径参数名
text_ref_path_default = None
# 参考音频文本参数名
text_ref_text_default = None
# 角色情绪参数名
text_emotion_default = None
# 待推理文本路径
text_test_content_default = None
# 请求并发数
slider_request_concurrency_num_default = 3
# 最大并发数
slider_request_concurrency_max_num = None

# -------------------第三步------------------------------

# 待asr的音频所在目录
text_asr_audio_dir_default = None
# 待分析的文件路径
text_text_similarity_analysis_path_default = None
# 文本相似度放大边界
slider_text_similarity_amplification_boundary_default = 0.90
# 文本相似度分析结果文件所在路径
text_text_similarity_result_path_default = None

# -------------------第四步------------------------------
# -------------------第五步------------------------------
# 模板内容
text_template_default = None


def empty_default(vale, default_value):
    if vale is None or vale == "":
        return default_value
    else:
        return vale


def init_base():
    global text_work_space_dir_default, text_role_default, base_dir_default, text_refer_audio_file_dir_default, text_inference_audio_file_dir_default

    text_work_space_dir_default = rw_param.read(rw_param.work_dir)
    text_role_default = rw_param.read(rw_param.role)
    base_dir_default = os.path.join(text_work_space_dir_default, text_role_default)

    text_refer_audio_file_dir_default = common.check_path_existence_and_return(
        os.path.join(base_dir_default, params.reference_audio_dir))

    text_inference_audio_file_dir_default = common.check_path_existence_and_return(
        os.path.join(base_dir_default, params.inference_audio_dir))


def init_first():
    global text_sample_dir_default, slider_subsection_num_default, slider_sample_num_default

    text_sample_dir_default = common.check_path_existence_and_return(
        os.path.join(base_dir_default, params.list_to_convert_reference_audio_dir))

    slider_subsection_num_default = empty_default(rw_param.read(rw_param.subsection_num), 5)

    slider_sample_num_default = empty_default(rw_param.read(rw_param.sample_num), 4)


def init_second():
    global text_api_set_model_base_url_default, text_api_gpt_param_default, text_api_sovits_param_default, text_api_v2_set_gpt_model_base_url_default, text_api_v2_gpt_model_param_default
    global text_api_v2_set_sovits_model_base_url_default, text_api_v2_sovits_model_param_default, text_url_default, text_whole_url_default, text_text_default, dropdown_refer_type_param_default, text_ref_path_default
    global text_ref_text_default, text_emotion_default, text_test_content_default, slider_request_concurrency_num_default, slider_request_concurrency_max_num

    text_api_set_model_base_url_default = empty_default(rw_param.read(rw_param.api_set_model_base_url),
                                                        'http://localhost:9880/set_model')
    text_api_gpt_param_default = empty_default(rw_param.read(rw_param.api_gpt_param), 'gpt_model_path')
    text_api_sovits_param_default = empty_default(rw_param.read(rw_param.api_sovits_param), 'sovits_model_path')

    text_api_v2_set_gpt_model_base_url_default = empty_default(rw_param.read(rw_param.api_v2_set_gpt_model_base_url),
                                                               'http://localhost:9880/set_gpt_weights')
    text_api_v2_gpt_model_param_default = empty_default(rw_param.read(rw_param.api_v2_gpt_model_param), 'weights_path')

    text_api_v2_set_sovits_model_base_url_default = empty_default(
        rw_param.read(rw_param.api_v2_set_sovits_model_base_url), 'http://localhost:9880/set_sovits_weights')
    text_api_v2_sovits_model_param_default = empty_default(rw_param.read(rw_param.api_v2_sovits_model_param), 'weights_path')

    text_url_default = empty_default(rw_param.read(rw_param.text_url),
                                     'http://localhost:9880?prompt_language=中文&text_language=中文&cut_punc=,.;?!、，。？！;：…')
    text_text_default = empty_default(rw_param.read(rw_param.text_param), 'text')
    dropdown_refer_type_param_default = empty_default(rw_param.read(rw_param.refer_type_param), '参考音频')

    text_ref_path_default = empty_default(rw_param.read(rw_param.ref_path_param), 'refer_wav_path')
    text_ref_text_default = empty_default(rw_param.read(rw_param.ref_text_param), 'prompt_text')
    text_emotion_default = empty_default(rw_param.read(rw_param.emotion_param), 'emotion')

    text_whole_url_default = whole_url(text_url_default, dropdown_refer_type_param_default, text_text_default,
                                       text_ref_path_default, text_ref_text_default, text_emotion_default)

    text_test_content_default = empty_default(rw_param.read(rw_param.test_content_path), params.default_test_text_path)

    slider_request_concurrency_max_num = multiprocessing.cpu_count()

    slider_request_concurrency_num_default = empty_default(rw_param.read(rw_param.request_concurrency_num), 3)

    slider_request_concurrency_num_default = min(int(slider_request_concurrency_num_default), slider_request_concurrency_max_num)


# 基于请求路径和参数，合成完整的请求路径
def whole_url(text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text, text_emotion):
    url_composer = audio_inference.TTSURLComposer(text_url, dropdown_refer_type_param, text_emotion, text_text,
                                                  text_ref_path, text_ref_text)
    if url_composer.is_emotion():
        text_whole_url = url_composer.build_url_with_emotion('测试内容', '情绪类型', False)
    else:
        text_whole_url = url_composer.build_url_with_ref('测试内容', '参考路径', '参考文本', False)
    return text_whole_url


def init_third():
    global text_asr_audio_dir_default, text_text_similarity_analysis_path_default, slider_text_similarity_amplification_boundary_default, text_text_similarity_result_path_default

    text_asr_audio_dir_default = common.check_path_existence_and_return(
        os.path.join(base_dir_default, params.inference_audio_dir, params.inference_audio_text_aggregation_dir))
    text_text_similarity_analysis_path_default = common.check_path_existence_and_return(
        os.path.join(base_dir_default, params.asr_filename + '.list'))
    slider_text_similarity_amplification_boundary_default = empty_default(
        rw_param.read(rw_param.text_similarity_amplification_boundary), 0.90)
    text_text_similarity_result_path_default = common.check_path_existence_and_return(
        os.path.join(base_dir_default, params.text_emotion_average_similarity_report_filename + '.txt'))


def init_fourth():
    pass


def init_fifth():
    global text_template_default

    default_template_path = params.default_template_path
    text_template_default = empty_default(rw_param.read(rw_param.text_template),
                                          common.read_file(default_template_path))


def init_all():
    init_base()
    init_first()
    init_second()
    init_third()
    init_fourth()
    init_fifth()
