import os.path
import os
import traceback

import gradio as gr

from Ref_Audio_Selector.config_param.log_config import logger

import Ref_Audio_Selector.common.model_manager as model_manager
import Ref_Audio_Selector.tool.audio_sample as audio_sample
import Ref_Audio_Selector.tool.audio_inference as audio_inference
import Ref_Audio_Selector.tool.audio_config as audio_config
import Ref_Audio_Selector.tool.audio_check as audio_check
import Ref_Audio_Selector.tool.text_check as text_check
import Ref_Audio_Selector.common.common as common
import Ref_Audio_Selector.config_param.config_params as params
import Ref_Audio_Selector.common.time_util as time_util
import Ref_Audio_Selector.ui_init.init_ui_param as init

from tools.i18n.i18n import I18nAuto
from config import python_exec, is_half
from tools import my_utils
from tools.asr.config import asr_dict
from subprocess import Popen

i18n = I18nAuto()
rw_param = params.config_manager.get_rw_param()

p_similarity = None
p_asr = None
p_text_similarity = None


# 校验基础信息
def check_base_info(text_work_space_dir, text_role):
    if text_work_space_dir is None or text_work_space_dir == '':
        raise Exception("工作目录不能为空")
    if text_role is None or text_role == '':
        raise Exception("角色不能为空")
    base_role_dir = os.path.join(text_work_space_dir, text_role)
    # 判断目录是否存在
    if not os.path.exists(base_role_dir):
        # 如果不存在，则创建目录
        os.makedirs(base_role_dir, exist_ok=True)
    return base_role_dir


# 从list文件，提取参考音频
def convert_from_list(text_work_space_dir, text_role, text_list_input):
    text_work_space_dir, text_list_input = common.batch_clean_paths([text_work_space_dir, text_list_input])

    text_convert_from_list_info = None
    text_sample_dir = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_list_input is None or text_list_input == '':
            raise Exception("list文件路径不能为空")

        ref_audio_all = os.path.join(base_role_dir,
                                     params.list_to_convert_reference_audio_dir)

        time_consuming, _ = time_util.time_monitor(audio_sample.convert_from_list)(text_list_input, ref_audio_all)

        text_convert_from_list_info = f"耗时：{time_consuming:0.1f}秒；转换成功：生成目录{ref_audio_all}"
        text_sample_dir = ref_audio_all

    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_convert_from_list_info = f"发生异常：{e}"
        text_sample_dir = ''
    return i18n(text_convert_from_list_info), text_sample_dir


def start_similarity_analysis(work_space_dir, sample_dir, speaker_verification, base_voice_path,
                              need_similarity_output):
    similarity_list = None
    similarity_file_dir = None

    similarity_dir = os.path.join(work_space_dir, params.audio_similarity_dir)
    os.makedirs(similarity_dir, exist_ok=True)

    base_voice_file_name = common.get_filename_without_extension(base_voice_path)
    similarity_file = os.path.join(similarity_dir, f'{base_voice_file_name}.txt')

    global p_similarity
    if p_similarity is None:
        cmd = f'"{python_exec}" Ref_Audio_Selector/tool/speaker_verification/voice_similarity.py '
        cmd += f' -r "{base_voice_path}"'
        cmd += f' -c "{sample_dir}"'
        cmd += f' -o "{similarity_file}"'
        cmd += f' -m {speaker_verification}'

        logger.info(cmd)
        p_similarity = Popen(cmd, shell=True)
        p_similarity.wait()

        similarity_list = audio_sample.parse_similarity_file(similarity_file)

        if need_similarity_output:
            similarity_file_dir = os.path.join(similarity_dir, base_voice_file_name)
            audio_sample.copy_and_move(similarity_file_dir, similarity_list)

        p_similarity = None
        return similarity_list, similarity_file, similarity_file_dir
    else:
        return similarity_list, None, None


# 基于一个基准音频，从参考音频目录中进行分段抽样
def sample(text_work_space_dir, text_role, text_sample_dir, dropdown_speaker_verification, text_base_voice_path,
           slider_subsection_num, slider_sample_num, checkbox_similarity_output):
    text_work_space_dir, text_sample_dir, text_base_voice_path \
        = common.batch_clean_paths([text_work_space_dir, text_sample_dir, text_base_voice_path])

    ref_audio_dir = None
    text_sample_info = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_sample_dir is None or text_sample_dir == '':
            raise Exception("参考音频抽样目录不能为空，请先完成上一步操作")
        if text_base_voice_path is None or text_base_voice_path == '':
            raise Exception("基准音频路径不能为空")
        if slider_subsection_num is None or slider_subsection_num == '':
            raise Exception("分段数不能为空")
        if slider_sample_num is None or slider_sample_num == '':
            raise Exception("每段随机抽样个数不能为空")
        if dropdown_speaker_verification is None or dropdown_speaker_verification == '':
            raise Exception("说话人确认算法不能为空")

        ref_audio_dir = os.path.join(base_role_dir, params.reference_audio_dir)

        time_consuming, (similarity_list, _, _) \
            = time_util.time_monitor(start_similarity_analysis)(base_role_dir, text_sample_dir,
                                                                dropdown_speaker_verification, text_base_voice_path,
                                                                checkbox_similarity_output)

        text_sample_info = f"耗时：{time_consuming:0.1f}秒；抽样成功：生成目录{ref_audio_dir}"

        if similarity_list is None:
            raise Exception("相似度分析失败")

        audio_sample.sample(ref_audio_dir, similarity_list, slider_subsection_num, slider_sample_num)

    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_sample_info = f"发生异常：{e}"
        ref_audio_dir = ''
    text_refer_audio_file_dir = ref_audio_dir
    return i18n(text_sample_info), text_refer_audio_file_dir


# 根据参考音频和测试文本，执行批量推理
def model_inference(text_work_space_dir, text_role, slider_request_concurrency_num, text_refer_audio_file_dir,
                    text_url, dropdown_refer_type_param,
                    text_text, text_ref_path, text_ref_text, text_emotion,
                    text_test_content_dir):
    text_work_space_dir, text_refer_audio_file_dir, text_test_content_dir \
        = common.batch_clean_paths([text_work_space_dir, text_refer_audio_file_dir, text_test_content_dir])

    inference_dir = None
    text_asr_audio_dir = None
    text_model_inference_info = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_refer_audio_file_dir is None or text_refer_audio_file_dir == '':
            raise Exception("待推理的参考音频所在目录不能为空，请先完成上一步操作")
        if text_url is None or text_url == '':
            raise Exception("推理服务请求地址不能为空")
        if text_text is None or text_text == '':
            raise Exception("文本参数名不能为空")
        if text_test_content_dir is None or text_test_content_dir == '':
            raise Exception("待推理文本路径不能为空")
        if (text_ref_path is None or text_ref_path == '') and (text_ref_text is None or text_ref_text == '') and (
                text_emotion is None or text_emotion == ''):
            raise Exception("参考音频路径/文本和角色情绪二选一填写，不能全部为空")

        inference_dir = os.path.join(base_role_dir, params.inference_audio_dir)
        text_asr_audio_dir = os.path.join(inference_dir,
                                          params.inference_audio_text_aggregation_dir)

        url_composer = audio_inference.TTSURLComposer(text_url, dropdown_refer_type_param, text_emotion, text_text,
                                                      text_ref_path, text_ref_text)
        url_composer.is_valid()
        text_list = common.read_text_file_to_list(text_test_content_dir)
        if text_list is None or len(text_list) == 0:
            raise Exception("待推理文本内容不能为空")
        ref_audio_manager = common.RefAudioListManager(text_refer_audio_file_dir)
        if len(ref_audio_manager.get_audio_list()) == 0:
            raise Exception("待推理的参考音频不能为空")

        time_consuming, _ = time_util.time_monitor(audio_inference.generate_audio_files_parallel)(url_composer,
                                                                                                  text_list,
                                                                                                  ref_audio_manager.get_ref_audio_list(),
                                                                                                  inference_dir,
                                                                                                  slider_request_concurrency_num)

        text_model_inference_info = f"耗时：{time_consuming:0.1f}秒；推理成功：生成目录{inference_dir}"

    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_model_inference_info = f"发生异常：{e}"
        text_asr_audio_dir = ''
    return i18n(text_model_inference_info), text_asr_audio_dir, inference_dir


# 对推理生成音频执行asr
def asr(text_work_space_dir, text_role, text_asr_audio_dir, dropdown_asr_model,
        dropdown_asr_size, dropdown_asr_lang):
    text_work_space_dir, text_asr_audio_dir \
        = common.batch_clean_paths([text_work_space_dir, text_asr_audio_dir])

    asr_file = None
    text_text_similarity_analysis_path = None
    text_asr_info = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_asr_audio_dir is None or text_asr_audio_dir == '':
            raise Exception("待asr的音频所在目录不能为空，请先完成上一步操作")
        if dropdown_asr_model is None or dropdown_asr_model == '':
            raise Exception("asr模型不能为空")
        if dropdown_asr_size is None or dropdown_asr_size == '':
            raise Exception("asr模型大小不能为空")
        if dropdown_asr_lang is None or dropdown_asr_lang == '':
            raise Exception("asr语言不能为空")

        time_consuming, asr_file = time_util.time_monitor(open_asr)(text_asr_audio_dir, base_role_dir,
                                                                    dropdown_asr_model, dropdown_asr_size,
                                                                    dropdown_asr_lang)

        text_text_similarity_analysis_path = asr_file
        text_asr_info = f"耗时：{time_consuming:0.1f}秒；asr成功：生成文件{asr_file}"
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_asr_info = f"发生异常：{e}"
        text_text_similarity_analysis_path = ''
    return i18n(text_asr_info), text_text_similarity_analysis_path


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang):
    global p_asr
    if p_asr is None:
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_py_path = asr_dict[asr_model]["path"]
        if asr_py_path == 'funasr_asr.py':
            asr_py_path = 'funasr_asr_multi_level_dir.py'
        if asr_py_path == 'fasterwhisper_asr.py':
            asr_py_path = 'fasterwhisper_asr_multi_level_dir.py'
        cmd = f'"{python_exec}" Ref_Audio_Selector/tool/asr/{asr_py_path} '
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += " -p %s" % ("float16" if is_half == True else "float32")

        logger.info(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None

        output_dir_abs = os.path.abspath(asr_opt_dir)
        output_file_name = os.path.basename(asr_inp_dir)
        # 构造输出文件路径
        output_file_path = os.path.join(output_dir_abs, f'{params.asr_filename}.list')
        return output_file_path

    else:
        return None


# 对asr生成的文件，与原本的文本内容，进行相似度分析
def text_similarity_analysis(text_work_space_dir, text_role, slider_text_similarity_amplification_boundary,
                             text_text_similarity_analysis_path):
    text_work_space_dir, text_text_similarity_analysis_path \
        = common.batch_clean_paths([text_work_space_dir, text_text_similarity_analysis_path])

    similarity_dir = None
    text_text_similarity_analysis_info = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_text_similarity_analysis_path is None or text_text_similarity_analysis_path == '':
            raise Exception("asr生成的文件路径不能为空，请先完成上一步操作")
        similarity_dir = os.path.join(base_role_dir, params.text_similarity_output_dir)

        time_consuming, _ = time_util.time_monitor(open_text_similarity_analysis)(text_text_similarity_analysis_path,
                                                                                  similarity_dir,
                                                                                  slider_text_similarity_amplification_boundary)

        average_similarity_file = os.path.join(similarity_dir,
                                               f'{params.text_emotion_average_similarity_report_filename}.txt')

        text_text_similarity_analysis_info = f"耗时：{time_consuming:0.1f}秒；相似度分析成功：生成目录{similarity_dir}"

    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_text_similarity_analysis_info = f"发生异常：{e}"
    return i18n(text_text_similarity_analysis_info), average_similarity_file


def open_text_similarity_analysis(asr_file_path, output_dir, similarity_enlarge_boundary=0.9):
    global p_text_similarity
    if p_text_similarity is None:
        cmd = f'"{python_exec}" Ref_Audio_Selector/tool/text_comparison/asr_text_process.py '
        cmd += f' -a "{asr_file_path}"'
        cmd += f' -o "{output_dir}"'
        cmd += f' -b {similarity_enlarge_boundary}'

        logger.info(cmd)
        p_text_similarity = Popen(cmd, shell=True)
        p_text_similarity.wait()
        p_text_similarity = None

        return output_dir

    else:
        return None


hide_voice_similarity_dir = ''


# 根据一个参考音频，对指定目录下的音频进行相似度分析，并输出到另一个目录
def similarity_audio_output(text_work_space_dir, text_role, text_base_audio_path,
                            text_compare_audio_dir, dropdown_speaker_verification):
    global hide_voice_similarity_dir
    text_work_space_dir, text_base_audio_path, text_compare_audio_dir \
        = common.batch_clean_paths([text_work_space_dir, text_base_audio_path, text_compare_audio_dir])

    text_similarity_audio_output_info = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_base_audio_path is None or text_base_audio_path == '':
            raise Exception("基准音频路径不能为空")
        if text_compare_audio_dir is None or text_compare_audio_dir == '':
            raise Exception("待分析的音频所在目录不能为空")
        if dropdown_speaker_verification is None or dropdown_speaker_verification == '':
            raise Exception("说话人验证模型不能为空")

        time_consuming, (similarity_list, similarity_file, similarity_file_dir) \
            = time_util.time_monitor(start_similarity_analysis)(base_role_dir, text_compare_audio_dir,
                                                                dropdown_speaker_verification, text_base_audio_path,
                                                                True)

        if similarity_list is None:
            raise Exception("相似度分析失败")

        text_similarity_audio_output_info = f'耗时：{time_consuming:0.1f}秒；相似度分析成功：生成目录{similarity_file_dir}，文件{similarity_file}'

        hide_voice_similarity_dir = os.path.join(base_role_dir, params.audio_similarity_dir)

    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_similarity_audio_output_info = f"发生异常：{e}"
    return i18n(text_similarity_audio_output_info)


# 根据参考音频目录的删除情况，将其同步到推理生成的音频目录中，即参考音频目录下，删除了几个参考音频，就在推理目录下，将这些参考音频生成的音频文件移除
def sync_ref_audio(text_work_space_dir, text_role, text_refer_audio_file_dir,
                   text_inference_audio_file_dir):
    text_work_space_dir, text_refer_audio_file_dir, text_inference_audio_file_dir \
        = common.batch_clean_paths([text_work_space_dir, text_refer_audio_file_dir, text_inference_audio_file_dir])

    text_sync_ref_audio_info = None
    try:
        check_base_info(text_work_space_dir, text_role)
        if text_refer_audio_file_dir is None or text_refer_audio_file_dir == '':
            raise Exception("参考音频目录不能为空")
        if text_inference_audio_file_dir is None or text_inference_audio_file_dir == '':
            raise Exception("推理生成的音频目录不能为空")
        time_consuming, (delete_text_wav_num, delete_emotion_dir_num) \
            = time_util.time_monitor(audio_check.sync_ref_audio)(text_refer_audio_file_dir,
                                                                 text_inference_audio_file_dir)

        text_sync_ref_audio_info = (f"耗时：{time_consuming:0.1f}秒；推理音频目录{text_inference_audio_file_dir}下，"
                                    f"text目录删除了{delete_text_wav_num}个推理音频，emotion目录下，删除了{delete_emotion_dir_num}个目录")
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_sync_ref_audio_info = f"发生异常：{e}"
    return i18n(text_sync_ref_audio_info)


hide_config_file = ''


# 根据模板和参考音频目录，生成参考音频配置内容
def create_config(text_work_space_dir, text_role, text_template, text_refer_audio_file_dir):
    text_work_space_dir, text_refer_audio_file_dir \
        = common.batch_clean_paths([text_work_space_dir, text_refer_audio_file_dir])

    global hide_config_file

    config_file = None
    text_create_config_info = None
    try:
        base_role_dir = check_base_info(text_work_space_dir, text_role)
        if text_template is None or text_template == '':
            raise Exception("参考音频抽样目录不能为空")
        if text_refer_audio_file_dir is None or text_refer_audio_file_dir == '':
            raise Exception("参考音频目录不能为空")
        config_file = os.path.join(base_role_dir, f'{params.reference_audio_config_filename}.json')
        ref_audio_manager = common.RefAudioListManager(text_refer_audio_file_dir)

        time_consuming, _ = time_util.time_monitor(audio_config.generate_audio_config)(base_role_dir, text_template,
                                                                                       ref_audio_manager.get_ref_audio_list(),
                                                                                       config_file)

        text_create_config_info = f"耗时：{time_consuming:0.1f}秒；配置生成成功：生成文件{config_file}"

        hide_config_file = config_file

    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_create_config_info = f"发生异常：{e}"
    return i18n(text_create_config_info)


# 基于请求路径和参数，合成完整的请求路径
def whole_url(text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text, text_emotion):
    url_composer = audio_inference.TTSURLComposer(text_url, dropdown_refer_type_param, text_emotion, text_text,
                                                  text_ref_path, text_ref_text)
    if url_composer.is_emotion():
        text_whole_url = url_composer.build_url_with_emotion('测试内容', '情绪类型', False)
    else:
        text_whole_url = url_composer.build_url_with_ref('测试内容', '参考路径', '参考文本', False)
    return text_whole_url


def start_api():
    text_start_api_info = None
    try:
        proc = common.start_new_service('api.py')
        text_start_api_info = "启动完成"
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_start_api_info = f"发生异常：{e}"
    return text_start_api_info


def refresh_api_model():
    return ({"choices": model_manager.get_gpt_model_names(), "__type__": "update"},
            {"choices": model_manager.get_sovits_model_names(), "__type__": "update"})


def api_set_model_whole_url(text_api_set_model_base_url, dropdown_api_gpt_models, dropdown_api_sovits_models,
                            text_api_gpt_param, text_api_sovits_param):
    url = audio_inference.SetModelURLComposer("all", text_api_set_model_base_url, text_api_gpt_param,
                                              text_api_sovits_param)
    return url.build_get_url([dropdown_api_gpt_models, dropdown_api_sovits_models], False)


def start_api_set_model(text_api_set_model_base_url, dropdown_api_gpt_models, dropdown_api_sovits_models,
                        text_api_gpt_param, text_api_sovits_param):
    text_api_start_set_model_request_info = None
    try:
        if dropdown_api_gpt_models is None or dropdown_api_gpt_models == '':
            raise Exception("GPT模型不能为空")
        if dropdown_api_sovits_models is None or dropdown_api_sovits_models == '':
            raise Exception("Sovits模型不能为空")
        url = audio_inference.SetModelURLComposer("all", text_api_set_model_base_url, text_api_gpt_param,
                                                  text_api_sovits_param)
        url.is_valid()
        time_consuming, result = time_util.time_monitor(audio_inference.start_api_set_model)(url,
                                                                                             dropdown_api_gpt_models,
                                                                                             dropdown_api_sovits_models)
        text_api_start_set_model_request_info = f"耗时：{time_consuming:0.1f}秒；请求结果：{result}"
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_api_start_set_model_request_info = f"发生异常：{e}"
    return text_api_start_set_model_request_info


def refresh_api_v2_gpt_model():
    return {"choices": model_manager.get_gpt_model_names(), "__type__": "update"}


def api_v2_set_gpt_whole_url(text_api_v2_set_gpt_model_base_url, text_api_v2_gpt_model_param,
                             dropdown_api_v2_gpt_models):
    url = audio_inference.SetModelURLComposer("gpt", text_api_v2_set_gpt_model_base_url, text_api_v2_gpt_model_param,
                                              None)
    return url.build_get_url([dropdown_api_v2_gpt_models], False)


def start_api_v2_set_gpt_model(text_api_v2_set_gpt_model_base_url, text_api_v2_gpt_model_param,
                               dropdown_api_v2_gpt_models):
    text_api_v2_start_set_gpt_model_request_info = None
    try:
        if dropdown_api_v2_gpt_models is None or dropdown_api_v2_gpt_models == '':
            raise Exception("GPT模型不能为空")
        url = audio_inference.SetModelURLComposer("gpt", text_api_v2_set_gpt_model_base_url,
                                                  text_api_v2_gpt_model_param, None)
        url.is_valid()
        time_consuming, result = time_util.time_monitor(audio_inference.start_api_v2_set_gpt_model)(url,
                                                                                                    dropdown_api_v2_gpt_models)
        text_api_v2_start_set_gpt_model_request_info = f"耗时：{time_consuming:0.1f}秒；请求结果：{result}"
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_api_v2_start_set_gpt_model_request_info = f"发生异常：{e}"
    return text_api_v2_start_set_gpt_model_request_info


def refresh_api_v2_sovits_model():
    return {"choices": model_manager.get_sovits_model_names(), "__type__": "update"}


def api_v2_set_sovits_whole_url(text_api_v2_set_sovits_model_base_url, text_api_v2_sovits_model_param,
                                dropdown_api_v2_sovits_models):
    url = audio_inference.SetModelURLComposer("sovits", text_api_v2_set_sovits_model_base_url, None,
                                              text_api_v2_sovits_model_param)
    return url.build_get_url([dropdown_api_v2_sovits_models], False)


def start_api_v2_set_sovits_model(text_api_v2_set_sovits_model_base_url, text_api_v2_sovits_model_param,
                                  dropdown_api_v2_sovits_models):
    text_api_v2_start_set_sovits_model_request_info = None
    try:
        if dropdown_api_v2_sovits_models is None or dropdown_api_v2_sovits_models == '':
            raise Exception("Sovits模型不能为空")
        url = audio_inference.SetModelURLComposer("sovits", text_api_v2_set_sovits_model_base_url, None,
                                                  text_api_v2_sovits_model_param)
        url.is_valid()
        time_consuming, result = time_util.time_monitor(audio_inference.start_api_v2_set_sovits_model)(url,
                                                                                                       dropdown_api_v2_sovits_models)
        text_api_v2_start_set_sovits_model_request_info = f"耗时：{time_consuming:0.1f}秒；请求结果：{result}"
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_api_v2_start_set_sovits_model_request_info = f"发生异常：{e}"
    return text_api_v2_start_set_sovits_model_request_info


def open_file(file_path):
    common.open_file(my_utils.clean_path(file_path))


def delete_ref_audio_below_boundary(ref_audio_path, text_text_similarity_result_path, text_inference_audio_file_dir,
                                    slider_audio_text_similarity_boundary):
    text_delete_ref_audio_below_boundary_info = None
    ref_audio_path, text_text_similarity_result_path, text_inference_audio_file_dir = common.batch_clean_paths(
        [ref_audio_path, text_text_similarity_result_path, text_inference_audio_file_dir])
    try:
        if ref_audio_path is None or ref_audio_path == '':
            raise Exception("参考音频路径不能为空")
        if text_text_similarity_result_path is None or text_text_similarity_result_path == '':
            raise Exception("文本相似度结果路径不能为空")
        time_consuming, count = time_util.time_monitor(text_check.delete_ref_audio_below_boundary)(ref_audio_path,
                                                                                                   text_text_similarity_result_path,
                                                                                                   text_inference_audio_file_dir,
                                                                                                   slider_audio_text_similarity_boundary)
        text_delete_ref_audio_below_boundary_info = f"耗时：{time_consuming:0.1f}秒；删除参考音频数量：{count}"
    except Exception as e:
        logger.error("发生异常: \n%s", traceback.format_exc())
        text_delete_ref_audio_below_boundary_info = f"发生异常：{e}"
    return text_delete_ref_audio_below_boundary_info


def change_lang_choices(key):  # 根据选择的模型修改可选的语言
    # return gr.Dropdown(choices=asr_dict[key]['lang'])
    return {"__type__": "update", "choices": asr_dict[key]['lang'], "value": asr_dict[key]['lang'][0]}


def change_size_choices(key):  # 根据选择的模型修改可选的模型尺寸
    # return gr.Dropdown(choices=asr_dict[key]['size'])
    return {"__type__": "update", "choices": asr_dict[key]['size']}


def save_work_dir(text_work_space_dir, text_role):
    text_work_space_dir = my_utils.clean_path(text_work_space_dir)
    rw_param.write(rw_param.work_dir, text_work_space_dir)
    if text_role is not None and text_role != '':
        return text_role
    else:
        role_dir = ''
        for i in range(1, 101):
            role_dir = f"role_{i}"
            dir_name = os.path.join(text_work_space_dir, role_dir)
            if not os.path.isdir(dir_name):
                break
        rw_param.write(rw_param.role, role_dir)
        return role_dir


def chang_refer_type_param(selected_value):
    rw_param.write(rw_param.refer_type_param, selected_value)
    if selected_value == "参考音频":
        return {"visible": True, "__type__": "update"}, {"visible": True, "__type__": "update"}, {"visible": False,
                                                                                                  "__type__": "update"}
    else:
        return {"visible": False, "__type__": "update"}, {"visible": False, "__type__": "update"}, {"visible": True,
                                                                                                    "__type__": "update"}


def switch_role_and_refresh():
    global hide_voice_similarity_dir, hide_config_file
    init.init_all()
    reset_list = []
    # 基础
    reset_list.extend([
        init.text_refer_audio_file_dir_default,
        init.text_inference_audio_file_dir_default
    ])
    # 第一步
    reset_list.extend([
        init.text_sample_dir_default,
        '',  # text_list_input
        '',  # text_base_voice_path
    ])
    # 第二步
    # 第三步
    reset_list.extend([
        init.text_asr_audio_dir_default,
        init.text_text_similarity_analysis_path_default
    ])
    # 第四步
    reset_list.extend([
        '',  # text_base_audio_path
        '',  # text_compare_audio_dir
    ])
    hide_voice_similarity_dir = ''
    hide_config_file = ''
    # 第五步
    return reset_list


def init_ui():
    init.init_all()

    with gr.Blocks() as app:
        gr.Markdown(value=i18n("基本介绍：这是一个从训练素材中，批量提取参考音频，并进行效果评估与配置生成的工具"))
        with gr.Accordion(label=i18n("基本信息")):
            with gr.Row():
                text_work_space_dir = gr.Text(label=i18n("工作目录，后续操作所生成文件都会保存在此目录下"),
                                              value=init.text_work_space_dir_default, scale=4)
                text_role = gr.Text(label=i18n("角色名称"), value=init.text_role_default, scale=4)
                button_switch_role_and_refresh = gr.Button(i18n("切换并刷新"), variant="primary", scale=1)
                text_work_space_dir.blur(save_work_dir, [text_work_space_dir, text_role], [text_role])
                text_role.blur(lambda value: rw_param.write(rw_param.role, value), [text_role], [])
            gr.Markdown(value=i18n("下方为公共参数，会随着进度自动填充，无需填写"))
            with gr.Row():
                text_refer_audio_file_dir = gr.Text(label=i18n("参考音频所在目录"),
                                                    value=init.text_refer_audio_file_dir_default)
                text_inference_audio_file_dir = gr.Text(label=i18n("推理音频所在目录"),
                                                        value=init.text_inference_audio_file_dir_default)
        with gr.Tab(label=i18n("第一步：基于训练素材，生成待选参考音频列表")):
            gr.Markdown(value=i18n("1.1：选择list文件，并提取3-10秒的素材作为参考候选"))
            text_list_input = gr.Text(label=i18n("请输入list文件路径"), value="")
            with gr.Row():
                button_convert_from_list = gr.Button(i18n("开始生成待参考列表"), variant="primary", scale=4)
                text_convert_from_list_info = gr.Text(label=i18n("参考列表生成结果"), value="", interactive=False,
                                                      scale=4)
                button_convert_from_list_result_dir = gr.Button(i18n("打开目录"), variant="primary", scale=1)
            gr.Markdown(value=i18n("1.2：选择基准音频，执行相似度匹配，并分段随机抽样"))
            with gr.Row():
                text_sample_dir = gr.Text(label=i18n("参考音频抽样目录"), value=init.text_sample_dir_default,
                                          interactive=True)
                dropdown_speaker_verification_1 = gr.Dropdown(label=i18n("说话人确认算法"),
                                                              choices=list(
                                                                  model_manager.speaker_verification_models.keys()),
                                                              value='speech_campplus_sv_zh-cn_16k-common',
                                                              interactive=True)
            button_convert_from_list_result_dir.click(open_file, [text_sample_dir], [])
            button_convert_from_list.click(convert_from_list, [text_work_space_dir, text_role, text_list_input],
                                           [text_convert_from_list_info, text_sample_dir])
            with gr.Row():
                text_base_voice_path = gr.Text(label=i18n("请输入基准音频路径"), value="")
                slider_subsection_num = gr.Slider(minimum=1, maximum=10, step=1, label=i18n("请输入分段数"),
                                                  value=init.slider_subsection_num_default,
                                                  interactive=True)
                slider_sample_num = gr.Slider(minimum=1, maximum=10, step=1, label=i18n("请输入每段随机抽样个数"),
                                              value=init.slider_sample_num_default, interactive=True)
                checkbox_similarity_output = gr.Checkbox(label=i18n("是否将相似度匹配结果输出到临时目录？"),
                                                         show_label=True)
                slider_subsection_num.change(lambda value: rw_param.write(rw_param.subsection_num, value),
                                             [slider_subsection_num], [])
                slider_sample_num.change(lambda value: rw_param.write(rw_param.sample_num, value), [slider_sample_num],
                                         [])
            with gr.Row():
                button_sample = gr.Button(i18n("开始分段随机抽样"), variant="primary", scale=4)
                text_sample_info = gr.Text(label=i18n("分段随机抽样结果"), value="", interactive=False, scale=4)
                button_sample_result_open = gr.Button(i18n("打开目录"), variant="primary", scale=1)
        with gr.Tab(label=i18n("第二步：基于参考音频和测试文本，执行批量推理")):
            gr.Markdown(value=i18n("2.1：启动推理服务，并配置模型参数"))
            with gr.Accordion(label=i18n("详情")):
                with gr.Tab(label=i18n("主项目下api.py服务")):
                    gr.Markdown(value=i18n("2.1.1：启动服务"))
                    with gr.Row():
                        button_start_api = gr.Button(i18n("启动api"), variant="primary")
                        text_start_api_info = gr.Text(label=i18n("api启动信息"), value="", interactive=False)
                        button_start_api.click(start_api, [], [text_start_api_info])
                    gr.Markdown(value=i18n("2.1.2：设置模型参数"))
                    text_api_set_model_base_url = gr.Text(label=i18n("请输入api服务模型切换接口地址"),
                                                          value=init.text_api_set_model_base_url_default,
                                                          interactive=True)
                    text_api_set_model_base_url.blur(
                        lambda value: rw_param.write(rw_param.api_set_model_base_url, value),
                        [text_api_set_model_base_url], [])
                    with gr.Row():
                        dropdown_api_gpt_models = gr.Dropdown(label=i18n("GPT模型列表"),
                                                              choices=model_manager.get_gpt_model_names(), value="",
                                                              interactive=True, scale=4)
                        dropdown_api_sovits_models = gr.Dropdown(label=i18n("SoVITS模型列表"),
                                                                 choices=model_manager.get_sovits_model_names(),
                                                                 value="", interactive=True, scale=4)
                        button_refresh_api_model = gr.Button(i18n("刷新模型路径"), variant="primary", scale=1)
                        button_refresh_api_model.click(refresh_api_model, [],
                                                       [dropdown_api_gpt_models, dropdown_api_sovits_models])
                    with gr.Row():
                        text_api_gpt_param = gr.Text(label=i18n("GPT模型参数名"), value=init.text_api_gpt_param_default,
                                                     interactive=True)
                        text_api_sovits_param = gr.Text(label=i18n("SoVITS模型参数名"),
                                                        value=init.text_api_sovits_param_default, interactive=True)
                        text_api_gpt_param.blur(lambda value: rw_param.write(rw_param.api_gpt_param, value),
                                                [text_api_gpt_param], [])
                        text_api_sovits_param.blur(lambda value: rw_param.write(rw_param.api_sovits_param, value),
                                                   [text_api_sovits_param], [])
                    gr.Markdown(value=i18n("2.1.3：发起设置请求"))
                    text_api_set_model_whole_url = gr.Text(label=i18n("完整的模型参数设置请求地址"), value="",
                                                           interactive=False)
                    dropdown_api_gpt_models.change(api_set_model_whole_url,
                                                   [text_api_set_model_base_url, dropdown_api_gpt_models,
                                                    dropdown_api_sovits_models, text_api_gpt_param,
                                                    text_api_sovits_param], [text_api_set_model_whole_url])
                    dropdown_api_sovits_models.change(api_set_model_whole_url,
                                                      [text_api_set_model_base_url, dropdown_api_gpt_models,
                                                       dropdown_api_sovits_models, text_api_gpt_param,
                                                       text_api_sovits_param], [text_api_set_model_whole_url])
                    text_api_gpt_param.input(api_set_model_whole_url,
                                             [text_api_set_model_base_url, dropdown_api_gpt_models,
                                              dropdown_api_sovits_models, text_api_gpt_param, text_api_sovits_param],
                                             [text_api_set_model_whole_url])
                    text_api_sovits_param.input(api_set_model_whole_url,
                                                [text_api_set_model_base_url, dropdown_api_gpt_models,
                                                 dropdown_api_sovits_models, text_api_gpt_param, text_api_sovits_param],
                                                [text_api_set_model_whole_url])
                    with gr.Row():
                        button_api_start_set_model_request = gr.Button(i18n("发起模型设置请求"), variant="primary")
                        text_api_start_set_model_request_info = gr.Text(label=i18n("设置请求结果"), value="",
                                                                        interactive=False)
                        button_api_start_set_model_request.click(start_api_set_model,
                                                                 [text_api_set_model_base_url, dropdown_api_gpt_models,
                                                                  dropdown_api_sovits_models, text_api_gpt_param,
                                                                  text_api_sovits_param],
                                                                 [text_api_start_set_model_request_info])
                with gr.Tab(label=i18n("fast项目下api_v2.py服务")):
                    gr.Markdown(value=i18n("2.1.1：请将训练完毕得模型，复制到你的项目文件下，启动服务"))
                    gr.Markdown(value=i18n("2.1.2：设置GPT模型参数"))
                    text_api_v2_set_gpt_model_base_url = gr.Text(label=i18n("请输入api服务GPT模型切换接口地址"),
                                                                 value=init.text_api_v2_set_gpt_model_base_url_default,
                                                                 interactive=True)
                    text_api_v2_set_gpt_model_base_url.blur(
                        lambda value: rw_param.write(rw_param.api_v2_set_gpt_model_base_url, value),
                        [text_api_v2_set_gpt_model_base_url], [])
                    with gr.Row():
                        text_api_v2_gpt_model_param = gr.Text(label=i18n("GPT模型参数名"),
                                                              value=init.text_api_v2_gpt_model_param_default,
                                                              interactive=True, scale=4)
                        dropdown_api_v2_gpt_models = gr.Dropdown(label=i18n("GPT模型列表"),
                                                                 choices=model_manager.get_gpt_model_names(), value="",
                                                                 interactive=True, scale=4)
                        text_api_v2_gpt_model_param.blur(
                            lambda value: rw_param.write(rw_param.api_v2_gpt_model_param, value),
                            [text_api_v2_gpt_model_param], [])
                        button_api_v2_refresh_gpt = gr.Button(i18n("刷新模型路径"), variant="primary", scale=1)
                        button_api_v2_refresh_gpt.click(refresh_api_v2_gpt_model, [], [dropdown_api_v2_gpt_models])
                    text_api_v2_set_gpt_model_whole_url = gr.Text(label=i18n("完整的GPT模型参数设置请求地址"), value="",
                                                                  interactive=False)
                    text_api_v2_gpt_model_param.input(api_v2_set_gpt_whole_url,
                                                      [text_api_v2_set_gpt_model_base_url, text_api_v2_gpt_model_param,
                                                       dropdown_api_v2_gpt_models],
                                                      [text_api_v2_set_gpt_model_whole_url])
                    dropdown_api_v2_gpt_models.change(api_v2_set_gpt_whole_url,
                                                      [text_api_v2_set_gpt_model_base_url, text_api_v2_gpt_model_param,
                                                       dropdown_api_v2_gpt_models],
                                                      [text_api_v2_set_gpt_model_whole_url])
                    with gr.Row():
                        button_api_v2_start_set_gpt_model_request = gr.Button(i18n("发起GPT模型设置请求"),
                                                                              variant="primary")
                        text_api_v2_start_set_gpt_model_request_info = gr.Text(label=i18n("设置请求结果"), value="",
                                                                               interactive=False)
                        button_api_v2_start_set_gpt_model_request.click(start_api_v2_set_gpt_model,
                                                                        [text_api_v2_set_gpt_model_base_url,
                                                                         text_api_v2_gpt_model_param,
                                                                         dropdown_api_v2_gpt_models],
                                                                        [text_api_v2_start_set_gpt_model_request_info])
                    gr.Markdown(value=i18n("2.1.3：设置SoVITS模型参数"))
                    text_api_v2_set_sovits_model_base_url = gr.Text(label=i18n("请输入api服务SoVITS模型切换接口地址"),
                                                                    value=init.text_api_v2_set_sovits_model_base_url_default,
                                                                    interactive=True)
                    text_api_v2_set_sovits_model_base_url.blur(
                        lambda value: rw_param.write(rw_param.api_v2_set_sovits_model_base_url, value),
                        [text_api_v2_set_sovits_model_base_url], [])
                    with gr.Row():
                        text_api_v2_sovits_model_param = gr.Text(label=i18n("SoVITS模型参数名"),
                                                                 value=init.text_api_v2_sovits_model_param_default,
                                                                 interactive=True, scale=4)
                        dropdown_api_v2_sovits_models = gr.Dropdown(label=i18n("SoVITS模型列表"),
                                                                    choices=model_manager.get_sovits_model_names(),
                                                                    value="", interactive=True, scale=4)
                        button_api_v2_refresh_sovits = gr.Button(i18n("刷新模型路径"), variant="primary", scale=1)
                        text_api_v2_sovits_model_param.blur(
                            lambda value: rw_param.write(rw_param.api_v2_sovits_model_param, value),
                            [text_api_v2_sovits_model_param], [])
                        button_api_v2_refresh_sovits.click(refresh_api_v2_sovits_model, [],
                                                           [dropdown_api_v2_sovits_models])
                    text_api_v2_set_sovits_model_whole_url = gr.Text(label=i18n("完整的SoVITS模型参数设置请求地址"),
                                                                     value="", interactive=False)
                    text_api_v2_sovits_model_param.input(api_v2_set_sovits_whole_url,
                                                         [text_api_v2_set_sovits_model_base_url,
                                                          text_api_v2_sovits_model_param,
                                                          dropdown_api_v2_sovits_models],
                                                         [text_api_v2_set_sovits_model_whole_url])
                    dropdown_api_v2_sovits_models.change(api_v2_set_sovits_whole_url,
                                                         [text_api_v2_set_sovits_model_base_url,
                                                          text_api_v2_sovits_model_param,
                                                          dropdown_api_v2_sovits_models],
                                                         [text_api_v2_set_sovits_model_whole_url])
                    with gr.Row():
                        button_api_v2_start_set_sovits_model_request = gr.Button(i18n("发起SoVITS模型设置请求"),
                                                                                 variant="primary")
                        text_api_v2_start_set_sovits_model_request_info = gr.Text(label=i18n("设置请求结果"), value="",
                                                                                  interactive=False)
                        button_api_v2_start_set_sovits_model_request.click(start_api_v2_set_sovits_model,
                                                                           [text_api_v2_set_sovits_model_base_url,
                                                                            text_api_v2_sovits_model_param,
                                                                            dropdown_api_v2_sovits_models], [
                                                                               text_api_v2_start_set_sovits_model_request_info])
                with gr.Tab(label=i18n("第三方推理服务")):
                    gr.Markdown(value=i18n("启动第三方推理服务，并完成参考音频打包，模型参数设置等操作"))
            gr.Markdown(value=i18n("2.2：配置推理服务参数信息，除api服务外，其他需要修改参数内容，参考音和角色情绪二选一，如果是角色情绪（第三方推理包），需要先执行第五步，"
                                   "将参考音频打包配置到推理服务下，在推理前，请确认完整请求地址是否与正常使用时的一致，包括角色名称，尤其是文本分隔符是否正确"))
            text_url = gr.Text(label=i18n("请输入推理服务请求地址与参数"),
                               value=init.text_url_default)
            with gr.Row():
                text_text = gr.Text(label=i18n("请输入文本参数名"), value=init.text_text_default)
                dropdown_refer_type_param = gr.Dropdown(label=i18n("类型"), choices=["参考音频", "角色情绪"],
                                                        value=init.dropdown_refer_type_param_default, interactive=True)
                text_ref_path = gr.Text(label=i18n("请输入参考音频路径参数名"),
                                        value=init.text_ref_path_default, visible=True)
                text_ref_text = gr.Text(label=i18n("请输入参考音频文本参数名"),
                                        value=init.text_ref_text_default, visible=True)
                text_emotion = gr.Text(label=i18n("请输入角色情绪参数名"), value=init.text_emotion_default,
                                       visible=False)
                dropdown_refer_type_param.change(chang_refer_type_param, [dropdown_refer_type_param],
                                                 [text_ref_path, text_ref_text, text_emotion])
            text_whole_url = gr.Text(label=i18n("完整地址"), value=init.text_whole_url_default, interactive=False)

            text_text.blur(lambda value: rw_param.write(rw_param.text_param, value), [text_text], [])
            text_ref_path.blur(lambda value: rw_param.write(rw_param.ref_path_param, value), [text_ref_path], [])
            text_ref_text.blur(lambda value: rw_param.write(rw_param.ref_text_param, value), [text_ref_text], [])
            text_emotion.blur(lambda value: rw_param.write(rw_param.emotion_param, value), [text_emotion], [])

            text_url.input(whole_url,
                           [text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text, text_emotion],
                           [text_whole_url])
            text_url.blur(lambda value: rw_param.write(rw_param.text_url, value), [text_url], [])
            text_text.input(whole_url, [text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text,
                                        text_emotion],
                            [text_whole_url])
            text_text.blur(lambda value: rw_param.write(rw_param.text_param, value), [text_text], [])
            dropdown_refer_type_param.change(whole_url,
                                             [text_url, dropdown_refer_type_param, text_text, text_ref_path,
                                              text_ref_text,
                                              text_emotion],
                                             [text_whole_url])
            text_ref_path.input(whole_url,
                                [text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text,
                                 text_emotion],
                                [text_whole_url])
            text_ref_path.blur(lambda value: rw_param.write(rw_param.ref_path_param, value), [text_ref_path], [])
            text_ref_text.input(whole_url,
                                [text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text,
                                 text_emotion],
                                [text_whole_url])
            text_ref_text.blur(lambda value: rw_param.write(rw_param.ref_text_param, value), [text_ref_text], [])
            text_emotion.input(whole_url, [text_url, dropdown_refer_type_param, text_text, text_ref_path, text_ref_text,
                                           text_emotion],
                               [text_whole_url])
            text_emotion.blur(lambda value: rw_param.write(rw_param.emotion_param, value), [text_emotion], [])
            gr.Markdown(value=i18n("2.3：配置待推理文本，一句一行，尽量保证文本多样性，不同情绪、不同类型的都来一点"))
            with gr.Row():
                text_test_content = gr.Text(label=i18n("请输入待推理文本路径"), value=init.text_test_content_default,
                                            scale=8)
                button_open_test_content_file = gr.Button(i18n("打开推理文本"), variant="primary", scale=1)
                button_open_test_content_file.click(open_file, [text_test_content], [])
                text_test_content.blur(lambda value: rw_param.write(rw_param.test_content_path, value),
                                       [text_test_content], [])
            gr.Markdown(value=i18n("2.4：开始批量推理，这个过程比较耗时，可以去干点别的"))
            slider_request_concurrency_num = gr.Slider(minimum=1, maximum=init.slider_request_concurrency_max_num,
                                                       step=1, label=i18n(
                    "请输入请求并发数，会根据此数创建对应数量的子进程并行发起推理请求"),
                                                       value=init.slider_request_concurrency_num_default,
                                                       interactive=True)
            slider_request_concurrency_num.change(lambda value: rw_param.write(rw_param.request_concurrency_num, value),
                                                  [slider_request_concurrency_num], [])
            with gr.Row():
                button_model_inference = gr.Button(i18n("开启批量推理"), variant="primary", scale=4)
                text_model_inference_info = gr.Text(label=i18n("批量推理结果"), value="", interactive=False, scale=4)
                button_model_inference_result_open = gr.Button(i18n("打开目录"), variant="primary", scale=1)
        with gr.Tab(label=i18n("第三步：进行参考音频推理效果准确度校验")):
            gr.Markdown(value=i18n("3.1：启动asr，获取推理音频文本"))
            text_asr_audio_dir = gr.Text(label=i18n("待asr的音频所在目录"), value=init.text_asr_audio_dir_default,
                                         interactive=True)
            with gr.Row():
                dropdown_asr_model = gr.Dropdown(
                    label=i18n("ASR 模型"),
                    choices=list(asr_dict.keys()),
                    interactive=True,
                    value="达摩 ASR (中文)"
                )
                dropdown_asr_size = gr.Dropdown(
                    label=i18n("ASR 模型尺寸"),
                    choices=["large"],
                    interactive=True,
                    value="large"
                )
                dropdown_asr_lang = gr.Dropdown(
                    label=i18n("ASR 语言设置"),
                    choices=["zh"],
                    interactive=True,
                    value="zh"
                )
                dropdown_asr_model.change(change_lang_choices, [dropdown_asr_model], [dropdown_asr_lang])
                dropdown_asr_model.change(change_size_choices, [dropdown_asr_model], [dropdown_asr_size])
            with gr.Row():
                button_asr = gr.Button(i18n("启动asr"), variant="primary", scale=4)
                text_asr_info = gr.Text(label=i18n("asr结果"), value="", interactive=False, scale=4)
                button_asr_result_open = gr.Button(i18n("打开文件"), variant="primary", scale=1)
            gr.Markdown(value=i18n("3.2：启动文本相似度分析"))
            with gr.Row():
                text_text_similarity_analysis_path = gr.Text(label=i18n("待分析的文件路径"),
                                                             value=init.text_text_similarity_analysis_path_default,
                                                             interactive=True)
                slider_text_similarity_amplification_boundary = gr.Slider(minimum=0, maximum=1, step=0.01,
                                                                          label=i18n(
                                                                              "文本相似度放大边界，因为原始模型输出的相似度差异太小，所以进行了一次放大，放大逻辑为，边界值以下归0，边界值到1的区间重新映射到0-1"),
                                                                          value=init.slider_text_similarity_amplification_boundary_default,
                                                                          interactive=True)
                slider_text_similarity_amplification_boundary.change(
                    lambda value: rw_param.write(rw_param.text_similarity_amplification_boundary, value),
                    [slider_text_similarity_amplification_boundary], [])
            button_asr.click(asr, [text_work_space_dir, text_role, text_asr_audio_dir, dropdown_asr_model,
                                   dropdown_asr_size, dropdown_asr_lang],
                             [text_asr_info, text_text_similarity_analysis_path])
            button_asr_result_open.click(open_file, [text_text_similarity_analysis_path], [])
            with gr.Row():
                button_text_similarity_analysis = gr.Button(i18n("启动文本相似度分析"), variant="primary")
                text_text_similarity_analysis_info = gr.Text(label=i18n("文本相似度分析结果"), value="",
                                                             interactive=False)
            gr.Markdown(value=i18n("3.3：根据相似度分析结果，重点检查最后几条是否存在复读等问题"))
            with gr.Row():
                text_text_similarity_result_path = gr.Text(label=i18n("文本相似度分析结果文件所在路径"),
                                                           value=init.text_text_similarity_result_path_default,
                                                           interactive=True, scale=7)
                button_open_text_similarity_result = gr.Button(i18n("打开结果文件"), variant="primary", scale=1)
                button_open_inference_dir = gr.Button(i18n("打开推理目录"), variant="primary", scale=1)

                button_text_similarity_analysis.click(text_similarity_analysis, [text_work_space_dir, text_role,
                                                                                 slider_text_similarity_amplification_boundary,
                                                                                 text_text_similarity_analysis_path],
                                                      [text_text_similarity_analysis_info,
                                                       text_text_similarity_result_path])

                button_open_text_similarity_result.click(open_file, [text_text_similarity_result_path], [])
                button_open_inference_dir.click(open_file, [text_inference_audio_file_dir], [])
            slider_audio_text_similarity_boundary = gr.Slider(minimum=0, maximum=1, step=0.001,
                                                              label=i18n("音频文本相似度边界值"), value=0.800,
                                                              interactive=True)
            with gr.Row():
                button_delete_ref_audio_below_boundary = gr.Button(i18n("删除音频文本相似度边界值以下的参考音频"),
                                                                   variant="primary")
                text_delete_ref_audio_below_boundary_info = gr.Text(label=i18n("删除结果"), value="", interactive=True)
                button_delete_ref_audio_below_boundary.click(delete_ref_audio_below_boundary,
                                                             [text_refer_audio_file_dir,
                                                              text_text_similarity_result_path,
                                                              text_inference_audio_file_dir,
                                                              slider_audio_text_similarity_boundary],
                                                             [text_delete_ref_audio_below_boundary_info])
        with gr.Tab(label=i18n("第四步：校验参考音频音质")):
            gr.Markdown(value=i18n("4.1：对结果按音频相似度排序，或许有用吧，主要还是耳朵听"))
            with gr.Row():
                text_base_audio_path = gr.Text(label=i18n("请输入基准音频"), value="")
                text_compare_audio_dir = gr.Text(label=i18n("请输入待比较的音频文件目录"), value="")
                dropdown_speaker_verification_2 = gr.Dropdown(label=i18n("说话人确认算法"),
                                                              choices=list(
                                                                  model_manager.speaker_verification_models.keys()),
                                                              value='speech_campplus_sv_zh-cn_16k-common',
                                                              interactive=True)
            with gr.Row():
                button_similarity_audio_output = gr.Button(i18n("输出相似度-参考音频到临时目录"), variant="primary",
                                                           scale=4)
                text_similarity_audio_output_info = gr.Text(label=i18n("输出结果"), value="", interactive=False,
                                                            scale=4)
                button_similarity_audio_output_result_open = gr.Button(i18n("打开目录"), variant="primary", scale=1)
                button_similarity_audio_output.click(similarity_audio_output,
                                                     [text_work_space_dir, text_role, text_base_audio_path,
                                                      text_compare_audio_dir, dropdown_speaker_verification_2],
                                                     [text_similarity_audio_output_info])
                button_similarity_audio_output_result_open.click(lambda: open_file(hide_voice_similarity_dir), [], [])
            gr.Markdown(value=i18n("4.2：如果发现存在低音质的推理音频，那么就去参考音频目录下，把原参考音频删了"))
            gr.Markdown(value=i18n("4.3：删除参考音频之后，按下面的操作，会将推理音频目录下对应的音频也删掉"))
            with gr.Row():
                button_sync_ref_audio = gr.Button(i18n("将参考音频的删除情况，同步到推理音频目录"), variant="primary")
                text_sync_ref_info = gr.Text(label=i18n("同步结果"), value="", interactive=False)
                button_sync_ref_audio.click(sync_ref_audio, [text_work_space_dir, text_role, text_refer_audio_file_dir,
                                                             text_inference_audio_file_dir], [text_sync_ref_info])
        with gr.Tab("第五步：生成参考音频配置文本"):
            gr.Markdown(value=i18n(
                "5.1：编辑模板，占位符说明：\${emotion}表示相对路径加音频文件名；\${ref_path}表示音频相对角色目录的文件路径；\${ref_text}：表示音频文本"))
            text_template = gr.Text(label=i18n("模板内容"), value=init.text_template_default, lines=10)
            text_template.blur(lambda value: rw_param.write(rw_param.text_template, value), [text_template], [])
            gr.Markdown(value=i18n("5.2：生成配置"))
            with gr.Row():
                button_create_config = gr.Button(i18n("生成配置"), variant="primary", scale=4)
                text_create_config_info = gr.Text(label=i18n("生成结果"), value="", interactive=False, scale=4)
                button_create_config_result_open = gr.Button(i18n("打开文件"), variant="primary", scale=1)
                button_create_config.click(create_config,
                                           [text_work_space_dir, text_role, text_template, text_refer_audio_file_dir],
                                           [text_create_config_info])
                button_create_config_result_open.click(lambda: open_file(hide_config_file), [], [])
        button_sample.click(sample, [text_work_space_dir, text_role, text_sample_dir, dropdown_speaker_verification_1,
                                     text_base_voice_path,
                                     slider_subsection_num, slider_sample_num, checkbox_similarity_output],
                            [text_sample_info, text_refer_audio_file_dir])
        button_sample_result_open.click(open_file, [text_refer_audio_file_dir], [])
        button_model_inference.click(model_inference,
                                     [text_work_space_dir, text_role, slider_request_concurrency_num,
                                      text_refer_audio_file_dir, text_url, dropdown_refer_type_param,
                                      text_text, text_ref_path, text_ref_text, text_emotion,
                                      text_test_content],
                                     [text_model_inference_info, text_asr_audio_dir, text_inference_audio_file_dir])
        button_model_inference_result_open.click(open_file, [text_inference_audio_file_dir], [])

        # 设置重置刷新事件
        refresh_list = []
        # 基础
        refresh_list.extend([
            text_refer_audio_file_dir,
            text_inference_audio_file_dir
        ])
        # 第一步
        refresh_list.extend([
            text_sample_dir,
            text_list_input,
            text_base_voice_path
        ])
        # 第二步
        # 第三步
        refresh_list.extend([
            text_asr_audio_dir,
            text_text_similarity_analysis_path
        ])
        # 第四步
        refresh_list.extend([
            text_base_audio_path,
            text_compare_audio_dir
        ])
        # 第五步

        button_switch_role_and_refresh.click(switch_role_and_refresh, [], refresh_list)

    app.launch(
        server_port=params.server_port,
        inbrowser=True,
        quiet=True,
    )


if __name__ == "__main__":
    init_ui()
