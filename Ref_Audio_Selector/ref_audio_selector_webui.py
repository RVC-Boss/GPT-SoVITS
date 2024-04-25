import os.path
import os
import traceback

import gradio as gr
import Ref_Audio_Selector.tool.audio_similarity as audio_similarity
import Ref_Audio_Selector.tool.audio_inference as audio_inference
import Ref_Audio_Selector.tool.audio_config as audio_config
import Ref_Audio_Selector.tool.delete_inference_with_ref as delete_inference_with_ref
import Ref_Audio_Selector.common.common as common
import Ref_Audio_Selector.config.config_params as params
from tools.i18n.i18n import I18nAuto
from config import python_exec, is_half
from tools import my_utils
from tools.asr.config import asr_dict
from subprocess import Popen

i18n = I18nAuto()

p_similarity = None
p_asr = None
p_text_similarity = None


# 校验基础信息
def check_base_info(text_work_space_dir):
    if text_work_space_dir is None or text_work_space_dir == '':
        raise Exception("工作目录不能为空")


# 从list文件，提取参考音频
def convert_from_list(text_work_space_dir, text_list_input):
    ref_audio_all = os.path.join(text_work_space_dir,
                                 params.list_to_convert_reference_audio_dir)
    text_convert_from_list_info = f"转换成功：生成目录{ref_audio_all}"
    text_sample_dir = ref_audio_all
    try:
        check_base_info(text_work_space_dir)
        if text_list_input is None or text_list_input == '':
            raise Exception("list文件路径不能为空")
        audio_similarity.convert_from_list(text_list_input, ref_audio_all)
    except Exception as e:
        traceback.print_exc()
        text_convert_from_list_info = f"发生异常：{e}"
        text_sample_dir = ''
    return i18n(text_convert_from_list_info), text_sample_dir


def start_similarity_analysis(work_space_dir, sample_dir, base_voice_path, need_similarity_output):
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
        cmd += f' -o {similarity_file}'

        print(cmd)
        p_similarity = Popen(cmd, shell=True)
        p_similarity.wait()

        if need_similarity_output:
            similarity_list = audio_similarity.parse_similarity_file(similarity_file)
            similarity_file_dir = os.path.join(similarity_dir, base_voice_file_name)
            audio_similarity.copy_and_move(similarity_file_dir, similarity_list)

        p_similarity = None
        return similarity_list, similarity_file, similarity_file_dir
    else:
        return similarity_list, None, None


# 基于一个基准音频，从参考音频目录中进行分段抽样
def sample(text_work_space_dir, text_sample_dir, text_base_voice_path,
           text_subsection_num, text_sample_num, checkbox_similarity_output):
    ref_audio_dir = os.path.join(text_work_space_dir, params.reference_audio_dir)
    text_sample_info = f"抽样成功：生成目录{ref_audio_dir}"
    try:
        check_base_info(text_work_space_dir)
        if text_sample_dir is None or text_sample_dir == '':
            raise Exception("参考音频抽样目录不能为空，请先完成上一步操作")
        if text_base_voice_path is None or text_base_voice_path == '':
            raise Exception("基准音频路径不能为空")
        if text_subsection_num is None or text_subsection_num == '':
            raise Exception("分段数不能为空")
        if text_sample_num is None or text_sample_num == '':
            raise Exception("每段随机抽样个数不能为空")

        similarity_list, _, _ = start_similarity_analysis(text_work_space_dir, text_sample_dir,
                                                          text_base_voice_path, checkbox_similarity_output)

        if similarity_list is None:
            raise Exception("相似度分析失败")

        audio_similarity.sample(ref_audio_dir, similarity_list, int(text_subsection_num), int(text_sample_num))

    except Exception as e:
        traceback.print_exc()
        text_sample_info = f"发生异常：{e}"
        ref_audio_dir = ''
    text_model_inference_voice_dir = ref_audio_dir
    text_sync_ref_audio_dir = ref_audio_dir
    text_sync_ref_audio_dir2 = ref_audio_dir
    return i18n(text_sample_info), text_model_inference_voice_dir, text_sync_ref_audio_dir, text_sync_ref_audio_dir2


# 根据参考音频和测试文本，执行批量推理
def model_inference(text_work_space_dir, text_model_inference_voice_dir, text_url,
                    text_text, text_ref_path, text_ref_text, text_emotion,
                    text_test_content):
    inference_dir = os.path.join(text_work_space_dir, params.inference_audio_dir)
    text_asr_audio_dir = os.path.join(inference_dir,
                                      params.inference_audio_text_aggregation_dir)
    text_model_inference_info = f"推理成功：生成目录{inference_dir}"
    try:
        check_base_info(text_work_space_dir)
        if text_model_inference_voice_dir is None or text_model_inference_voice_dir == '':
            raise Exception("待推理的参考音频所在目录不能为空，请先完成上一步操作")
        if text_url is None or text_url == '':
            raise Exception("推理服务请求地址不能为空")
        if text_text is None or text_text == '':
            raise Exception("文本参数名不能为空")
        if text_test_content is None or text_test_content == '':
            raise Exception("待推理文本路径不能为空")
        if (text_ref_path is None or text_ref_path == '') and (text_ref_text is None or text_ref_text == '') and (
                text_emotion is None or text_emotion == ''):
            raise Exception("参考音频路径/文本和角色情绪二选一填写，不能全部为空")
        url_composer = audio_inference.URLComposer(text_url, text_emotion, text_text, text_ref_path, text_ref_text)
        url_composer.is_valid()
        text_list = common.read_text_file_to_list(text_test_content)
        if text_list is None or len(text_list) == 0:
            raise Exception("待推理文本内容不能为空")
        ref_audio_manager = common.RefAudioListManager(text_model_inference_voice_dir)
        if len(ref_audio_manager.get_audio_list()) == 0:
            raise Exception("待推理的参考音频不能为空")
        audio_inference.generate_audio_files(url_composer, text_list, ref_audio_manager.get_ref_audio_list(),
                                             inference_dir)
    except Exception as e:
        traceback.print_exc()
        text_model_inference_info = f"发生异常：{e}"
        text_asr_audio_dir = ''
    return i18n(text_model_inference_info), text_asr_audio_dir


# 对推理生成音频执行asr
def asr(text_work_space_dir, text_asr_audio_dir, dropdown_asr_model,
        dropdown_asr_size, dropdown_asr_lang):
    asr_file = None
    text_text_similarity_analysis_path = None
    text_asr_info = None
    try:
        check_base_info(text_work_space_dir)
        if text_asr_audio_dir is None or text_asr_audio_dir == '':
            raise Exception("待asr的音频所在目录不能为空，请先完成上一步操作")
        if dropdown_asr_model is None or dropdown_asr_model == '':
            raise Exception("asr模型不能为空")
        if dropdown_asr_size is None or dropdown_asr_size == '':
            raise Exception("asr模型大小不能为空")
        if dropdown_asr_lang is None or dropdown_asr_lang == '':
            raise Exception("asr语言不能为空")
        asr_file = open_asr(text_asr_audio_dir, text_work_space_dir, dropdown_asr_model, dropdown_asr_size,
                            dropdown_asr_lang)
        text_text_similarity_analysis_path = asr_file
        text_asr_info = f"asr成功：生成文件{asr_file}"
    except Exception as e:
        traceback.print_exc()
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
        if asr_py_path == 'fasterwhisper.py':
            asr_py_path = 'fasterwhisper_asr_multi_level_dir.py'
        cmd = f'"{python_exec}" Ref_Audio_Selector/tool/asr/{asr_py_path} '
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += " -p %s" % ("float16" if is_half == True else "float32")

        print(cmd)
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
def text_similarity_analysis(text_work_space_dir,
                             text_text_similarity_analysis_path):
    similarity_dir = os.path.join(text_work_space_dir, params.text_similarity_output_dir)
    text_text_similarity_analysis_info = f"相似度分析成功：生成目录{similarity_dir}"
    try:
        check_base_info(text_work_space_dir)
        if text_text_similarity_analysis_path is None or text_text_similarity_analysis_path == '':
            raise Exception("asr生成的文件路径不能为空，请先完成上一步操作")
        open_text_similarity_analysis(text_text_similarity_analysis_path, similarity_dir)
    except Exception as e:
        traceback.print_exc()
        text_text_similarity_analysis_info = f"发生异常：{e}"
    return i18n(text_text_similarity_analysis_info)


def open_text_similarity_analysis(asr_file_path, output_dir, similarity_enlarge_boundary=0.8):
    global p_text_similarity
    if p_text_similarity is None:
        cmd = f'"{python_exec}" Ref_Audio_Selector/tool/text_comparison/asr_text_process.py '
        cmd += f' -a "{asr_file_path}"'
        cmd += f' -o "{output_dir}"'
        cmd += f' -b {similarity_enlarge_boundary}'

        print(cmd)
        p_text_similarity = Popen(cmd, shell=True)
        p_text_similarity.wait()
        p_text_similarity = None

        return output_dir

    else:
        return None


# 根据一个参考音频，对指定目录下的音频进行相似度分析，并输出到另一个目录
def similarity_audio_output(text_work_space_dir, text_base_audio_path,
                            text_compare_audio_dir):
    text_similarity_audio_output_info = None
    try:
        check_base_info(text_work_space_dir)
        if text_base_audio_path is None or text_base_audio_path == '':
            raise Exception("基准音频路径不能为空")
        if text_compare_audio_dir is None or text_compare_audio_dir == '':
            raise Exception("待分析的音频所在目录不能为空")
        similarity_list, similarity_file, similarity_file_dir = start_similarity_analysis(
            text_work_space_dir, text_compare_audio_dir, text_base_audio_path, True)

        if similarity_list is None:
            raise Exception("相似度分析失败")

        text_similarity_audio_output_info = f'相似度分析成功：生成目录{similarity_file_dir}，文件{similarity_file}'

    except Exception as e:
        traceback.print_exc()
        text_similarity_audio_output_info = f"发生异常：{e}"
    return i18n(text_similarity_audio_output_info)


# 根据参考音频目录的删除情况，将其同步到推理生成的音频目录中，即参考音频目录下，删除了几个参考音频，就在推理目录下，将这些参考音频生成的音频文件移除
def sync_ref_audio(text_work_space_dir, text_sync_ref_audio_dir,
                   text_sync_inference_audio_dir):
    text_sync_ref_audio_info = None
    try:
        check_base_info(text_work_space_dir)
        if text_sync_ref_audio_dir is None or text_sync_ref_audio_dir == '':
            raise Exception("参考音频目录不能为空")
        if text_sync_inference_audio_dir is None or text_sync_inference_audio_dir == '':
            raise Exception("推理生成的音频目录不能为空")
        delete_text_wav_num, delete_emotion_dir_num = delete_inference_with_ref.sync_ref_audio(text_sync_ref_audio_dir,
                                                                                               text_sync_inference_audio_dir)
        text_sync_ref_audio_info = f"推理音频目录{text_sync_inference_audio_dir}下，text目录删除了{delete_text_wav_num}个参考音频，emotion目录下，删除了{delete_emotion_dir_num}个目录"
    except Exception as e:
        traceback.print_exc()
        text_sync_ref_audio_info = f"发生异常：{e}"
    return i18n(text_sync_ref_audio_info)


# 根据模板和参考音频目录，生成参考音频配置内容
def create_config(text_work_space_dir, text_template, text_sync_ref_audio_dir2):
    config_file = os.path.join(text_work_space_dir, f'{params.reference_audio_config_filename}.json')
    text_create_config_info = f"配置生成成功：生成文件{config_file}"
    try:
        check_base_info(text_work_space_dir)
        if text_template is None or text_template == '':
            raise Exception("参考音频抽样目录不能为空")
        if text_sync_ref_audio_dir2 is None or text_sync_ref_audio_dir2 == '':
            raise Exception("参考音频目录不能为空")
        ref_audio_manager = common.RefAudioListManager(text_sync_ref_audio_dir2)
        audio_config.generate_audio_config(text_work_space_dir, text_template, ref_audio_manager.get_ref_audio_list(),
                                           config_file)
    except Exception as e:
        traceback.print_exc()
        text_create_config_info = f"发生异常：{e}"
    return i18n(text_create_config_info)


# 基于请求路径和参数，合成完整的请求路径
def whole_url(text_url, text_text, text_ref_path, text_ref_text, text_emotion):
    url_composer = audio_inference.URLComposer(text_url, text_emotion, text_text, text_ref_path, text_ref_text)
    if url_composer.is_emotion():
        text_whole_url = url_composer.build_url_with_emotion('测试内容', '情绪类型', False)
    else:
        text_whole_url = url_composer.build_url_with_ref('测试内容', '参考路径', '参考文本', False)
    return text_whole_url


with gr.Blocks() as app:
    gr.Markdown(value=i18n("基本介绍：这是一个从训练素材中，批量提取参考音频，并进行效果评估与配置生成的工具"))
    text_work_space_dir = gr.Text(label=i18n("工作目录，后续操作所生成文件都会保存在此目录下"), value="")
    with gr.Accordion(label=i18n("第一步：基于训练素材，生成待选参考音频列表"), open=False):
        gr.Markdown(value=i18n("1.1：选择list文件，并提取3-10秒的素材作为参考候选"))
        text_list_input = gr.Text(label=i18n("请输入list文件路径"), value="")
        with gr.Row():
            button_convert_from_list = gr.Button(i18n("开始生成待参考列表"), variant="primary")
            text_convert_from_list_info = gr.Text(label=i18n("参考列表生成结果"), value="", interactive=False)
        gr.Markdown(value=i18n("1.2：选择基准音频，执行相似度匹配，并分段随机抽样"))
        text_sample_dir = gr.Text(label=i18n("参考音频抽样目录"), value="", interactive=False)
        button_convert_from_list.click(convert_from_list, [text_work_space_dir, text_list_input],
                                       [text_convert_from_list_info, text_sample_dir])
        with gr.Row():
            text_base_voice_path = gr.Text(label=i18n("请输入基准音频路径"), value="")
            text_subsection_num = gr.Text(label=i18n("请输入分段数"), value="10")
            text_sample_num = gr.Text(label=i18n("请输入每段随机抽样个数"), value="4")
            checkbox_similarity_output = gr.Checkbox(label=i18n("是否将相似度匹配结果输出到临时目录？"), show_label=True)
        with gr.Row():
            button_sample = gr.Button(i18n("开始分段随机抽样"), variant="primary")
            text_sample_info = gr.Text(label=i18n("分段随机抽样结果"), value="", interactive=False)
    with gr.Accordion(label=i18n("第二步：基于参考音频和测试文本，执行批量推理"), open=False):
        gr.Markdown(value=i18n("2.1：配置推理服务参数信息，参考音频路径/文本和角色情绪二选一，如果是角色情绪，需要先执行第四步，"
                               "将参考音频打包配置到推理服务下，在推理前，请确认完整请求地址是否与正常使用时的一致，包括角色名称，尤其是文本分隔符是否正确"))
        text_model_inference_voice_dir = gr.Text(label=i18n("待推理的参考音频所在目录"), value="", interactive=True)
        text_url = gr.Text(label=i18n("请输入推理服务请求地址与参数"), value="")
        with gr.Row():
            text_text = gr.Text(label=i18n("请输入文本参数名"), value="text")
            text_ref_path = gr.Text(label=i18n("请输入参考音频路径参数名"), value="")
            text_ref_text = gr.Text(label=i18n("请输入参考音频文本参数名"), value="")
            text_emotion = gr.Text(label=i18n("请输入角色情绪参数名"), value="emotion")
        text_whole_url = gr.Text(label=i18n("完整地址"), value="", interactive=False)
        text_url.input(whole_url, [text_url, text_text, text_ref_path, text_ref_text, text_emotion],
                       [text_whole_url])
        text_text.input(whole_url, [text_url, text_text, text_ref_path, text_ref_text, text_emotion],
                        [text_whole_url])
        text_ref_path.input(whole_url, [text_url, text_text, text_ref_path, text_ref_text, text_emotion],
                            [text_whole_url])
        text_ref_text.input(whole_url, [text_url, text_text, text_ref_path, text_ref_text, text_emotion],
                            [text_whole_url])
        text_emotion.input(whole_url, [text_url, text_text, text_ref_path, text_ref_text, text_emotion],
                           [text_whole_url])
        gr.Markdown(value=i18n("2.2：配置待推理文本，一句一行，不要太多，10条即可"))
        default_test_content_path = params.default_test_text_path
        text_test_content = gr.Text(label=i18n("请输入待推理文本路径"), value=default_test_content_path)
        gr.Markdown(value=i18n("2.3：启动推理服务，如果还没启动的话"))
        gr.Markdown(value=i18n("2.4：开始批量推理，这个过程比较耗时，可以去干点别的"))
        with gr.Row():
            button_model_inference = gr.Button(i18n("开启批量推理"), variant="primary")
            text_model_inference_info = gr.Text(label=i18n("批量推理结果"), value="", interactive=False)
    with gr.Accordion(label=i18n("第三步：进行参考音频效果校验与筛选"), open=False):
        gr.Markdown(value=i18n("3.1：启动asr，获取推理音频文本"))
        text_asr_audio_dir = gr.Text(label=i18n("待asr的音频所在目录"), value="", interactive=False)
        button_model_inference.click(model_inference,
                                     [text_work_space_dir, text_model_inference_voice_dir, text_url,
                                      text_text, text_ref_path, text_ref_text, text_emotion,
                                      text_test_content], [text_model_inference_info, text_asr_audio_dir])
        with gr.Row():
            dropdown_asr_model = gr.Dropdown(
                label=i18n("ASR 模型"),
                choices=[],
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
        with gr.Row():
            button_asr = gr.Button(i18n("启动asr"), variant="primary")
            text_asr_info = gr.Text(label=i18n("asr结果"), value="", interactive=False)
        gr.Markdown(value=i18n("3.2：启动文本相似度分析"))
        text_text_similarity_analysis_path = gr.Text(label=i18n("待分析的文件路径"), value="", interactive=False)
        button_asr.click(asr, [text_work_space_dir, text_asr_audio_dir, dropdown_asr_model,
                               dropdown_asr_size, dropdown_asr_lang],
                         [text_asr_info, text_text_similarity_analysis_path])
        with gr.Row():
            button_text_similarity_analysis = gr.Button(i18n("启动文本相似度分析"), variant="primary")
            text_text_similarity_analysis_info = gr.Text(label=i18n("文本相似度分析结果"), value="", interactive=False)
            button_text_similarity_analysis.click(text_similarity_analysis, [text_work_space_dir,
                                                                             text_text_similarity_analysis_path],
                                                  [text_text_similarity_analysis_info])
        gr.Markdown(value=i18n("3.3：根据相似度分析结果，重点检查最后几条是否存在复读等问题"))
        gr.Markdown(value=i18n("3.4：对结果按音频相似度排序，筛选低音质音频"))
        with gr.Row():
            text_base_audio_path = gr.Text(label=i18n("请输入基准音频"), value="")
            text_compare_audio_dir = gr.Text(label=i18n("请输入待比较的音频文件目录"), value="")
        with gr.Row():
            button_similarity_audio_output = gr.Button(i18n("输出相似度-参考音频到临时目录"), variant="primary")
            text_similarity_audio_output_info = gr.Text(label=i18n("输出结果"), value="", interactive=False)
            button_similarity_audio_output.click(similarity_audio_output,
                                                 [text_work_space_dir, text_base_audio_path,
                                                  text_compare_audio_dir], [text_similarity_audio_output_info])
        with gr.Row():
            text_sync_ref_audio_dir = gr.Text(label=i18n("参考音频路径"), value="", interactive=True)
            text_sync_inference_audio_dir = gr.Text(label=i18n("被同步的推理音频路径"), value="", interactive=False)
        with gr.Row():
            button_sync_ref_audio = gr.Button(i18n("将参考音频的删除情况，同步到推理音频目录"), variant="primary")
            text_sync_ref_info = gr.Text(label=i18n("同步结果"), value="", interactive=False)
            button_sync_ref_audio.click(sync_ref_audio, [text_work_space_dir, text_sync_ref_audio_dir,
                                                         text_sync_inference_audio_dir], [text_sync_ref_info])
    with gr.Accordion("第四步：生成参考音频配置文本", open=False):
        gr.Markdown(value=i18n("4.1：编辑模板"))
        default_template_path = params.default_template_path
        default_template_content = common.read_file(default_template_path)
        text_template_path = gr.Text(label=i18n("模板文件路径"), value=default_template_path, interactive=False)
        text_template = gr.Text(label=i18n("模板内容"), value=default_template_content, lines=10)
        gr.Markdown(value=i18n("4.2：生成配置"))
        text_sync_ref_audio_dir2 = gr.Text(label=i18n("参考音频路径"), value="", interactive=True)
        with gr.Row():
            button_create_config = gr.Button(i18n("生成配置"), variant="primary")
            text_create_config_info = gr.Text(label=i18n("生成结果"), value="", interactive=False)
            button_create_config.click(create_config,
                                       [text_work_space_dir, text_template, text_sync_ref_audio_dir2],
                                       [text_create_config_info])
    button_sample.click(sample, [text_work_space_dir, text_sample_dir, text_base_voice_path,
                                 text_subsection_num, text_sample_num, checkbox_similarity_output],
                        [text_sample_info, text_model_inference_voice_dir, text_sync_ref_audio_dir,
                         text_sync_ref_audio_dir2])

app.launch(
    server_port=9423,
    quiet=True,
)
