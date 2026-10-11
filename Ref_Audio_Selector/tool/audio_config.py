import os
import platform


def generate_audio_config(work_space_dir, template_str, audio_list, output_file_path):
    # 定义一个空字符串来存储最终要写入文件的内容
    file_content = ""

    # 遍历参考音频列表
    for audio_info in audio_list:
        emotion = audio_info['emotion']
        ref_path = audio_info['ref_path']
        ref_text = audio_info['ref_text']

        relative_path = os.path.relpath(ref_path, work_space_dir)
        if platform.system() == 'Windows':
            relative_path = relative_path.replace('\\', '/')

        # 使用字符串模板替换变量
        formatted_line = template_str.replace('${emotion}', emotion).replace('${ref_path}', relative_path).replace(
            '${ref_text}', ref_text)

        # 将格式化后的行添加到内容中，使用逗号和换行符分隔
        file_content += formatted_line + ",\n"

    # 删除最后一个逗号和换行符，确保格式整洁
    file_content = file_content[:-2]

    # 将内容写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(file_content)
