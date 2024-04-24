import os
import shutil
from config import python_exec
from subprocess import Popen

def convert_from_list(list_file, output_dir):
    # 创建输出目录，如果它不存在的话
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 解析.list文件，并操作文件
    with open(list_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('|')
        if len(parts) != 4:
            print(f"Line format incorrect: {line}")
            continue

        audio_path, _, _, transcription = parts

        # 构建新的文件名和路径
        new_filename = transcription + '.wav'
        # new_filename = new_filename.replace(' ', '_')  # 移除空格
        # new_filename = ''.join(e for e in new_filename if e.isalnum() or e in ['_', '.'])  # 移除非法字符
        new_path = os.path.join(output_dir, new_filename)

        # 如果目标文件已存在，不要覆盖
        if os.path.exists(new_path):
            print(f"File already exists: {new_path}")
            continue

        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"Audio file does not exist: {audio_path}")
                continue

            # 复制音频文件到output目录并重命名
            shutil.copy2(audio_path, new_path)
            print(f"File copied and renamed to: {new_path}")
        except Exception as e:
            print(f"An error occurred while processing: {audio_path}")
            print(e)

    print("Processing complete.")


def sample(output_audio_dir, similarity_list, subsection_num, sample_num):
    # 按照相似度分值降序排序相似度列表
    similarity_list.sort(key=lambda x: x['score'], reverse=True)

    # 计算每段的起始索引
    step = len(similarity_list) // subsection_num
    if len(similarity_list) % subsection_num != 0:
        step += 1

    # 分段并随机采样
    for i in range(subsection_num):
        start = i * step
        end = (i + 1) * step
        end = min(end, len(similarity_list))  # 防止最后一段越界
        
        num = min(sample_num, len(similarity_list[start:end]))

        # 随机采样
        random.shuffle(similarity_list[start:end])
        sampled_subsection = similarity_list[start:start+num]

        # 创建并进入子目录
        subdir_name = f'subsection_{i+1}'
        subdir_path = os.path.join(output_audio_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)

        # 复制采样结果的音频到子目录
        for item in sampled_subsection:
            src_path = item['wav_path']
            dst_path = os.path.join(subdir_path, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)

    print("Sampling completed.")



def start_similarity_analysis(work_space_dir, sample_dir, base_voice_path, need_similarity_output):

    similarity_list = None

    similarity_dir = os.path.join(work_space_dir, 'similarity')
    os.makedirs(similarity_dir, exist_ok=True)

    base_voice_file_name = ref_audio_opt.get_filename_without_extension(base_voice_path)
    similarity_file = os.path.join(similarity_dir, f'{base_voice_file_name}.txt')

    global p_similarity
    if(p_similarity==None):
        cmd = f'"{python_exec}" tools/speaker_verification/audio_similarity.py '
        cmd += f' -r "{base_voice_path}"'
        cmd += f' -c "{sample_dir}"'
        cmd += f' -o {similarity_file}'

        print(cmd)
        p_similarity = Popen(cmd, shell=True)
        p_similarity.wait()

        if need_similarity_output:
            similarity_list = ref_audio_opt.parse_similarity_file(similarity_file)
            similarity_file_dir = os.path.dirname(similarity_dir, base_voice_file_name)
            ref_audio_opt.copy_and_move(similarity_file_dir, similarity_list)

        p_similarity=None
        return similarity_list
    else:
        return similarity_list


def parse_similarity_file(file_path):
    """
    解析指定文本文件，将其中的内容以元组形式存入列表。

    参数:
        file_path (str): 待解析的文本文件路径。

    返回:
        list[tuple[float, str]]: 存储浮点数和路径的元组列表。
    """
    result_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾换行符并按'|'分割
            score, filepath = line.strip().split('|')

            # 将浮点数字符串转换为浮点数类型
            score = float(score)

            # 将得分和路径作为元组添加到结果列表
            result_list.append({
                'score': score,
                'wav_path': filepath
            })

    return result_list


def copy_and_move(output_audio_directory, similarity_scores):

    # 确保新目录存在
    if not os.path.exists(output_audio_directory):
        os.makedirs(output_audio_directory)

    # 遍历并复制文件
    for item in similarity_scores:
        # 构造新的文件名
        base_name = os.path.basename(item['wav_path'])[:-4]  # 去掉.wav扩展名
        new_name = f"{item['score']}-{base_name}.wav"

        # 新文件的完整路径
        new_path = os.path.join(output_audio_directory, new_name)

        # 复制文件到新目录
        shutil.copyfile(item['wav_path'], new_path)

    print("已完成复制和重命名操作。")


def get_filename_without_extension(file_path):
    """
    Given a file path string, returns the file name without its extension.

    Parameters:
    file_path (str): The full path to the file.

    Returns:
    str: The file name without its extension.
    """
    base_name = os.path.basename(file_path)  # Get the base name (file name with extension)
    file_name, file_extension = os.path.splitext(base_name)  # Split the base name into file name and extension
    return file_name  # Return the file name without extension


