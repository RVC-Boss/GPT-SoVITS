import os
import shutil
import random
import librosa
from Ref_Audio_Selector.config_param.log_config import logger


def check_audio_duration(path, min_duration=3, max_duration=10):
    try:

        # 直接计算音频文件的时长（单位：秒）
        duration = librosa.get_duration(filename=path)

        # 判断时长是否在3s至10s之间
        if min_duration <= duration <= max_duration:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"无法打开或处理音频文件：{e}")
        return None


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
            logger.error(f"Line format incorrect: {line}")
            continue

        audio_path, _, _, transcription = parts

        # 构建新的文件名和路径
        new_filename = transcription.strip() + '.wav'
        # new_filename = new_filename.replace(' ', '_')  # 移除空格
        # new_filename = ''.join(e for e in new_filename if e.isalnum() or e in ['_', '.'])  # 移除非法字符
        new_path = os.path.join(output_dir, new_filename)

        # 如果目标文件已存在，不要覆盖
        if os.path.exists(new_path):
            logger.info(f"File already exists: {new_path}")
            continue

        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                logger.info(f"Audio file does not exist: {audio_path}")
                continue

            if check_audio_duration(audio_path):
                # 复制音频文件到output目录并重命名
                shutil.copy2(audio_path, new_path)
                logger.info(f"File copied and renamed to: {new_path}")
            else:
                logger.info(f"File skipped due to duration: {audio_path}")

        except Exception as e:
            logger.error(f"An error occurred while processing: {audio_path}")
            logger.error(e)

    logger.info("Processing complete.")


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

        # 创建子列表
        subsection = similarity_list[start:end]
        # 在子列表上随机打乱
        random.shuffle(subsection)

        # 从打乱后的子列表中抽取相应数量的个体
        num = min(sample_num, len(subsection))
        sampled_subsection = subsection[:num]

        # 创建并进入子目录
        subdir_name = f'emotion_{i + 1}'
        subdir_path = os.path.join(output_audio_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)

        # 复制采样结果的音频到子目录
        for item in sampled_subsection:
            src_path = item['wav_path']
            dst_path = os.path.join(subdir_path, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)

    logger.info("Sampling completed.")


def parse_similarity_file(file_path):
    """
    解析指定文本文件，将其中的内容以元组形式存入列表。

    参数:
        file_path (str): 待解析的文本文件路径。

    返回:
        list[tuple[float, str]]: 存储浮点数和路径的元组列表。
    """
    result_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
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
        new_name = f"{item['score'] * 10000:04.0f}-{base_name}.wav"

        # 新文件的完整路径
        new_path = os.path.join(output_audio_directory, new_name)

        # 复制文件到新目录
        shutil.copyfile(item['wav_path'], new_path)

    logger.info("已完成复制和重命名操作。")


if __name__ == '__main__':
    similarity_list = parse_similarity_file("D:/tt/similarity/啊，除了伊甸和樱，竟然还有其他人会提起我？.txt")
    sample('D:/tt/similarity/output', similarity_list, 10, 4)
