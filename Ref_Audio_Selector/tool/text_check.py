import os
import Ref_Audio_Selector.common.common as common
import Ref_Audio_Selector.tool.audio_check as audio_check
from Ref_Audio_Selector.config_param.log_config import logger


def parse_text_similarity_result_txt(file_path):
    """
    解析指定格式的txt文件，每行格式：f"{item['average_similarity_score']}|{item['count']}|{item['emotion']}"

    :param file_path: txt文件的路径
    :return: 包含解析后数据的字典列表
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用'|'作为分隔符分割每行数据
            parts = line.strip().split('|')
            if len(parts) == 3:
                # 将分割后的字符串转换为浮点数、整数和字符串
                try:
                    item = {
                        'average_similarity_score': float(parts[0]),
                        'count': int(parts[1]),
                        'emotion': parts[2]
                    }
                    data_list.append(item)
                except ValueError as e:
                    # 如果转换失败，打印错误信息并跳过该行
                    logger.error(f"Error parsing line: {line.strip()} - {e}")

    return data_list


def remove_low_similarity_files(ref_audio_list, report_list, audio_text_similarity_boundary):
    """
    根据条件删除低相似度音频文件并返回删除数量。
    
    :param ref_audio_list: 包含音频路径和情感属性的列表
    :param report_list: 包含相似度评分和情感属性的列表
    :param audio_text_similarity_boundary: 相似度阈值
    :return: 删除的文件数量
    """
    deleted_count = 0

    # 筛选出平均相似度低于阈值的报告
    low_similarity_reports = [report for report in report_list if
                              report['average_similarity_score'] < audio_text_similarity_boundary]

    # 遍历低相似度报告，查找并删除对应音频文件
    for report in low_similarity_reports:
        emotion = report['emotion']
        # 查找ref_audio_list中相同情感的音频文件路径
        matching_refs = [ref for ref in ref_audio_list if ref['emotion'] == emotion]
        for match in matching_refs:
            ref_path = match['ref_path']
            # 检查文件是否存在，然后尝试删除
            if os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                    deleted_count += 1
                    logger.info(f"Deleted file: {ref_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {ref_path}: {e}")
            else:
                logger.error(f"File not found: {ref_path}")

    return deleted_count


def delete_ref_audio_below_boundary(ref_audio_path, text_similarity_result_path, sync_inference_audio_dir,
                                    audio_text_similarity_boundary):
    ref_audio_list = common.RefAudioListManager(ref_audio_path).get_ref_audio_list()
    report_list = parse_text_similarity_result_txt(text_similarity_result_path)
    count = remove_low_similarity_files(ref_audio_list, report_list, audio_text_similarity_boundary)
    audio_check.sync_ref_audio(ref_audio_path, sync_inference_audio_dir)
    return count
