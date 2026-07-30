import os
import argparse
from collections import defaultdict
from operator import itemgetter
from Ref_Audio_Selector.common.time_util import timeit_decorator
import Ref_Audio_Selector.tool.text_comparison.text_comparison as text_comparison
import Ref_Audio_Selector.config_param.config_params as params
import Ref_Audio_Selector.common.common as common
from Ref_Audio_Selector.config_param.log_config import logger


def parse_asr_file(file_path):
    output = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 假设每行都是正确的格式，且"|"'是固定分隔符
            input_file_path, original_text, language, asr_text = line.strip().split('|')

            emotion = common.get_filename_without_extension(input_file_path)

            # 将解析出的数据构造成新的字典或元组等结构
            parsed_data = {
                'emotion': emotion,
                'input_file_path': input_file_path,
                'original_text': original_text,
                'language': language,
                'asr_text': asr_text,
                'similarity_score': 0
            }

            output.append(parsed_data)

    return output


@timeit_decorator
def calculate_similarity_and_append_to_list(input_list, boundary):
    all_count = len(input_list)
    has_been_processed_count = 0
    for item in input_list:
        original_score, similarity_score = text_comparison.calculate_result(item['original_text'], item['asr_text'], boundary)
        item['similarity_score'] = similarity_score
        item['original_score'] = original_score
        has_been_processed_count += 1
        logger.info(f'进度：{has_been_processed_count}/{all_count}')

    return input_list


def calculate_average_similarity_by_emotion(data_list):
    result_dict = defaultdict(list)

    for item in data_list:
        emotion = item['emotion']
        similarity_score = item['similarity_score']
        result_dict[emotion].append(similarity_score)

    average_scores = [{'emotion': emotion, 'average_similarity_score': sum(scores) / len(scores), 'count': len(scores)}
                      for emotion, scores in result_dict.items()]

    average_scores.sort(key=lambda x: x['average_similarity_score'], reverse=True)

    return average_scores


def group_and_sort_by_field(data, group_by_field):
    # 创建一个空的结果字典，键是group_by_field指定的字段，值是一个列表
    result_dict = defaultdict(list)

    # 遍历输入列表
    for item in data:
        # 根据指定的group_by_field将当前元素添加到对应键的列表中
        key_to_group = item[group_by_field]
        result_dict[key_to_group].append(item)

    # 对每个键对应的列表中的元素按similarity_score降序排序
    for key in result_dict:
        result_dict[key].sort(key=itemgetter('similarity_score'), reverse=True)

    # 将结果字典转换为列表，每个元素是一个包含键（emotion或original_text）和排序后数组的元组
    result_list = [(k, v) for k, v in result_dict.items()]

    return result_list


def format_list_to_text(data_list, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write('放大后的相似度分值|原始分值|ASR文本|原文文本\n')
        for key, items in data_list:
            # 写入情绪标题
            output_file.write(key + '\n')

            # 写入每条记录
            for item in items:
                formatted_line = f"{item['similarity_score']}|{item['original_score']}|{item['asr_text']}|{item['original_text']}\n"
                output_file.write(formatted_line)


def format_list_to_emotion(data_list, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write('放大后的相似度分值|原始分值|ASR文本|情绪类型\n')
        for key, items in data_list:
            # 写入情绪标题
            output_file.write(key + '\n')

            # 写入每条记录
            for item in items:
                formatted_line = f"{item['similarity_score']}|{item['original_score']}|{item['asr_text']}|{item['emotion']}\n"
                output_file.write(formatted_line)


@timeit_decorator
def process(asr_file_path, output_dir, similarity_enlarge_boundary):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    records = parse_asr_file(asr_file_path)
    calculate_similarity_and_append_to_list(records, similarity_enlarge_boundary)
    average_similarity_list = calculate_average_similarity_by_emotion(records)

    average_similarity_file = os.path.join(output_dir,
                                           f'{params.text_emotion_average_similarity_report_filename}.txt')
    average_similarity_content = \
        '\n'.join([f"{item['average_similarity_score']}|{item['count']}|{item['emotion']}" for item in average_similarity_list])
    common.write_text_to_file(average_similarity_content, average_similarity_file)

    emotion_detail_list = group_and_sort_by_field(records, 'emotion')

    emotion_detail_file = os.path.join(output_dir, f'{params.text_similarity_by_emotion_detail_filename}.txt')
    format_list_to_text(emotion_detail_list, emotion_detail_file)

    original_text_detail_list = group_and_sort_by_field(records, 'original_text')

    original_text_detail_file = os.path.join(output_dir, f'{params.text_similarity_by_text_detail_filename}.txt')
    format_list_to_emotion(original_text_detail_list, original_text_detail_file)

    logger.info('文本相似度分析完成。')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process ASR files and analyze similarity.")

    parser.add_argument("-a", "--asr_file_path", type=str, required=True,
                        help="Path to the directory containing ASR files or path to a single ASR file.")

    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Path to the directory where the analysis results should be saved.")

    parser.add_argument("-b", "--similarity_enlarge_boundary", type=float, required=True,
                        help="Similarity score boundary value to be used in your calculations.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cmd = parse_arguments()
    # print(cmd)
    process(cmd.asr_file_path, cmd.output_dir, cmd.similarity_enlarge_boundary)
