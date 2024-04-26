import argparse
import os
from Ref_Audio_Selector.common.time_util import timeit_decorator

from modelscope.pipelines import pipeline

sv_pipeline = pipeline(
    task='speaker-verification',
    model='Ref_Audio_Selector/tool/speaker_verification/models/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)


@timeit_decorator
def compare_audio_and_generate_report(reference_audio_path, comparison_dir_path, output_file_path):
    # Step 1: 获取比较音频目录下所有音频文件的路径
    comparison_audio_paths = [os.path.join(comparison_dir_path, f) for f in os.listdir(comparison_dir_path) if
                              f.endswith('.wav')]

    # Step 2: 用参考音频依次比较音频目录下的每个音频，获取相似度分数及对应路径
    similarity_scores = []
    for audio_path in comparison_audio_paths:
        score = sv_pipeline([reference_audio_path, audio_path])['score']
        similarity_scores.append({
            'score': score,
            'path': audio_path
        })
        print(f'similarity score: {score}, path: {audio_path}')

    # Step 3: 根据相似度分数降序排列
    similarity_scores.sort(key=lambda x: x['score'], reverse=True)

    # Step 4: 处理输出文件不存在的情况，创建新文件
    if not os.path.exists(output_file_path):
        open(output_file_path, 'w').close()  # Create an empty file

    # Step 5: 将排序后的结果写入输出结果文件（支持中文）
    formatted_scores = [f'{item["score"]}|{item["path"]}' for item in similarity_scores]
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 使用'\n'将每个字符串分开，使其写入不同行
        content = '\n'.join(formatted_scores)
        f.write(content)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio processing script arguments")

    # Reference audio path
    parser.add_argument("-r", "--reference_audio", type=str, required=True,
                        help="Path to the reference WAV file.")

    # Comparison directory path
    parser.add_argument("-c", "--comparison_dir", type=str, required=True,
                        help="Path to the directory containing comparison WAV files.")

    # Output file path
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="Path to the output file where results will be written.")

    return parser.parse_args()


if __name__ == '__main__':
    cmd = parse_arguments()
    print(cmd)
    compare_audio_and_generate_report(
        reference_audio_path=cmd.reference_audio,
        comparison_dir_path=cmd.comparison_dir,
        output_file_path=cmd.output_file,
    )
