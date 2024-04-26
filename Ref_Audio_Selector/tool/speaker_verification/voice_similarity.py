import argparse
import os
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
import Ref_Audio_Selector.config_param.config_params as params
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

    # 因为这个模型是基于16k音频数据训练的，为了避免后续比较时，每次都对参考音频进行重采样，所以，提前进行了采样
    reference_audio_16k = ensure_16k_wav(reference_audio_path)

    # Step 2: 用参考音频依次比较音频目录下的每个音频，获取相似度分数及对应路径
    similarity_scores = []
    for audio_path in comparison_audio_paths:
        score = sv_pipeline([reference_audio_16k, audio_path])['score']
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


def ensure_16k_wav(audio_file_path, target_sample_rate=16000):
    # 读取音频文件信息
    sample_rate, audio_data = sf.read(audio_file_path)

    # 检查采样率是否为16kHz
    if sample_rate == target_sample_rate:
        # 是16kHz采样率，直接返回原始文件路径
        return audio_file_path

    # 设置临时文件名
    temp_file_path = os.path.join(params.temp_dir, os.path.basename(audio_file_path))

    # 重采样至16kHz并保存到临时文件
    sf.write(temp_file_path, audio_data, samplerate=target_sample_rate, format="WAV")

    return temp_file_path


def ensure_16k_wav_2(audio_file_path, target_sample_rate=16000):
    """
    输入一个音频文件地址，判断其采样率并决定是否进行重采样，然后将结果保存到指定的输出文件。

    参数:
        audio_file_path (str): 音频文件路径。
        output_file_path (str): 保存重采样后音频数据的目标文件路径。
        target_sample_rate (int, optional): 目标采样率，默认为16000Hz。
    """
    # 读取音频文件并获取其采样率
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # 判断是否需要重采样
    if sample_rate == target_sample_rate:
        return audio_file_path
    else:

        # 创建Resample实例
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)

        # 应用重采样
        resampled_waveform = resampler(waveform)

        # 设置临时文件名
        temp_file_path = os.path.join(params.temp_dir, os.path.basename(audio_file_path))

        # 保存重采样后的音频到指定文件
        torchaudio.save(temp_file_path, resampled_waveform, target_sample_rate)

    return temp_file_path


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
