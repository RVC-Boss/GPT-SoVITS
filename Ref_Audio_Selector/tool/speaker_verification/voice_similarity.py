import argparse
import os
import torchaudio
import torchaudio.transforms as T
import platform
import Ref_Audio_Selector.config_param.config_params as params
import Ref_Audio_Selector.config_param.log_config as log_config
from Ref_Audio_Selector.common.time_util import timeit_decorator
from Ref_Audio_Selector.common.model_manager import speaker_verification_models as models

from modelscope.pipelines import pipeline


def init_model(model_type='speech_campplus_sv_zh-cn_16k-common'):
    log_config.logger.info(f'人声识别模型类型：{model_type}')
    return pipeline(
        task=models[model_type]['task'],
        model=models[model_type]['model'],
        model_revision=models[model_type]['model_revision']
    )


@timeit_decorator
def compare_audio_and_generate_report(reference_audio_path, comparison_dir_path, output_file_path, model_type):
    sv_pipeline = init_model(model_type)

    # Step 1: 获取比较音频目录下所有音频文件的路径
    comparison_audio_paths = [os.path.join(comparison_dir_path, f) for f in os.listdir(comparison_dir_path) if
                              f.endswith('.wav')]

    if platform.system() == 'Windows':
        # 因为这个模型是基于16k音频数据训练的，为了避免后续比较时，每次都对参考音频进行重采样，所以，提前进行了采样
        # windows不支持torchaudio.sox_effects.apply_effects_tensor，所以改写了依赖文件中的重采样方法
        # 改用torchaudio.transforms.Resample进行重采样，如果在非windows环境下，没有更改依赖包的采样方法的话，
        # 使用这段代码进行预采样会出现因为采样方法不同，而导致的模型相似度计算不准确的问题
        # 当然如果在windows下，使用了其他的采样方法，也会出现不准确的问题
        if params.enable_pre_sample == 'true':
            reference_audio_16k = ensure_16k_wav(reference_audio_path)
        else:
            reference_audio_16k = reference_audio_path
    else:
        reference_audio_16k = reference_audio_path

    # Step 2: 用参考音频依次比较音频目录下的每个音频，获取相似度分数及对应路径
    all_count = len(comparison_audio_paths)
    has_processed_count = 0
    similarity_scores = []
    for audio_path in comparison_audio_paths:
        score = sv_pipeline([reference_audio_16k, audio_path])['score']
        similarity_scores.append({
            'score': score,
            'path': audio_path
        })
        has_processed_count += 1
        log_config.logger.info(f'进度：{has_processed_count}/{all_count}')

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

        # 创建临时文件夹
        os.makedirs(params.temp_dir, exist_ok=True)

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

    # Model Type
    parser.add_argument("-m", "--model_type", type=str, required=True,
                        help="Path to the model type.")

    return parser.parse_args()


if __name__ == '__main__':
    cmd = parse_arguments()
    compare_audio_and_generate_report(
        reference_audio_path=cmd.reference_audio,
        comparison_dir_path=cmd.comparison_dir,
        output_file_path=cmd.output_file,
        model_type=cmd.model_type,
    )

    # compare_audio_and_generate_report(
    #     reference_audio_path="D:/tt/渡鸦/refer_audio_all/也对，你的身份和我们不同吗？.wav",
    #     comparison_dir_path='D:/tt/渡鸦/refer_audio_all',
    #     output_file_path='D:/tt/渡鸦/test.txt',
    # )
