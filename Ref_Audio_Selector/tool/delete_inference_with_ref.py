import os
import shutil
import Ref_Audio_Selector.common.common as common
import Ref_Audio_Selector.config.config_params as params


def remove_matching_audio_files_in_text_dir(text_dir, emotions_list):
    count = 0
    for root, dirs, files in os.walk(text_dir):
        for emotion_dict in emotions_list:
            emotion_tag = emotion_dict['emotion']
            wav_file_name = f"{emotion_tag}.wav"
            file_path = os.path.join(root, wav_file_name)
            if os.path.exists(file_path):
                print(f"Deleting file: {file_path}")
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
    return count


def delete_emotion_subdirectories(emotion_dir, emotions_list):
    """
    根据给定的情绪数组，删除emotion目录下对应情绪标签的子目录。

    参数:
    emotions_list (List[Dict]): 每个字典包含'emotion'字段。
    base_dir (str): 子目录所在的基础目录，默认为'emotion')。

    返回:
    None
    """
    count = 0
    for emotion_dict in emotions_list:
        emotion_folder = emotion_dict['emotion']
        folder_path = os.path.join(emotion_dir, emotion_folder)

        # 检查emotion子目录是否存在
        if os.path.isdir(folder_path):
            print(f"Deleting directory: {folder_path}")
            try:
                # 使用shutil.rmtree删除整个子目录及其内容
                shutil.rmtree(folder_path)
                count += 1
            except Exception as e:
                print(f"Error deleting directory {folder_path}: {e}")
    return count


def sync_ref_audio(ref_audio_dir, inference_audio_dir):
    ref_audio_manager = common.RefAudioListManager(ref_audio_dir)
    ref_list = ref_audio_manager.get_ref_audio_list()
    text_dir = os.path.join(inference_audio_dir, params.inference_audio_text_aggregation_dir)
    emotion_dir = os.path.join(inference_audio_dir, params.inference_audio_emotion_aggregation_dir)
    delete_text_wav_num = remove_matching_audio_files_in_text_dir(text_dir, ref_list)
    delete_emotion_dir_num = delete_emotion_subdirectories(emotion_dir, ref_list)
    return delete_text_wav_num, delete_emotion_dir_num
