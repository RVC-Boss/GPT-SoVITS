import os
import shutil
import Ref_Audio_Selector.common.common as common
import Ref_Audio_Selector.config_param.config_params as params
from Ref_Audio_Selector.config_param.log_config import logger


def remove_matching_audio_files_in_text_dir(text_dir, emotions_list):
    count = 0
    emotions = [item['emotion'] for item in emotions_list]
    for root, dirs, files in os.walk(text_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion_tag = os.path.basename(file)[:-4]
                if emotion_tag not in emotions:
                    file_path = os.path.join(root, file)
                    logger.info(f"Deleting file: {file_path}")
                    try:
                        os.remove(file_path)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")

    return count


def delete_emotion_subdirectories(emotion_dir, emotions_list):
    count = 0

    emotions = [item['emotion'] for item in emotions_list]

    for entry in os.listdir(emotion_dir):
        entry_path = os.path.join(emotion_dir, entry)
        if os.path.isdir(entry_path):
            if entry not in emotions:
                logger.info(f"Deleting directory: {entry_path}")
                try:
                    # 使用shutil.rmtree删除整个子目录及其内容
                    shutil.rmtree(entry_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting directory {entry_path}: {e}")

    return count


def sync_ref_audio(ref_audio_dir, inference_audio_dir):
    ref_audio_manager = common.RefAudioListManager(ref_audio_dir)
    ref_list = ref_audio_manager.get_ref_audio_list()
    text_dir = os.path.join(inference_audio_dir, params.inference_audio_text_aggregation_dir)
    emotion_dir = os.path.join(inference_audio_dir, params.inference_audio_emotion_aggregation_dir)
    delete_text_wav_num = remove_matching_audio_files_in_text_dir(text_dir, ref_list)
    delete_emotion_dir_num = delete_emotion_subdirectories(emotion_dir, ref_list)
    return delete_text_wav_num, delete_emotion_dir_num
