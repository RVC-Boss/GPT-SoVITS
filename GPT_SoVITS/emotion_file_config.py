import os
import shutil

class EmotionFile:
    def __init__(self) -> None:
        pass

    @staticmethod
    def process_audio_folder(folder_path):
        # 创建一个字典用于存储文件名和对应的路径
        audio_dict = {}
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 遍历参考音频文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 如果当前项是文件夹
                if os.path.isdir(root):
                    # 遍历文件夹中的文件
                    for subdir_root, _, subdir_files in os.walk(root):
                        for subdir_file in subdir_files:
                            # 将文件名和路径存储到字典中
                            # 去除后辍名
                            clean_subdir_file = subdir_file.split('.')[0]
                            audio_dict[clean_subdir_file] = os.path.join(subdir_root, subdir_file)
                else:
                    # 创建其他文件夹路径
                    other_folder_path = os.path.join(folder_path, '其他')
                    if not os.path.exists(other_folder_path):
                        os.makedirs(other_folder_path)
                    
                    # 移动文件到其他文件夹
                    shutil.move(root, other_folder_path)

        return audio_dict
    
    @staticmethod
    def getCurrentPath(audio_dict:dict[str, str],key:str)->str:
        return audio_dict[key]
    
    @staticmethod
    def getAllKeys(audio_dict:dict[str, str])->list[str]:
        return list(audio_dict.keys())

# 测试
reference_audio_folder = '参考音频'
audio_files_dict = EmotionFile.process_audio_folder(reference_audio_folder)
result = EmotionFile.getAllKeys(audio_files_dict)
print(result)
