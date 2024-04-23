import os
import shutil


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
