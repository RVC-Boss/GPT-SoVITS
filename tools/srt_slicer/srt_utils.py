import srt
import shutil

def parse_srt_with_lib(content):

    subtitles = list(srt.parse(content))
    return subtitles

def generate_srt_with_lib(subtitles):
    content = srt.compose(subtitles)
    return content

def merge_subtitles_with_lib(subtitles, short_interval, max_interval, max_text_length=30, add_period=True, merge_zero_interval=True):
    # 标点符号
    punctuations = ["。","!", "！", "？", "?", "；", ";",  "…"]
    punctuations_extanded = punctuations
    punctuations_extanded.extend([ "：", ":", "，", ",", "—",])
    
    # 直接合并间隔特别短的字幕
    if merge_zero_interval:
        eps = short_interval
        for i in range(len(subtitles) - 1, 0, -1):
            if subtitles[i-1].content[-1] in punctuations_extanded:
                continue
            if abs(subtitles[i].start.total_seconds() - subtitles[i-1].end.total_seconds()) < eps:
                subtitles[i - 1].end = subtitles[i].end
                subtitles[i - 1].content += subtitles[i].content
                subtitles.pop(i)
                
    merged_subtitles = []
    current_subtitle = None
    for subtitle in subtitles:
        if current_subtitle is None:
            current_subtitle = subtitle
        else:
            current_end = current_subtitle.end.total_seconds()
            next_start = subtitle.start.total_seconds()
            if current_subtitle.content[-1] not in punctuations and (next_start - current_end <= max_interval and count_words_multilang(current_subtitle.content + subtitle.content) < max_text_length):
                current_subtitle.end = subtitle.end
                comma = '，' if current_subtitle.content[-1] not in punctuations_extanded else ''
                current_subtitle.content += comma + subtitle.content
                
            else:
                if add_period and current_subtitle.content[-1] not in punctuations_extanded:
                    current_subtitle.content += '。'
                merged_subtitles.append(current_subtitle)
                current_subtitle = subtitle
    if current_subtitle is not None:
        merged_subtitles.append(current_subtitle)
    # 重新分配id，因为srt.compose需要id连续
    for i, subtitle in enumerate(merged_subtitles, start=1):
        subtitle.index = i
    return merged_subtitles



def count_words_multilang(text):
    # 初始化计数器
    word_count = 0
    in_word = False
    
    for char in text:
        if char.isspace():  # 如果当前字符是空格
            in_word = False
        elif char.isascii() and not in_word:  # 如果是ASCII字符（英文）并且不在单词内
            word_count += 1  # 新的英文单词
            in_word = True
        elif not char.isascii():  # 如果字符非英文
            word_count += 1  # 每个非英文字符单独计为一个字
    
    return word_count

import pydub, os

def slice_audio_with_lib(audio_path, save_folder, format, subtitles, pre_preserve_time, post_preserve_time, pre_silence_time, post_silence_time, language='auto', character='character'):
    list_file = os.path.join(save_folder, 'datamapping.list')
    with open(list_file, 'w', encoding="utf-8") as f:
        for i in range(len(subtitles)):
            subtitle = subtitles[i]
            start = subtitle.start.total_seconds() - pre_preserve_time
            end = subtitle.end.total_seconds() + post_preserve_time
            if i < len(subtitles) - 1:
                next_subtitle = subtitles[i + 1]
                end = min(end, 1.0/2*(subtitle.end.total_seconds()+next_subtitle.start.total_seconds()))
            if i > 0:
                prev_subtitle = subtitles[i - 1]
                start = max(start, 1.0/2*(prev_subtitle.end.total_seconds()+subtitle.start.total_seconds()))
            try:
                audio = pydub.AudioSegment.from_file(audio_path)
                sliced_audio = audio[int(start * 1000):int(end * 1000)]
                file_name = f'{character}_{i + 1:03d}.{format}'
                save_path = os.path.join(save_folder, file_name)
                sliced_audio.export(save_path, format=format)
                f.write(f"{file_name}|{character}|{language}|{subtitle.content}\n")
            except Exception as e:
                raise e
        
def merge_list_folders(first_list_file, second_list_file, character, first_folder, second_folder):
    merged_lines = []
    character1 = ""
    filenames = set()
    with open(first_list_file, 'r', encoding="utf-8") as f:
        first_list = f.readlines()
        for line in first_list:
            filename, character1, language, content = line.split('|')
            filenames.add(filename)
            if character=="" or character is None:
                character = character1
            new_line = f"{filename}|{character}|{language}|{content}"
            merged_lines.append(new_line)
    with open(second_list_file, 'r', encoding="utf-8") as f:
        second_list = f.readlines()
        for line in second_list:
            filename, _, language, content = line.split('|')
            orig_filename = filename
            num = 1
            while filename in filenames:
                filename = f"{filename.rsplit('.', 1)[0]}_{num}.{filename.rsplit('.', 1)[1]}"
                num += 1
            try:
                os.rename(os.path.join(second_folder, orig_filename), os.path.join(first_folder, filename))
            except Exception as e:
                raise e
            new_line = f"{filename}|{character}|{language}|{content}"
            merged_lines.append(new_line)
    os.remove(second_list_file)
    if not os.listdir(second_folder):
        os.rmdir(second_folder)
    with open(first_list_file, 'w', encoding="utf-8") as f:
        f.writelines(merged_lines)
    return "\n".join(merged_lines)
            
        
    