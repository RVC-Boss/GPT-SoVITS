import os


# Supporting third-party datasets with the format where each audio file corresponds to a text file.
# For example, voice01.wav -> voice01.txt.
def convert_dataset(input_folder, output_file, language='zh'):
    audio_files = []
    for f in os.listdir(input_folder):
        if f.endswith('.wav'):
            audio_files.append(f)

    with open(output_file, 'w', encoding='utf-8') as output:

        for audio_file in audio_files:
            audio_path = os.path.join(input_folder, audio_file)
            text_file = os.path.join(input_folder, audio_file.replace('.wav', '.txt'))

            with open(text_file, 'r', encoding='utf-8') as text_content:
                text = text_content.read().replace('\n', '')

            speaker_name = os.path.splitext(audio_file)[0]

            output_line = f'{audio_path}|{speaker_name}|{language}|{text}\n'
            output.write(output_line)

