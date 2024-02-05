import os
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from glob import glob
from faster_whisper import WhisperModel

def main(input_folder, output_folder, output_filename, language):
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    output_file = os.path.join(output_folder, output_filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_file, 'w', encoding='utf-8') as f:
        for file in glob(os.path.join(input_folder, '**/*.wav'), recursive=True):
            segments, _ = model.transcribe(file, beam_size=10, vad_filter=True,
                                           vad_parameters=dict(min_silence_duration_ms=700), language=language)
            segments = list(segments)

            filename = os.path.basename(file).replace('.wav', '')
            directory = os.path.dirname(file)

            result_line = f"{file}|{language.upper()}|{segments[0].text}\n"
            f.write(result_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the folder containing WAV files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder to store transcriptions.")
    parser.add_argument("-f", "--output_filename", type=str, default="transcriptions.txt", help="Name of the output text file.")
    parser.add_argument("-l", "--language", type=str, default='zh', choices=['zh', 'en', ...],
                        help="Language of the audio files.")

    cmd = parser.parse_args()

    input_folder = cmd.input_folder
    output_folder = cmd.output_folder
    output_filename = cmd.output_filename
    language = cmd.language
    main(input_folder, output_folder, output_filename, language)