import argparse
import os
import sys
from tempfile import NamedTemporaryFile

import soundfile as sf

# This CLI lives in the GPT_SoVITS folder but needs to import the tools module, which is located outside that folder.
# GPT_SoVITS is not a namespace package, so we cannot use relative imports. As a workaround, we can add the current
# directory to the system path and invoke the CLI from the root of the project like this:
# > python GPT_SoVITS/inference_cli.py [arguments]
sys.path.append(os.getcwd())

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()


def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path,
               target_language, output_path, how_to_cut, top_k, top_p, temperature, ref_free, speed,
               additional_inp_refs):
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # Gradio prepares a list of NamedTemporaryFile objects for the additional reference audio files uploaded to the UI.
    # get_tts_wav expects the additional reference audio files to be supplied in this way, so, for compatibility, the
    # CLI must also wrap the specified file paths in NamedTemporaryFile objects.
    def make_tempfile(file_path):
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.name = file_path
        return temp_file
    additional_inputs = [make_tempfile(file_path) for file_path in additional_inp_refs] if additional_inp_refs else None

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language=i18n(ref_language), 
                                   text=target_text, 
                                   text_language=i18n(target_language),
                                   how_to_cut=how_to_cut,
                                   top_k=top_k,
                                   top_p=top_p,
                                   temperature=temperature,
                                   ref_free=ref_free,
                                   speed=speed,
                                   inp_refs=additional_inputs)

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--gpt_model', required=True, help="Path to the GPT model file")
    parser.add_argument('--sovits_model', required=True, help="Path to the SoVITS model file")
    parser.add_argument('--ref_audio', required=True, help="Path to the reference audio file")
    parser.add_argument('--ref_text', required=True, help="Path to the reference text file")
    parser.add_argument('--ref_language', required=True, choices=["中文", "英文", "日文", "粤语", "韩文", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], help="Language of the reference audio")
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--target_language', required=True, choices=["中文", "英文", "日文", "粤语", "韩文", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], help="Language of the target text")
    parser.add_argument('--output_path', required=True, help="Path to the output directory, where generated audio files will be saved.")
    parser.add_argument('--speed', type=float, default=1.0, help="Adjusts the speed of the generated audio without changing its pitch. Higher numbers = faster.")
    parser.add_argument('--how_to_cut', default="凑四句一切", choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"], help="The desired strategy for slicing up the prompt text. Audio will be generated for each slice and then concatenated together.")
    parser.add_argument('--top_k', type=int, default=15, help="Parameter for top-K filtering")
    parser.add_argument('--top_p', type=float, default=1.0, help="Parameter for nucleus filtering")
    parser.add_argument('--temperature', type=float, default=1.0, help="Inverse scale factor for the logits. Temperatures between 0 and 1.0 produce audio that sounds more like the speaker in the reference audio but may introduce pronunciation errors. Temperature > 1.0 tends to fix pronunciation but sounds less like the speaker.")
    parser.add_argument('--ref_free', action='store_true', default=False, help="Instructs the application to ignore the reference audio transcript.")
    parser.add_argument('--additional_inp_refs', nargs='*', help='Paths to additional reference audio files. The average "Tone" of these files will guide the tone of the generated audio. If none are provided, then ref_audio will be used instead.')

    args = parser.parse_args()

    synthesize(args.gpt_model, args.sovits_model, args.ref_audio, args.ref_text, args.ref_language, args.target_text,
               args.target_language, args.output_path, args.how_to_cut, args.top_k, args.top_p, args.temperature,
               args.ref_free, args.speed, args.additional_inp_refs)


if __name__ == '__main__':
    main()

