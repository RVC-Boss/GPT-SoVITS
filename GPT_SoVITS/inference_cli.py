import argparse
import os
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()


def synthesize(
    GPT_model_path,
    SoVITS_model_path,
    ref_audio_path,
    ref_text,
    ref_language,
    target_text,
    target_language,
    output_path,
):

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        top_p=1,
        temperature=1,
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")

    # input settings
    parser.add_argument("--ref_audio", required=True, help="Path to the reference audio file")
    parser.add_argument("--ref_text", required=True, help="Transcript of the reference audio")
    parser.add_argument("--ref_language", required=True,
                       choices=["中文", "英文", "日文", "韩文", "粤语"], help="Language of the reference audio")

    # output settings
    parser.add_argument("--target_text", required=True, help="Text to be synthesized")
    parser.add_argument("--target_language", required=True,
                       choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
                       help="Language of the target text")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")

    sub = parser.add_subparsers(dest="mode", required=True)

    # Mode 1: provide model paths directly
    p_paths = sub.add_parser("paths", help="Use explicit model file paths")
    p_paths.add_argument("--gpt_path", required=True, help="Path to the GPT model file")
    p_paths.add_argument("--sovits_path", required=True, help="Path to the SoVITS model file")

    # Mode 2: select by experiment/version
    p_sel = sub.add_parser("select", help="Select models by experiment/version")
    p_sel.add_argument("--exp_name", required=True, help="Experiment name")
    available_gpt_versions = ["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"]
    p_sel.add_argument("--gpt_version", required=True, choices=available_gpt_versions, help="Version of the GPT model")
    available_sovits_versions = ["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"]
    p_sel.add_argument("--sovits_version", required=True, choices=available_sovits_versions, help="Version of the SoVITS model")
    p_sel.add_argument("--gpt_epoch", type=int, help="Epoch of the GPT model")
    p_sel.add_argument("--sovits_epoch", type=int, help="Epoch of the SoVITS model")

    return parser


def get_model_path(args)->argparse.Namespace:
    """
    Get the model path from exp_name, version and epoch

    Args:
        args: argparse.Namespace

    Returns:
        args: argparse.Namespace
    """
    exist_gpt_path = []
    exist_sovits_path = []

    def _get_model_dir(model_type, version):
        if version == "v1":
            return f"{model_type}_weights"
        else:
            return f"{model_type}_weights_{version}"

    # get all the model paths with the same exp_name
    for files in os.listdir(_get_model_dir("GPT", args.gpt_version)):
        if args.exp_name in files and files.endswith(".ckpt"):
            exist_gpt_path.append(os.path.join(_get_model_dir("GPT", args.gpt_version), files))
    for files in os.listdir(_get_model_dir("SoVITS", args.sovits_version)):
        if args.exp_name in files and files.endswith(".pth"):
            exist_sovits_path.append(os.path.join(_get_model_dir("SoVITS", args.sovits_version), files))

    # get the largest epoch if not specified
    if args.gpt_epoch:
        args.gpt_path = [i for i in exist_gpt_path if f"e{args.gpt_epoch}" in i]
    else:
        args.gpt_path = sorted(exist_gpt_path)[-1]
    if args.sovits_epoch:
        args.sovits_path = [i for i in exist_sovits_path if f"e{args.sovits_epoch}" in i]
    else:
        args.sovits_path = sorted(exist_sovits_path)[-1]

    if not args.gpt_path or not args.sovits_path:
        raise ValueError("No model found")
    
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(args)
    if args.mode == "select":
        args = get_model_path(args)
    
    args.target_text = args.target_text.replace("'", "").replace('"', "")


    synthesize(
        args.gpt_path,
        args.sovits_path,
        args.ref_audio,
        args.ref_text,
        args.ref_language,
        args.target_text,
        args.target_language,
        args.output_path,
    )


if __name__ == "__main__":
    main()
