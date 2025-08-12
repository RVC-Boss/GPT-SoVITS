import argparse
import os
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

LANGUAGE_CHOICES = ["中文", "英文", "日文", "韩文", "粤语"]
MIXED_LANGUAGE_CHOICES = ["中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合"]

SLICE_METHOD_CHOICES = ["凑四句一切", "凑50字一切", "按标点符号切", "按中文句号。切", "按英文句号.切"]

def synthesize(args: argparse.Namespace):

    # Change model weights
    change_gpt_weights(gpt_path=args.gpt_path)
    change_sovits_weights(sovits_path=args.sovits_path)

    params = {
        "ref_wav_path": args.ref_audio,
        "prompt_text": args.ref_text,
        "prompt_language": i18n(args.ref_language),
        "text": args.target_text,
        "text_language": i18n(args.target_language),
    }

    # region - optional params
    if args.slicer: params["how_to_cut"] = i18n(args.slicer)
    if args.top_k: params["top_k"] = args.top_k
    if args.top_p: params["top_p"] = args.top_p
    if args.temperature: params["temperature"] = args.temperature
    if args.ref_free: params["ref_free"] = args.ref_free
    if args.speed: params["speed"] = args.speed
    if args.if_freeze: params["if_freeze"] = args.if_freeze
    if args.inp_refs: params["inp_refs"] = args.inp_refs
    if args.sample_steps: params["sample_steps"] = args.sample_steps
    if args.if_sr: params["if_sr"] = args.if_sr
    if args.pause_second: params["pause_second"] = args.pause_second
    # endregion - optional params

    # Synthesize audio
    synthesis_result = get_tts_wav(**params)

    result_list = list(synthesis_result)

    if result_list:
        os.makedirs(args.output_path, exist_ok=True) # Create output directory if it doesn't exist
        output_wav_path = os.path.join(args.output_path, "output.wav")
        last_sampling_rate, last_audio_data = result_list[-1]
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")

    # reference settings
    parser.add_argument("--ref_audio", required=True, help="Path to the reference audio file")
    parser.add_argument("--ref_text", required=True, help="Transcript of the reference audio")
    parser.add_argument("--ref_language", required=True,
                       choices=LANGUAGE_CHOICES, help="Language of the reference audio")

    # output settings
    parser.add_argument("--target_text", required=True, help="Text to be synthesized")
    parser.add_argument("--target_language", required=True,
                       choices=LANGUAGE_CHOICES+MIXED_LANGUAGE_CHOICES,
                       help="Language of the target text")
    parser.add_argument("--slicer", required=False,
                       choices=SLICE_METHOD_CHOICES, help="Slicer method")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")

    # region - inference settings
    parser.add_argument("--top_k", required=False, type=int, help="Top-k value")
    parser.add_argument("--top_p", required=False, type=float, help="Top-p value")
    parser.add_argument("--temperature", required=False, type=float, help="Temperature value")
    parser.add_argument("--ref_free", required=False, type=bool, help="Reference free value")
    parser.add_argument("--speed", required=False, type=float, help="Speed value")
    parser.add_argument("--if_freeze", required=False, type=bool, help="If freeze value")
    parser.add_argument("--inp_refs", required=False, type=str, help="Input references")
    parser.add_argument("--sample_steps", required=False, type=int, help="Sample steps value")
    parser.add_argument("--if_sr", required=False, type=bool, help="If super resolution value")
    parser.add_argument("--pause_second", required=False, type=float, help="Pause second value")
    # endregion - inference settings

    # region - model selection
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
    # endregion - model selection

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


    synthesize(args)


if __name__ == "__main__":
    main()
