"""Model construction, checkpoint loading, and hot-switch lifecycle."""

import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from GPT_SoVITS.AR.models.t2s_inference import Text2SemanticInference
from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from GPT_SoVITS.feature_extractor.cnhubert import CNHubert
from GPT_SoVITS.inference.checkpoints import get_sovits_version_from_path_fast, load_sovits_new
from GPT_SoVITS.inference.speaker import SV
from GPT_SoVITS.inference.super_resolution import AP_BWE
from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
from GPT_SoVITS.module.models_v5 import V5_VERSIONS
from GPT_SoVITS.runtime.i18n import I18nAuto

now_dir = os.getcwd()
language = os.environ.get("language", "Auto")
i18n = I18nAuto(language=language)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class ModelRuntimeMixin:
    def _init_models(
        self,
    ):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

    def init_cnhuhbert_weights(self, base_path: str):
        print(f"Loading CNHuBERT weights from {base_path}")
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model = self.cnhuhbert_model.eval()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.cnhuhbert_model = self.cnhuhbert_model.half()

    def init_bert_weights(self, base_path: str):
        print(f"Loading BERT weights from {base_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModel.from_pretrained(base_path, add_pooling_layer=False)
        self.bert_model = self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.bert_model = self.bert_model.half()

    def init_vits_weights(self, weights_path: str):
        self.configs.vits_weights_path = weights_path
        version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(weights_path)
        if "Pro" in model_version:
            self.init_sv_model()
        path_sovits = self.configs.default_configs[model_version]["vits_weights_path"]

        if if_lora_v3 == True and os.path.exists(path_sovits) == False:
            info = path_sovits + i18n("SoVITS %s 底模缺失，无法加载相应 LoRA 权重" % model_version)
            raise FileNotFoundError(info)

        # dict_s2 = torch.load(weights_path, map_location=self.configs.device,weights_only=False)
        dict_s2 = load_sovits_new(weights_path)
        hps = dict_s2["config"]
        hps["model"]["semantic_frame_rate"] = "25hz"
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps["model"]["version"] = "v2"  # v3model,v2sybomls
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps["model"]["version"] = "v1"
        else:
            hps["model"]["version"] = "v2"
        version = hps["model"]["version"]
        v3v4set = {"v3", "v4"} | V5_VERSIONS
        if model_version not in v3v4set:
            if "Pro" not in model_version:
                model_version = version
            else:
                hps["model"]["version"] = model_version
        else:
            hps["model"]["version"] = model_version

        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.semantic_frame_rate = hps["model"]["semantic_frame_rate"]
        kwargs = hps["model"]
        # print(f"self.configs.sampling_rate:{self.configs.sampling_rate}")

        self.configs.update_version(model_version)

        # print(f"model_version:{model_version}")
        # print(f'hps["model"]["version"]:{hps["model"]["version"]}')
        if model_version not in v3v4set:
            vits_model = SynthesizerTrn(
                self.configs.filter_length // 2 + 1,
                **kwargs,
            )
            self.configs.use_vocoder = False
        else:
            kwargs["version"] = model_version
            vits_model = SynthesizerTrnV3(
                self.configs.filter_length // 2 + 1,
                **kwargs,
            )
            self.configs.use_vocoder = True
            self.init_vocoder(model_version)

        self.is_v2pro = model_version in {"v2Pro", "v2ProPlus"}

        if if_lora_v3 == False:
            print(
                f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )
        else:
            print(
                f"Loading VITS pretrained weights from {weights_path}. {vits_model.load_state_dict(load_sovits_new(path_sovits)['weight'], strict=False)}"
            )
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vits_model.cfm = get_peft_model(vits_model.cfm, lora_config)
            print(
                f"Loading LoRA weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )

            vits_model.cfm = vits_model.cfm.merge_and_unload()

        vits_model = vits_model.to(self.configs.device)
        vits_model = vits_model.eval()

        self.vits_model = vits_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.vits_model = self.vits_model.half()

    def init_t2s_weights(self, weights_path: str):
        if self.streaming_accel is not None:
            self.streaming_accel.close()
            self.streaming_accel = None
        if self.t2s_accel is not None:
            self.t2s_accel.close()
            self.t2s_accel = None
        print(f"Loading Text2Semantic weights from {weights_path}")
        self.configs.t2s_weights_path = weights_path
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device, weights_only=False)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticInference(config)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model = t2s_model.eval()
        self.t2s_model = t2s_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.t2s_model = self.t2s_model.half()

        codebook = t2s_model.model.ar_audio_embedding.weight.clone()
        mute_emb = codebook[self.configs.mute_tokens[self.configs.version]].unsqueeze(0)
        sim_matrix = F.cosine_similarity(mute_emb.float(), codebook.float(), dim=-1)
        self.configs.mute_emb_sim_matrix = sim_matrix

    def init_vocoder(self, version: str):
        if version == "v3":
            if self.vocoder is not None and self.vocoder.__class__.__name__ == "BigVGAN":
                return
            if self.vocoder is not None:
                self.vocoder.cpu()
                del self.vocoder
                self.empty_cache()

            self.vocoder = BigVGAN.from_pretrained(
                "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
                use_cuda_kernel=False,
            )
            # remove weight norm in the model and set to eval mode
            self.vocoder.remove_weight_norm()

            self.vocoder_configs["sr"] = 24000
            self.vocoder_configs["T_ref"] = 468
            self.vocoder_configs["T_chunk"] = 934
            self.vocoder_configs["upsample_rate"] = 256
            self.vocoder_configs["overlapped_len"] = 12

        elif version in ({"v4"} | V5_VERSIONS):
            if (self.vocoder is not None and self.vocoder.__class__.__name__ == "Generator"
                    and getattr(self, "_vocoder_version", None) == version):
                return
            if self.vocoder is not None:
                self.vocoder.cpu()
                del self.vocoder
                self.empty_cache()

            self.vocoder = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            self.vocoder.remove_weight_norm()
            state_dict_g = torch.load(
                "%s/GPT_SoVITS/pretrained_models/gsv-%s-pretrained/vocoder.pth" %
                (now_dir, "v5" if version in V5_VERSIONS else version),
                map_location="cpu",
                weights_only=False,
            )
            print("loading vocoder", self.vocoder.load_state_dict(state_dict_g))

            self.vocoder_configs["sr"] = 48000
            self.vocoder_configs["T_ref"] = 500
            self.vocoder_configs["T_chunk"] = 1000
            self.vocoder_configs["upsample_rate"] = 480
            self.vocoder_configs["overlapped_len"] = 12

        self.vocoder = self.vocoder.eval()
        self._vocoder_version = version
        if self.configs.is_half == True:
            self.vocoder = self.vocoder.half().to(self.configs.device)
        else:
            self.vocoder = self.vocoder.to(self.configs.device)

    def init_sr_model(self):
        if self.sr_model is not None:
            return
        try:
            self.sr_model: AP_BWE = AP_BWE(self.configs.device, DictToAttrRecursive)
            self.sr_model_not_exist = False
        except FileNotFoundError:
            print(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            self.sr_model_not_exist = True

    def init_sv_model(self):
        if self.sv_model is not None:
            return
        self.sv_model = SV(self.configs.device, self.configs.is_half)
