
from subprocess import Popen
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix
from train_base import open1abc, open1Ba, open1Bb, gpus, default_batch_size, open_slice, open_asr
import os

current_working_directory = os.getcwd()
inp_text_dir = current_working_directory + "/" + "output/asr_opt"
inp_text = inp_text_dir + "/slicer_opt.list"
inp_wav_dir = current_working_directory + "/" + "output/slicer_opt"
exp_name = "jax_clone_voice"
gpu_numbers="%s-%s"%(gpus,gpus)
bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
ssl_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
pretrained_s2G_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"

def slice_audio(inp,opt_root=inp_wav_dir):
    openSliceGenerator = open_slice(inp,opt_root,"-34","4000","300","10","500",0.9,0.25,4)
    for value in openSliceGenerator:
        print(value)

def asr(asr_inp_dir=inp_wav_dir, asr_opt_dir=inp_text_dir, asr_model="达摩 ASR (中文)", asr_model_size="large", asr_lang="zh"):
    openASRGenerator =  open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang)
    for value in openASRGenerator:
        print(value)

def train_prepare(inp_text=inp_text,inp_wav_dir =inp_wav_dir ,exp_name = exp_name,gpu_numbers1a = gpu_numbers,gpu_numbers1Ba=gpu_numbers,gpu_numbers1c=gpu_numbers,bert_pretrained_dir=bert_pretrained_dir,ssl_pretrained_dir=ssl_pretrained_dir,pretrained_s2G_path=pretrained_s2G_path):
    open1abcGenerator = open1abc(inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G_path)
    for value in open1abcGenerator:
        print(value)

batch_size = default_batch_size
total_epoch = 8
text_low_lr_rate = 0.4
if_save_latest = True
if_save_every_weights = True
save_every_epoch = 4
gpu_numbers1Ba = "%s" % (gpus)
pretrained_s2D = "GPT_SoVITS/pretrained_models/s2D488k.pth"
pretrained_s2G = "GPT_SoVITS/pretrained_models/s2G488k.pth"

def train_SoVITS(batch_size=batch_size,total_epoch=total_epoch,exp_name=exp_name,text_low_lr_rate=text_low_lr_rate,if_save_latest=if_save_latest,if_save_every_weights=if_save_every_weights,save_every_epoch=save_every_epoch,gpu_numbers1Ba=gpu_numbers1Ba,pretrained_s2G=pretrained_s2G,pretrained_s2D=pretrained_s2D):
    open1BaGenerator = open1Ba(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D)
    for value in open1BaGenerator:
        print(value)

gpt_batch_size = default_batch_size
gpt_total_epoch = 15
gpt_if_save_latest = True
gpt_if_save_every_weights = True
gpt_save_every_epoch = 5
gpu_numbers1Bb = "%s" % (gpus)
pretrained_s1 = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
if_dpo = False


def train_GPT(batch_size=default_batch_size,total_epoch=gpt_total_epoch,exp_name=exp_name,if_dpo=if_dpo, if_save_latest=gpt_if_save_latest,
              if_save_every_weights=gpt_if_save_every_weights,save_every_epoch=gpt_save_every_epoch,gpu_numbers=gpu_numbers1Bb,pretrained_s1=pretrained_s1):
    open1BbGenerator = open1Bb(batch_size, total_epoch, exp_name, if_dpo, if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1)
    for value in open1BbGenerator:
        print(value)






# train_prepare(inp_text=inp_text, inp_wav_dir=inp_wav_dir, exp_name=exp_name, gpu_numbers1a=gpu_numbers, gpu_numbers1Ba=gpu_numbers, gpu_numbers1c=gpu_numbers,
#               bert_pretrained_dir=bert_pretrained_dir, ssl_pretrained_dir=ssl_pretrained_dir, pretrained_s2G_path=pretrained_s2G_path )

# train_SoVITS(batch_size=batch_size, total_epoch=total_epoch, exp_name=exp_name, 
#              text_low_lr_rate=text_low_lr_rate, if_save_latest=if_save_latest, 
#              if_save_every_weights=if_save_every_weights, save_every_epoch=save_every_epoch,
#                gpu_numbers1Ba=gpu_numbers1Ba, pretrained_s2D=pretrained_s2D, pretrained_s2G=pretrained_s2G)


# train_GPT(batch_size=gpt_batch_size, total_epoch=gpt_total_epoch, exp_name=exp_name, if_dpo=if_dpo, if_save_latest=gpt_if_save_latest,
#           if_save_every_weights=gpt_if_save_every_weights, save_every_epoch=gpt_save_every_epoch, 
#           gpu_numbers=gpu_numbers1Bb, pretrained_s1=pretrained_s1
#            )