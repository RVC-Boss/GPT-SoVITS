import os,sys
if len(sys.argv)==1:sys.argv.append('v2')
version="v1"if sys.argv[1]=="v1" else"v2"
os.environ["version"]=version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings
warnings.filterwarnings("ignore")
import json,yaml,torch,pdb,re,shutil
import platform
import psutil
import signal
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if(os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if(name=="jieba.cache"):continue
        path="%s/%s"%(tmp,name)
        delete=os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
import traceback
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError as e:
            traceback.print_exc()
from tools import my_utils
import shutil
import pdb
from subprocess import Popen
import signal
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from tools.i18n.i18n import I18nAuto, scan_language_list
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"]=language
i18n = I18nAuto(language=language)
from scipy.io import wavfile
from tools.my_utils import load_audio, check_for_existance, check_details
from multiprocessing import cpu_count
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...
import gradio as gr
n_cpu=cpu_count()
           
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords={"10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060","H"}
set_gpu_numbers=set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    set_gpu_numbers.add(0)
    default_batch_size = int(psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2)
gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers=str(sorted(list(set_gpu_numbers))[0])
def fix_gpu_number(input):#将越界的number强制改到界内
    try:
        if(int(input)not in set_gpu_numbers):return default_gpu_numbers
    except:return input
    return input
def fix_gpu_numbers(inputs):
    output=[]
    try:
        for input in inputs.split(","):output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs

pretrained_sovits_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]

pretrained_model_list = (pretrained_sovits_name[-int(version[-1])+2],pretrained_sovits_name[-int(version[-1])+2].replace("s2G","s2D"),pretrained_gpt_name[-int(version[-1])+2],"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large","GPT_SoVITS/pretrained_models/chinese-hubert-base")

_=''
for i in pretrained_model_list:
    if os.path.exists(i):...
    else:_+=f'\n    {i}'
if _:
    print("warning:",i18n('以下模型不存在:')+_)

_ =[[],[]]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):_[0].append(pretrained_gpt_name[i])
    else:_[0].append("")##没有下pretrained模型的，说不定他们是想自己从零训底模呢
    if os.path.exists(pretrained_sovits_name[i]):_[-1].append(pretrained_sovits_name[i])
    else:_[-1].append("")
pretrained_gpt_name,pretrained_sovits_name = _

SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
for root in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(root,exist_ok=True)
def get_weights_names():
    SoVITS_names = [name for name in pretrained_sovits_name if name!=""]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [name for name in pretrained_gpt_name if name!=""]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names

SoVITS_names,GPT_names = get_weights_names()
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path,exist_ok=True)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}

p_label=None
p_uvr5=None
p_asr=None
p_denoise=None
p_tts_inference=None

def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)
    

def change_label(path_list):
    global p_label
    if(p_label==None):
        check_for_existance([path_list])
        path_list=my_utils.clean_path(path_list)
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s'%(python_exec,path_list,webui_port_subfix,is_share)
        yield i18n("打标工具WebUI已开启"), {'__type__':'update','visible':False}, {'__type__':'update','visible':True}
        print(cmd)
        p_label = Popen(cmd, shell=True)
    elif(p_label!=None):
        kill_process(p_label.pid)
        p_label=None
        yield i18n("打标工具WebUI已关闭"), {'__type__':'update','visible':True}, {'__type__':'update','visible':False}

def change_uvr5():
    global p_uvr5
    if(p_uvr5==None):
        cmd = '"%s" tools/uvr5/webui.py "%s" %s %s %s'%(python_exec,infer_device,is_half,webui_port_uvr5,is_share)
        yield i18n("UVR5已开启"), {'__type__':'update','visible':False}, {'__type__':'update','visible':True}
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    elif(p_uvr5!=None):
        kill_process(p_uvr5.pid)
        p_uvr5=None
        yield i18n("UVR5已关闭"), {'__type__':'update','visible':True}, {'__type__':'update','visible':False}

def change_tts_inference(bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path, batched_infer_enabled):
    global p_tts_inference
    if batched_infer_enabled:
        cmd = '"%s" GPT_SoVITS/inference_webui_fast.py "%s"'%(python_exec, language)
    else:
        cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"'%(python_exec, language)
    if(p_tts_inference==None):
        os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=fix_gpu_number(gpu_number)
        os.environ["is_half"]=str(is_half)
        os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
        os.environ["is_share"]=str(is_share)
        yield i18n("TTS推理进程已开启"), {'__type__':'update','visible':False}, {'__type__':'update','visible':True}
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
    elif(p_tts_inference!=None):
        kill_process(p_tts_inference.pid)
        p_tts_inference=None
        yield i18n("TTS推理进程已关闭"), {'__type__':'update','visible':True}, {'__type__':'update','visible':False}

from tools.asr.config import asr_dict
def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if(p_asr==None):
        asr_inp_dir=my_utils.clean_path(asr_inp_dir)
        asr_opt_dir=my_utils.clean_path(asr_opt_dir)
        check_for_existance([asr_inp_dir])
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += f" -p {asr_precision}"
        output_file_name = os.path.basename(asr_inp_dir)
        output_folder = asr_opt_dir or "output/asr_opt"
        output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')
        yield "ASR任务开启：%s"%cmd, {"__type__":"update","visible":False}, {"__type__":"update","visible":True}, {"__type__":"update"}, {"__type__":"update"}, {"__type__":"update"}
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr=None
        yield f"ASR任务完成, 查看终端进行下一步", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}, {"__type__":"update","value":output_file_path}, {"__type__":"update","value":output_file_path}, {"__type__":"update","value":asr_inp_dir}
    else:
        yield "已有正在进行的ASR任务，需先终止才能开启下一次任务", {"__type__":"update","visible":False}, {"__type__":"update","visible":True}, {"__type__":"update"}, {"__type__":"update"}, {"__type__":"update"}
        # return None

def close_asr():
    global p_asr
    if(p_asr!=None):
        kill_process(p_asr.pid)
        p_asr=None
    return "已终止ASR进程", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if(p_denoise==None):
        denoise_inp_dir=my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir=my_utils.clean_path(denoise_opt_dir)
        check_for_existance([denoise_inp_dir])
        cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s'%(python_exec,denoise_inp_dir,denoise_opt_dir,"float16"if is_half==True else "float32")

        yield "语音降噪任务开启：%s"%cmd, {"__type__":"update","visible":False}, {"__type__":"update","visible":True}, {"__type__":"update"}, {"__type__":"update"}
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise=None
        yield f"语音降噪任务完成, 查看终端进行下一步", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}, {"__type__":"update","value":denoise_opt_dir}, {"__type__":"update","value":denoise_opt_dir}
    else:
        yield "已有正在进行的语音降噪任务，需先终止才能开启下一次任务", {"__type__":"update","visible":False}, {"__type__":"update","visible":True}, {"__type__":"update"}, {"__type__":"update"}
        # return None

def close_denoise():
    global p_denoise
    if(p_denoise!=None):
        kill_process(p_denoise.pid)
        p_denoise=None
    return "已终止语音降噪进程", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}

p_train_SoVITS=None
def open1Ba(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D):
    global p_train_SoVITS
    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        if check_for_existance([s2_dir],is_train=True):
            check_details([s2_dir],is_train=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["model"]["version"]=version
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root[-int(version[-1])+2]
        data["name"]=exp_name
        data["version"]=version
        tmp_config_path="%s/tmp_s2.json"%tmp
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        yield "SoVITS训练开始：%s"%cmd, {"__type__":"update","visible":False}, {"__type__":"update","visible":True}
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        yield "SoVITS训练完成", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
    else:
        yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务", {"__type__":"update","visible":False}, {"__type__":"update","visible":True}

def close1Ba():
    global p_train_SoVITS
    if(p_train_SoVITS!=None):
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS=None
    return "已终止SoVITS训练", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}

p_train_GPT=None
def open1Bb(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1):
    global p_train_GPT
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml"if version=="v1"else "GPT_SoVITS/configs/s1longer-v2.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if check_for_existance([s1_dir],is_train=True):
            check_details([s1_dir],is_train=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=GPT_weight_root[-int(version[-1])+2]
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"]=fix_gpu_numbers(gpu_numbers.replace("-",","))
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%tmp
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        yield "GPT训练开始：%s"%cmd, {"__type__":"update","visible":False}, {"__type__":"update","visible":True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        yield "GPT训练完成", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
    else:
        yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务", {"__type__":"update","visible":False}, {"__type__":"update","visible":True}

def close1Bb():
    global p_train_GPT
    if(p_train_GPT!=None):
        kill_process(p_train_GPT.pid)
        p_train_GPT=None
    return "已终止GPT训练", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}

ps_slice=[]
def open_slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if(os.path.exists(inp)==False):
        yield "输入路径不存在", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}, {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        return
    if os.path.isfile(inp):n_parts=1
    elif os.path.isdir(inp):pass
    else:
        yield "输入路径存在但既不是文件也不是文件夹", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}, {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (python_exec,inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "切割执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}, {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        for p in ps_slice:
            p.wait()
        ps_slice=[]
        yield "切割结束", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}, {"__type__": "update", "value":opt_root}, {"__type__": "update", "value":opt_root}, {"__type__": "update", "value":opt_root}
    else:
        yield "已有正在进行的切割任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}, {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice=[]
    return "已终止所有切割进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1a=[]
def open1a(inp_text,inp_wav_dir,exp_name,gpu_numbers,bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text,inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text,inp_wav_dir], is_dataset_processing=True)
    if (ps1a == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "bert_pretrained_dir":bert_pretrained_dir,
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    "is_half": str(is_half)
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield "文本进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a=[]
        if len("".join(opt)) > 0:
            yield "文本进程成功", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        else:
            yield "文本进程失败", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的文本任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1a():
    global ps1a
    if (ps1a != []):
        for p1a in ps1a:
            try:
                kill_process(p1a.pid)
            except:
                traceback.print_exc()
        ps1a=[]
    return "已终止所有1a进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1b=[]
def open1b(inp_text,inp_wav_dir,exp_name,gpu_numbers,ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text,inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text,inp_wav_dir], is_dataset_processing=True)
    if (ps1b == []):
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":"%s/%s"%(exp_root,exp_name),
            "cnhubert_base_dir":ssl_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield "SSL提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1b:
            p.wait()
        ps1b=[]
        yield "SSL提取进程结束", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
    else:
        yield "已有正在进行的SSL提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1b():
    global ps1b
    if (ps1b != []):
        for p1b in ps1b:
            try:
                kill_process(p1b.pid)
            except:
                traceback.print_exc()
        ps1b=[]
    return "已终止所有1b进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1c=[]
def open1c(inp_text,exp_name,gpu_numbers,pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if check_for_existance([inp_text,''], is_dataset_processing=True):
        check_details([inp_text,''], is_dataset_processing=True)
    if (ps1c == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "pretrained_s2G":pretrained_s2G_path,
            "s2config_path":"GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield "语义token提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c=[]
        yield "语义token提取进程结束", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
    else:
        yield "已有正在进行的语义token提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1c():
    global ps1c
    if (ps1c != []):
        for p1c in ps1c:
            try:
                kill_process(p1c.pid)
            except:
                traceback.print_exc()
        ps1c=[]
    return "已终止所有语义token进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
#####inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G
ps1abc=[]
def open1abc(inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text,inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text,inp_wav_dir], is_dataset_processing=True)
    if (ps1abc == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
            yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            for p in ps1abc:p.wait()
            yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G_path,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        except:
            traceback.print_exc()
            close1abc()
            yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc=[]
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

def switch_version(version_):
    os.environ['version']=version_
    global version
    version = version_
    if pretrained_sovits_name[-int(version[-1])+2] !='' and pretrained_gpt_name[-int(version[-1])+2] !='':...
    else:   
        gr.Warning(i18n(f'未下载{version.upper()}模型'))
    return  {'__type__':'update', 'value':pretrained_sovits_name[-int(version[-1])+2]}, {'__type__':'update', 'value':pretrained_sovits_name[-int(version[-1])+2].replace("s2G","s2D")}, {'__type__':'update', 'value':pretrained_gpt_name[-int(version[-1])+2]}, {'__type__':'update', 'value':pretrained_gpt_name[-int(version[-1])+2]}, {'__type__':'update', 'value':pretrained_sovits_name[-int(version[-1])+2]}

if os.path.exists('GPT_SoVITS/text/G2PWModel'):...
else:
    cmd = '"%s" GPT_SoVITS/download.py'%python_exec
    p = Popen(cmd, shell=True)
    p.wait()

def sync(text):
    return {'__type__':'update','value':text}
with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    gr.Markdown(
        value=
            i18n("中文教程文档：https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e")
    )

    with gr.Tabs():
        with gr.TabItem(i18n("0-前置数据集获取工具")):#提前随机切片防止uvr5爆内存->uvr5->slicer->asr->打标
            gr.Markdown(value=i18n("0a-UVR5人声伴奏分离&去混响去延迟工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        uvr5_info = gr.Textbox(label=i18n("UVR5进程输出信息"))
                open_uvr5 = gr.Button(value=i18n("开启UVR5-WebUI"),variant="primary",visible=True)
                close_uvr5 = gr.Button(value=i18n("关闭UVR5-WebUI"),variant="primary",visible=False)
            gr.Markdown(value=i18n("0b-语音切分工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        slice_inp_path=gr.Textbox(label=i18n("音频自动切分输入路径，可文件可文件夹"),value="")
                        slice_opt_root=gr.Textbox(label=i18n("切分后的子音频的输出根目录"),value="output/slicer_opt")
                    with gr.Row():
                        threshold=gr.Textbox(label=i18n("threshold:音量小于这个值视作静音的备选切割点"),value="-34")
                        min_length=gr.Textbox(label=i18n("min_length:每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值"),value="4000")
                        min_interval=gr.Textbox(label=i18n("min_interval:最短切割间隔"),value="300")
                        hop_size=gr.Textbox(label=i18n("hop_size:怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）"),value="10")
                        max_sil_kept=gr.Textbox(label=i18n("max_sil_kept:切完后静音最多留多长"),value="500")
                    with gr.Row():
                        _max=gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("max:归一化后最大值多少"),value=0.9,interactive=True)
                        alpha=gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("alpha_mix:混多少比例归一化后音频进来"),value=0.25,interactive=True)   
                    with gr.Row():
                        n_process=gr.Slider(minimum=1,maximum=n_cpu,step=1,label=i18n("切割使用的进程数"),value=4,interactive=True)
                        slicer_info = gr.Textbox(label=i18n("语音切割进程输出信息"))
                open_slicer_button=gr.Button(i18n("开启语音切割"), variant="primary",visible=True)
                close_slicer_button=gr.Button(i18n("终止语音切割"), variant="primary",visible=False)
            gr.Markdown(value=i18n("0bb-语音降噪工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        denoise_input_dir=gr.Textbox(label=i18n("降噪音频文件输入文件夹"),value="")
                        denoise_output_dir=gr.Textbox(label=i18n("降噪结果输出文件夹"),value="output/denoise_opt")
                    with gr.Row():
                        denoise_info = gr.Textbox(label=i18n("语音降噪进程输出信息"))
                open_denoise_button = gr.Button(i18n("开启语音降噪"), variant="primary",visible=True)
                close_denoise_button = gr.Button(i18n("终止语音降噪进程"), variant="primary",visible=False)
            gr.Markdown(value=i18n("0c-中文批量离线ASR工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        asr_inp_dir = gr.Textbox(
                            label=i18n("输入文件夹路径"),
                            value="D:\\GPT-SoVITS\\raw\\xxx",
                            interactive=True,
                        )
                        asr_opt_dir = gr.Textbox(
                            label       = i18n("输出文件夹路径"),
                            value       = "output/asr_opt",
                            interactive = True,
                        )
                    with gr.Row():
                        asr_model = gr.Dropdown(
                            label       = i18n("ASR 模型"),
                            choices     = list(asr_dict.keys()),
                            interactive = True,
                            value="达摩 ASR (中文)"
                        )
                        asr_size = gr.Dropdown(
                            label       = i18n("ASR 模型尺寸"),
                            choices     = ["large"],
                            interactive = True,
                            value="large"
                        )
                        asr_lang = gr.Dropdown(
                            label       = i18n("ASR 语言设置"),
                            choices     = ["zh","yue"],
                            interactive = True,
                            value="zh"
                        )
                        asr_precision = gr.Dropdown(
                            label       = i18n("数据类型精度"),
                            choices     = ["float32"],
                            interactive = True,
                            value="float32"
                        ) 
                    with gr.Row():
                        asr_info = gr.Textbox(label=i18n("ASR进程输出信息"))     
                open_asr_button = gr.Button(i18n("开启离线批量ASR"), variant="primary",visible=True)
                close_asr_button = gr.Button(i18n("终止ASR进程"), variant="primary",visible=False)                  

                def change_lang_choices(key): #根据选择的模型修改可选的语言
                    # return gr.Dropdown(choices=asr_dict[key]['lang'])
                    return {"__type__": "update", "choices": asr_dict[key]['lang'],"value":asr_dict[key]['lang'][0]}
                def change_size_choices(key): # 根据选择的模型修改可选的模型尺寸
                    # return gr.Dropdown(choices=asr_dict[key]['size'])
                    return {"__type__": "update", "choices": asr_dict[key]['size'],"value":asr_dict[key]['size'][-1]}
                def change_precision_choices(key): #根据选择的模型修改可选的语言
                    if key =="Faster Whisper (多语种)":
                        if default_batch_size <= 4:
                            precision = 'int8'
                        elif is_half:
                            precision = 'float16'
                        else:
                            precision = 'float32'
                    else:
                        precision = 'float32'
                    # return gr.Dropdown(choices=asr_dict[key]['precision'])
                    return {"__type__": "update", "choices": asr_dict[key]['precision'],"value":precision}
                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                asr_model.change(change_precision_choices, [asr_model], [asr_precision])

                
            gr.Markdown(value=i18n("0d-语音文本校对标注工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        path_list = gr.Textbox(
                        label=i18n(".list标注文件的路径"),
                        value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                        interactive=True,
                    )
                        label_info = gr.Textbox(label=i18n("打标工具进程输出信息"))
                
                open_label = gr.Button(value=i18n("开启打标WebUI"),variant="primary",visible=True)
                close_label = gr.Button(value=i18n("关闭打标WebUI"),variant="primary",visible=False)
            open_label.click(change_label, [path_list], [label_info,open_label,close_label])
            close_label.click(change_label, [path_list], [label_info,open_label,close_label])
            open_uvr5.click(change_uvr5, [], [uvr5_info,open_uvr5,close_uvr5])
            close_uvr5.click(change_uvr5, [], [uvr5_info,open_uvr5,close_uvr5])

        with gr.TabItem(i18n("1-GPT-SoVITS-TTS")):
            with gr.Row():
                with gr.Row():
                    exp_name = gr.Textbox(label=i18n("*实验/模型名"), value="xxx", interactive=True)
                    gpu_info = gr.Textbox(label=i18n("显卡信息"), value=gpu_info, visible=True, interactive=False)
                    version_checkbox = gr.Radio(label=i18n("版本"),value=version,choices=['v1','v2'])
                with gr.Row():
                    pretrained_s2G = gr.Textbox(label=i18n("预训练的SoVITS-G模型路径"), value=pretrained_sovits_name[-int(version[-1])+2], interactive=True, lines=2, max_lines=3,scale=9)
                    pretrained_s2D = gr.Textbox(label=i18n("预训练的SoVITS-D模型路径"), value=pretrained_sovits_name[-int(version[-1])+2].replace("s2G","s2D"), interactive=True, lines=2, max_lines=3,scale=9)
                    pretrained_s1 = gr.Textbox(label=i18n("预训练的GPT模型路径"), value=pretrained_gpt_name[-int(version[-1])+2], interactive=True, lines=2, max_lines=3,scale=10)
            with gr.TabItem(i18n("1A-训练集格式化工具")):
                gr.Markdown(value=i18n("输出logs/实验名目录下应有23456开头的文件和文件夹"))
                with gr.Row():
                    with gr.Row():
                        inp_text = gr.Textbox(label=i18n("*文本标注文件"),value=r"D:\RVC1006\GPT-SoVITS\raw\xxx.list",interactive=True,scale=10)
                    with gr.Row():
                        inp_wav_dir = gr.Textbox(
                            label=i18n("*训练集音频文件目录"),
                            # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                            interactive=True,
                            placeholder=i18n("填切割后音频所在目录！读取的音频文件完整路径=该目录-拼接-list文件里波形对应的文件名（不是全路径）。如果留空则使用.list文件里的绝对全路径。"), scale=10
                        )
                gr.Markdown(value=i18n("1Aa-文本内容"))
                with gr.Row():
                    with gr.Row():
                        gpu_numbers1a = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"),value="%s-%s"%(gpus,gpus),interactive=True)
                    with gr.Row():
                        bert_pretrained_dir = gr.Textbox(label=i18n("预训练的中文BERT模型路径"),value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",interactive=False,lines=2)
                    with gr.Row():
                        button1a_open = gr.Button(i18n("开启文本获取"), variant="primary",visible=True)
                        button1a_close = gr.Button(i18n("终止文本获取进程"), variant="primary",visible=False)
                    with gr.Row():
                        info1a=gr.Textbox(label=i18n("文本进程输出信息"))
                gr.Markdown(value=i18n("1Ab-SSL自监督特征提取"))
                with gr.Row():
                    with gr.Row():
                        gpu_numbers1Ba = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"),value="%s-%s"%(gpus,gpus),interactive=True)
                    with gr.Row():
                        cnhubert_base_dir = gr.Textbox(label=i18n("预训练的SSL模型路径"),value="GPT_SoVITS/pretrained_models/chinese-hubert-base",interactive=False,lines=2)
                    with gr.Row():
                        button1b_open = gr.Button(i18n("开启SSL提取"), variant="primary",visible=True)
                        button1b_close = gr.Button(i18n("终止SSL提取进程"), variant="primary",visible=False)
                    with gr.Row():
                        info1b=gr.Textbox(label=i18n("SSL进程输出信息"))
                gr.Markdown(value=i18n("1Ac-语义token提取"))
                with gr.Row():
                    with gr.Row():
                        gpu_numbers1c = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"),value="%s-%s"%(gpus,gpus),interactive=True)
                    with gr.Row():
                        pretrained_s2G_ = gr.Textbox(label=i18n("预训练的SoVITS-G模型路径"), value=pretrained_sovits_name[-int(version[-1])+2], interactive=False,lines=2)
                    with gr.Row():
                        button1c_open = gr.Button(i18n("开启语义token提取"), variant="primary",visible=True)
                        button1c_close = gr.Button(i18n("终止语义token提取进程"), variant="primary",visible=False)
                    with gr.Row():
                        info1c=gr.Textbox(label=i18n("语义token提取进程输出信息"))
                gr.Markdown(value=i18n("1Aabc-训练集格式化一键三连"))
                with gr.Row():
                    with gr.Row():
                        button1abc_open = gr.Button(i18n("开启一键三连"), variant="primary",visible=True)
                        button1abc_close = gr.Button(i18n("终止一键三连"), variant="primary",visible=False)
                    with gr.Row():
                        info1abc=gr.Textbox(label=i18n("一键三连进程输出信息"))

            pretrained_s2G.change(sync,[pretrained_s2G],[pretrained_s2G_])
            open_asr_button.click(open_asr, [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision], [asr_info,open_asr_button,close_asr_button,path_list,inp_text,inp_wav_dir])
            close_asr_button.click(close_asr, [], [asr_info,open_asr_button,close_asr_button])
            open_slicer_button.click(open_slice, [slice_inp_path,slice_opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_process], [slicer_info,open_slicer_button,close_slicer_button,asr_inp_dir,denoise_input_dir,inp_wav_dir])
            close_slicer_button.click(close_slice, [], [slicer_info,open_slicer_button,close_slicer_button])
            open_denoise_button.click(open_denoise, [denoise_input_dir,denoise_output_dir], [denoise_info,open_denoise_button,close_denoise_button,asr_inp_dir,inp_wav_dir])
            close_denoise_button.click(close_denoise, [], [denoise_info,open_denoise_button,close_denoise_button])
            
            button1a_open.click(open1a, [inp_text,inp_wav_dir,exp_name,gpu_numbers1a,bert_pretrained_dir], [info1a,button1a_open,button1a_close])
            button1a_close.click(close1a, [], [info1a,button1a_open,button1a_close])
            button1b_open.click(open1b, [inp_text,inp_wav_dir,exp_name,gpu_numbers1Ba,cnhubert_base_dir], [info1b,button1b_open,button1b_close])
            button1b_close.click(close1b, [], [info1b,button1b_open,button1b_close])
            button1c_open.click(open1c, [inp_text,exp_name,gpu_numbers1c,pretrained_s2G], [info1c,button1c_open,button1c_close])
            button1c_close.click(close1c, [], [info1c,button1c_open,button1c_close])
            button1abc_open.click(open1abc, [inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G], [info1abc,button1abc_open,button1abc_close])
            button1abc_close.click(close1abc, [], [info1abc,button1abc_open,button1abc_close])
            with gr.TabItem(i18n("1B-微调训练")):
                gr.Markdown(value=i18n("1Ba-SoVITS训练。用于分享的模型文件输出在SoVITS_weights下。"))
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            batch_size = gr.Slider(minimum=1,maximum=40,step=1,label=i18n("每张显卡的batch_size"),value=default_batch_size,interactive=True)
                            total_epoch = gr.Slider(minimum=1,maximum=25,step=1,label=i18n("总训练轮数total_epoch，不建议太高"),value=8,interactive=True)
                        with gr.Row():
                            text_low_lr_rate = gr.Slider(minimum=0.2,maximum=0.6,step=0.05,label=i18n("文本模块学习率权重"),value=0.4,interactive=True)
                            save_every_epoch = gr.Slider(minimum=1,maximum=25,step=1,label=i18n("保存频率save_every_epoch"),value=4,interactive=True)
                    with gr.Column():     
                        with gr.Column():                   
                            if_save_latest = gr.Checkbox(label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"), value=True, interactive=True, show_label=True)
                            if_save_every_weights = gr.Checkbox(label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"), value=True, interactive=True, show_label=True)
                        with gr.Row():
                            gpu_numbers1Ba = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"), value="%s" % (gpus), interactive=True)
                with gr.Row():
                    with gr.Row():
                        button1Ba_open = gr.Button(i18n("开启SoVITS训练"), variant="primary",visible=True)
                        button1Ba_close = gr.Button(i18n("终止SoVITS训练"), variant="primary",visible=False)
                    with gr.Row():
                        info1Ba=gr.Textbox(label=i18n("SoVITS训练进程输出信息"))
                gr.Markdown(value=i18n("1Bb-GPT训练。用于分享的模型文件输出在GPT_weights下。"))
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            batch_size1Bb = gr.Slider(minimum=1,maximum=40,step=1,label=i18n("每张显卡的batch_size"),value=default_batch_size,interactive=True)
                            total_epoch1Bb = gr.Slider(minimum=2,maximum=50,step=1,label=i18n("总训练轮数total_epoch"),value=15,interactive=True)
                        with gr.Row():
                            save_every_epoch1Bb = gr.Slider(minimum=1,maximum=50,step=1,label=i18n("保存频率save_every_epoch"),value=5,interactive=True) 
                            if_dpo = gr.Checkbox(label=i18n("是否开启dpo训练选项(实验性)"), value=False, interactive=True, show_label=True)   
                    with gr.Column():
                        with gr.Column():
                                if_save_latest1Bb = gr.Checkbox(label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"), value=True, interactive=True, show_label=True)
                                if_save_every_weights1Bb = gr.Checkbox(label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"), value=True, interactive=True, show_label=True)
                        with gr.Row():                            
                            gpu_numbers1Bb = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"), value="%s" % (gpus), interactive=True)
                with gr.Row():
                    with gr.Row():
                        button1Bb_open = gr.Button(i18n("开启GPT训练"), variant="primary",visible=True)
                        button1Bb_close = gr.Button(i18n("终止GPT训练"), variant="primary",visible=False)
                    with gr.Row():
                        info1Bb=gr.Textbox(label=i18n("GPT训练进程输出信息"))
            button1Ba_open.click(open1Ba, [batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D], [info1Ba,button1Ba_open,button1Ba_close])
            button1Ba_close.click(close1Ba, [], [info1Ba,button1Ba_open,button1Ba_close])
            button1Bb_open.click(open1Bb, [batch_size1Bb,total_epoch1Bb,exp_name,if_dpo,if_save_latest1Bb,if_save_every_weights1Bb,save_every_epoch1Bb,gpu_numbers1Bb,pretrained_s1],   [info1Bb,button1Bb_open,button1Bb_close])
            button1Bb_close.click(close1Bb, [], [info1Bb,button1Bb_open,button1Bb_close])
            with gr.TabItem(i18n("1C-推理")):
                gr.Markdown(value=i18n("选择训练完存放在SoVITS_weights和GPT_weights下的模型。默认的一个是底模，体验5秒Zero Shot TTS用。"))
                with gr.Row():
                    with gr.Row():
                        GPT_dropdown = gr.Dropdown(label=i18n("*GPT模型列表"), choices=sorted(GPT_names,key=custom_sort_key),value=pretrained_gpt_name[0],interactive=True)
                        SoVITS_dropdown = gr.Dropdown(label=i18n("*SoVITS模型列表"), choices=sorted(SoVITS_names,key=custom_sort_key),value=pretrained_sovits_name[0],interactive=True)
                    with gr.Row():
                        gpu_number_1C=gr.Textbox(label=i18n("GPU卡号,只能填1个整数"), value=gpus, interactive=True)
                        refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                    refresh_button.click(fn=change_choices,inputs=[],outputs=[SoVITS_dropdown,GPT_dropdown])
                with gr.Row():
                    with gr.Row():
                        batched_infer_enabled = gr.Checkbox(label=i18n("启用并行推理版本(推理速度更快)"), value=False, interactive=True, show_label=True)
                    with gr.Row():
                        open_tts = gr.Button(value=i18n("开启TTS推理WebUI"),variant='primary',visible=True)
                        close_tts = gr.Button(value=i18n("关闭TTS推理WebUI"),variant='primary',visible=False)
                    with gr.Row():
                        tts_info = gr.Textbox(label=i18n("TTS推理WebUI进程输出信息"))
                    open_tts.click(change_tts_inference, [bert_pretrained_dir,cnhubert_base_dir,gpu_number_1C,GPT_dropdown,SoVITS_dropdown, batched_infer_enabled], [tts_info,open_tts,close_tts])
                    close_tts.click(change_tts_inference, [bert_pretrained_dir,cnhubert_base_dir,gpu_number_1C,GPT_dropdown,SoVITS_dropdown, batched_infer_enabled], [tts_info,open_tts,close_tts])
            version_checkbox.change(switch_version,[version_checkbox],[pretrained_s2G,pretrained_s2D,pretrained_s1,GPT_dropdown,SoVITS_dropdown])
        with gr.TabItem(i18n("2-GPT-SoVITS-变声")):gr.Markdown(value=i18n("施工中，请静候佳音"))
    app.queue().launch(#concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        quiet=True,
    )
