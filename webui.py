
import os
import gradio as gr
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()
from train_base import gpu_info, n_cpu, SoVITS_names, pretrained_sovits_name, pretrained_gpt_name, custom_sort_key, GPT_names, default_batch_size, kill_process, SoVITS_weight_root, GPT_weight_root, change_choices, change_label, change_uvr5, open_asr, open1Ba, open1Bb, close1Bb, open_slice, close_asr, open1a, close1a, open1b, close1Ba, close_slice, close1b, open1c, close1c, open1abc, close1abc, gpus
from tools.asr.config import asr_dict
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from subprocess import Popen

def change_tts_inference(if_tts,bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path):
    global p_tts_inference
    if(if_tts==True and p_tts_inference==None):
        os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_number
        os.environ["is_half"]=str(is_half)
        os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
        os.environ["is_share"]=str(is_share)
        cmd = '"%s" GPT_SoVITS/inference_webui.py'%(python_exec)
        yield i18n("TTS推理进程已开启")
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
    elif(if_tts==False and p_tts_inference!=None):
        kill_process(p_tts_inference.pid)
        p_tts_inference=None
        yield i18n("TTS推理进程已关闭")

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
                if_uvr5 = gr.Checkbox(label=i18n("是否开启UVR5-WebUI"),show_label=True)
                uvr5_info = gr.Textbox(label=i18n("UVR5进程输出信息"))
            gr.Markdown(value=i18n("0b-语音切分工具"))
            with gr.Row():
                with gr.Row():
                    slice_inp_path=gr.Textbox(label=i18n("音频自动切分输入路径，可文件可文件夹"),value="")
                    slice_opt_root=gr.Textbox(label=i18n("切分后的子音频的输出根目录"),value="output/slicer_opt")
                    threshold=gr.Textbox(label=i18n("threshold:音量小于这个值视作静音的备选切割点"),value="-34")
                    min_length=gr.Textbox(label=i18n("min_length:每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值"),value="4000")
                    min_interval=gr.Textbox(label=i18n("min_interval:最短切割间隔"),value="300")
                    hop_size=gr.Textbox(label=i18n("hop_size:怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）"),value="10")
                    max_sil_kept=gr.Textbox(label=i18n("max_sil_kept:切完后静音最多留多长"),value="500")
                with gr.Row():
                    open_slicer_button=gr.Button(i18n("开启语音切割"), variant="primary",visible=True)
                    close_slicer_button=gr.Button(i18n("终止语音切割"), variant="primary",visible=False)
                    _max=gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("max:归一化后最大值多少"),value=0.9,interactive=True)
                    alpha=gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("alpha_mix:混多少比例归一化后音频进来"),value=0.25,interactive=True)
                    n_process=gr.Slider(minimum=1,maximum=n_cpu,step=1,label=i18n("切割使用的进程数"),value=4,interactive=True)
                    slicer_info = gr.Textbox(label=i18n("语音切割进程输出信息"))
            gr.Markdown(value=i18n("0c-中文批量离线ASR工具"))
            with gr.Row():
                open_asr_button = gr.Button(i18n("开启离线批量ASR"), variant="primary",visible=True)
                close_asr_button = gr.Button(i18n("终止ASR进程"), variant="primary",visible=False)
                with gr.Column():
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
                            choices     = ["zh"],
                            interactive = True,
                            value="zh"
                        )
                    with gr.Row():
                        asr_info = gr.Textbox(label=i18n("ASR进程输出信息"))

                def change_lang_choices(key): #根据选择的模型修改可选的语言
                    # return gr.Dropdown(choices=asr_dict[key]['lang'])
                    return {"__type__": "update", "choices": asr_dict[key]['lang'],"value":asr_dict[key]['lang'][0]}
                def change_size_choices(key): # 根据选择的模型修改可选的模型尺寸
                    # return gr.Dropdown(choices=asr_dict[key]['size'])
                    return {"__type__": "update", "choices": asr_dict[key]['size']}
                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                
            gr.Markdown(value=i18n("0d-语音文本校对标注工具"))
            with gr.Row():
                if_label = gr.Checkbox(label=i18n("是否开启打标WebUI"),show_label=True)
                path_list = gr.Textbox(
                    label=i18n(".list标注文件的路径"),
                    value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                    interactive=True,
                )
                label_info = gr.Textbox(label=i18n("打标工具进程输出信息"))
            if_label.change(change_label, [if_label,path_list], [label_info])
            if_uvr5.change(change_uvr5, [if_uvr5], [uvr5_info])
            open_asr_button.click(open_asr, [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang], [asr_info,open_asr_button,close_asr_button])
            close_asr_button.click(close_asr, [], [asr_info,open_asr_button,close_asr_button])
            open_slicer_button.click(open_slice, [slice_inp_path,slice_opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_process], [slicer_info,open_slicer_button,close_slicer_button])
            close_slicer_button.click(close_slice, [], [slicer_info,open_slicer_button,close_slicer_button])
        with gr.TabItem(i18n("1-GPT-SoVITS-TTS")):
            with gr.Row():
                exp_name = gr.Textbox(label=i18n("*实验/模型名"), value="xxx", interactive=True)
                gpu_info = gr.Textbox(label=i18n("显卡信息"), value=gpu_info, visible=True, interactive=False)
                pretrained_s2G = gr.Textbox(label=i18n("预训练的SoVITS-G模型路径"), value="GPT_SoVITS/pretrained_models/s2G488k.pth", interactive=True)
                pretrained_s2D = gr.Textbox(label=i18n("预训练的SoVITS-D模型路径"), value="GPT_SoVITS/pretrained_models/s2D488k.pth", interactive=True)
                pretrained_s1 = gr.Textbox(label=i18n("预训练的GPT模型路径"), value="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", interactive=True)
            with gr.TabItem(i18n("1A-训练集格式化工具")):
                gr.Markdown(value=i18n("输出logs/实验名目录下应有23456开头的文件和文件夹"))
                with gr.Row():
                    inp_text = gr.Textbox(label=i18n("*文本标注文件"),value=r"D:\RVC1006\GPT-SoVITS\raw\xxx.list",interactive=True)
                    inp_wav_dir = gr.Textbox(
                        label=i18n("*训练集音频文件目录"),
                        # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                        interactive=True,
                        placeholder=i18n("填切割后音频所在目录！读取的音频文件完整路径=该目录-拼接-list文件里波形对应的文件名（不是全路径）。如果留空则使用.list文件里的绝对全路径。")
                    )
                gr.Markdown(value=i18n("1Aa-文本内容"))
                with gr.Row():
                    gpu_numbers1a = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"),value="%s-%s"%(gpus,gpus),interactive=True)
                    bert_pretrained_dir = gr.Textbox(label=i18n("预训练的中文BERT模型路径"),value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",interactive=False)
                    button1a_open = gr.Button(i18n("开启文本获取"), variant="primary",visible=True)
                    button1a_close = gr.Button(i18n("终止文本获取进程"), variant="primary",visible=False)
                    info1a=gr.Textbox(label=i18n("文本进程输出信息"))
                gr.Markdown(value=i18n("1Ab-SSL自监督特征提取"))
                with gr.Row():
                    gpu_numbers1Ba = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"),value="%s-%s"%(gpus,gpus),interactive=True)
                    cnhubert_base_dir = gr.Textbox(label=i18n("预训练的SSL模型路径"),value="GPT_SoVITS/pretrained_models/chinese-hubert-base",interactive=False)
                    button1b_open = gr.Button(i18n("开启SSL提取"), variant="primary",visible=True)
                    button1b_close = gr.Button(i18n("终止SSL提取进程"), variant="primary",visible=False)
                    info1b=gr.Textbox(label=i18n("SSL进程输出信息"))
                gr.Markdown(value=i18n("1Ac-语义token提取"))
                with gr.Row():
                    gpu_numbers1c = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"),value="%s-%s"%(gpus,gpus),interactive=True)
                    button1c_open = gr.Button(i18n("开启语义token提取"), variant="primary",visible=True)
                    button1c_close = gr.Button(i18n("终止语义token提取进程"), variant="primary",visible=False)
                    info1c=gr.Textbox(label=i18n("语义token提取进程输出信息"))
                gr.Markdown(value=i18n("1Aabc-训练集格式化一键三连"))
                with gr.Row():
                    button1abc_open = gr.Button(i18n("开启一键三连"), variant="primary",visible=True)
                    button1abc_close = gr.Button(i18n("终止一键三连"), variant="primary",visible=False)
                    info1abc=gr.Textbox(label=i18n("一键三连进程输出信息"))
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
                    batch_size = gr.Slider(minimum=1,maximum=40,step=1,label=i18n("每张显卡的batch_size"),value=default_batch_size,interactive=True)
                    total_epoch = gr.Slider(minimum=1,maximum=25,step=1,label=i18n("总训练轮数total_epoch，不建议太高"),value=8,interactive=True)
                    text_low_lr_rate = gr.Slider(minimum=0.2,maximum=0.6,step=0.05,label=i18n("文本模块学习率权重"),value=0.4,interactive=True)
                    save_every_epoch = gr.Slider(minimum=1,maximum=25,step=1,label=i18n("保存频率save_every_epoch"),value=4,interactive=True)
                    if_save_latest = gr.Checkbox(label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"), value=True, interactive=True, show_label=True)
                    if_save_every_weights = gr.Checkbox(label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"), value=True, interactive=True, show_label=True)
                    gpu_numbers1Ba = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"), value="%s" % (gpus), interactive=True)
                with gr.Row():
                    button1Ba_open = gr.Button(i18n("开启SoVITS训练"), variant="primary",visible=True)
                    button1Ba_close = gr.Button(i18n("终止SoVITS训练"), variant="primary",visible=False)
                    info1Ba=gr.Textbox(label=i18n("SoVITS训练进程输出信息"))
                gr.Markdown(value=i18n("1Bb-GPT训练。用于分享的模型文件输出在GPT_weights下。"))
                with gr.Row():
                    batch_size1Bb = gr.Slider(minimum=1,maximum=40,step=1,label=i18n("每张显卡的batch_size"),value=default_batch_size,interactive=True)
                    total_epoch1Bb = gr.Slider(minimum=2,maximum=50,step=1,label=i18n("总训练轮数total_epoch"),value=15,interactive=True)
                    if_dpo = gr.Checkbox(label=i18n("是否开启dpo训练选项(实验性)"), value=False, interactive=True, show_label=True)
                    if_save_latest1Bb = gr.Checkbox(label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"), value=True, interactive=True, show_label=True)
                    if_save_every_weights1Bb = gr.Checkbox(label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"), value=True, interactive=True, show_label=True)
                    save_every_epoch1Bb = gr.Slider(minimum=1,maximum=50,step=1,label=i18n("保存频率save_every_epoch"),value=5,interactive=True)
                    gpu_numbers1Bb = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"), value="%s" % (gpus), interactive=True)
                with gr.Row():
                    button1Bb_open = gr.Button(i18n("开启GPT训练"), variant="primary",visible=True)
                    button1Bb_close = gr.Button(i18n("终止GPT训练"), variant="primary",visible=False)
                    info1Bb=gr.Textbox(label=i18n("GPT训练进程输出信息"))
            button1Ba_open.click(open1Ba, [batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D], [info1Ba,button1Ba_open,button1Ba_close])
            button1Ba_close.click(close1Ba, [], [info1Ba,button1Ba_open,button1Ba_close])
            button1Bb_open.click(open1Bb, [batch_size1Bb,total_epoch1Bb,exp_name,if_dpo,if_save_latest1Bb,if_save_every_weights1Bb,save_every_epoch1Bb,gpu_numbers1Bb,pretrained_s1],   [info1Bb,button1Bb_open,button1Bb_close])
            button1Bb_close.click(close1Bb, [], [info1Bb,button1Bb_open,button1Bb_close])
            with gr.TabItem(i18n("1C-推理")):
                gr.Markdown(value=i18n("选择训练完存放在SoVITS_weights和GPT_weights下的模型。默认的一个是底模，体验5秒Zero Shot TTS用。"))
                with gr.Row():
                    GPT_dropdown = gr.Dropdown(label=i18n("*GPT模型列表"), choices=sorted(GPT_names,key=custom_sort_key),value=pretrained_gpt_name,interactive=True)
                    SoVITS_dropdown = gr.Dropdown(label=i18n("*SoVITS模型列表"), choices=sorted(SoVITS_names,key=custom_sort_key),value=pretrained_sovits_name,interactive=True)
                    gpu_number_1C=gr.Textbox(label=i18n("GPU卡号,只能填1个整数"), value=gpus, interactive=True)
                    refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                    refresh_button.click(fn=change_choices,inputs=[],outputs=[SoVITS_dropdown,GPT_dropdown])
                with gr.Row():
                    if_tts = gr.Checkbox(label=i18n("是否开启TTS推理WebUI"), show_label=True)
                    tts_info = gr.Textbox(label=i18n("TTS推理WebUI进程输出信息"))
                    if_tts.change(change_tts_inference, [if_tts,bert_pretrained_dir,cnhubert_base_dir,gpu_number_1C,GPT_dropdown,SoVITS_dropdown], [tts_info])
        with gr.TabItem(i18n("2-GPT-SoVITS-变声")):gr.Markdown(value=i18n("施工中，请静候佳音"))
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        quiet=True,
    )
