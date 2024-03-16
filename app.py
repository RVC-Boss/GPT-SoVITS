frontend_version = "2.2.3 240316"

from datetime import datetime
import gradio as gr
import json, os
import requests
import numpy as np
from string import Template
import  wave

# 在开头加入路径
import os, sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "Inference/src"))

# 取得模型文件夹路径
config_path = "Inference/config.json"

# 读取config.json
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        _config = json.load(f)
        locale_language = str(_config.get("locale", "auto"))
        locale_language = None if locale_language.lower() == "auto" else locale_language
        tts_port = _config.get("tts_port", 5000)
        max_text_length = _config.get("max_text_length", 70)
        default_batch_size = _config.get("batch_size", 10)
        default_word_count = _config.get("max_word_count", 80)
        is_share = _config.get("is_share", "false").lower() == "true"
        is_classic = _config.get("classic_inference", "false").lower() == "true"
        enable_auth = _config.get("enable_auth", "false").lower() == "true"
        users = _config.get("user", {})
        try:
            default_username = list(users.keys())[0]
            default_password = users[default_username]
        except:
            default_username = "admin"
            default_password = "admin123"

from tools.i18n.i18n import I18nAuto
i18n = I18nAuto(locale_language , "Inference/i18n/locale")

language_list = ["auto", "zh", "en", "ja", "all_zh", "all_ja"]
translated_language_list = [i18n("auto"), i18n("zh"), i18n("en"), i18n("ja"), i18n("all_zh"), i18n("all_ja")] # 由于i18n库的特性，这里需要全部手输一遍
language_dict = dict(zip(translated_language_list, language_list))

cut_method_list = ["auto_cut", "cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]
translated_cut_method_list = [i18n("auto_cut"), i18n("cut0"), i18n("cut1"), i18n("cut2"), i18n("cut3"), i18n("cut4"), i18n("cut5")]
cut_method_dict = dict(zip(translated_cut_method_list, cut_method_list))

tts_port = 5000



def load_character_emotions(character_name, characters_and_emotions):
    emotion_options = ["default"]
    emotion_options = characters_and_emotions.get(character_name, ["default"])

    return gr.Dropdown(emotion_options, value="default")




from load_infer_info import get_wav_from_text_api, update_character_info, load_character, character_name, models_path
import soundfile as sf
import io

def send_request(
    endpoint,
    endpoint_data,
    text,
    cha_name,
    text_language,
    batch_size,
    speed_factor,
    top_k,
    top_p,
    temperature,
    character_emotion,
    cut_method,
    word_count,
    seed,
    stream="False",
):
    global character_name
    global models_path
    text_language = language_dict[text_language]
    cut_method = cut_method_dict[cut_method]
    if cut_method == "auto_cut":
        cut_method = f"{cut_method}_{word_count}"
    # Using Template to fill in variables
    
    
    expected_path = os.path.join(models_path, cha_name) if cha_name else None

    # 检查cha_name和路径
    if cha_name and cha_name != character_name and expected_path and os.path.exists(expected_path):
        character_name = cha_name
        print(f"Loading character {character_name}")
        load_character(character_name)  
    elif expected_path and not os.path.exists(expected_path):
        gr.Warning("Directory {expected_path} does not exist. Using the current character.")

    
    
    stream = stream.lower() in ('true', '1', 't', 'y', 'yes')
    
    
    params = {
        "text": text,
        "text_language": text_language,
        
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "character_emotion": character_emotion,
        "cut_method": cut_method,
        "stream": stream
    }
    # 如果不是经典模式，则添加额外的参数
    if not is_classic:
        params["batch_size"] = batch_size
        params["speed_factor"] = speed_factor
        params["seed"] = seed
    gen = get_wav_from_text_api(**params)
    sampling_rate, audio_data = next(gen)
    wav = io.BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)
    return sampling_rate, np.frombuffer(wav.read(), dtype=np.int16)
    

def stopAudioPlay():
    return


global characters_and_emotions_dict
characters_and_emotions_dict = {}

def get_characters_and_emotions(character_list_url):
    global characters_and_emotions_dict
    # 直接检查字典是否为空，如果不是，直接返回，避免重复获取
    if characters_and_emotions_dict == {}:
        # 假设 update_character_info 是一个函数，需要传递 URL 参数
        characters_and_emotions_dict = update_character_info()['characters_and_emotions']
        print(characters_and_emotions_dict)
   
    return characters_and_emotions_dict
    


def change_character_list(
    character_list_url, cha_name="", auto_emotion=False, character_emotion="default"
):

    characters_and_emotions = {}

    try:
        characters_and_emotions = get_characters_and_emotions(character_list_url)
        character_names = [i for i in characters_and_emotions]
        if len(character_names) != 0:
            if cha_name in character_names:
                character_name_value = cha_name
            else:
                character_name_value = character_names[0]
        else:
            character_name_value = ""
        emotions = characters_and_emotions.get(character_name_value, ["default"])
        emotion_value = character_emotion
        if auto_emotion == False and emotion_value not in emotions:
            emotion_value = "default"
    except:
        character_names = []
        character_name_value = ""
        emotions = ["default"]
        emotion_value = "default"
        characters_and_emotions = {}
    if auto_emotion:
        return (
            gr.Dropdown(character_names, value=character_name_value, label=i18n("选择角色")),
            gr.Checkbox(auto_emotion, label=i18n("是否自动匹配情感"), visible=False, interactive=False),
            gr.Dropdown(["auto"], value="auto", label=i18n("情感列表"), interactive=False),
            characters_and_emotions,
        )
    return (
        gr.Dropdown(character_names, value=character_name_value, label=i18n("选择角色")),
        gr.Checkbox(auto_emotion, label=i18n("是否自动匹配情感"),visible=False, interactive=False),
        gr.Dropdown(emotions, value=emotion_value, label=i18n("情感列表"), interactive=True),
        characters_and_emotions,
    )


def change_endpoint(url):
    url = url.strip()
    return gr.Textbox(f"{url}/tts"), gr.Textbox(f"{url}/character_list")


def change_batch_size(batch_size):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
        with open(config_path, "w", encoding="utf-8") as f:
            _config["batch_size"] = batch_size
            json.dump(_config, f, ensure_ascii=False, indent=4)
    except:
        pass
    return

def change_word_count(word_count):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
        with open(config_path, "w", encoding="utf-8") as f:
            _config["max_word_count"] = word_count
            json.dump(_config, f, ensure_ascii=False, indent=4)
    except:
        pass
    return


def cut_sentence_multilang(text, max_length=30):
    # 初始化计数器
    word_count = 0
    in_word = False
    
    
    for index, char in enumerate(text):
        if char.isspace():  # 如果当前字符是空格
            in_word = False
        elif char.isascii() and not in_word:  # 如果是ASCII字符（英文）并且不在单词内
            word_count += 1  # 新的英文单词
            in_word = True
        elif not char.isascii():  # 如果字符非英文
            word_count += 1  # 每个非英文字符单独计为一个字
        if word_count > max_length:
            return text[:index], text[index:]
    
    return text, ""

default_request_url = f"http://127.0.0.1:{tts_port}"
default_character_info_url = f"{default_request_url}/character_list"
default_endpoint = f"{default_request_url}/tts"
default_endpoint_data = """{
    "method": "POST",
    "body": {
        "cha_name": "${chaName}",
        "character_emotion": "${characterEmotion}",
        "text": "${speakText}",
        "text_language": "${textLanguage}",
        "batch_size": ${batch_size},
        "speed": ${speed_factor},
        "top_k": ${topK},
        "top_p": ${topP},
        "temperature": ${temperature},
        "stream": "${stream}",
        "cut_method": "${cut_method}",
        "save_temp": "False"
    }
}"""
default_text = i18n("我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样。")

if "。" not in default_text:
    _sentence_list = default_text.split(".")
    default_text = ".".join(_sentence_list[:1]) + "."
else:
    _sentence_list = default_text.split("。")
    default_text = "。".join(_sentence_list[:2]) + "。"

information = ""

try:
    with open("Information.md", "r", encoding="utf-8") as f:
        information = f.read()
except:
    pass


with gr.Blocks() as app:
    gr.Markdown(information)
    with gr.Row():
        text = gr.Textbox(
            value=default_text, label=i18n("输入文本")+"( "+ i18n("最大允许长度")  + f": {max_text_length} )", interactive=True, lines=8
        )
        text.blur(lambda x: gr.update(value=cut_sentence_multilang(x,max_length=max_text_length)[0]), [text], [text])
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label=i18n("基础选项")):
                    with gr.Group():
                        text_language = gr.Dropdown(
                            translated_language_list,
                            value=translated_language_list[0],
                            label=i18n("文本语言"),
                        )
                        
                    with gr.Group():
                        (
                            cha_name,
                            auto_emotion_checkbox,
                            character_emotion,
                            characters_and_emotions_,
                        ) = change_character_list(default_character_info_url)
                        characters_and_emotions = gr.State(characters_and_emotions_)
                        scan_character_list = gr.Button(i18n("扫描人物列表"), variant="secondary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label=i18n("基础选项")):
                    gr.Textbox(
                        value=i18n("您在使用经典推理模式，部分选项不可用"),
                        label=i18n("提示"),
                        interactive=False,
                        visible=is_classic,
                    )
                    with gr.Group():
                        speed_factor = gr.Slider(
                            minimum=0.25,
                            maximum=4,
                            value=1,
                            label=i18n("语速"),
                            step=0.05,
                            visible=not is_classic,
                        )
                    with gr.Group():

                        cut_method = gr.Dropdown(
                            translated_cut_method_list,
                            value=translated_cut_method_list[0],
                            label=i18n("切句方式"),
                            visible=not is_classic,
                        )
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=35,
                            value=default_batch_size,
                            label=i18n("batch_size，1代表不并行，越大越快，但是越可能出问题"),
                            step=1,
                            visible=not is_classic,
                        )
                        word_count = gr.Slider(
                            minimum=5,maximum=500,value=default_word_count,label=i18n("每句允许最大切分字词数"),step=1, visible=not is_classic,
                        )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label=i18n("高级选项")):


                    with gr.Group():
                        seed = gr.Number(
                            -1,
                            label=i18n("种子"),
                            visible=not is_classic,
                            interactive=True,
                        )
                    
   
                    with gr.Group():
                        top_k = gr.Slider(minimum=1, maximum=30, value=6, label=i18n("Top K"), step=1)
                        top_p = gr.Slider(minimum=0, maximum=1, value=0.8, label=i18n("Top P"))
                        temperature = gr.Slider(
                            minimum=0, maximum=1, value=0.8, label=i18n("Temperature")
                        )
            batch_size.release(change_batch_size, inputs=[batch_size])
            word_count.release(change_word_count, inputs=[word_count])
            cut_method.input(lambda x: gr.update(visible=(cut_method_dict[x]=="auto_cut")),  [cut_method], [word_count])
        with gr.Column(visible=False):
            with gr.Tabs():

                with gr.Tab(label=i18n("网址设置")):
                    gr.Textbox(
                        value=i18n("这是展示页面的版本，并未使用后端服务，下面参数无效。"),
                        label=i18n("提示"),
                        interactive=False,
                    )
                    request_url_input = gr.Textbox(
                        value=default_request_url, label=i18n("请求网址"), interactive=False
                    )
                    endpoint = gr.Textbox(
                        value=default_endpoint, label=i18n("Endpoint"), interactive=False
                    )
                    character_list_url = gr.Textbox(
                        value=default_character_info_url,
                        label=i18n("人物情感列表网址"),
                        interactive=False,
                    )
                    request_url_input.blur(
                        change_endpoint,
                        inputs=[request_url_input],
                        outputs=[endpoint, character_list_url],
                    )
                with gr.Tab(label=i18n("认证信息"),visible=False):
                    gr.Textbox(
                        value=i18n("认证信息已启用，您可以在config.json中关闭。\n但是这个功能还没做好，只是摆设"),
                        label=i18n("认证信息"),
                        interactive=False
                    )
                    username = gr.Textbox(
                        value=default_username, label=i18n("用户名"), interactive=False
                    )
                    password = gr.Textbox(
                        value=default_password, label=i18n("密码"), interactive=False
                    )
                with gr.Tab(label=i18n("json设置（一般不动）"),visible=False):
                    endpoint_data = gr.Textbox(
                        value=default_endpoint_data, label=i18n("发送json格式"), lines=10
                    )
    with gr.Tabs():
        with gr.Tab(label=i18n("请求完整音频")):
            with gr.Row():
                sendRequest = gr.Button(i18n("发送请求"), variant="primary")
                audioRecieve = gr.Audio(
                    None, label=i18n("音频输出"), type="filepath", streaming=False
                )
        with gr.Tab(label=i18n("流式音频"),interactive=False,visible=False):
            with gr.Row():
                sendStreamRequest = gr.Button(
                    i18n("发送并开始播放"), variant="primary", interactive=True
                )
                stopStreamButton = gr.Button(i18n("停止播放"), variant="secondary")
            with gr.Row():
                audioStreamRecieve = gr.Audio(None, label=i18n("音频输出"), interactive=False)
    gr.HTML("<hr style='border-top: 1px solid #ccc; margin: 20px 0;' />")
    gr.HTML(
        f"""<p>{i18n("这是一个由")} <a href="{i18n("https://space.bilibili.com/66633770")}">XTer</a> {i18n("提供的推理特化包，当前版本：")}<a href="https://www.yuque.com/xter/zibxlp/awo29n8m6e6soru9">{frontend_version}</a>  {i18n("项目开源地址：")} <a href="https://github.com/X-T-E-R/TTS-for-GPT-soVITS">Github</a></p>
            <p>{i18n("吞字漏字属于正常现象，太严重可尝试换行、加句号或调节batch size滑条。")}</p>
            <p>{i18n("若有疑问或需要进一步了解，可参考文档：")}<a href="{i18n("https://www.yuque.com/xter/zibxlp")}">{i18n("点击查看详细文档")}</a>。</p>"""
    )
    # 以下是事件绑定
    app.load(
        change_character_list,
        inputs=[character_list_url, cha_name, auto_emotion_checkbox, character_emotion],
        outputs=[
            cha_name,
            auto_emotion_checkbox,
            character_emotion,
            characters_and_emotions,
        ]
    )            
    sendRequest.click(lambda: gr.update(interactive=False), None, [sendRequest]).then(
        send_request,
        inputs=[
            endpoint,
            endpoint_data,
            text,
            cha_name,
            text_language,
            batch_size,
            speed_factor,
            top_k,
            top_p,
            temperature,
            character_emotion,
            cut_method,
            word_count,
            seed,
            gr.State("False"),
        ],
        outputs=[audioRecieve],
    ).then(lambda: gr.update(interactive=True), None, [sendRequest])
    sendStreamRequest.click(
        lambda: gr.update(interactive=False), None, [sendStreamRequest]
    ).then(
        send_request,
        inputs=[
            endpoint,
            endpoint_data,
            text,
            cha_name,
            text_language,
            batch_size,
            speed_factor,
            top_k,
            top_p,
            temperature,
            character_emotion,
            cut_method,
            word_count,
            seed,
            gr.State("True"),
        ],
        outputs=[audioStreamRecieve],
    ).then(
        lambda: gr.update(interactive=True), None, [sendStreamRequest]
    )
    stopStreamButton.click(stopAudioPlay, inputs=[])
    cha_name.change(
        load_character_emotions,
        inputs=[cha_name, characters_and_emotions],
        outputs=[character_emotion],
    )
    character_list_url.change(
        change_character_list,
        inputs=[character_list_url, cha_name, auto_emotion_checkbox, character_emotion],
        outputs=[
            cha_name,
            auto_emotion_checkbox,
            character_emotion,
            characters_and_emotions,
        ],
    )
    scan_character_list.click(
        change_character_list,
        inputs=[character_list_url, cha_name, auto_emotion_checkbox, character_emotion],
        outputs=[
            cha_name,
            auto_emotion_checkbox,
            character_emotion,
            characters_and_emotions,
        ],
    )
    auto_emotion_checkbox.input(
        change_character_list,
        inputs=[character_list_url, cha_name, auto_emotion_checkbox, character_emotion],
        outputs=[
            cha_name,
            auto_emotion_checkbox,
            character_emotion,
            characters_and_emotions,
        ],
    )


app.launch(show_error=True, share=is_share, inbrowser=True)

