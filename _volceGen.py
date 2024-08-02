import requests

class GSV_Gen:

    def __init__(self) -> None:
        self.api_url = "http://127.0.0.1:9880?text={text}&text_language={text_language}"
        self.emotion_api_url = "http://127.0.0.1:9880?refer_wav_path={wav_path}&prompt_text={prompt_text}。&prompt_language={prompt_language}&text={text}&text_language={text_language}"
        self.change_api_url = "http://127.0.0.1:9880/change_refer?refer_wav_path={refer_wav_path}&prompt_text={prompt_text}&prompt_language={prompt_language}"
        self.command_endpoint = "http://127.0.0.1:9880/control?command={command}"
        self.base_path = r"F:\BaiduNetdiskDownload\GPT-SoVITS\beta\GPT-SoVITS-beta0706\character\符玄\参考音频"

        # 这个字典是存储情感参考音频和文本的文件路径
        self.defaultEmoDICT: dict[str,tuple[str,str]] = {
            "happy": (f"{self.base_path}/激动说话-如此境地，还要处理将军交来的星核猎手，可不是大祸临头？.wav", "如此境地，还要处理将军交来的星核猎手，可不是大祸临头？"),
            "sad": (f"{self.base_path}/难过/【难过】涨落在乾、震之间。行有眚，无攸利。.wav", "涨落在乾、震之间。行有眚，无攸利。"),
            "angry": (f"{self.base_path}/生气/【生气】青雀！又是你，是你把这些外人带进司部的吗？你把我平日里所立的规矩都当作耳旁风了吗？.wav","青雀！又是你，是你把这些外人带进司部的吗？你把我平日里所立的规矩都当作耳旁风了吗？"),
            "normally": (f"{self.base_path}/中立/【中立】然后加水，加糖，糖要加到…加到茶水里再也溶不进一粒糖为止。.wav", "然后加水，加糖，糖要加到…加到茶水里再也溶不进一粒糖为止。"),
            "fear": (f"{self.base_path}/恐惧/【恐惧】做出了很多荒唐的事，真是颜面尽失。.wav","做出了很多荒唐的事，真是颜面尽失。"),
        }

    def endRunning(self):
        command = "exit"
        command_url = self.command_endpoint.format(command=command)
        requests.get(command_url)

    def restart(self):
        command = "restart"
        command_url = self.command_endpoint.format(command=command)
        requests.get(command_url)

    def GenVoice(self, text: str, language: str):
        url = self.api_url.format(text=text, text_language=language)
        try:
            resp = requests.get(url)
            return resp.content
        except Exception as e:
           return None

    def GenVoicewithEmotion(self,wav_path: str, prompt_text: str, prompt_language: str, text: str, language: str):
        url = self.emotion_api_url.format(wav_path=wav_path, prompt_text=prompt_text, prompt_language=prompt_language, text=text, text_language=language)
        try:
            resp = requests.get(url)
            return resp.content
        except Exception as e:
            return None

    def EmotionGen(self,emotion: str, text: str, language: str):
        if emotion is None or "":
           return self.GenVoice(text=text,language=language)
        wav_path = self.defaultEmoDICT[emotion][0]
        prompt_text = self.defaultEmoDICT[emotion][1]
        return self.GenVoicewithEmotion(wav_path=wav_path, prompt_text=prompt_text, prompt_language="zh", text=text, language=language)