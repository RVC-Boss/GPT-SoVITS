### 20240121更新

1-config添加is_share，诸如colab等场景可以将此改为True，来使得webui映射到公网

2-WebUI添加英文系统英文翻译适配

3-cmd-asr自动判断是否已自带damo模型，如不在默认目录上将从modelscope自带下载

4-[SoVITS训练报错ZeroDivisionError](https://github.com/RVC-Boss/GPT-SoVITS/issues/79) 尝试修复（过滤长度0的样本等）

5-清理TEMP文件夹缓存音频等文件

6-大幅削弱合成音频包含参考音频结尾的问题

### 20240122更新

1-修复过短输出文件返回重复参考音频的问题。

2-经测试，英文日文训练原生支持（日文训练需要根目录不含非英文等特殊字符）。

3-音频路径检查。如果尝试读取输入错的路径报错路径不存在，而非ffmpeg错误。

待修复：-hubert提取在half下出现nan概率更高的问题

