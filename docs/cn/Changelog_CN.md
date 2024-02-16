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

### 20240123更新

1-解决hubert提取nan导致SoVITS/GPT训练报错ZeroDivisionError的问题

2-支持推理界面快速切换模型

3-优化模型文件排序逻辑

4-中文分词使用jieba_fast代替jieba

### 20240126更新

1-支持输出文本中英混合、日英混合

2-输出可选切分模式

3-修复uvr5读取到目录自动跳出的问题

4-修复多个换行导致推理报错

5-去除推理界面大量冗余log

6-支持mac训练推理

7-自动识别不支持半精度的卡强制单精度。cpu推理下强制单精度。

### 20240128更新

1-修复数字转汉字念法问题

2-修复句首少量字容易吞字的问题

3-通过限制排除不合理的参考音频长度

4-修复GPT训练不保存ckpt的问题

5-完善Dockerfile的下载模型流程

### 20240129更新

1-16系等半精度训练有问题的显卡把训练配置改为单精度训练

2-测试更新可用的colab版本

3-修复git clone modelscope funasr仓库+老版本funasr导致接口不对齐报错的问题


### 20240130更新

1-所有涉及路径的地方双引号自动去除,小白复制路径带双引号不会报错

2-修复中英文标点切割问题和句首句尾补标点的问题

3-增加按标点符号切分

### 20240201更新

1-修复uvr5读取格式错误导致分离失败的问题

2-支持中日英混合多种文本自动切分识别语种

### 20240202更新

1-修复asr路径尾缀带/保存文件名报错

2-引入paddlespeech的Normalizer https://github.com/RVC-Boss/GPT-SoVITS/pull/377 修复一些问题，例如：xx.xx%(带百分号类)，元/吨 会读成 元吨 而不是元每吨,下划线不再会报错

### 20240207更新

1-修正语种传参混乱导致中文推理效果下降 https://github.com/RVC-Boss/GPT-SoVITS/issues/391

2-uvr5适配高版本librosa https://github.com/RVC-Boss/GPT-SoVITS/pull/403

3-修复uvr5 inf everywhere报错的问题(is_half传参未转换bool导致恒定半精度推理，16系显卡会inf) https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8

4-优化英文文本前端

5-修复gradio依赖

6-支持三连根目录留空自动读取.list全路径

7-集成faster whisper ASR日文英文

### 20240208更新

1-GPT训练卡死（win10 1909）和https://github.com/RVC-Boss/GPT-SoVITS/issues/232 （系统语言繁体）GPT训练报错，[尝试修复](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b)。

### 20240212更新

1-faster whisper和funasr逻辑优化。faster whisper转镜像站下载，规避huggingface连不上的问题。

2-DPO Loss实验性训练选项开启，通过构造负样本训练缓解GPT重复漏字问题。推理界面公开几个推理参数。 https://github.com/RVC-Boss/GPT-SoVITS/pull/457

### 20240214更新

1-训练支持中文实验名（原来会报错）

2-DPO训练改为可勾选选项而非必须。如勾选batch size自动减半。修复推理界面新参数不传参的问题。

### 20240216更新

1-支持无参考文本输入

2-修复中文文本前端bug https://github.com/RVC-Boss/GPT-SoVITS/issues/475

todolist：

1-中文多音字推理优化



