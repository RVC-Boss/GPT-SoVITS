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

3-[修复uvr5 inf everywhere报错的问题(is_half传参未转换bool导致恒定半精度推理，16系显卡会inf)](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8)

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

### 20240221更新

1-数据处理添加语音降噪选项（降噪为只剩16k采样率，除非底噪很大先不急着用哦。）

2-中文日文前端处理优化 https://github.com/RVC-Boss/GPT-SoVITS/pull/559 https://github.com/RVC-Boss/GPT-SoVITS/pull/556 https://github.com/RVC-Boss/GPT-SoVITS/pull/532 https://github.com/RVC-Boss/GPT-SoVITS/pull/507 https://github.com/RVC-Boss/GPT-SoVITS/pull/509

3-mac CPU推理更快因此把推理设备从mps改到CPU

4-colab修复不开启公网url

### 20240306更新

1-推理加速50%（RTX3090+pytorch2.2.1+cu11.8+win10+py39 tested）https://github.com/RVC-Boss/GPT-SoVITS/pull/672

2-如果用faster whisper非中文ASR不再需要先下中文funasr模型

3-修复uvr5去混响模型 是否混响 反的 https://github.com/RVC-Boss/GPT-SoVITS/pull/610

4-faster whisper如果无cuda可用自动cpu推理 https://github.com/RVC-Boss/GPT-SoVITS/pull/675

5-修改is_half的判断使在Mac上能正常CPU推理 https://github.com/RVC-Boss/GPT-SoVITS/pull/573

### 202403/202404/202405更新

2个重点

1-修复sovits训练未冻结vq的问题（可能造成效果下降）

2-增加一个快速推理分支

以下都是小修补

1-修复无参考文本模式问题

2-优化中英文文本前端

3-api格式优化

4-cmd格式问题修复

5-训练数据处理阶段不支持的语言提示报错

6-nan自动转fp32阶段的hubert提取bug修复

### 20240610

小问题修复：

1-完善纯标点、多标点文本输入的判断逻辑 https://github.com/RVC-Boss/GPT-SoVITS/pull/1168 https://github.com/RVC-Boss/GPT-SoVITS/pull/1169

2-uvr5中的mdxnet去混响cmd格式修复，兼容路径带空格  [#501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232)

3-s2训练进度条逻辑修复 https://github.com/RVC-Boss/GPT-SoVITS/pull/1159

大问题修复：

4-修复了webui的GPT中文微调没读到bert导致和推理不一致，训练太多可能效果还会变差的问题。如果大量数据微调的建议重新微调模型得到质量优化 [#99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a)

### 20240706

小问题修复：

1-[修正CPU推理默认bs小数](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041)

2-修复降噪、asr中途遇到异常跳出所有需处理的音频文件的问题 https://github.com/RVC-Boss/GPT-SoVITS/pull/1258 https://github.com/RVC-Boss/GPT-SoVITS/pull/1265 https://github.com/RVC-Boss/GPT-SoVITS/pull/1267

3-修复按标点符号切分时小数会被切分 https://github.com/RVC-Boss/GPT-SoVITS/pull/1253

4-[多卡训练多进程保存逻辑修复](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca)

5-移除冗余my_utils https://github.com/RVC-Boss/GPT-SoVITS/pull/1251

重点：

6-倍速推理代码经过验证后推理效果和base完全一致，合并进main。使用的代码：https://github.com/RVC-Boss/GPT-SoVITS/pull/672 。支持无参考文本模式也倍速。

后面会逐渐验证快速推理分支的推理改动的一致性

### 20240727

1-清理冗余i18n代码 https://github.com/RVC-Boss/GPT-SoVITS/pull/1298

2-修复用户打文件及路径在结尾添加/会导致命令行报错的问题 https://github.com/RVC-Boss/GPT-SoVITS/pull/1299

3-修复GPT训练的step计算逻辑 https://github.com/RVC-Boss/GPT-SoVITS/pull/756

重点：

4-[支持合成语速调节。支持冻结随机性只调节语速，](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2)并将其更新到api.py上https://github.com/RVC-Boss/GPT-SoVITS/pull/1340


### 20240806

1-增加bs-roformer人声伴奏分离模型支持。 https://github.com/RVC-Boss/GPT-SoVITS/pull/1306 https://github.com/RVC-Boss/GPT-SoVITS/pull/1356 [支持fp16推理。](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c)

2-更好的中文文本前端。 https://github.com/RVC-Boss/GPT-SoVITS/pull/987 https://github.com/RVC-Boss/GPT-SoVITS/pull/1351 https://github.com/RVC-Boss/GPT-SoVITS/pull/1404 优化多音字逻辑（v2版本特供）。 https://github.com/RVC-Boss/GPT-SoVITS/pull/488

3-自动填充下一步的文件路径 https://github.com/RVC-Boss/GPT-SoVITS/pull/1355

4-增加喂饭逻辑，用户瞎写显卡序号也可以正常运作 [bce451a](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299) [4c8b761](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78)

5-增加粤语ASR支持 [8a10147](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3)

6-GPT-SoVITS-v2支持

7-计时逻辑优化 https://github.com/RVC-Boss/GPT-SoVITS/pull/1387

### 20240821

1-fast_inference分支合并进main：https://github.com/RVC-Boss/GPT-SoVITS/pull/1490

2-支持通过ssml标签优化数字、电话、时间日期等：https://github.com/RVC-Boss/GPT-SoVITS/issues/1508

3-api修复优化：https://github.com/RVC-Boss/GPT-SoVITS/pull/1503

4-修复了参考音频混合只能上传一条的bug:https://github.com/RVC-Boss/GPT-SoVITS/pull/1422

5-增加了各种数据集检查,若缺失会弹出warning:https://github.com/RVC-Boss/GPT-SoVITS/pull/1422

### 20250211

增加gpt-sovits-v3模型，需要14G显存可以微调

### 20250212

sovits-v3微调支持开启梯度检查点，需要12G显存可以微调https://github.com/RVC-Boss/GPT-SoVITS/pull/2040

### 20250214

优化多语种混合文本切分策略a https://github.com/RVC-Boss/GPT-SoVITS/pull/2047

### 20250217

优化文本里的数字和英文处理逻辑https://github.com/RVC-Boss/GPT-SoVITS/pull/2062

### 20250218

优化多语种混合文本切分策略b https://github.com/RVC-Boss/GPT-SoVITS/pull/2073

### 20250223

1-sovits-v3微调支持lora训练，需要8G显存可以微调，效果比全参微调更好

2-人声背景音分离增加mel band roformer模型支持https://github.com/RVC-Boss/GPT-SoVITS/pull/2078

### 20250226

https://github.com/RVC-Boss/GPT-SoVITS/pull/2112 https://github.com/RVC-Boss/GPT-SoVITS/pull/2114

修复中文路径下mecab的报错（具体表现为日文韩文、文本混合语种切分可能会遇到的报错）

### 20250227

针对v3生成24k音频感觉闷的问题https://github.com/RVC-Boss/GPT-SoVITS/issues/2085 https://github.com/RVC-Boss/GPT-SoVITS/issues/2117 ,支持使用24k to 48k的音频超分模型缓解。


### 20250228

修复短文本语种选择出错 https://github.com/RVC-Boss/GPT-SoVITS/pull/2122

修复v3sovits未传参以支持调节语速

### 202503

修复一批由依赖的库版本不对导致的问题https://github.com/RVC-Boss/GPT-SoVITS/commit/6c468583c5566e5fbb4fb805e4cc89c403e997b8

修复模型加载异步逻辑https://github.com/RVC-Boss/GPT-SoVITS/commit/03b662a769946b7a6a8569a354860e8eeeb743aa

修复其他若干bug

重点更新：

1-v3支持并行推理 https://github.com/RVC-Boss/GPT-SoVITS/commit/03b662a769946b7a6a8569a354860e8eeeb743aa

2-整合包修复onnxruntime GPU推理的支持，影响：（1）g2pw有个onnx模型原先是CPU推理现在用GPU，显著降低推理的CPU瓶颈 （2）foxjoy去混响模型现在可使用GPU推理
