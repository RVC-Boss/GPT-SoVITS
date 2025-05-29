# 更新日志

## 20240121

1-config添加is_share, 诸如colab等场景可以将此改为True, 来使得webui映射到公网

2-WebUI添加英文系统英文翻译适配

3-cmd-asr自动判断是否已自带damo模型, 如不在默认目录上将从modelscope自带下载

4-[SoVITS训练报错ZeroDivisionError](https://github.com/RVC-Boss/GPT-SoVITS/issues/79) 尝试修复(过滤长度0的样本等)

5-清理TEMP文件夹缓存音频等文件

6-大幅削弱合成音频包含参考音频结尾的问题

## 20240122

1-修复过短输出文件返回重复参考音频的问题.

2-经测试, 英文日文训练原生支持(日文训练需要根目录不含非英文等特殊字符).

3-音频路径检查.如果尝试读取输入错的路径报错路径不存在, 而非ffmpeg错误.

## 20240123

1-解决hubert提取nan导致SoVITS/GPT训练报错ZeroDivisionError的问题

2-支持推理界面快速切换模型

3-优化模型文件排序逻辑

4-中文分词使用jieba_fast代替jieba

## 20240126

1-支持输出文本中英混合、日英混合

2-输出可选切分模式

3-修复uvr5读取到目录自动跳出的问题

4-修复多个换行导致推理报错

5-去除推理界面大量冗余log

6-支持mac训练推理

7-自动识别不支持半精度的卡强制单精度.cpu推理下强制单精度.

## 20240128

1-修复数字转汉字念法问题

2-修复句首少量字容易吞字的问题

3-通过限制排除不合理的参考音频长度

4-修复GPT训练不保存ckpt的问题

5-完善Dockerfile的下载模型流程

## 20240129

1-16系等半精度训练有问题的显卡把训练配置改为单精度训练

2-测试更新可用的colab版本

3-修复git clone modelscope funasr仓库+老版本funasr导致接口不对齐报错的问题


## 20240130

1-所有涉及路径的地方双引号自动去除,小白复制路径带双引号不会报错

2-修复中英文标点切割问题和句首句尾补标点的问题

3-增加按标点符号切分

## 20240201

1-修复uvr5读取格式错误导致分离失败的问题

2-支持中日英混合多种文本自动切分识别语种

## 20240202

1-修复asr路径尾缀带/保存文件名报错

2-引入paddlespeech的Normalizer https://github.com/RVC-Boss/GPT-SoVITS/pull/377 修复一些问题, 例如: xx.xx%(带百分号类), 元/吨 会读成 元吨 而不是元每吨,下划线不再会报错

## 20240207

1-修正语种传参混乱导致中文推理效果下降 https://github.com/RVC-Boss/GPT-SoVITS/issues/391

2-uvr5适配高版本librosa https://github.com/RVC-Boss/GPT-SoVITS/pull/403

3-[修复uvr5 inf everywhere报错的问题(is_half传参未转换bool导致恒定半精度推理, 16系显卡会inf)](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8)

4-优化英文文本前端

5-修复gradio依赖

6-支持三连根目录留空自动读取.list全路径

7-集成faster whisper ASR日文英文

## 20240208

1-GPT训练卡死 (win10 1909) 和https://github.com/RVC-Boss/GPT-SoVITS/issues/232 (系统语言繁体) GPT训练报错, [尝试修复](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b).

## 20240212

1-faster whisper和funasr逻辑优化.faster whisper转镜像站下载, 规避huggingface连不上的问题.

2-DPO Loss实验性训练选项开启, 通过构造负样本训练缓解GPT重复漏字问题.推理界面公开几个推理参数. https://github.com/RVC-Boss/GPT-SoVITS/pull/457

## 20240214

1-训练支持中文实验名 (原来会报错)

2-DPO训练改为可勾选选项而非必须.如勾选batch size自动减半.修复推理界面新参数不传参的问题.

## 20240216

1-支持无参考文本输入

2-修复中文文本前端bug https://github.com/RVC-Boss/GPT-SoVITS/issues/475

## 20240221

1-数据处理添加语音降噪选项 (降噪为只剩16k采样率, 除非底噪很大先不急着用哦).

2-中文日文前端处理优化 https://github.com/RVC-Boss/GPT-SoVITS/pull/559 https://github.com/RVC-Boss/GPT-SoVITS/pull/556 https://github.com/RVC-Boss/GPT-SoVITS/pull/532 https://github.com/RVC-Boss/GPT-SoVITS/pull/507 https://github.com/RVC-Boss/GPT-SoVITS/pull/509

3-mac CPU推理更快因此把推理设备从mps改到CPU

4-colab修复不开启公网url

## 20240306

1-推理加速50% (RTX3090+pytorch2.2.1+cu11.8+win10+py39 tested) https://github.com/RVC-Boss/GPT-SoVITS/pull/672

2-如果用faster whisper非中文ASR不再需要先下中文funasr模型

3-修复uvr5去混响模型 是否混响 反的 https://github.com/RVC-Boss/GPT-SoVITS/pull/610

4-faster whisper如果无cuda可用自动cpu推理 https://github.com/RVC-Boss/GPT-SoVITS/pull/675

5-修改is_half的判断使在Mac上能正常CPU推理 https://github.com/RVC-Boss/GPT-SoVITS/pull/573

## 202403/202404/202405

2个重点

1-修复sovits训练未冻结vq的问题 (可能造成效果下降)

2-增加一个快速推理分支

以下都是小修补

1-修复无参考文本模式问题

2-优化中英文文本前端

3-api格式优化

4-cmd格式问题修复

5-训练数据处理阶段不支持的语言提示报错

6-nan自动转fp32阶段的hubert提取bug修复

## 20240610

小问题修复:

1-完善纯标点、多标点文本输入的判断逻辑 https://github.com/RVC-Boss/GPT-SoVITS/pull/1168 https://github.com/RVC-Boss/GPT-SoVITS/pull/1169

2-uvr5中的mdxnet去混响cmd格式修复, 兼容路径带空格  [#501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232)

3-s2训练进度条逻辑修复 https://github.com/RVC-Boss/GPT-SoVITS/pull/1159

大问题修复:

4-修复了webui的GPT中文微调没读到bert导致和推理不一致, 训练太多可能效果还会变差的问题.如果大量数据微调的建议重新微调模型得到质量优化 [#99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a)

## 20240706

小问题修复:

1-[修正CPU推理默认bs小数](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041)

2-修复降噪、asr中途遇到异常跳出所有需处理的音频文件的问题 https://github.com/RVC-Boss/GPT-SoVITS/pull/1258 https://github.com/RVC-Boss/GPT-SoVITS/pull/1265 https://github.com/RVC-Boss/GPT-SoVITS/pull/1267

3-修复按标点符号切分时小数会被切分 https://github.com/RVC-Boss/GPT-SoVITS/pull/1253

4-[多卡训练多进程保存逻辑修复](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca)

5-移除冗余my_utils https://github.com/RVC-Boss/GPT-SoVITS/pull/1251

重点:

6-倍速推理代码经过验证后推理效果和base完全一致, 合并进main.使用的代码: https://github.com/RVC-Boss/GPT-SoVITS/pull/672 .支持无参考文本模式也倍速.

后面会逐渐验证快速推理分支的推理改动的一致性

## 20240727

1-清理冗余i18n代码 https://github.com/RVC-Boss/GPT-SoVITS/pull/1298

2-修复用户打文件及路径在结尾添加/会导致命令行报错的问题 https://github.com/RVC-Boss/GPT-SoVITS/pull/1299

3-修复GPT训练的step计算逻辑 https://github.com/RVC-Boss/GPT-SoVITS/pull/756

重点:

4-[支持合成语速调节.支持冻结随机性只调节语速, ](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2)并将其更新到api.py上https://github.com/RVC-Boss/GPT-SoVITS/pull/1340

- 2024.07.27 [PR#1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306), [PR#1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356): 增加 BS-Roformer 人声伴奏分离模型支持.
  - 类型: 新功能
  - 提交: KamioRinn
- 2024.07.27 [PR#1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351): 更好的中文文本前端.
  - 类型: 新功能
  - 提交: KamioRinn

## 202408 (V2 版本)

- 2024.08.01 [PR#1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1355): 添加自动填充下一步文件路径的功能.
  - 类型: 杂项
  - 提交: XXXXRT666
- 2024.08.01 [Commit#e62e9653](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c): 支持 BS-Roformer 的 FP16 推理.
  - 类型: 性能优化
  - 提交: RVC-Boss
- 2024.08.01 [Commit#bce451a2](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit#4c8b7612](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78): 增加用户友好逻辑, 对用户随意输入的显卡序号也能正常运行.
  - 类型: 杂项
  - 提交: RVC-Boss
- 2024.08.02 [Commit#ff6c193f](https://github.com/RVC-Boss/GPT-SoVITS/commit/ff6c193f6fb99d44eea3648d82ebcee895860a22)~[Commit#de7ee7c7](https://github.com/RVC-Boss/GPT-SoVITS/commit/de7ee7c7c15a2ec137feb0693b4ff3db61fad758): **新增 GPT-SoVITS V2 模型.**
  - 类型: 新功能
  - 提交: RVC-Boss
- 2024.08.03 [Commit#8a101474](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3): 增加粤语 FunASR 支持.
  - 类型: 新功能
  - 提交: RVC-Boss
- 2024.08.03 [PR#1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387), [PR#1388](https://github.com/RVC-Boss/GPT-SoVITS/pull/1388): 优化界面, 优化计时逻辑.
  - 类型: 杂项
  - 提交: XXXXRT666
- 2024.08.06 [PR#1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404), [PR#987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987), [PR#488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488): 优化多音字逻辑 (V2 版本特供).
  - 类型: 修复, 新功能
  - 提交: KamioRinn, RVC-Boss
- 2024.08.13 [PR#1422](https://github.com/RVC-Boss/GPT-SoVITS/pull/1422): 修复参考音频混合只能上传一条的错误, 添加数据集检查, 缺失会弹出警告窗口.
  - 类型: 修复, 杂项
  - 提交: XXXXRT666
- 2024.08.20 [Issue#1508](https://github.com/RVC-Boss/GPT-SoVITS/issues/1508): 上游 LangSegment 库支持通过 SSML 标签优化数字、电话、时间日期等.
  - 类型: 新功能
  - 提交: juntaosun
- 2024.08.20 [PR#1503](https://github.com/RVC-Boss/GPT-SoVITS/pull/1503): 修复并优化 API.
  - 类型: 修复
  - 提交: KamioRinn
- 2024.08.20 [PR#1490](https://github.com/RVC-Boss/GPT-SoVITS/pull/1490): 合并 fast_inference 分支.
  - 类型: 重构
  - 提交: ChasonJiang
- 2024.08.21 **正式发布 GPT-SoVITS V2 版本.**

## 202502 (V3 版本)

- 2025.02.11 [Commit#ed207c4b](https://github.com/RVC-Boss/GPT-SoVITS/commit/ed207c4b879d5296e9be3ae5f7b876729a2c43b8)~[Commit#6e2b4918](https://github.com/RVC-Boss/GPT-SoVITS/commit/6e2b49186c5b961f0de41ea485d398dffa9787b4): **新增 GPT-SoVITS V3 模型, 需要 14G 显存进行微调.**
  - 类型: 新功能 (特性参阅 [Wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)))
  - 提交: RVC-Boss
- 2025.02.12 [PR#2032](https://github.com/RVC-Boss/GPT-SoVITS/pull/2032): 更新项目多语言文档.
  - 类型: 文档
  - 提交: StaryLan
- 2025.02.12 [PR#2033](https://github.com/RVC-Boss/GPT-SoVITS/pull/2033): 更新日语文档.
  - 类型: 文档
  - 提交: Fyphen
- 2025.02.12 [PR#2010](https://github.com/RVC-Boss/GPT-SoVITS/pull/2010): 优化注意力计算逻辑.
  - 类型: 性能优化
  - 提交: wzy3650
- 2025.02.12 [PR#2040](https://github.com/RVC-Boss/GPT-SoVITS/pull/2040): 微调添加梯度检查点支持, 需要 12G 显存进行微调.
  - 类型: 新功能
  - 提交: Kakaru Hayate
- 2025.02.14 [PR#2047](https://github.com/RVC-Boss/GPT-SoVITS/pull/2047), [PR#2062](https://github.com/RVC-Boss/GPT-SoVITS/pull/2062), [PR#2073](https://github.com/RVC-Boss/GPT-SoVITS/pull/2073): 切换新的语言分割工具, 优化多语种混合文本切分策略, 优化文本里的数字和英文处理逻辑.
  - 类型: 新功能
  - 提交: KamioRinn
- 2025.02.23 [Commit#56509a17](https://github.com/RVC-Boss/GPT-SoVITS/commit/56509a17c918c8d149c48413a672b8ddf437495b)~[Commit#514fb692](https://github.com/RVC-Boss/GPT-SoVITS/commit/514fb692db056a06ed012bc3a5bca2a5b455703e): **GPT-SoVITS V3 模型支持 LoRA 训练, 需要 8G 显存进行微调.**
  - 类型: 新功能
  - 提交: RVC-Boss
- 2025.02.23 [PR#2078](https://github.com/RVC-Boss/GPT-SoVITS/pull/2078): 人声背景音分离增加 Mel Band Roformer 模型支持.
  - 类型: 新功能
  - 提交: Sucial
- 2025.02.26 [PR#2112](https://github.com/RVC-Boss/GPT-SoVITS/pull/2112), [PR#2114](https://github.com/RVC-Boss/GPT-SoVITS/pull/2114): 修复中文路径下 Mecab 的报错 (具体表现为日文韩文、文本混合语种切分可能会遇到的报错).
  - 类型: 修复
  - 提交: KamioRinn
- 2025.02.27 [Commit#92961c3f](https://github.com/RVC-Boss/GPT-SoVITS/commit/92961c3f68b96009ff2cd00ce614a11b6c4d026f)~[Commit#](https://github.com/RVC-Boss/GPT-SoVITS/commit/250b1c73cba60db18148b21ec5fbce01fd9d19bc): **支持使用 24KHz 转 48kHz 的音频超分模型**, 缓解 V3 模型生成音频感觉闷的问题.
  - 类型: 新功能
  - 提交: RVC-Boss
  - 关联: [Issue#2085](https://github.com/RVC-Boss/GPT-SoVITS/issues/2085), [Issue#2117](https://github.com/RVC-Boss/GPT-SoVITS/issues/2117)
- 2025.02.28 [PR#2123](https://github.com/RVC-Boss/GPT-SoVITS/pull/2123): 更新项目多语言文档
  - 类型: 文档
  - 提交: StaryLan
- 2025.02.28 [PR#2122](https://github.com/RVC-Boss/GPT-SoVITS/pull/2122): 对于模型无法判断的CJK短字符采用规则判断.
  - 类型: 修复
  - 提交: KamioRinn
  - 关联: [Issue#2116](https://github.com/RVC-Boss/GPT-SoVITS/issues/2116)
- 2025.02.28 [Commit#c38b1690](https://github.com/RVC-Boss/GPT-SoVITS/commit/c38b16901978c1db79491e16905ea3a37a7cf686), [Commit#a32a2b89](https://github.com/RVC-Boss/GPT-SoVITS/commit/a32a2b893436fad56cc82409121c7fa36a1815d5): 增加语速传参以支持调整合成语速.
  - 类型: 修复
  - 提交: RVC-Boss
- 2025.02.28 **正式发布 GPT-SoVITS V3**.

## 202503

- 2025.03.31 [PR#2236](https://github.com/RVC-Boss/GPT-SoVITS/pull/2236): 修复一批由依赖的库版本不对导致的问题.
  - 类型: 修复
  - 提交: XXXXRT666
  - 关联:
    - PyOpenJTalk: [Issue#1131](https://github.com/RVC-Boss/GPT-SoVITS/issues/1131), [Issue#2231](https://github.com/RVC-Boss/GPT-SoVITS/issues/2231), [Issue#2233](https://github.com/RVC-Boss/GPT-SoVITS/issues/2233).
    - ONNX: [Issue#492](https://github.com/RVC-Boss/GPT-SoVITS/issues/492), [Issue#671](https://github.com/RVC-Boss/GPT-SoVITS/issues/671), [Issue#1192](https://github.com/RVC-Boss/GPT-SoVITS/issues/1192), [Issue#1819](https://github.com/RVC-Boss/GPT-SoVITS/issues/1819), [Issue#1841](https://github.com/RVC-Boss/GPT-SoVITS/issues/1841).
    - Pydantic: [Issue#2230](https://github.com/RVC-Boss/GPT-SoVITS/issues/2230), [Issue#2239](https://github.com/RVC-Boss/GPT-SoVITS/issues/2239).
    - PyTorch-Lightning: [Issue#2174](https://github.com/RVC-Boss/GPT-SoVITS/issues/2174).
- 2025.03.31 [PR#2241](https://github.com/RVC-Boss/GPT-SoVITS/pull/2241): **为 SoVITS v3 适配并行推理**.
  - 类型: 新功能
  - 提交: ChasonJiang

- 修复其他若干错误.

- 整合包修复 onnxruntime GPU 推理的支持
  - 类型: 修复
  - 内容:
    - G2PW 内的 ONNX 模型由 CPU 推理 换为 GPU, 显著降低推理的 CPU 瓶颈;
    - foxjoy 去混响模型现在可使用 GPU 推理

## 202504 (V4 版本)

- 2025.04.01 [Commit#6a60e5ed](https://github.com/RVC-Boss/GPT-SoVITS/commit/6a60e5edb1817af4a61c7a5b196c0d0f1407668f): 解锁 SoVITS v3 并行推理, 修复模型加载异步逻辑.
  - 类型: 修复
  - 提交: RVC-Boss
- 2025.04.07 [PR#2255](https://github.com/RVC-Boss/GPT-SoVITS/pull/2255): Ruff 格式化代码, 更新 G2PW 链接.
  - 类型: 风格
  - 提交: XXXXRT666
- 2025.04.15 [PR#2290](https://github.com/RVC-Boss/GPT-SoVITS/pull/2290): 清理文档, 支持 Python 3.11, 更新安装文件.
  - 类型: 杂项
  - 提交: XXXXRT666
- 2025.04.20 [PR#2300](https://github.com/RVC-Boss/GPT-SoVITS/pull/2300): 更新 Colab, 安装文件和模型下载.
  - 类型: 杂项
  - 提交: XXXXRT666
- 2025.04.20 [Commit#e0c452f0](https://github.com/RVC-Boss/GPT-SoVITS/commit/e0c452f0078e8f7eb560b79a54d75573fefa8355)~[Commit#9d481da6](https://github.com/RVC-Boss/GPT-SoVITS/commit/9d481da610aa4b0ef8abf5651fd62800d2b4e8bf): **新增 GPT-SoVITS V4 模型**.
  - 类型: 新功能
  - 提交: RVC-Boss
- 2025.04.21 [Commit#8b394a15](https://github.com/RVC-Boss/GPT-SoVITS/commit/8b394a15bce8e1d85c0b11172442dbe7a6017ca2)~[Commit#bc2fe5ec](https://github.com/RVC-Boss/GPT-SoVITS/commit/bc2fe5ec86536c77bb3794b4be263ac87e4fdae6), [PR#2307](https://github.com/RVC-Boss/GPT-SoVITS/pull/2307): 适配 V4 并行推理.
  - 类型: 新功能
  - 提交: RVC-Boss, ChasonJiang
- 2025.04.22 [Commit#7405427a](https://github.com/RVC-Boss/GPT-SoVITS/commit/7405427a0ab2a43af63205df401fd6607a408d87)~[Commit#590c83d7](https://github.com/RVC-Boss/GPT-SoVITS/commit/590c83d7667c8d4908f5bdaf2f4c1ba8959d29ff), [PR#2309](https://github.com/RVC-Boss/GPT-SoVITS/pull/2309): 修复模型版本传参.
  - 类型: 修复
  - 提交: RVC-Boss, ChasonJiang
- 2025.04.22 [Commit#fbdab94e](https://github.com/RVC-Boss/GPT-SoVITS/commit/fbdab94e17d605d85841af6f94f40a45976dd1d9), [PR#2310](https://github.com/RVC-Boss/GPT-SoVITS/pull/2310): 修复 Numpy 与 Numba 版本不匹配问题, 更新 librosa 版本.
  - 类型: 修复
  - 提交: RVC-Boss, XXXXRT666
  - 关联: [Issue#2308](https://github.com/RVC-Boss/GPT-SoVITS/issues/2308)
- **2024.04.22 正式发布 GPT-SoVITS V4**.
- 2025.04.22 [PR#2311](https://github.com/RVC-Boss/GPT-SoVITS/pull/2311): 更新 Gradio 参数.
  - 类型: 杂项
  - 提交: XXXXRT666
- 2025.04.25 [PR#2322](https://github.com/RVC-Boss/GPT-SoVITS/pull/2322): 完善 Colab/Kaggle Notebook 脚本.
  - 类型: 杂项
  - 提交: XXXXRT666

## 202505

- 2025.05.26 [PR#2351](https://github.com/RVC-Boss/GPT-SoVITS/pull/2351): 完善 Docker, Windows 自动构建脚本, Pre-Commit 格式化.
  - 类型: 杂项
  - 提交: XXXXRT666
- 2025.05.26 [PR#2408](https://github.com/RVC-Boss/GPT-SoVITS/pull/2408): 优化混合语种切分识别逻辑.
  - 类型: 修复
  - 提交: KamioRinn
  - 关联: [Issue#2404](https://github.com/RVC-Boss/GPT-SoVITS/issues/2404)
- 2025.05.26 [PR#2377](https://github.com/RVC-Boss/GPT-SoVITS/pull/2377): 通过缓存策略使 SoVITS V3/V4 推理提速 10%.
  - 类型: 性能优化
  - 提交: Kakaru Hayate
- 2025.05.26 [Commit#4d9d56b1](https://github.com/RVC-Boss/GPT-SoVITS/commit/4d9d56b19638dc434d6eefd9545e4d8639a3e072), [Commit#8c705784](https://github.com/RVC-Boss/GPT-SoVITS/commit/8c705784c50bf438c7b6d0be33a9e5e3cb90e6b2), [Commit#fafe4e7f](https://github.com/RVC-Boss/GPT-SoVITS/commit/fafe4e7f120fba56c5f053c6db30aa675d5951ba): 更新标注界面, 增加友情提示, 即标注完每一面都要点 Submit Text 否则修改无效.
  - 类型: 修复
  - 提交: RVC-Boss
- 2025.05.29 [Commit#1934fc1e](https://github.com/RVC-Boss/GPT-SoVITS/commit/1934fc1e1b22c4c162bba1bbe7d7ebb132944cdc): 修复 UVR5 和 ONNX 去混响模型使用 FFmpeg 编码 MP3 和 M4A 原路径带空格时的错误.
  - 类型: 修复
  - 提交: RVC-Boss

**预告：端午后基于V2版本进行重大优化更新！**
