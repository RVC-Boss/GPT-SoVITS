# GPT-SoVITS_inference

- 项目基于[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)进行修改。
- 项目应用场景为使用GPT-SoVITS完成语音模型的训练之后需要整合到其他项目之中，故而删除了训练部分以及基于webUI推理部分的代码。
- 项目在完成环境配置（详见[GPT-SoVITS文档](./README_origin.md)）后，修改inference.py中line 42行部分的speakers。该部分主要包含训练完成的模型地址以及参考语音地址等相关的内容。
- 直接运行可以得到output.wav，该文件即为模型生成的语音文件。