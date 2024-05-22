# api

` python api.py -dr "123.wav" -dt "一二三。" -dl "zh" `

## 执行参数:

`-s` - `SoVITS模型路径, 可在 config.py 中指定`  
`-g` - `GPT模型路径, 可在 config.py 中指定`  

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`  
`-dt` - `默认参考音频文本`  
`-dl` - `默认参考音频语种, "中文","英文","日文","zh","en","ja"`  

`-d` - `推理设备, "cuda","cpu"`  
`-a` - `绑定地址, 默认"127.0.0.1"`  
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`  
`-fp` - `覆盖 config.py 使用全精度`  
`-hp` - `覆盖 config.py 使用半精度`  
`-sm` - `流式返回模式, 默认不启用, "close","c", "normal","n", "keepalive","k"`  
·-mt` - `返回的音频编码格式, 流式默认ogg, 非流式默认wav, "wav", "ogg", "aac"`  
·-cp` - `文本切分符号设定, 默认为空, 以",.，。"字符串的方式传入`  

`-hb` - `cnhubert路径`  
`-b` - `bert路径`  

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
- GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`

- POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

使用执行参数指定的参考音频并设定分割符号:
- GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh&cut_punc=，。`
- POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh",
    "cut_punc": "，。"
}
```

手动指定当次推理所使用的参考音频:
- GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
- POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh",
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

RESP:
- 成功: 直接返回 wav 音频流， http code 200
- 失败: 返回包含错误信息的 json, http code 400


### 更换默认参考音频

endpoint: `/change_refer`

key与推理端一样

- GET:
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh`
- POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh"
}
```

RESP:
成功: json, http code 200
失败: json, 400


### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

- GET:
    `http://127.0.0.1:9880/control?command=restart`
- POST:
```json
{
    "command": "restart"
}
```

RESP: 无