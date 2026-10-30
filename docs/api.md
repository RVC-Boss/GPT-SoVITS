# HTTP 推理 API

服务默认地址为 `http://127.0.0.1:9880`。模型随服务启动加载；以下请求使用当前进程中已加载的模型。交互式接口定义位于 `/docs`，机器可读定义位于 `/openapi.json`。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET、POST | `/tts` | 合成完整音频或流式音频 |
| GET | `/set_refer_audio` | 预加载参考音频特征 |
| GET | `/set_gpt_weights` | 切换 GPT 权重 |
| GET | `/set_sovits_weights` | 切换 SoVITS 权重和对应模型版本 |

## 请求中的路径与语言

`ref_audio_path`、`aux_ref_audio_paths`、`refer_audio_path`、`weights_path` 都是**服务端本地路径**。相对路径从仓库根目录解析；容器内路径从容器的仓库目录解析，客户端机器上的路径不会自动映射进去。参考音频和权重应提前放到服务器或挂载到容器中。主参考音频经解码后的长度须在 **3～10 秒，包含边界**。

`text_lang` 与 `prompt_lang` 使用下表的小写代码。当前语言集合由 SoVITS 模型版本决定。

| 代码 | 文本处理方式 | 可用版本 |
| --- | --- | --- |
| `en` | 整段按英文处理 | 全部 |
| `zh`、`ja` | 英文片段保留英文，其余片段按指定的中文、日文处理 | 全部 |
| `all_zh`、`all_ja` | 以中文、日文作为分段器默认语言，数字和非英文片段使用该语言；检测出的英文片段仍保留英文 | 全部 |
| `auto` | 自动分段并识别语言 | 全部；v1 的模型语言范围为中、英、日 |
| `yue`、`ko` | 英文片段保留英文，其余片段按粤语、韩文处理 | v2、v2Pro、v2ProPlus、v3、v4、v5dev、v5turbo |
| `all_yue`、`all_ko` | 以粤语、韩文作为非英文片段及数字的默认语言 | 同上 |
| `auto_yue` | 自动识别后，将识别为中文的片段按粤语处理 | 同上 |

v1 接口允许的完整代码集合是 `auto`、`en`、`zh`、`ja`、`all_zh`、`all_ja`。其他版本增加 `auto_yue`、`yue`、`ko`、`all_yue`、`all_ko`。GET 会把语言代码转成小写；POST 应直接使用表中的小写值。

中文前端会自动规范化数字和日期等文本，包括整数、小数、数字范围、日期、时间、温度、分数、百分比、金额、电话及编号。例如 `1995年3月1日` 会按年月日读法处理，`62％` 会按百分数读法处理。若某个编号、缩写或数字有特定读法，可以在 `text` 中直接写出希望读出的文字。`prompt_text` 则应填写参考音频实际说出的内容。

## `GET /tts` 与 `POST /tts`

POST 使用 `Content-Type: application/json`，请求体为一个对象。GET 使用同名查询参数。一次请求的 `text` 是一个字符串；`batch_size` 控制文本切分后的小批量推理，不表示传入多个独立请求。

### 必需字段

这四项在接口模型中的缺省值都是 `null`，但处理器会拒绝省略值或空字符串并返回 HTTP 400。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `text` | string | 要合成的文本 |
| `text_lang` | string | 目标文本语言代码 |
| `ref_audio_path` | string | 主参考音频路径；调用过 `/set_refer_audio` 后仍须提供 |
| `prompt_lang` | string | 参考文本语言代码；即使省略 `prompt_text` 也须提供 |

### 可选字段及完整默认值

默认值来自 HTTP 请求模型和 GET 参数签名；`sample_steps`、`cfg_rate` 的版本相关有效值在表内说明。省略字段与显式传 JSON `null` 并不总是等价：只有 `sample_steps`、`cfg_rate` 声明为可空数值；其他字段应使用表中类型，未使用的 `aux_ref_audio_paths` 可以省略或传 `[]`。

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `aux_ref_audio_paths` | array of string | `null`，运行时按 `[]` 处理 | 附加参考音频，用于支持多参考音色融合的合成路径；主参考音频仍由 `ref_audio_path` 指定 |
| `prompt_text` | string | `""` | 主参考音频的文字。v1/v2/Pro 系列允许空值；v3/v4/v5 需要有效参考文本 |
| `top_k` | integer | `15` | GPT 采样的候选 token 数 |
| `top_p` | number | `1` | GPT nucleus sampling 参数 |
| `temperature` | number | `1` | GPT 采样温度 |
| `text_split_method` | string | `"cut5"` | 文本切分方法，见下表 |
| `batch_size` | integer | `1` | 文本段的批大小，使用正整数；模式 2/3 使用 `1` |
| `batch_threshold` | number | `0.75` | 按文本长度分桶时的批分组阈值 |
| `split_bucket` | boolean | `true` | 按长度分桶并在输出时恢复顺序；流式/分段返回、变速及部分声码器并行路径会自动关闭 |
| `speed_factor` | number | `1.0` | 语速系数，`1` 为原速，较大值加快 |
| `fragment_interval` | number | `0.3` | 文本片段后的静音间隔参数，单位秒；不是每个 HTTP 数据块之间的延时 |
| `seed` | integer | `-1` | `-1` 随机选择种子；指定值可固定采样种子 |
| `media_type` | string | `"wav"` | `wav`、`raw`、`ogg`、`aac` |
| `streaming_mode` | boolean or integer | `false` | `false`/`0`、`true`/`1`、`2`、`3`，见流式模式表 |
| `parallel_infer` | boolean | `true` | 普通/分段路径的批量并行推理；token 流式路径会自动关闭 |
| `use_cuda_graph` | boolean | `true` | 请求使用可用的 CUDA Graph AR 加速 |
| `repetition_penalty` | number | `1.35` | GPT token 重复惩罚 |
| `sample_steps` | integer or null | `null` | v3/v4/v5 的声学采样步数；省略时 v5turbo 为 `4`，其他版本为 `32` |
| `cfg_rate` | number or null | `null` | V5 声学 CFG 系数；省略时 v5dev 为 `1.30`，其他版本为 `0.0`。v3/v4 合成路径固定使用 `0`，不采用请求中的覆盖值 |
| `super_sampling` | boolean | `false` | v3 的音频超采样；需要配套超采样权重。其他版本不启用该步骤 |
| `overlap_length` | integer | `2` | v1/v2/Pro token 流式音频解码的重叠 token 数；V5 语义流式使用秒数窗口参数 |
| `min_chunk_length` | integer | `16` | token 流式的最小/固定块长；GPT 语义流式更新间隔取 `min(16, min_chunk_length)` 个 token |
| `gpt_streaming` | boolean | `false` | 启用独立 AR 加速引擎的 GPT 语义流式路径 |
| `streaming_chunk_seconds` | number | `2.0` | V5 语义流式固定声学窗口长度，或每个文本段的首个 hybrid 窗口长度 |
| `streaming_window_mode` | string | `"fixed"` | GPT 语义流式窗口策略：`fixed` 或 `hybrid` |
| `streaming_lookahead_seconds` | number | `0.12` | V5 声学窗口使用的未来上下文；hybrid 在后续窗口中不再使用此额外前瞻 |
| `streaming_overlap_seconds` | number | `0.04` | V5 声学窗口衔接的音频交叠长度 |
| `gpt_streaming_lookahead_seconds` | number | `0.5` | V5 流式时保留尚未提交的未来语义上下文，单位秒 |
| `use_flash_attention` | boolean | `true` | 请求使用可用的 FlashAttention AR 加速 |

`use_cuda_graph` 与 `use_flash_attention` 是加速请求开关，不会改变服务启动时选定的设备和模型精度。普通推理在加速不可用时使用普通 AR；GPT 语义流式需要可用的加速后端。

### 文本切分

| 值 | 切分策略 |
| --- | --- |
| `cut0` | 不主动按标点切段 |
| `cut1` | 按标点得到的句子每四句组成一段 |
| `cut2` | 累积到约 50 字后切段，较短尾段并入前段 |
| `cut3` | 按中文句号 `。` 切段 |
| `cut4` | 按英文句号 `.` 切段，避开数字内部的小数点 |
| `cut5` | 按中英文标点切段，保留数字内部的小数点 |

切分之后还会过滤无效文本、合并过短片段，并对过长文本做前端处理，因此上述策略不是最终 HTTP 数据块的边界。

### 流式模式

| `streaming_mode` | 实际行为 |
| --- | --- |
| `false` 或 `0` | 完成整次合成后返回一个音频响应 |
| `true` 或 `1` | 分段返回：一个文本批次完成 GPT 和声学合成后发送音频；`batch_size=1` 时按文本段返回 |
| `2` | token 流式：在最小块长之后参考静音 token 相似度选择切点，逐块生成音频 |
| `3` | token 流式：按目标 token 块长切块，末块可较短 |

`true` 就是模式 **1**，不会启用模式 2、3 或 `gpt_streaming`。模式 2/3 的低延迟表现取决于模型、切分、参考音频、硬件和客户端接收速度，不能由模式编号推导固定首包延迟或音质等级。

未启用 `gpt_streaming` 时，v3、v4、v5dev、v5turbo 使用声码器的路径会将模式 2/3 自动回退为模式 1 的分段返回。响应仍是流式 HTTP 响应，但要等每段的完整语义和声学合成结束。

### GPT 语义流式

`gpt_streaming=true` 选择独立加速引擎提供语义 token 更新，再驱动音频窗口生成。请求须满足以下接口约束：

- 当前模型为 **v2ProPlus、v5dev 或 v5turbo**。
- `streaming_mode` 为整数 `2` 或 `3`；POST 使用 JSON 数字，不能用 `true` 或字符串 `"2"`。GET 使用 `streaming_mode=2` 或 `3`。
- `batch_size=1`，`media_type` 为 `raw` 或 `wav`，`streaming_window_mode` 为 `fixed` 或 `hybrid`。
- V5 的 `prompt_text` 去掉首尾空白后非空，且不使用 `aux_ref_audio_paths`。
- 使用支持的 CUDA AR 加速路径：可用的 FlashAttention，或能够捕获/运行的静态 CUDA Graph。CPU 和当前 ROCm 路径无法提供此功能；普通推理仍可使用这些设备。

GPT 语义流式还会检查以下数值，边界值包含在允许范围内：

| 参数 | 允许值 |
| --- | --- |
| `streaming_chunk_seconds` | 有限数，`0.2～5.0` |
| `streaming_lookahead_seconds` | 有限数，`0.0～0.5` |
| `streaming_overlap_seconds` | 有限数，`0.0～0.2` |
| `gpt_streaming_lookahead_seconds` | 有限数，`0.16～2.0` |
| `speed_factor` | 有限数，`0.1～4.0` |
| `fragment_interval` | 有限数，`0.0～10.0` |
| `min_chunk_length`、`overlap_length` | 正整数 |
| `sample_steps` | 省略/`null`，或正整数 |
| `cfg_rate` | 省略/`null`，或有限非负数 |

这些检查会应用到全部 GPT 语义流式请求，即使其中某些窗口参数只在 V5 路径使用。

v2ProPlus 使用 `min_chunk_length` 控制初始 token 块长；`fixed` 保持目标块长，模式 2 仍会等待合适切点；`hybrid` 根据已生成音频的播放余量调整后续目标块长，最多为初始值的四倍。V5 使用秒数窗口，模式 2 和模式 3 在这条声学窗口路径中行为相同；`fixed` 使用固定窗口，`hybrid` 可调整后续窗口。窗口策略不会读取远端播放器的实际缓冲区。

`cut0`～`cut5` 都可用于 GPT 语义流式。每个文本段独立建立 GPT 会话，多个文本段按顺序输出到同一 HTTP 响应。`cut0` 会把换行替换为空格，并要求前端处理后仍是单段；过长时应缩短文本或改用其他切分方法。

## 响应音频

正常音频响应的状态码是 HTTP 200，`Content-Type` 为 `audio/wav`、`audio/raw`、`audio/ogg` 或 `audio/aac`。响应体只有音频字节，没有 JSON 包装或 base64。

| `media_type` | 非流式 | 流式 |
| --- | --- | --- |
| `raw` | 单声道、有符号 16-bit PCM；当前 Windows/Linux 小端环境下为 `s16le` | 同一 PCM 字节流，没有每块头部 |
| `wav` | 使用实际采样率和完整长度封装的 WAV，单声道 PCM 16-bit | 开头发送一个 WAV 头，后续发送 RAW PCM；文件长度字段的边界见下文 |
| `ogg` | 通过 libsndfile 封装 OGG | 每个生成块独立封装 OGG |
| `aac` | FFmpeg 编码的 192 kbit/s AAC，ADTS 格式 | 每个生成块独立调用 FFmpeg 编码为 ADTS AAC |

RAW 不包含采样率元信息，服务也没有额外的采样率响应头；客户端需按当前模型配置解码。

| 模型/配置 | 正常合成输出采样率 |
| --- | --- |
| v1、v2、v2Pro、v2ProPlus | 读取 SoVITS checkpoint 的 `data.sampling_rate`，常见配套模型为 32000 Hz |
| v3 | 24000 Hz |
| v3 且成功启用配套超采样模型 | 配套超采样配置的 `hr_sampling_rate`，标准 24k→48k 模型为 48000 Hz |
| v4、v5dev、v5turbo | 48000 Hz |

HTTP 传输块与模型生成块没有一一对应关系。客户端应连续消费字节；处理 RAW PCM 时要保留不足一个 16-bit 样本的尾字节，不能假设一次网络读取一定以样本边界结束。

## 参考音频与模型热切换

所有接口共享**单个 TTS pipeline**。推理、参考音频设置和模型切换使用同一把锁；流式推理持锁至响应结束或断开后的清理完成。请求的 `parallel_infer` 不会让多个 HTTP 推理请求同时修改 pipeline。

| 接口 | 查询参数 | 成功响应 |
| --- | --- | --- |
| `GET /set_refer_audio` | `refer_audio_path`：主参考音频路径，注意这里叫 `refer_`，而 `/tts` 使用 `ref_audio_path` | `{"message":"success"}` |
| `GET /set_gpt_weights` | `weights_path`：GPT checkpoint 路径 | `{"message":"success"}` |
| `GET /set_sovits_weights` | `weights_path`：SoVITS checkpoint 路径 | `{"message":"success"}` |

成功状态均为 HTTP 200。权重接口的路径为空时返回 HTTP 400；参考音频读取失败、权重缺失或加载失败也由对应接口返回 HTTP 400 JSON。切换只改变当前进程，重启后重新读取启动配置。

切换一整套声音时，由调用方暂停新推理请求，并按顺序等待每一步成功：

1. `/set_sovits_weights`：先确定目标 SoVITS 版本及相关声码器。
2. `/set_gpt_weights`：加载匹配 GPT 权重，并按当前版本更新 GPT 推理相关状态。
3. `/set_refer_audio`：用新模型重新提取参考音频特征。
4. 恢复 `/tts` 请求，继续携带 `ref_audio_path`、匹配的 `prompt_text` 与 `prompt_lang`。

`/set_refer_audio` 只预加载参考音频特征，不设置参考文本。它不会改变 `/tts` 的必填字段校验；后续请求仍必须携带 `ref_audio_path`。模型切换的事务边界见下文。

## HTTP 错误

| 情况 | 实际响应 |
| --- | --- |
| 必需字段省略或空字符串、不支持的语言/格式/切分方法、模式编号不在 0～3 | HTTP 400，JSON 中有 `message` |
| GPT 语义流式约束不满足 | HTTP 400，`{"message":"具体原因"}` |
| FastAPI/Pydantic 无法解析类型、无效 JSON 请求体等 | HTTP 422，JSON 中有 `detail`；例如 `batch_size=abc` |
| 在音频响应开始前被 `/tts` 处理器捕获的异常 | HTTP 400，`{"message":"tts failed","Exception":"具体原因"}` |
| 参考音频或权重切换失败 | HTTP 400，含对应 `message`，加载异常通常还含 `Exception` |

GPT 语义流式会先取得首个非空 PCM 块再创建响应；这个阶段失败可以返回 400，包括完全没有生成 PCM 的情况。普通分段/token 流式不做相同的首块预取。流开始后的行为以及普通合成中的静音回退见 `Limitations / Known Issues`，不能只用状态码判断音频是否完整。

## 调用示例

示例假定服务已经加载匹配权重，`references/reference.wav` 是服务器上的 3～10 秒参考音频。将示例中的参考文本替换成该音频实际说出的内容。

### curl：完整 WAV

在仓库根目录使用共享请求文件 [examples/tts_request.json](../examples/tts_request.json)：

```bash
curl --fail-with-body --show-error \
  -H 'Content-Type: application/json' \
  --data-binary @examples/tts_request.json \
  --output output.wav \
  http://127.0.0.1:9880/tts
```

PowerShell 使用真实的 `curl.exe`，可以写成一行：

```powershell
curl.exe --fail-with-body --show-error -H "Content-Type: application/json" --data-binary "@examples/tts_request.json" --output output.wav http://127.0.0.1:9880/tts
```

`--fail-with-body` 让 HTTP 错误产生非零退出码，但错误 JSON 仍可能被写入 `--output` 指定的文件；出现错误时先阅读该文件，不要当作 WAV 播放。

### curl：GET 列表参数

GET 列表通过**重复同名参数**传递，不用逗号拼接，也不传 JSON 字符串：

```bash
curl --get --fail-with-body --show-error \
  --data-urlencode 'text=你好，这是接口测试。' \
  --data-urlencode 'text_lang=all_zh' \
  --data-urlencode 'ref_audio_path=references/reference.wav' \
  --data-urlencode 'prompt_lang=all_zh' \
  --data-urlencode 'prompt_text=请替换为参考音频实际说出的内容。' \
  --data-urlencode 'aux_ref_audio_paths=references/aux1.wav' \
  --data-urlencode 'aux_ref_audio_paths=references/aux2.wav' \
  --data-urlencode 'batch_size=1' \
  --data-urlencode 'parallel_infer=false' \
  --data-urlencode 'media_type=wav' \
  --data-urlencode 'streaming_mode=0' \
  --output output.wav \
  http://127.0.0.1:9880/tts
```

### Python：审查请求、下载 WAV 或保存流式 PCM 为标准 WAV

将下面代码保存为客户端脚本，从仓库根目录运行。`--dry-run` 仅打印请求，不连接服务，也不需要权重；默认下载非流式 WAV。`--stream` 请求普通模式 3，并将收到的 RAW PCM 写入带正确长度的本地 WAV。`--gpt-stream` 在此基础上开启 GPT 语义流式，服务模型须满足前述约束。

```python
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:9880")
parser.add_argument("--request", default="examples/tts_request.json")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--stream", action="store_true")
parser.add_argument("--gpt-stream", action="store_true")
parser.add_argument("--sample-rate", type=int, default=32000)
args = parser.parse_args()

payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
stream = args.stream or args.gpt_stream
payload.update(batch_size=1, parallel_infer=False)
payload.update(media_type="raw" if stream else "wav", streaming_mode=3 if stream else 0)
payload["gpt_streaming"] = args.gpt_stream
url = args.base_url.rstrip("/") + "/tts"
body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
if args.dry_run:
    print("POST", url)
    print(body.decode("utf-8"))
    raise SystemExit(0)

request = Request(url, data=body, headers={"Content-Type": "application/json"})
try:
    with urlopen(request, timeout=600) as response:
        if not stream:
            Path(args.output).write_bytes(response.read())
        else:
            with wave.open(args.output, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(args.sample_rate)
                pending = b""
                while True:
                    block = response.read(8192)
                    if not block:
                        break
                    pending += block
                    complete = len(pending) // 2 * 2
                    output.writeframesraw(pending[:complete])
                    pending = pending[complete:]
                if pending:
                    raise RuntimeError("PCM ended inside a 16-bit sample")
except HTTPError as error:
    print(error.code, error.read().decode("utf-8", errors="replace"))
    raise SystemExit(1)
```

`--sample-rate` 必须与服务当前输出一致，例如 v2ProPlus 常见配置用 `32000`，v5dev/v5turbo 用 `48000`。该脚本不会从 RAW 响应自动识别采样率。传输中断会抛出异常；已写入的文件应视为不完整结果。

### curl：按顺序热切换

以下 Bash 示例逐次检查 HTTP 结果。权重路径需替换成服务器上的匹配文件，执行期间由调用方暂停新推理请求。

```bash
set -e
base=http://127.0.0.1:9880
curl --get --fail-with-body --show-error \
  --data-urlencode 'weights_path=GPT_SoVITS/pretrained_models/model.pth' \
  "$base/set_sovits_weights"
curl --get --fail-with-body --show-error \
  --data-urlencode 'weights_path=GPT_SoVITS/pretrained_models/model.ckpt' \
  "$base/set_gpt_weights"
curl --get --fail-with-body --show-error \
  --data-urlencode 'refer_audio_path=references/reference.wav' \
  "$base/set_refer_audio"
```

## Limitations / Known Issues

- **普通模式 2/3 使用 `batch_size=1`。** 当前非 GPT 语义流式路径只对每批第一个文本段进行 token 流式合成；更大的批大小可能跳过同批其他文本段。接口仅在 `gpt_streaming=true` 时强制检查批大小为 1。
- **流式 WAV 的头部长度不是最终文件长度。** 服务用空帧创建 WAV 头后追加 PCM，不会回写 RIFF/data 长度。部分播放器可按流播放，严格 WAV 读取器可能认为没有数据。需要保存标准文件时，使用非流式 WAV，或像 Python 示例一样将 RAW PCM 重新封装。
- **OGG/AAC 流的每块分别编码。** 本接口没有提供容器重封装或块索引协议，不能保证直接拼接响应就得到所有播放器都接受的标准文件。需要完整文件时使用非流式编码；需要连续 PCM 时使用 RAW。
- **流开始后发生异常，HTTP 状态不能改成 400。** 已发出的 200 头部和音频不会撤回，也没有尾部 JSON 错误事件。客户端可能看到提前结束、连接中断或部分音频；应处理读取异常，不能把“曾收到 200”当作合成完整性的保证。
- **非 GPT 路径仍有内部静音回退。** 某些合成阶段异常会先生成 16000 Hz 的一秒静音，再在继续迭代时尝试重载模型并抛错。非流式处理器只取一次结果，可能返回 200 静音；普通流式中也可能夹入这段静音后断开。这不是正常成功承诺。无有效音素的部分输入也可能产生静音，应检查输入、服务日志和输出内容。
- **热切换不是多请求事务。** SoVITS 与 GPT 是两个独立接口，两次调用之间可以插入其他客户端请求。单次加载出错也不保证恢复之前的所有模型/配置状态；不要在切换失败后假定旧模型仍完整可用，应重新加载一套已知匹配的权重与参考音频，必要时按已知配置重启。接口没有全套模型切换的原子提交或回滚操作。
- **语义流式受上下文容量约束。** 每段 GPT 的文本、参考提示及生成 token 共用 2048 token 上下文，生成上限还受 1500 token 和模型最大时长约束；到达上限会报错。V5 的参考声学特征、首窗口和前瞻合计不得超过 1000 帧（100 帧/秒）；参考特征最多保留 500 帧。因此即使单个参数在允许范围内，组合仍可能超出容量。
- **普通推理不做完整数值范围校验。** 类型能被解析不代表任意数值组合都有效；GPT 语义流式表中的范围检查不应当视为所有模式的统一校验。固定 `seed` 也不保证不同硬件、精度、后端或窗口模式下得到逐字节相同的音频。
- **v3 超采样权重缺失时会跳过超采样。** 请求中的 `super_sampling=true` 不保证输出为 48000 Hz，应以实际模型配置或 WAV 采样率为准。

本页依据当前 [API 路由](../api_v2.py)、[TTS pipeline](../GPT_SoVITS/TTS_infer_pack/TTS.py)、[语义流式实现](../GPT_SoVITS/TTS_infer_pack/streaming.py) 和 [文本切分实现](../GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py) 描述。
