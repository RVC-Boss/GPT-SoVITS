# Accel 通用运行说明

本目录及配套修改代码统一用于 `gsvv5/py312` 和 `gsvv5/py39` 下的四套包。各自 runtime 保留：py312 为 Torch 2.7.1+cu118 / 2.7.1+cu128，py39 为 Torch 2.0.0+cu118 / 2.7.0+cu128。运行依赖不随源码统一而替换。

## 目录与依赖

- `GPT_SoVITS/Accel` 包含 PyTorch/MLX 后端、适配器、日志模块和 `vendor/cuda/bin/cudart64_12.dll`。
- py312 的 FlashAttention 2.8.3 通过标准 pip 安装在各自 `runtime/Lib/site-packages`；不从外置测试目录导入。两套 py39 未安装 FlashAttention，对应控件自动隐藏。
- `tools/acceleration.py` 负责能力选择，`tools/portable_runtime.py` 负责便携环境、DLL 搜索路径和 Transformers 兼容初始化。
- 正常推理不依赖 `others`；测试、备份、网络 requirements 和补丁 ZIP 均放在 `others`。
- 超分配置 `tools/AP_BWE_main/24kto48k/config.json` 保留。原版 BigVGAN/UVR 运行文件、模型权重及 runtime 未删除。

## 自动推理路线

Graph 和外部 FlashAttention 分别识别，按实际推理设备、精度与用户开关选择：

| 启用 FlashAttention | 有效 CUDA Graph | 非流式 AR |
| --- | --- | --- |
| 是 | 是 | FlashAttention + Graph |
| 是 | 否 | FlashAttention eager |
| 否 | 是 | 纯 Torch SDPA Graph |
| 否 | 否 | 原始普通 AR |

主 WebUI 未勾选“启用并行推理版本”时，普通子 WebUI 固定逐句 `bs=1`，Accel 容量为 1，只捕获单批 Graph，不提供 batch size 或并行开关。勾选后启动快速子 WebUI，保留原 `batch_size` 滑块（1–200，默认 20），固定并行，Accel 容量随滑块调整。

两个子 WebUI 都只提供 `CUDA Graph` 和 `flash_attn` 两个加速开关。启动时分别判断当前设备能力；支持时默认勾选、显示且可修改，不支持时默认关闭、隐藏且不可修改。用户可以独立关闭任一能力，`CUDAGraph=0` 仍禁用 Graph。真正 token 流式保留原实现，API 的 `parallel_infer` 参数保留。

Graph 实际捕获失败时，Flash 继续 eager，纯 Torch Graph 回普通 AR；记录失败状态，避免反复捕获。权重加载和实际推理错误正常抛出，不隐藏为能力缺失。尊重传入 dtype，不强制 FP32 转 FP16。换权重、设备或精度时关闭旧引擎。

## 中文 G2PW DLL 修复

普通版 ONNX Runtime 1.18.0 需要 CUDA 11 / cuDNN 8，而 Torch 自带 cuDNN 9。配套 cuDNN 8 已安装在 `runtime/Lib/site-packages/nvidia/*/bin`。初始化现会注册存在的 DLL 目录，并加入当前进程 PATH，以支持 cuDNN 动态加载其组件 DLL。

此处理也适用于 CUDA 12 环境的 Torch DLL。未修改系统 PATH、CUDA_PATH、库版本或 site-packages，未把 G2PW 改成 CPU。UVR5 的 `onnx_dereverb_By_FoxJoy` 创建 ONNX Runtime 会话前也调用同一 DLL 初始化，因此该模型可使用 CUDA/cuDNN。已经运行的推理窗口需要重新启动才能加载修改。

## 兼容与安装

Accel 源码采用 Python 3.9 兼容语法；旧 Torch 不支持的六项 `_inductor.config` 内部设置已移除，权重加载仅在支持时传 `mmap`。语法兼容不代表旧 py39 环境的全部功能已实测。

以下五项不再设置：`preferred_blas_library`、`ExecutionTraceObserver`、`torch.compiler.disable` 包装、`allow_fp16_accumulation=True`、`allow_fp16_bf16_reduction_math_sdp(True)`。原有可选 Torch profiler/TensorBoard 保留，uvloop 是可选依赖。

四份网络安装清单位于 `others/requirements-py312-win-flash_attention`，按 cu118/cu128 和国内 ModelScope/国外 Hugging Face 选择。适用于 Windows x64、CPython 3.12、Torch 2.7.1；先安装匹配 Torch，再使用清单。现有 FlashAttention cp312 Windows wheel 不能用于 Python 3.9 或 Linux。网络清单尚未在空白环境完成全部安装测试。

## 通用补丁

`gsvv5/py312/others/add_accel/GPT-SoVITS_add_accel.zip` 是四套包共用的唯一当前补丁。ZIP 保留项目相对路径，直接解压至目标项目根目录。四套 runtime 不互相覆盖。

包含 Accel 完整发布内容（Python 源码、此说明及 vendor CUDA DLL），加上目录外新增 3 个、修改 9 个文件；其中 `tools/portable_runtime.py` 已包含最新 G2PW DLL 修复，`tools/uvr5/mdxnet.py` 已包含 FoxJoy ONNX CUDA DLL 修复。不包含 runtime、site-packages、FlashAttention wheel、模型权重、测试结果或 `__pycache__`。

## 已完成的运行测试

本机 RTX 3090，两套 Python 3.12 环境均完成适配器模式切换、无参考文本、v2 TTS 四种 Graph/并行组合；各完成八种能力缺失/捕获失败/流式切换场景，均产生音频。

最新中文测试使用 `s1v3.ckpt` 与 `s2Gv2ProPlus.pth`：两套 G2PW 均使用 CUDAExecutionProvider，成功处理中文文本，并完成并行 Graph 和串行 eager 完整 TTS。

py312 普通子 WebUI 已完成四种 Flash/Graph 开关组合及重新开启双开的真实合成，均为 `bs=1`；关闭两者时使用原始 AR。快速子 WebUI 保留默认 `bs=20` 滑块，已检查 1、20、40、200 的参数绑定；本轮未重复执行用户已确认成功的四环境并行 TTS。py312 控件默认双开，py39 缺 Flash 时正确隐藏该控件，CPU 下两个控件均默认关闭且隐藏。

四套 runtime 均通过真实 `tools/uvr5/webui.py` 的 FoxJoy 去混响入口，性能记录每套含 278 次 `CUDAExecutionProvider` 节点执行、无 CPU 模型节点。人声与伴奏输出均为 44.1 kHz 双声道；py39 NVIDIA 的 MDX 初始化调用已补齐。测试输出位于各自 `others/<包名>/foxjoy_gpu_20261005_current`。

未测试所有模型版本、其他 GPU、Linux/MLX；未试听或证明不同路线音质/输出等价。浏览器测试页停留在 Gradio 加载画面，控件与调用检查取自实际 Gradio 配置和入口函数，本轮不宣称浏览器端完整交互已通过。

运行记录与报告位于 `others/<对应包名>`、`others/auto_acceleration_changes_20261005.md` 和 `others/onnx_chinese_fix_20261005.md`。历史分版结果保留在外置目录，不放进通用补丁。
