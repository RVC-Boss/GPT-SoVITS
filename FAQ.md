> 基本是从[白菜的用户指南](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/pgah3gvetrdy8ryt)搬运的

# Table of Contents

<!-- toc -->

- [任何时候都有可能的：](#部署-gpt-sovits)
  - [OutOfMEemoryError: CUDA out of memory](#setup-the-development-environment)
  - [RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED](#runtimeerrorcuda-error-cublas_status_not_initialized)
  - [RuntimeError: bad allocation](#runtimeerror-bad-allocation)
  - [页面文件太小](#页面文件太小)
  - [MemoryError](#memoryerror)
  - [RuntimeError: DataLoader worker (pid(s) XXX) exited unexpectedly](#runtimeerrordata-loader-worker-pid-s-exited-unexpectedly)
  - [RuntimeErrorr: Failed to initalize Mecab (PyOpenJTalk报错,日语才有)](#runtimeerrorr-failed-to-initalize-mecab-pyopenjtalk报错-日语才有)
  - [RuntimeError: CUDA error: no kernel image is available for execution on the device](#runtimeerror-cuda-error-no-kernel-image-is-available-for-execution-on-the-device)
  - [OSError：［winError 10014］系统检测到在一个调用中尝试使用指针参数时的无效指针地址。](#oserrorwinerror-10014系统检测到在一个调用中尝试使用指针参数时的无效指针地址)
  - [NotADirectoryError：［winError 267］目录名称无效](#notadirectoryerrorwinerror-267目录名称无效)
- [Nightly Checkout & Pull](#nightly-checkout--pull)
- [Codebase structure](#codebase-structure)
- [Unit testing](#unit-testing)
  - [Python Unit Testing](#python-unit-testing)


<!-- tocstop -->

## 任何时候都有可能的：

### OutOfMEemoryError: CUDA out of memory

爆显存了，若在预处理和推理都能爆显存，那这张卡基本没法训练，若在训练，则检查数据集中要没有超过 10 秒的音频，有的话手动切分至 10 秒以下，重新开始 ASR。并且在训练时**调低 batch_size**。如果显存较低，如 4 G，勿关闭共享显存

### RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED

见 [CUDA out of memory](#outofmeemoryerror-cuda-out-of-memory)

### RuntimeError: bad allocation

见 [CUDA out of memory](#outofmeemoryerror-cuda-out-of-memory)

### 页面文件太小

打开电脑设置—系统—关于—高级系统设置，性能—设置—高级，虚拟内存—更改。设置为自动管理所有驱动器的分页文件大小。如果还是报错那就选择一个空间较大的盘自定义大小初始大小建议高于 30720，最大值适当即可

### MemoryError

见 [页面文件太小](#页面文件太小)

### RuntimeError: DataLoader worker (pid(s) XXX) exited unexpectedly

见 [页面文件太小](#页面文件太小)

### RuntimeErrorr: Failed to initalize Mecab (PyOpenJTalk报错,日语才有)

GPT-Sovits, 即整合包文件夹绝对路径不要含有中文以及除下划线外字符,然后更新代码至 GitHub 版本,从 GitHub 下载代码压缩包解压覆盖

### RuntimeError: CUDA error: no kernel image is available for execution on the device

驱动太旧，或者显卡不支持 cuda118

### OSError：［winError 10014］系统检测到在一个调用中尝试使用指针参数时的无效指针地址。

以管理员身份开启一个新的 cmd，输入 `netsh winsock reset`

### NotADirectoryError：［winError 267］目录名称无效

要求填写正确的目录路径，不是文件路径，或者单纯不存在这个路径


### 和 nltk 相关

运行报错中给出的提示代码，重启 webui 即可

```py
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
```

或者在 shell 中：
```shell
    python -m nltk.downloader averaged_perceptron_tagger_eng
```

## 推理结果

### 英文读音问题

项目使用的是英式发音，请以英式发音作为参考判断是否存在问题。
