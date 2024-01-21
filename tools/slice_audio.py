import os,sys,numpy as np
import traceback
from scipy.io import wavfile
# parent_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(parent_directory)
from my_utils import load_audio
from slicer2 import Slicer

def slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,i_part,all_part):
    os.makedirs(opt_root,exist_ok=True)
    if os.path.isfile(inp):
        input=[inp]
    elif os.path.isdir(inp):
        input=[os.path.join(inp, name) for name in sorted(list(os.listdir(inp)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=      int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=     int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=   int(min_interval),  # 最短切割间隔
        hop_size=       int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=   int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max=float(_max)
    alpha=float(alpha)
    for inp_path in input[int(i_part)::int(all_part)]:
        # print(inp_path)
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            # print(audio.shape)
            for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if(tmp_max>1):chunk/=tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%s_%s.wav" % (opt_root, name, start, end),
                    32000,
                    # chunk.astype(np.float32),
                    (chunk * 32767).astype(np.int16),
                )
        except:
            print(inp_path,"->fail->",traceback.format_exc())
    return "执行完毕，请检查输出文件"

print(slice(*sys.argv[1:]))

