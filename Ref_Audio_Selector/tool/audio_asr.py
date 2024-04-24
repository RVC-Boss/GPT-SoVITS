import os
from config import python_exec,is_half
from tools import my_utils
from tools.asr.config import asr_dict
from subprocess import Popen
def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang):
    global p_asr
    if(p_asr==None):
        asr_inp_dir=my_utils.clean_path(asr_inp_dir)
        asr_py_path = asr_dict[asr_model]["path"]
        if asr_py_path == 'funasr_asr.py':
            asr_py_path = 'funasr_asr_multi_level_dir.py'
        if asr_py_path == 'fasterwhisper.py':
            asr_py_path = 'fasterwhisper_asr_multi_level_dir.py'
        cmd = f'"{python_exec}" tools/asr/{asr_py_path}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += " -p %s"%("float16"if is_half==True else "float32")

        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr=None

        output_dir_abs = os.path.abspath(asr_opt_dir)
        output_file_name = os.path.basename(asr_inp_dir)
        # 构造输出文件路径
        output_file_path = os.path.join(output_dir_abs, f'{output_file_name}.list')
        return output_file_path
        
    else:
        return None