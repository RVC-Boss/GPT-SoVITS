import zipfile
from . import file_lib as fl
from . import time_lib as tl
from . import info_lib as il
import os
from typing import Union
import numpy as np
import torch

POOL:set = set() 
def get_unique_name(name,MySet:set=set()):
    _id = 1
    if name not in POOL and name not in MySet:
        POOL.add(name)
        return name
    while name in POOL or name in MySet:
        _id += 1
        name = f'{name}_{_id}'
    POOL.add(name)
    return name

TEMP_DIR = fl.merge_dir_txt2(fl.get_my_dir(), "Temp")
TEMP_ZIP_DIR = fl.merge_dir_txt2(TEMP_DIR, "ZipTemp")
def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    cloned = tensor.clone().detach()
    np_array = cloned.cpu().numpy()
    return np_array

def save_np(path: str, np_array: np.ndarray) -> None:
    np.save(path, np_array)

class ZIP_File:
    def __init__(self, path: str,name:str,MySet:set=set()):
        self.path = path
        if not os.path.exists(self.path):
            with zipfile.ZipFile(self.path, 'w') as zipf:
                pass
        self.name = get_unique_name(name,MySet=MySet)#MySet用于补充命名集合，防止文件夹混淆
        self.temp_write = fl.merge_dir_txt2(TEMP_ZIP_DIR, self.name)

        if not os.path.exists(self.temp_write):
            os.makedirs(self.temp_write)

    def release(self):
        '''relaese the zip file, extract it to temp dir'''
        if os.path.exists(self.temp_write):
            fl.delete_dir(self.temp_write)
            fl.create_dir(self.temp_write)
        with zipfile.ZipFile(self.path, 'r') as zipf:
            zipf.extractall(self.temp_write)
        #fl.delete_file(self.path)
    def create_dir(self, dir_:str):
        dir_path = fl.merge_dir_txt2(self.temp_write, dir_)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path,exist_ok=True)

    def create_file(self, file_name:str,location:str=''):
        if location == '':
            file_path = fl.merge_dir_txt2(self.temp_write,file_name)
        else:
            file_path = fl.merge_dir_txt2(self.temp_write, location, file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w') as f:
            pass

    def get_file_path(self, file_name:str,location:str=''):
        if location == '':
            file_path = fl.merge_dir_txt2(self.temp_write,file_name)
        else:
            file_path = fl.merge_dir_txt2(self.temp_write, location, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        return file_path
    
    def get_file_obj(self, file_name:str,location:str='',mode:str='r'):
        if location == '':
            file_path = fl.merge_dir_txt2(self.temp_write,file_name)
        else:
            file_path = fl.merge_dir_txt2(self.temp_write, location, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        return open(file_path, mode)
    
    def save_file(self, obj):
        obj.close()
    
    def save_zip(self):
        with zipfile.ZipFile(self.path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.temp_write):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_write)
                    zipf.write(file_path, arcname)
        #fl.delete_dir(self.temp_write)

    def close(self):
        self.save_zip()
        fl.delete_dir(self.temp_write)
        POOL.remove(self.name)

def save_tensor(path: str, tensors: Union[torch.Tensor, list],name:str,MySet:set=set(),file_names:Union[str,list,None]=None,**info_save) -> None:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if not file_names:
        return
    if isinstance(file_names, str):
        files = [file_names]
    else:
        files = file_names

    if len(tensors) != len(files):
        raise ValueError("The number of tensors and files must be the same.")
    np_arrays = []
    for tensor in tensors:
        np_array = _tensor_to_numpy(tensor)
        np_arrays.append(np_array)
    zf = ZIP_File(path, name, MySet=MySet)
    zf.create_file("voice.json")
    info = {'name': name}
    info.update(info_save)
    il.save_info(info, str(zf.get_file_path("voice.json")))
    for i in range(len(files)):
        file_name = files[i]
        np_array = np_arrays[i]
        zf.create_file(file_name)
        save_np(str(zf.get_file_path(file_name)), np_array)
    zf.close()
    del zf

def load_tensor(path: str,name:str,find_func,MySet:set=set()) -> list[torch.Tensor]:
    zf = ZIP_File(path, name, MySet=MySet)
    zf.release()
    voice_path = find_func(zf,il)
    tensors = []
    for i in range(len(voice_path)):
        v = voice_path[i]
        np_array = np.load(v,allow_pickle=True)
        tensor = torch.from_numpy(np_array)
        tensors.append(tensor)
    zf.close()
    del zf
    return tensors