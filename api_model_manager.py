#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("gpt-sovits-api")

class ModelManager:
    """
    GPT-SoVITS模型管理器
    用于管理GPT和SoVITS模型的映射关系
    """
    def __init__(self):
        self.gpt_weights_dir = "GPT_weights"
        self.sovits_weights_dir = "SoVITS_weights"
        
        # 扫描多个版本的模型目录
        self.gpt_dirs = [
            "GPT_weights", 
            "GPT_weights_v2", 
            "GPT_weights_v3", 
            "GPT_weights_v4"
        ]
        
        self.sovits_dirs = [
            "SoVITS_weights", 
            "SoVITS_weights_v2", 
            "SoVITS_weights_v3", 
            "SoVITS_weights_v4"
        ]
        
        # 模型映射缓存
        self.model_mapping = {}
        self.voice_info = {}
        
        # 加载模型映射
        self.load_model_mapping()
    
    def _extract_model_info(self, filename: str) -> Dict:
        """
        从模型文件名中提取信息
        支持多种命名格式:
        1. 模型名_e迭代次数_s批次.pth 
        2. 模型名-e迭代次数.ckpt
        
        Args:
            filename: 模型文件名
        
        Returns:
            Dict: 包含模型名称、迭代次数和批次的字典
        """
        basename = os.path.basename(filename)
        name_parts = basename.split('.')
        base_name = name_parts[0]
        
        # 尝试匹配迭代次数 (e参数)，支持连字符(-)和下划线(_)
        e_match = re.search(r"[-_]e(\d+)", base_name)
        
        # 尝试匹配批次 (s参数)，主要在SoVITS模型中使用
        s_match = re.search(r"[-_]s(\d+)", base_name)
        
        # 提取模型名称（去掉e和s参数部分）
        model_name = base_name
        
        # 如果找到了e参数
        if e_match:
            # 获取e参数之前的部分作为模型名称
            e_pos = base_name.find(e_match.group(0))
            if e_pos > 0:
                separator = base_name[e_pos]  # 获取分隔符 (- 或 _)
                model_name = base_name.split(f"{separator}e")[0]
        
        # 提取扩展名
        ext = os.path.splitext(basename)[1].lower()
        
        iteration = int(e_match.group(1)) if e_match else 0
        batch = int(s_match.group(1)) if s_match else 0
        
        logger.debug(f"解析模型: {basename} -> 名称={model_name}, 迭代={iteration}, 批次={batch}")
        
        return {
            "name": model_name,
            "iteration": iteration,
            "batch": batch,
            "filename": filename
        }
    
    def load_model_mapping(self):
        """
        扫描模型目录，创建模型映射关系
        将相同名称的GPT和SoVITS模型进行匹配
        """
        # 扫描GPT模型
        gpt_models = {}
        for dir_path in self.gpt_dirs:
            if not os.path.exists(dir_path):
                continue
            
            for file_path in glob.glob(f"{dir_path}/*.ckpt"):
                model_info = self._extract_model_info(file_path)
                model_name = model_info["name"]
                
                # 使用更高迭代次数和批次的模型
                if model_name not in gpt_models or \
                   (model_info["iteration"] > gpt_models[model_name]["iteration"] or \
                   (model_info["iteration"] == gpt_models[model_name]["iteration"] and \
                    model_info["batch"] > gpt_models[model_name]["batch"])):
                    gpt_models[model_name] = model_info
        
        # 扫描SoVITS模型
        sovits_models = {}
        for dir_path in self.sovits_dirs:
            if not os.path.exists(dir_path):
                continue
            
            for file_path in glob.glob(f"{dir_path}/*.pth"):
                model_info = self._extract_model_info(file_path)
                model_name = model_info["name"]
                
                # 使用更高迭代次数和批次的模型
                if model_name not in sovits_models or \
                   (model_info["iteration"] > sovits_models[model_name]["iteration"] or \
                   (model_info["iteration"] == sovits_models[model_name]["iteration"] and \
                    model_info["batch"] > sovits_models[model_name]["batch"])):
                    sovits_models[model_name] = model_info
        
        # 创建映射关系
        for name in set(list(gpt_models.keys()) + list(sovits_models.keys())):
            gpt_model = gpt_models.get(name)
            sovits_model = sovits_models.get(name)
            
            if gpt_model and sovits_model:
                self.model_mapping[name] = {
                    "gpt_path": gpt_model["filename"],
                    "sovits_path": sovits_model["filename"],
                    "iteration": min(gpt_model["iteration"], sovits_model["iteration"]),
                    "batch": min(gpt_model["batch"], sovits_model["batch"])
                }
                
                self.voice_info[name] = {
                    "id": name,
                    "name": name,
                    "iteration": min(gpt_model["iteration"], sovits_model["iteration"]),
                    "batch": min(gpt_model["batch"], sovits_model["batch"])
                }
        
        logger.info(f"已加载 {len(self.model_mapping)} 个模型映射")
    
    def get_model_paths(self, voice_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        获取指定voice对应的GPT和SoVITS模型路径
        
        Args:
            voice_name: 声音名称
        
        Returns:
            Tuple[str, str]: (GPT模型路径, SoVITS模型路径)
        """
        if voice_name in self.model_mapping:
            return (
                self.model_mapping[voice_name]["gpt_path"],
                self.model_mapping[voice_name]["sovits_path"]
            )
        return None, None
    
    def get_all_voices(self) -> List[Dict]:
        """
        获取所有可用的声音列表
        
        Returns:
            List[Dict]: 声音信息列表
        """
        return [self.voice_info[name] for name in self.voice_info]
    
    def get_voice_details(self, voice_name: str) -> Optional[Dict]:
        """
        获取指定声音的详细信息
        
        Args:
            voice_name: 声音名称
        
        Returns:
            Dict: 声音详细信息
        """
        if voice_name in self.voice_info:
            info = self.voice_info[voice_name].copy()
            info.update({
                "gpt_path": self.model_mapping[voice_name]["gpt_path"],
                "sovits_path": self.model_mapping[voice_name]["sovits_path"]
            })
            return info
        return None

# 单例模式
model_manager = ModelManager()

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    manager = ModelManager()
    voices = manager.get_all_voices()
    print(f"发现 {len(voices)} 个声音模型:")
    for voice in voices:
        print(f"- {voice['name']}, 迭代次数: {voice['iteration']}, 批次: {voice['batch']}")
        gpt_path, sovits_path = manager.get_model_paths(voice['name'])
        print(f"  GPT: {gpt_path}")
        print(f"  SoVITS: {sovits_path}") 