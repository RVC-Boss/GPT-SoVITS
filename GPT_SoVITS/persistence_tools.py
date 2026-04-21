# -*- coding: utf-8 -*-
"""
GPT-SoVITS 持久化工具类
包含：模型配置、参考音频、推理参数 的持久化读写与管理
抽离自主文件，减少主文件臃肿，方便后续维护
"""
import json
import yaml
import hashlib
import os
import shutil
import random
from pathlib import Path

# ===================== 全局配置（统一管理所有持久化文件路径） =====================
# 模型持久化配置文件
LAST_SELECTED_MODELS_JSON = Path("./last_selected_models.json")
# 参考预设最后选中配置文件
LAST_SELECTED_PRESET_JSON = Path("./last_selected_preset.json")
# 参考音频持久化目录
REF_AUDIO_DIR = Path("GPT_SoVITS/ref_audios")
# 参考预设配置文件
REF_PRESETS_YAML = Path("GPT_SoVITS/configs/ref_audios_presets.yaml")
# 推理参数配置文件
INFER_SETTINGS_JSON = Path("GPT_SoVITS/configs/infer_settings.json")

# 参考音频配置常量
MAX_FILENAME_LENGTH = 40
INVALID_FILE_CHARS = set(r'\/:*?"<>|')

# 默认推理参数
DEFAULT_INFER_SETTINGS = {
    "batch_size": 20,
    "sample_steps": 32,
    "fragment_interval": 0.2,
    "speed_factor": 1.0,
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 1.0,
    "repetition_penalty": 1.35,
    "how_to_cut": "凑四句一切",
    "super_sampling": False,
    "parallel_infer": True,
    "split_bucket": True,
    "seed": -1,
    "keep_random": True
}

# ===================== 通用工具函数（抽离重复逻辑） =====================
def sanitize_filename(name):
    """清理文件名中的非法字符，替换为下划线"""
    if not name:
        return "unnamed_preset"
    return ''.join(c if c not in INVALID_FILE_CHARS else '_' for c in name)

def get_audio_md5(file_path, chunk_size=4096):
    """计算音频文件的MD5值（取前8位），用于区分不同音频内容"""
    if not os.path.exists(file_path):
        return "invalid_file"
    try:
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()[:8]
    except Exception as e:
        print(f"计算音频MD5失败：{e}")
        return f"err_{random.randint(10000000, 99999999)}"

def ensure_dir_exists(dir_path):
    """确保目录存在，不存在则创建"""
    if dir_path and not dir_path.exists():
        dir_path.mkdir(exist_ok=True, parents=True)

# ===================== 1. 模型配置持久化（last_selected_models.json） =====================
def init_last_selected_models(gpt_default, sovits_default, current_version):
    """初始化模型配置文件，写入默认模型路径"""
    ensure_dir_exists(LAST_SELECTED_MODELS_JSON.parent)
    init_data = {
        "gpt_model_path": gpt_default,
        "sovits_model_path": sovits_default,
        "version": current_version
    }
    with open(LAST_SELECTED_MODELS_JSON, "w", encoding="utf-8") as f:
        json.dump(init_data, f, ensure_ascii=False, indent=4)
    print(f"首次生成模型配置文件：{LAST_SELECTED_MODELS_JSON}")
    return init_data

def read_last_selected_models():
    """读取模型配置文件中的路径"""
    if not LAST_SELECTED_MODELS_JSON.exists():
        return None
    try:
        with open(LAST_SELECTED_MODELS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 校验必要字段
        required_fields = ["gpt_model_path", "sovits_model_path", "version"]
        for field in required_fields:
            if field not in data:
                return None
        return data
    except Exception as e:
        print(f"读取模型配置失败：{e}")
        return None

def write_last_selected_models(gpt_path_new, sovits_path_new, current_version):
    """写入新的模型路径到配置文件"""
    ensure_dir_exists(LAST_SELECTED_MODELS_JSON.parent)
    try:
        data = read_last_selected_models() or {}
        data["gpt_model_path"] = gpt_path_new
        data["sovits_model_path"] = sovits_path_new
        data["version"] = current_version
        with open(LAST_SELECTED_MODELS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"写入模型配置失败：{e}")

# ===================== 2. 参考音频预设持久化（last_selected_preset.json + ref_audios_presets.yaml） =====================
# 2.1 最后选中预设的读写清
def read_last_selected_preset():
    """读取最后一次选中的预设名称"""
    if not LAST_SELECTED_PRESET_JSON.exists():
        return None
    try:
        with open(LAST_SELECTED_PRESET_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_selected_preset")
    except Exception as e:
        print(f"读取最后选中预设失败：{e}")
        return None

def write_last_selected_preset(preset_name):
    """写入最后一次选中的预设名称"""
    ensure_dir_exists(LAST_SELECTED_PRESET_JSON.parent)
    try:
        data = {"last_selected_preset": preset_name.strip()}
        with open(LAST_SELECTED_PRESET_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"已记录最后选中的预设：{preset_name.strip()}")
    except Exception as e:
        print(f"写入最后选中预设失败：{e}")

def clear_last_selected_preset():
    """清空最后选中的预设记录"""
    if not LAST_SELECTED_PRESET_JSON.exists():
        return
    try:
        with open(LAST_SELECTED_PRESET_JSON, "w", encoding="utf-8") as f:
            json.dump({"last_selected_preset": ""}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"清空最后选中预设失败：{e}")

# 2.2 参考预设配置的加载/保存/删除
def load_ref_presets():
    """加载多组参考预设配置"""
    ensure_dir_exists(REF_PRESETS_YAML.parent)
    
    # 新增：配置文件不存在时，自动创建空文件
    if not REF_PRESETS_YAML.exists():
        with open(REF_PRESETS_YAML, "w", encoding="utf-8") as f:
            yaml.dump([], f, indent=4, allow_unicode=True)
        print(f"暂未检测到参考预设配置文件，已自动创建空文件：{REF_PRESETS_YAML}")
        return []
    
    try:
        with open(REF_PRESETS_YAML, "r", encoding="utf-8") as f:
            presets = yaml.load(f, Loader=yaml.FullLoader) or []
        
        # 兼容旧格式转换
        if isinstance(presets, dict):
            presets = [{"name": "旧配置转换", "ref_audio_path": presets.get("ref_audio_path"),
                        "prompt_text": presets.get("prompt_text", ""), "prompt_language": presets.get("prompt_language", "中文")}]
        
        # 补充缺失字段 + 校验音频路径
        default_template = {"name": "", "ref_audio_path": None, "prompt_text": "", "prompt_language": "中文"}
        for preset in presets:
            for key, value in default_template.items():
                preset.setdefault(key, value)
            # 校验音频路径有效性
            audio_path = preset["ref_audio_path"]
            if audio_path and not os.path.exists(str(audio_path)):
                preset["ref_audio_path"] = None
        
        # 清理冗余音频
        clean_unreferenced_audios(presets)
        print(f"参考预设加载成功，共 {len(presets)} 组")
        return presets
    except Exception as e:
        print(f"参考预设加载失败：{e}")
        return []

def get_preset_by_name(preset_name, presets=None):
    """根据配置名称查询对应的配置详情"""
    # 核心修复：先判断 preset_name 是否为 None，避免 AttributeError
    if preset_name is None:
        return {"name": "", "ref_audio_path": None, "prompt_text": "", "prompt_language": "中文"}
    
    if not presets:
        presets = load_ref_presets()
    
    # 现在再调用 strip()，确保 preset_name 不是 None
    preset_name_str = preset_name.strip()
    for preset in presets:
        if preset["name"].strip() == preset_name_str:
            return preset
    
    # 无匹配预设时，返回空的合法预设字典
    return {"name": "", "ref_audio_path": None, "prompt_text": "", "prompt_language": "中文"}

def save_ref_preset_core(preset_name, ref_audio_path, prompt_text, prompt_language, confirm_override=False):
    """保存/覆盖参考预设核心逻辑（返回：提示信息、是否成功、预设列表）"""
    ensure_dir_exists(REF_AUDIO_DIR)
    presets = load_ref_presets()
    preset_name = preset_name.strip()
    
    # 前置校验
    if not ref_audio_path or not os.path.exists(str(ref_audio_path)):
        return "保存失败！请先上传有效的主参考音频文件。", False, [p["name"] for p in presets]
    if not preset_name:
        return "保存失败！配置名称不能为空。", False, [p["name"] for p in presets]
    
    # 音频持久化处理
    persistent_audio_path = get_persistent_audio_path(ref_audio_path, preset_name)
    if not persistent_audio_path:
        return "保存失败！音频文件持久化存储失败。", False, [p["name"] for p in presets]
    
    # 同名检测
    preset_index = -1
    for idx, p in enumerate(presets):
        if p["name"].strip() == preset_name:
            preset_index = idx
            break
    
    if preset_index >= 0 and not confirm_override:
        return f"配置「{preset_name}」已存在，如需替换请确认覆盖！", False, [p["name"] for p in presets]
    
    # 构造新配置
    new_preset = {
        "name": preset_name,
        "ref_audio_path": persistent_audio_path,
        "prompt_text": prompt_text,
        "prompt_language": prompt_language
    }
    
    # 更新配置列表
    is_new_preset = preset_index < 0
    if preset_index >= 0:
        presets[preset_index] = new_preset
        tip = "同名配置已覆盖！"
    else:
        presets.append(new_preset)
        tip = "新配置已新增！"
    
    # 写入配置文件
    try:
        with open(REF_PRESETS_YAML, "w", encoding="utf-8") as f:
            yaml.dump(presets, f, indent=4, allow_unicode=True)
        
        # 新增预设自动记录为最后选中
        if is_new_preset:
            write_last_selected_preset(preset_name)
        
        preset_names = [p["name"] for p in presets]
        return f"配置保存成功！{tip}", True, preset_names
    except Exception as e:
        return f"保存失败：{str(e)}", False, [p["name"] for p in presets]

def delete_ref_preset_core(preset_name):
    """删除参考预设核心逻辑（返回：提示信息、预设列表、默认选中预设）"""
    presets = load_ref_presets()
    preset_name = preset_name.strip()
    
    if not presets:
        return "暂无配置可删除！", [], None
    
    # 获取待删除音频路径
    target_audio_path = None
    for p in presets:
        if p["name"].strip() == preset_name:
            target_audio_path = p.get("ref_audio_path")
            break
    
    # 过滤删除
    presets = [p for p in presets if p["name"].strip() != preset_name]
    
    # 写入配置文件
    try:
        with open(REF_PRESETS_YAML, "w", encoding="utf-8") as f:
            yaml.dump(presets, f, indent=4, allow_unicode=True)
        
        # 删除对应音频
        if target_audio_path and os.path.exists(target_audio_path):
            os.unlink(target_audio_path)
            print(f"同步删除配置对应音频：{target_audio_path}")
        
        # 清空最后选中记录（若删除的是最后选中的预设）
        last_selected = read_last_selected_preset()
        if last_selected and last_selected == preset_name:
            clear_last_selected_preset()
        
        preset_names = [p["name"] for p in presets]
        new_selected = preset_names[0] if preset_names else None
        tip = "配置删除成功！已同步清理对应音频文件" if preset_names else "配置删除成功！已同步清理对应音频文件，当前无剩余配置"
        return tip, preset_names, new_selected
    except Exception as e:
        return f"删除失败：{str(e)}", [p["name"] for p in presets], preset_name

# 2.3 参考音频文件管理
def get_persistent_audio_path(src_audio_path, preset_name):
    """获取音频持久化路径，清理同配置名旧音频"""
    if not src_audio_path or not os.path.exists(src_audio_path):
        return None
    
    # 清理文件名
    safe_preset_name = sanitize_filename(preset_name)
    safe_preset_name = safe_preset_name[:MAX_FILENAME_LENGTH]
    
    # 提取后缀
    src_suffix = Path(src_audio_path).suffix.lower()
    if not src_suffix or src_suffix not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        src_suffix = ".wav"
    
    # 计算MD5
    audio_md5 = get_audio_md5(src_audio_path)
    dst_filename = f"{safe_preset_name}_{audio_md5}{src_suffix}"
    dst_path = REF_AUDIO_DIR / dst_filename
    
    # 清理同配置名旧音频
    for old_audio in REF_AUDIO_DIR.glob(f"{safe_preset_name}_*"):
        if old_audio.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            try:
                old_audio.unlink()
            except Exception as e:
                print(f"清理旧音频失败：{e}")
    
    # 复制新音频
    try:
        shutil.copy2(src_audio_path, dst_path)
        return str(dst_path)
    except Exception as e:
        print(f"音频持久化复制失败：{e}")
        return None

def clean_unreferenced_audios(presets):
    """清理未被任何预设引用的冗余音频"""
    if not REF_AUDIO_DIR.exists():
        return
    
    # 收集已引用音频
    referenced = set()
    for preset in presets:
        audio_path = preset.get("ref_audio_path")
        if audio_path and os.path.exists(audio_path):
            referenced.add(Path(audio_path).absolute())
    
    # 删除未引用音频
    deleted_count = 0
    for audio_file in REF_AUDIO_DIR.glob("*"):
        if audio_file.is_file() and audio_file.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            if audio_file.absolute() not in referenced:
                try:
                    audio_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"清理冗余音频失败：{e}")
    
    if deleted_count > 0:
        print(f"清理冗余未引用音频 {deleted_count} 个")

# ===================== 3. 推理参数持久化（infer_settings.json） =====================
def load_infer_settings():
    """加载推理参数配置"""
    ensure_dir_exists(INFER_SETTINGS_JSON.parent)
    if not INFER_SETTINGS_JSON.exists():
        return DEFAULT_INFER_SETTINGS
    try:
        with open(INFER_SETTINGS_JSON, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULT_INFER_SETTINGS, **saved}
    except Exception as e:
        print(f"加载推理参数失败，使用默认值：{e}")
        return DEFAULT_INFER_SETTINGS

def save_infer_settings_core(settings):
    """保存推理参数核心逻辑（返回：提示信息）"""
    ensure_dir_exists(INFER_SETTINGS_JSON.parent)
    try:
        with open(INFER_SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        
        # 精简日志输出
        print(f"✅ 推理配置保存成功：{INFER_SETTINGS_JSON.absolute()}")
        return "推理设置保存成功！已覆盖原有配置文件。"
    except Exception as e:
        print(f"❌ 推理配置保存失败：{e}")
        return f"推理设置保存失败：{str(e)}"

def restore_default_infer_settings_core():
    """恢复推理参数默认值核心逻辑（返回：默认参数列表）"""
    ensure_dir_exists(INFER_SETTINGS_JSON.parent)
    try:
        with open(INFER_SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_INFER_SETTINGS, f, indent=4, ensure_ascii=False)
        print(f"✅ 推理配置已恢复默认值：{INFER_SETTINGS_JSON.absolute()}")
    except Exception as e:
        print(f"❌ 推理配置恢复默认失败：{e}")
    
    # 返回默认参数（按顺序对应UI组件）
    return [
        DEFAULT_INFER_SETTINGS["batch_size"],
        DEFAULT_INFER_SETTINGS["sample_steps"],
        DEFAULT_INFER_SETTINGS["fragment_interval"],
        DEFAULT_INFER_SETTINGS["speed_factor"],
        DEFAULT_INFER_SETTINGS["top_k"],
        DEFAULT_INFER_SETTINGS["top_p"],
        DEFAULT_INFER_SETTINGS["temperature"],
        DEFAULT_INFER_SETTINGS["repetition_penalty"],
        DEFAULT_INFER_SETTINGS["how_to_cut"],
        DEFAULT_INFER_SETTINGS["super_sampling"],
        DEFAULT_INFER_SETTINGS["parallel_infer"],
        DEFAULT_INFER_SETTINGS["split_bucket"],
        DEFAULT_INFER_SETTINGS["seed"],
        DEFAULT_INFER_SETTINGS["keep_random"]
    ]