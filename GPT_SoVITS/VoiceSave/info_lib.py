import json

def load_info(info_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return info

def save_info(info, info_path):
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)