import ast
import json
from collections import OrderedDict
import os

# locale_path = "./i18n/locale" # The path to the i18n locale directory, you can change it to your own path
# scan_list = ["./",
#              "GPT_SoVITS/",
#              "tools/"
#              ]  # The path to the directory you want to scan, you can change it to your own path
# scan_subfolders = False  # Whether to scan subfolders

locale_path = "./tools/srt_slicer/i18n/locale"
scan_list = ["./tools/srt_slicer"]  # The path to the directory you want to scan, you can change it to your own path
scan_subfolders = True

special_words_to_keep = {
    "auto": "自动判断",
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "all_zh": "只有中文",
    "all_ja": "只有日文",
    "auto_cut": "智能切分",
    "cut0": "仅凭换行切分",
    "cut1": "凑四句一切",
    "cut2": "凑50字一切",
    "cut3": "按中文句号。切",
    "cut4": "按英文句号.切",
    "cut5": "按标点符号切",
    
}


def extract_i18n_strings(node):
    i18n_strings = []

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
    ):
        for arg in node.args:
            if isinstance(arg, ast.Str):
                i18n_strings.append(arg.s)

    for child_node in ast.iter_child_nodes(node):
        i18n_strings.extend(extract_i18n_strings(child_node))

    return i18n_strings

strings = []

# for each file, parse the code into an AST
# for each AST, extract the i18n strings
def scan_i18n_strings(filename):
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
        if "I18nAuto" in code:
            tree = ast.parse(code)
            i18n_strings = extract_i18n_strings(tree)
            print(filename, len(i18n_strings))
            strings.extend(i18n_strings)


# scan the directory for all .py files (recursively)
if scan_subfolders:
    for folder in scan_list:
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".py")]:
                scan_i18n_strings(os.path.join(dirpath, filename))
else:
    for folder in scan_list:
        for filename in os.listdir(folder):
            if filename.endswith(".py"):
                scan_i18n_strings(os.path.join(folder, filename))
        
code_keys = set(strings)
"""
n_i18n.py
gui_v1.py 26
app.py 16
infer-web.py 147
scan_i18n.py 0
i18n.py 0
lib/train/process_ckpt.py 1
"""
print()
print("Total unique:", len(code_keys))


standard_file = os.path.join(locale_path, "zh_CN.json")
try:
    with open(standard_file, "r", encoding="utf-8") as f:
        standard_data = json.load(f, object_pairs_hook=OrderedDict)
    standard_keys = set(standard_data.keys())
except FileNotFoundError:
    standard_keys = set()
# Define the standard file name
unused_keys = standard_keys - code_keys
print("Unused keys:", len(unused_keys))
for unused_key in unused_keys:
    print("\t", unused_key)

missing_keys = code_keys - standard_keys
print("Missing keys:", len(missing_keys))
for missing_key in missing_keys:
    print("\t", missing_key)
    


code_keys_dict = OrderedDict()
for s in strings:
    if s in special_words_to_keep:
        code_keys_dict[s] = special_words_to_keep[s]
    else:    
        code_keys_dict[s] = s

# write back
os.makedirs(locale_path, exist_ok=True)
with open(standard_file, "w", encoding="utf-8") as f:
    json.dump(code_keys_dict, f, ensure_ascii=False, indent=4, sort_keys=True)
    f.write("\n")
