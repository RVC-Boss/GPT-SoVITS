import ast
import code
import glob
import json
import os
from collections import OrderedDict

I18N_JSON_DIR    = os.path.join(os.path.dirname(os.path.relpath(__file__)), 'locale')
DEFAULT_LANGUAGE = "zh_CN"
TITLE_LEN        = 100
SHOW_KEYS        = False # 是否显示键信息

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

# scan the directory for all .py files (recursively)
# for each file, parse the code into an AST
# for each AST, extract the i18n strings

def scan_i18n_strings():
    strings = []
    print(" Scanning Files and Extracting i18n Strings ".center(TITLE_LEN, "="))
    for filename in glob.iglob("**/*.py", recursive=True):
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
            if "I18nAuto" in code:
                tree = ast.parse(code)
                i18n_strings = extract_i18n_strings(tree)
                print(f"{filename.ljust(30)}: {len(i18n_strings)}")
                strings.extend(i18n_strings)

    code_keys = set(strings)
    print(f"{'Total Unique'.ljust(30)}: {len(code_keys)}")
    return code_keys

def update_i18n_json(json_file, standard_keys):
    print(f" Process {json_file} ".center(TITLE_LEN, "="))
    # 读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)
    # 打印处理前的 JSON 条目数
    print(f"{'Total Keys (Before)'.ljust(20)}: {len(json_data)}")
    # 识别缺失的键并补全
    miss_keys = set(standard_keys) - set(json_data.keys())
    if len(miss_keys) > 0:
        print(f"{'Missing Keys (+)'.ljust(20)}: {len(miss_keys)}")
        for key in miss_keys:
            if DEFAULT_LANGUAGE in json_file:
                # 默认语言的键值相同.
                json_data[key] = key
            else:
                # 其他语言的值设置为 #! + 键名以标注未被翻译.
                json_data[key] = "#!" + key
            if SHOW_KEYS:
                print(f"{'Added Missing Key'.ljust(20)}: {key}")
    # 识别多余的键并删除
    diff_keys = set(json_data.keys()) - set(standard_keys)
    if len(diff_keys) > 0:
        print(f"{'Unused Keys  (-)'.ljust(20)}: {len(diff_keys)}")    
        for key in diff_keys:
            del json_data[key]
            if SHOW_KEYS:
                print(f"{'Removed Unused Key'.ljust(20)}: {key}")
    # 按键顺序排序
    json_data = OrderedDict(
        sorted(json_data.items(), 
        key=lambda x: list(standard_keys).index(x[0])))
    # 打印处理后的 JSON 条目数
    print(f"{'Total Keys (After)'.ljust(20)}: {len(json_data)}")
    # 识别有待翻译的键
    num_miss_translation = 0
    for key, value in json_data.items():
        if value.startswith("#!"):
            num_miss_translation += 1
            if SHOW_KEYS:
                print(f"{'Missing Translation'.ljust(20)}: {key}")
    if num_miss_translation > 0:
        print(f"{'Missing Translation'.ljust(20)}: {num_miss_translation}")
    # 将处理后的结果写入 JSON 文件
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
    print(f" Updated {json_file} ".center(TITLE_LEN, "=") + '\n')

if __name__ == "__main__":
    code_keys = scan_i18n_strings()
    for json_file in os.listdir(I18N_JSON_DIR):
        if json_file.endswith(r".json"):
            json_file = os.path.join(I18N_JSON_DIR, json_file)
            update_i18n_json(json_file, code_keys)
        else:
            pass