import ast
import glob
import json
from collections import OrderedDict


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

strings = []
for filename in glob.iglob("**/*.py", recursive=True):
    with open(filename, "r") as f:
        code = f.read()
        if "I18nAuto" in code:
            tree = ast.parse(code)
            i18n_strings = extract_i18n_strings(tree)
            print(filename, len(i18n_strings))
            strings.extend(i18n_strings)
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


standard_file = "i18n/locale/zh_CN.json"
with open(standard_file, "r", encoding="utf-8") as f:
    standard_data = json.load(f, object_pairs_hook=OrderedDict)
standard_keys = set(standard_data.keys())

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
    code_keys_dict[s] = s

# write back
with open(standard_file, "w", encoding="utf-8") as f:
    json.dump(code_keys_dict, f, ensure_ascii=False, indent=4, sort_keys=True)
    f.write("\n")
