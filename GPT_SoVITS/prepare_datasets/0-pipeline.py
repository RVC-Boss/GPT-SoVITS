import os, torch, sys
from subprocess import Popen

now_dir = os.getcwd()
sys.path.append(now_dir)
from config import (
    text_path,
    wav_dir,
    n_card,
    exp_name,
    n_parts,
    exp_dir,
)

os.makedirs("%s/logs_s1" % exp_dir, exist_ok=True)
os.makedirs("%s/logs_s2" % exp_dir, exist_ok=True)
##############step1
ps = []
for i_part in range(n_parts):
    cmd = "python prepare/1-get-text.py %s %s %s %s %s %s" % (
        text_path,
        wav_dir,
        exp_name,
        i_part,
        n_parts,
        i_part % n_card,
    )
    print(cmd)
    p = Popen(cmd, shell=True)
    ps.append(p)
for p in ps:
    p.wait()

opt = []
for i_part in range(n_parts):
    txt_path = "%s/2-name2text-%s.txt" % (exp_dir, i_part)
    with open(txt_path, "r") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(txt_path)
with open("%s/2-name2text.txt" % exp_dir, "w") as f:
    f.write("\n".join(opt) + "\n")

############step2
ps = []
for i_part in range(n_parts):
    cmd = "python prepare/2-get-hubert-wav32k.py %s %s %s %s %s %s" % (
        text_path,
        wav_dir,
        exp_name,
        i_part,
        n_parts,
        i_part % n_card,
    )
    print(cmd)
    p = Popen(cmd, shell=True)
    ps.append(p)
for p in ps:
    p.wait()
#############step3
ps = []
for i_part in range(n_parts):
    cmd = "python prepare/3-get-semantic.py %s %s %s %s %s" % (
        text_path,
        exp_name,
        i_part,
        n_parts,
        i_part % n_card,
    )
    print(cmd)
    p = Popen(cmd, shell=True)
    ps.append(p)
for p in ps:
    p.wait()
opt = ["item_name	semantic_audio"]
for i_part in range(n_parts):
    semantic_path = "%s/6-name2semantic-%s.tsv" % (exp_dir, i_part)
    with open(semantic_path, "r") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(semantic_path)
with open("%s/6-name2semantic.tsv" % exp_dir, "w") as f:
    f.write("\n".join(opt) + "\n")
