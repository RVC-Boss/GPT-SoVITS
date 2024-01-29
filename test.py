import re

def add_period(text):
    if not re.search(r'[^\w\s]', text[-1]):
        text += '。'
    return text

def cut5(inp):
   inp = add_period(inp)
   inp = inp.strip("\n")
   punds = r'[、，。？！;：]'
   items = re.split(f'({punds})', inp)
   items = ["".join(group) for group in zip(items[::2], items[1::2])]
   opt = "\n".join(items)
   return opt

print(cut5("测试"))