



import re
from typing import Callable
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

METHODS = dict()

def get_method(name:str)->Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method

def register_method(name):
    def decorator(func):
        METHODS[name] = func
        return func
    return decorator

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split_big_text(text, max_len=510):
    # 定义全角和半角标点符号
    punctuation = "".join(splits)

    # 切割文本
    segments = re.split('([' + punctuation + '])', text)
    
    # 初始化结果列表和当前片段
    result = []
    current_segment = ''
    
    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment
    
    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)
    
    return result



def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


# 不切
@register_method("cut0")
def cut0(inp):
    return inp


# 凑四句一切
@register_method("cut1")
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    # split_idx[-1] = None
    split_idx.append(None)
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)

# 凑50字一切
@register_method("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)

# 按中文句号。切
@register_method("cut3")
def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])

#按英文句号.切
@register_method("cut4")
def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

# 按标点符号切
# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
@register_method("cut5")
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    # punds = r'[,.;?!、，。？！;：…]'
    punds = r'[,.;?!、，。？！；：:…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt

def num_to_chinese(num):
    chinese_nums = {
        0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九',
    }
    units = ['', '十', '百', '千', '万', '亿']
    num_str = str(num)
    num_str_rev = num_str[::-1]
    result = ''
    for i, digit in enumerate(num_str_rev):
        if i == 0 and digit == '0':
            continue
        if i > 0 and digit == '0' and result[0] != '零':
            result = '零' + result
        digit_chinese = chinese_nums[int(digit)]
        unit = units[i % 4]
        if i % 4 == 0:
            unit = units[i % 4 + int(i / 4)]
        result = digit_chinese + unit + result
    return result

# 支持语言混合切，按照约10个字一组，拆分更多的文本支持batch并行推理
@register_method("mixed_cut")
def mixed_cut(inp):
    def re_exp_japenese_char():
        #日文中带有中文字符的情况，依赖short合并把中文合并到上一个日文分组中
        return '[\u3040-\u30FF\uFF66-\uFF9D]'
    def re_exp_chinese_char():
        return '[\u4e00-\u9fa5]'
    def re_exp_alpha():
        return '[a-zA-Z]'
    def re_exp_digit():
        return '[0-9]'
    bad_case_ignore = [ "...","~","——","……" ]
    for ss in bad_case_ignore:
        inp = inp.replace(ss, "。")
    result = []
    last_s = ""
    last_c_type = ""
    #按连续字符进行分组
    for char in inp:
        c_type = "unknow"
        if char == " ":
            last_s += char
            continue
        elif re.match(re_exp_japenese_char(), char):
            c_type="jps"
        elif re.match(re_exp_chinese_char(), char):
            c_type="hans"
        elif re.match(re_exp_alpha(), char):
            c_type="alpha"
        elif re.match(re_exp_digit(), char):
            c_type="digit"
        if (c_type != last_c_type and c_type != "unknow" and len(last_c_type) > 0):
            result.append(last_s)
            last_s = ""
        last_s += char
        if c_type != "unknow":
            last_c_type = c_type
        elif len(last_s) > 10:
            result.append(last_s)
            last_s = ""
    result.append(last_s)

    def s_type(s):
        if len(s) > 0:
            if re.compile(re_exp_japenese_char()).search(s) is not None:
                return "jps"
            elif re.compile(re_exp_chinese_char()).search(s) is not None:
                return "hans"
            elif re.compile(re_exp_alpha()).search(s) is not None:
                return "alpha"
            elif re.compile(re_exp_digit()).search(s) is not None:
                return "digit"
        return "unknow"    
    #数组合并至前项，并支持念出中文数字
    new_result = []
    n = 0
    while n < len(result):
        this_s = result[n]
        this_s_type = s_type(this_s)
        before_s = ""
        if n > 0:
            before_s = result[n-1]
        before_s_type = s_type(this_s)
        next_s = ""
        if n < (len(result)-1):
            next_s = result[n+1]
        next_s_type = s_type(this_s)
        if this_s_type == "digit":
            if before_s == "":
                new_result.append(before_s)
            if before_s_type == "hans" or next_s_type == "hans":
                ss = num_to_chinese(this_s)
            else:
                ss = this_s
            if before_s == "" or before_s_type == next_s_type:
                ss += next_s
                n+=1
            new_result[len(new_result)-1]+=ss
        else:
            new_result.append(this_s)
        n+=1
    opt = "\n".join(new_result)
    return opt

if __name__ == '__main__':
    method = get_method("mixed_cut")
    print(method("你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。") + "\n===\n")
    print(method("你好，我是小明") + "\n===\n")
    print(method("12345") + "\n===\n")
    print(method("123，不许动") + "\n===\n")
    print(method("你好，我是小明。我今年20岁了") + "\n===\n")
    print(method("你好，我是Maxwell, nice to meet you") + "\n===\n")
    print(method("你好，我是Maxwell。我今年20岁了") + "\n===\n")
    print(method("你好，我是小明。こんにちは、シャオミンです。") + "\n===\n")
    print(method("こんにちは、シャオミンです。 今年で20周年") + "\n===\n")
    print(method("こんにちは、シャオミンです。 今年で20周年， nice to meet you") + "\n===\n")
    print(method("こんにちは、シャオミンです。nice to meet you") + "\n===\n")
    print(method("Hello, I am Maxwell. 20 years old，中文名叫小明") + "\n===\n")
    
