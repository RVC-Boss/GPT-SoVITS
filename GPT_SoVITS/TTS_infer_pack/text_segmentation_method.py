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

# contributed by XTer
# 简单的按长度切分，不希望出现超长的句子
def split_long_sentence(text, max_length=510):
    
    opts = []
    sentences = text.split('\n')
    for sentence in sentences:
        while len(sentence) > max_length:
            part = sentence[:max_length]
            opts.append(part)
            sentence = sentence[max_length:]
        if sentence:
            opts.append(sentence)
    return "\n".join(opts)

# 不切
@register_method("cut0")
def cut0(inp):
    return inp


# 凑四句一切
@register_method("cut1")
def cut1(inp):
    inp = split_long_sentence(inp).strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


# 凑50字一切
@register_method("cut2")
def cut2(inp, max_length=50):
    inp = split_long_sentence(inp).strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > max_length:
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
    inp = split_long_sentence(inp).strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


# 按英文句号.切
@register_method("cut4")
def cut4(inp):
    inp = split_long_sentence(inp).strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

# 按标点符号切
# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
@register_method("cut5")
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = split_long_sentence(inp).strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt

# contributed by https://github.com/X-T-E-R/GPT-SoVITS-Inference/blob/main/GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py
@register_method("auto_cut")
def auto_cut(inp, max_length=30):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    erase_punds = r'[“”"‘’\'（）()【】[\]{}<>《》〈〉〔〕〖〗〘〙〚〛〛〞〟]'
    inp = re.sub(erase_punds, '', inp)
    split_punds = r'[?!。？！~：]'
    if inp[-1] not in split_punds:
        inp+="。"
    items = re.split(f'({split_punds})', inp)
    items = ["".join(group) for group in zip(items[::2], items[1::2])]

    def process_commas(text, max_length):
    
        # Define separators and the regular expression for splitting
        separators = ['，', ',', '、', '——', '…']
        # 使用正则表达式的捕获组来保留分隔符，分隔符两边的括号就是所谓的捕获组
        regex_pattern = '(' + '|'.join(map(re.escape, separators)) + ')'
        # 使用re.split函数分割文本，由于使用了捕获组，分隔符也会作为分割结果的一部分返回
        sentences = re.split(regex_pattern, text)
  
        processed_text = "" 
        current_line = "" 
        
        final_sentences = []
        
        for sentence in sentences:
            if len(sentence)>max_length:
                
                final_sentences+=split_long_sentence(sentence,max_length=max_length).split("\n")
            else:
                final_sentences.append(sentence)
        
        for sentence in final_sentences:
            # Add the length of the sentence plus one for the space or newline that will follow
            if len(current_line) + len(sentence) <= max_length:
                # If adding the next sentence does not exceed max length, add it to the current line
                current_line += sentence
            else:
                # If the current line is too long, start a new line
                processed_text += current_line.strip() + '\n'
                current_line = sentence + " "  # Start the new line with the current sentence
        
        # Add any remaining text in current_line to processed_text
        processed_text += current_line.strip()

        return processed_text

    final_items = []
    for item in items:
        final_items+=process_commas(item,max_length=max_length).split("\n")
    
    final_items = [item for item in final_items if item.strip() and not (len(item.strip()) == 1 and item.strip() in "?!，,。？！~：")]

    return "\n".join(final_items)


if __name__ == '__main__':
    method = get_method("cut1")
    str1="""一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十
    """
    print("|\n|".join(method(str1).split("\n")))
