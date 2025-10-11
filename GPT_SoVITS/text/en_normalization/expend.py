# by https://github.com/Cosmo-klara

from __future__ import print_function

import re
import inflect
import unicodedata

# 后缀计量单位替换表
measurement_map = {
    "m": ["meter", "meters"],
    "km": ["kilometer", "kilometers"],
    "km/h": ["kilometer per hour", "kilometers per hour"],
    "ft": ["feet", "feet"],
    "L": ["liter", "liters"],
    "tbsp": ["tablespoon", "tablespoons"],
    "tsp": ["teaspoon", "teaspoons"],
    "h": ["hour", "hours"],
    "min": ["minute", "minutes"],
    "s": ["second", "seconds"],
    "°C": ["degree celsius", "degrees celsius"],
    "°F": ["degree fahrenheit", "degrees fahrenheit"],
}


# 识别 12,000 类型
_inflect = inflect.engine()

# 转化数字序数词
_ordinal_number_re = re.compile(r"\b([0-9]+)\. ")

# 我听说好像对于数字正则识别其实用 \d 会好一点

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")

# 时间识别
_time_re = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")

# 后缀计量单位识别
_measurement_re = re.compile(r"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b")

# 前后 £ 识别 ( 写了识别两边某一边的，但是不知道为什么失败了┭┮﹏┭┮ )
_pounds_re_start = re.compile(r"£([0-9\.\,]*[0-9]+)")
_pounds_re_end = re.compile(r"([0-9\.\,]*[0-9]+)£")

# 前后 $ 识别
_dollars_re_start = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_dollars_re_end = re.compile(r"([(0-9\.\,]*[0-9]+)\$")

# 小数的识别
_decimal_number_re = re.compile(r"([0-9]+\.\s*[0-9]+)")

# 分数识别 (形式 "3/4" )
_fraction_re = re.compile(r"([0-9]+/[0-9]+)")

# 序数词识别
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")

# 数字处理
_number_re = re.compile(r"[0-9]+")


def _convert_ordinal(m):
    """
    标准化序数词, 例如: 1. 2. 3. 4. 5. 6.
    Examples:
        input: "1. "
        output: "1st"
    然后在后面的 _expand_ordinal, 将其转化为 first 这类的
    """
    ordinal = _inflect.ordinal(m.group(1))
    return ordinal + ", "


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_time(m):
    """
    将 24 小时制的时间转换为 12 小时制的时间表示方式。

    Examples:
        input: "13:00 / 4:00 / 13:30"
        output: "one o'clock p.m. / four o'clock am. / one thirty p.m."
    """
    hours, minutes = map(int, m.group(1, 2))
    period = "a.m." if hours < 12 else "p.m."
    if hours > 12:
        hours -= 12

    hour_word = _inflect.number_to_words(hours)
    minute_word = _inflect.number_to_words(minutes) if minutes != 0 else ""

    if minutes == 0:
        return f"{hour_word} o'clock {period}"
    else:
        return f"{hour_word} {minute_word} {period}"


def _expand_measurement(m):
    """
    处理一些常见的测量单位后缀, 目前支持: m, km, km/h, ft, L, tbsp, tsp, h, min, s, °C, °F
    如果要拓展的话修改: _measurement_re 和 measurement_map
    """
    sign = m.group(3)
    ptr = 1
    # 想不到怎么方便的取数字，又懒得改正则，诶，1.2 反正也是复数读法，干脆直接去掉 "."
    num = int(m.group(1).replace(sign, "").replace(".", ""))
    decimal_part = m.group(2)
    # 上面判断的漏洞，比如 0.1 的情况，在这里排除了
    if decimal_part == None and num == 1:
        ptr = 0
    return m.group(1).replace(sign, " " + measurement_map[sign][ptr])


def _expand_pounds(m):
    """
    没找到特别规范的说明，和美元的处理一样，其实可以把两个合并在一起
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " pounds"  # Unexpected format
    pounds = int(parts[0]) if parts[0] else 0
    pence = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if pounds and pence:
        pound_unit = "pound" if pounds == 1 else "pounds"
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s and %s %s" % (pounds, pound_unit, pence, penny_unit)
    elif pounds:
        pound_unit = "pound" if pounds == 1 else "pounds"
        return "%s %s" % (pounds, pound_unit)
    elif pence:
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s" % (pence, penny_unit)
    else:
        return "zero pounds"


def _expand_dollars(m):
    """
    change: 美分是 100 的限值, 应该要做补零的吧
    Example:
        input: "32.3$ / $6.24"
        output: "thirty-two dollars and thirty cents" / "six dollars and twenty-four cents"
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s and %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


# 小数的处理
def _expand_decimal_number(m):
    """
    Example:
        input: "13.234"
        output: "thirteen point two three four"
    """
    match = m.group(1)
    parts = match.split(".")
    words = []
    # 遍历字符串中的每个字符
    for char in parts[1]:
        if char == ".":
            words.append("point")
        else:
            words.append(char)
    return parts[0] + " point " + " ".join(words)


# 分数的处理
def _expend_fraction(m):
    """
    规则1: 分子使用基数词读法, 分母用序数词读法.
    规则2: 如果分子大于 1, 在读分母的时候使用序数词复数读法.
    规则3: 当分母为2的时候, 分母读做 half, 并且当分子大于 1 的时候, half 也要用复数读法, 读为 halves.
    Examples:

    | Written |	Said |
    |:---:|:---:|
    | 1/3 | one third |
    | 3/4 | three fourths |
    | 5/6 | five sixths |
    | 1/2 | one half |
    | 3/2 | three halves |
    """
    match = m.group(0)
    numerator, denominator = map(int, match.split("/"))

    numerator_part = _inflect.number_to_words(numerator)
    if denominator == 2:
        if numerator == 1:
            denominator_part = "half"
        else:
            denominator_part = "halves"
    elif denominator == 1:
        return f"{numerator_part}"
    else:
        denominator_part = _inflect.ordinal(_inflect.number_to_words(denominator))
        if numerator > 1:
            denominator_part += "s"

    return f"{numerator_part} {denominator_part}"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


class CharMapper:
    """
    字符映射追踪器，用于记录文本标准化过程中字符位置的变化
    
    核心思想：维护从原始文本到当前文本的映射
    - orig_to_curr[i] 表示原始文本第i个字符对应当前文本的位置
    """
    def __init__(self, text):
        self.original_text = text
        self.text = text
        # 初始化：每个原始字符映射到自己
        self.orig_to_curr = list(range(len(text)))
        
    def apply_sub(self, pattern, replacement_func):
        """
        应用正则替换并更新映射
        
        关键：需要通过旧映射来更新新映射
        支持捕获组的特殊处理（如大写字母拆分）
        """
        new_text = ""
        # curr_to_new[i] 表示当前文本第i个字符在新文本中的位置
        curr_to_new = [-1] * len(self.text)
        
        pos = 0
        for match in pattern.finditer(self.text):
            # 处理匹配前的未变化文本
            for i in range(pos, match.start()):
                curr_to_new[i] = len(new_text)
                new_text += self.text[i]
            
            # 处理匹配的部分
            replacement = replacement_func(match)
            replacement_start_pos = len(new_text)
            
            # 特殊处理：如果替换文本包含原始文本的字符（例如 "A" -> " A"）
            # 尝试找到对应关系
            match_text = match.group(0)
            if len(match.groups()) > 0 and match.group(1) in replacement:
                # 有捕获组，尝试精确映射
                # 例如: "(?<!^)(?<![\s])([A-Z])" 匹配 "P"，替换为 " P"
                captured = match.group(1)
                replacement_idx = replacement.find(captured)
                if replacement_idx >= 0:
                    # 捕获的字符在替换文本中
                    for i in range(match.start(), match.end()):
                        char = self.text[i]
                        if char in captured:
                            # 这个字符在捕获组中，映射到替换文本中的对应位置
                            char_idx_in_replacement = replacement.find(char, replacement_idx)
                            if char_idx_in_replacement >= 0:
                                curr_to_new[i] = replacement_start_pos + char_idx_in_replacement
                            else:
                                curr_to_new[i] = replacement_start_pos
                        else:
                            curr_to_new[i] = replacement_start_pos
                else:
                    # 捕获的字符不在替换文本中，都映射到起始位置
                    for i in range(match.start(), match.end()):
                        curr_to_new[i] = replacement_start_pos
            else:
                # 没有捕获组或不包含原始字符，匹配部分的所有字符都映射到替换文本的起始位置
                for i in range(match.start(), match.end()):
                    curr_to_new[i] = replacement_start_pos
            
            new_text += replacement
            pos = match.end()
        
        # 处理剩余文本
        for i in range(pos, len(self.text)):
            curr_to_new[i] = len(new_text)
            new_text += self.text[i]
        
        # 更新原始到当前的映射：orig -> old_curr -> new_curr
        new_orig_to_curr = []
        for orig_idx in range(len(self.original_text)):
            old_curr_idx = self.orig_to_curr[orig_idx]
            if old_curr_idx >= 0 and old_curr_idx < len(curr_to_new):
                new_orig_to_curr.append(curr_to_new[old_curr_idx])
            else:
                new_orig_to_curr.append(-1)
        
        self.text = new_text
        self.orig_to_curr = new_orig_to_curr
        
    def apply_char_filter(self, keep_pattern):
        """
        应用字符过滤（只保留符合模式的字符）并更新映射
        
        keep_pattern: 正则表达式字符串，如 "[ A-Za-z'.,?!-]"
        """
        new_text = ""
        curr_to_new = []
        
        for i, char in enumerate(self.text):
            if re.match(keep_pattern, char):
                curr_to_new.append(len(new_text))
                new_text += char
            else:
                # 字符被删除
                if new_text:
                    curr_to_new.append(len(new_text) - 1)
                else:
                    curr_to_new.append(-1)
        
        # 更新原始映射
        new_orig_to_curr = []
        for orig_idx in range(len(self.original_text)):
            old_curr_idx = self.orig_to_curr[orig_idx]
            if old_curr_idx >= 0 and old_curr_idx < len(curr_to_new):
                new_orig_to_curr.append(curr_to_new[old_curr_idx])
            else:
                new_orig_to_curr.append(-1)
        
        self.text = new_text
        self.orig_to_curr = new_orig_to_curr
        
    def get_norm_to_orig(self):
        """
        构建标准化文本到原始文本的反向映射
        """
        if not self.text:
            return []
        
        norm_to_orig = [-1] * len(self.text)
        for orig_idx, norm_idx in enumerate(self.orig_to_curr):
            if 0 <= norm_idx < len(self.text):
                # 如果多个原始字符映射到同一个标准化位置，取第一个
                if norm_to_orig[norm_idx] == -1:
                    norm_to_orig[norm_idx] = orig_idx
        
        return norm_to_orig


def normalize_with_map(text):
    """
    带字符映射的标准化函数
    
    返回:
        normalized_text: 标准化后的文本
        char_mappings: 字典，包含:
            - "orig_to_norm": list[int], 原始文本每个字符对应标准化文本的位置
            - "norm_to_orig": list[int], 标准化文本每个字符对应原始文本的位置
    """
    mapper = CharMapper(text)
    
    # 按照 normalize() 的顺序应用所有转换
    mapper.apply_sub(_ordinal_number_re, _convert_ordinal)
    mapper.apply_sub(re.compile(r"(?<!\d)-|-(?!\d)"), lambda m: " minus ")
    mapper.apply_sub(_comma_number_re, _remove_commas)
    mapper.apply_sub(_time_re, _expand_time)
    mapper.apply_sub(_measurement_re, _expand_measurement)
    mapper.apply_sub(_pounds_re_start, _expand_pounds)
    mapper.apply_sub(_pounds_re_end, _expand_pounds)
    mapper.apply_sub(_dollars_re_start, _expand_dollars)
    mapper.apply_sub(_dollars_re_end, _expand_dollars)
    mapper.apply_sub(_decimal_number_re, _expand_decimal_number)
    mapper.apply_sub(_fraction_re, _expend_fraction)
    mapper.apply_sub(_ordinal_re, _expand_ordinal)
    mapper.apply_sub(_number_re, _expand_number)
    
    # Strip accents - 需要手动处理映射
    normalized_nfd = unicodedata.normalize("NFD", mapper.text)
    new_text = ""
    curr_to_new = []
    for i, char in enumerate(normalized_nfd):
        if unicodedata.category(char) != "Mn":
            curr_to_new.append(len(new_text))
            new_text += char
        else:
            # 重音符号被删除，映射到前一个字符
            if new_text:
                curr_to_new.append(len(new_text) - 1)
            else:
                curr_to_new.append(-1)
    
    # 更新原始映射 - 需要处理 NFD 可能改变字符数的情况
    # 简化处理：假设 NFD 不会显著改变字符数（对于英文通常是这样）
    if len(curr_to_new) >= len(mapper.text):
        # NFD 展开了一些字符
        new_orig_to_curr = []
        for orig_idx in range(len(mapper.original_text)):
            old_curr_idx = mapper.orig_to_curr[orig_idx]
            if old_curr_idx >= 0 and old_curr_idx < len(curr_to_new):
                new_orig_to_curr.append(curr_to_new[old_curr_idx])
            else:
                new_orig_to_curr.append(-1)
        mapper.orig_to_curr = new_orig_to_curr
    mapper.text = new_text
    
    # 继续其他替换
    mapper.apply_sub(re.compile("%"), lambda m: " percent")
    
    # 删除非法字符 - 使用 apply_char_filter
    mapper.apply_char_filter(r"[ A-Za-z'.,?!\-]")
    
    mapper.apply_sub(re.compile(r"(?i)i\.e\."), lambda m: "that is")
    mapper.apply_sub(re.compile(r"(?i)e\.g\."), lambda m: "for example")
    mapper.apply_sub(re.compile(r"(?<!^)(?<![\s])([A-Z])"), lambda m: " " + m.group(1))
    
    norm_to_orig = mapper.get_norm_to_orig()
    
    return mapper.text, {
        "orig_to_norm": mapper.orig_to_curr,
        "norm_to_orig": norm_to_orig
    }


def normalize(text):
    """
    !!! 所有的处理都需要正确的输入 !!!
    可以添加新的处理，只需要添加正则表达式和对应的处理函数即可
    """

    text = re.sub(_ordinal_number_re, _convert_ordinal, text)
    text = re.sub(r"(?<!\d)-|-(?!\d)", " minus ", text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_pounds_re_start, _expand_pounds, text)
    text = re.sub(_pounds_re_end, _expand_pounds, text)
    text = re.sub(_dollars_re_start, _expand_dollars, text)
    text = re.sub(_dollars_re_end, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_number, text)
    text = re.sub(_fraction_re, _expend_fraction, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)

    text = "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )  # Strip accents

    text = re.sub("%", " percent", text)
    text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
    text = re.sub(r"(?i)i\.e\.", "that is", text)
    text = re.sub(r"(?i)e\.g\.", "for example", text)
    # 增加纯大写单词拆分
    text = re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)
    return text


if __name__ == "__main__":
    # 我觉得其实可以把切分结果展示出来（只读，或者修改不影响传给TTS的实际text）
    # 然后让用户确认后再输入给 TTS，可以让用户检查自己有没有不标准的输入
    print(normalize("1. test ordinal number 1st"))
    print(normalize("32.3$, $6.24, 1.1£, £7.14."))
    print(normalize("3/23, 1/2, 3/2, 1/3, 6/1"))
    print(normalize("1st, 22nd"))
    print(normalize("a test 20h, 1.2s, 1L, 0.1km"))
    print(normalize("a test of time 4:00, 13:00, 13:30"))
    print(normalize("a test of temperature 4°F, 23°C, -19°C"))
