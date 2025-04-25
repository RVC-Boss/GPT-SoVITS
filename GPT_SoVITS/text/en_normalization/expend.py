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
