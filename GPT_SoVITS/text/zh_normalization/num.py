# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rules to verbalize numbers into Chinese characters.
https://zh.wikipedia.org/wiki/中文数字#現代中文
"""
import re
from collections import OrderedDict
from typing import List

DIGITS = {str(i): tran for i, tran in enumerate('零一二三四五六七八九')}
UNITS = OrderedDict({
    1: '十',
    2: '百',
    3: '千',
    4: '万',
    8: '亿',
})

COM_QUANTIFIERS = '(处|台|架|枚|趟|幅|平|方|堵|间|床|株|批|项|例|列|篇|栋|注|亩|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)'

# 分数表达式
RE_FRAC = re.compile(r'(-?)(\d+)/(\d+)')


def replace_frac(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    nominator = match.group(2)
    denominator = match.group(3)
    sign: str = "负" if sign else ""
    nominator: str = num2str(nominator)
    denominator: str = num2str(denominator)
    result = f"{sign}{denominator}分之{nominator}"
    return result


# 百分数表达式
RE_PERCENTAGE = re.compile(r'(-?)(\d+(\.\d+)?)%')


def replace_percentage(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    percent = match.group(2)
    sign: str = "负" if sign else ""
    percent: str = num2str(percent)
    result = f"{sign}百分之{percent}"
    return result


# 整数表达式
# 带负号的整数 -10
RE_INTEGER = re.compile(r'(-)' r'(\d+)')


def replace_negative_num(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    number = match.group(2)
    sign: str = "负" if sign else ""
    number: str = num2str(number)
    result = f"{sign}{number}"
    return result


# 编号-无符号整形
# 00078
RE_DEFAULT_NUM = re.compile(r'\d{3}\d*')


def replace_default_num(match):
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    number = match.group(0)
    return verbalize_digit(number, alt_one=True)


# 加减乘除
# RE_ASMD = re.compile(
#     r'((-?)((\d+)(\.\d+)?)|(\.(\d+)))([\+\-\×÷=])((-?)((\d+)(\.\d+)?)|(\.(\d+)))')
RE_ASMD = re.compile(
    r'((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))([\+\-\×÷=])((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))')

asmd_map = {
    '+': '加',
    '-': '减',
    '×': '乘',
    '÷': '除',
    '=': '等于'
}

def replace_asmd(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    result = match.group(1) + asmd_map[match.group(8)] + match.group(9)
    return result


# 次方专项
RE_POWER = re.compile(r'[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+')

power_map = {
    '⁰': '0',
    '¹': '1',
    '²': '2',
    '³': '3',
    '⁴': '4',
    '⁵': '5',
    '⁶': '6',
    '⁷': '7',
    '⁸': '8',
    '⁹': '9',
    'ˣ': 'x',
    'ʸ': 'y',
    'ⁿ': 'n'
}

def replace_power(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    power_num = ""
    for m in match.group(0):
        power_num += power_map[m]
    result = "的" + power_num + "次方"
    return result


# 数字表达式
# 纯小数
RE_DECIMAL_NUM = re.compile(r'(-?)((\d+)(\.\d+))' r'|(\.(\d+))')
# 正整数 + 量词
RE_POSITIVE_QUANTIFIERS = re.compile(r"(\d+)([多余几\+])?" + COM_QUANTIFIERS)
RE_NUMBER = re.compile(r'(-?)((\d+)(\.\d+)?)' r'|(\.(\d+))')


def replace_positive_quantifier(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    number = match.group(1)
    match_2 = match.group(2)
    if match_2 == "+":
        match_2 = "多"
    match_2: str = match_2 if match_2 else ""
    quantifiers: str = match.group(3)
    number: str = num2str(number)
    result = f"{number}{match_2}{quantifiers}"
    return result


def replace_number(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    number = match.group(2)
    pure_decimal = match.group(5)
    if pure_decimal:
        result = num2str(pure_decimal)
    else:
        sign: str = "负" if sign else ""
        number: str = num2str(number)
        result = f"{sign}{number}"
    return result


# 范围表达式
# match.group(1) and match.group(8) are copy from RE_NUMBER

RE_RANGE = re.compile(
    r"""
    (?<![\d\+\-\×÷=])      # 使用反向前瞻以确保数字范围之前没有其他数字和操作符
    ((-?)((\d+)(\.\d+)?))  # 匹配范围起始的负数或正数（整数或小数）
    [-~]                   # 匹配范围分隔符
    ((-?)((\d+)(\.\d+)?))  # 匹配范围结束的负数或正数（整数或小数）
    (?![\d\+\-\×÷=])       # 使用正向前瞻以确保数字范围之后没有其他数字和操作符
    """, re.VERBOSE)


def replace_range(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    first, second = match.group(1), match.group(6)
    first = RE_NUMBER.sub(replace_number, first)
    second = RE_NUMBER.sub(replace_number, second)
    result = f"{first}到{second}"
    return result


# ~至表达式
RE_TO_RANGE = re.compile(
    r'((-?)((\d+)(\.\d+)?)|(\.(\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)[~]((-?)((\d+)(\.\d+)?)|(\.(\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)')

def replace_to_range(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    result = match.group(0).replace('~', '至')
    return result


def _get_value(value_string: str, use_zero: bool=True) -> List[str]:
    stripped = value_string.lstrip('0')
    if len(stripped) == 0:
        return []
    elif len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS['0'], DIGITS[stripped]]
        else:
            return [DIGITS[stripped]]
    else:
        largest_unit = next(
            power for power in reversed(UNITS.keys()) if power < len(stripped))
        first_part = value_string[:-largest_unit]
        second_part = value_string[-largest_unit:]
        return _get_value(first_part) + [UNITS[largest_unit]] + _get_value(
            second_part)


def verbalize_cardinal(value_string: str) -> str:
    if not value_string:
        return ''

    # 000 -> '零' , 0 -> '零'
    value_string = value_string.lstrip('0')
    if len(value_string) == 0:
        return DIGITS['0']

    result_symbols = _get_value(value_string)
    # verbalized number starting with '一十*' is abbreviated as `十*`
    if len(result_symbols) >= 2 and result_symbols[0] == DIGITS[
            '1'] and result_symbols[1] == UNITS[1]:
        result_symbols = result_symbols[1:]
    return ''.join(result_symbols)


def verbalize_digit(value_string: str, alt_one=False) -> str:
    result_symbols = [DIGITS[digit] for digit in value_string]
    result = ''.join(result_symbols)
    if alt_one:
        result = result.replace("一", "幺")
    return result


def num2str(value_string: str) -> str:
    integer_decimal = value_string.split('.')
    if len(integer_decimal) == 1:
        integer = integer_decimal[0]
        decimal = ''
    elif len(integer_decimal) == 2:
        integer, decimal = integer_decimal
    else:
        raise ValueError(
            f"The value string: '${value_string}' has more than one point in it."
        )

    result = verbalize_cardinal(integer)

    decimal = decimal.rstrip('0')
    if decimal:
        # '.22' is verbalized as '零点二二'
        # '3.20' is verbalized as '三点二
        result = result if result else "零"
        result += '点' + verbalize_digit(decimal)
    return result
