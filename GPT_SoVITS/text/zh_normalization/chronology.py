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
import re

from .num import DIGITS
from .num import num2str
from .num import verbalize_cardinal
from .num import verbalize_digit


def _time_num2str(num_string: str) -> str:
    """A special case for verbalizing number in time."""
    result = num2str(num_string.lstrip("0"))
    if num_string.startswith("0"):
        result = DIGITS["0"] + result
    return result


# 时刻表达式
RE_TIME = re.compile(
    r"([0-1]?[0-9]|2[0-3])"
    r":([0-5][0-9])"
    r"(:([0-5][0-9]))?"
)

# 时间范围，如8:30-12:30
RE_TIME_RANGE = re.compile(
    r"([0-1]?[0-9]|2[0-3])"
    r":([0-5][0-9])"
    r"(:([0-5][0-9]))?"
    r"(~|-)"
    r"([0-1]?[0-9]|2[0-3])"
    r":([0-5][0-9])"
    r"(:([0-5][0-9]))?"
)


def replace_time(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """

    is_range = len(match.groups()) > 5

    hour = match.group(1)
    minute = match.group(2)
    second = match.group(4)

    if is_range:
        hour_2 = match.group(6)
        minute_2 = match.group(7)
        second_2 = match.group(9)

    result = f"{num2str(hour)}点"
    if minute.lstrip("0"):
        if int(minute) == 30:
            result += "半"
        else:
            result += f"{_time_num2str(minute)}分"
    if second and second.lstrip("0"):
        result += f"{_time_num2str(second)}秒"

    if is_range:
        result += "至"
        result += f"{num2str(hour_2)}点"
        if minute_2.lstrip("0"):
            if int(minute) == 30:
                result += "半"
            else:
                result += f"{_time_num2str(minute_2)}分"
        if second_2 and second_2.lstrip("0"):
            result += f"{_time_num2str(second_2)}秒"

    return result


RE_DATE = re.compile(
    r"(\d{4}|\d{2})年"
    r"((0?[1-9]|1[0-2])月)?"
    r"(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?"
)


def replace_date(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    year = match.group(1)
    month = match.group(3)
    day = match.group(5)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}{match.group(9)}"
    return result


# 用 / 或者 - 分隔的 YY/MM/DD 或者 YY-MM-DD 日期
RE_DATE2 = re.compile(r"(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])")


def replace_date2(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    year = match.group(1)
    month = match.group(3)
    day = match.group(4)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}日"
    return result
