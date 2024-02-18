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

from .num import num2str

# 温度表达式，温度会影响负号的读法
# -3°C 零下三度
RE_TEMPERATURE = re.compile(r'(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)')
measure_dict = {
    "cm2": "平方厘米",
    "cm²": "平方厘米",
    "cm3": "立方厘米",
    "cm³": "立方厘米",
    "cm": "厘米",
    "db": "分贝",
    "ds": "毫秒",
    "kg": "千克",
    "km": "千米",
    "m2": "平方米",
    "m²": "平方米",
    "m³": "立方米",
    "m3": "立方米",
    "ml": "毫升",
    "m": "米",
    "mm": "毫米",
    "s": "秒"
}


def replace_temperature(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    temperature = match.group(2)
    unit = match.group(3)
    sign: str = "零下" if sign else ""
    temperature: str = num2str(temperature)
    unit: str = "摄氏度" if unit == "摄氏度" else "度"
    result = f"{sign}{temperature}{unit}"
    return result


def replace_measure(sentence) -> str:
    for q_notation in measure_dict:
        if q_notation in sentence:
            sentence = sentence.replace(q_notation, measure_dict[q_notation])
    return sentence
