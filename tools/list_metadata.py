from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ListLine:
    """一条 `.list` 标注行（兼容 4/5/6 列）。

    列定义: wav_path|speaker_name|language|text[|emotion|remark]
    训练只用前 4 列（training_fields）；emotion / remark 为元数据，
    不参与模型训练，用于标注与参考音频展示。
    """

    wav_path: str
    speaker_name: str
    language: str
    text: str
    emotion: str = ""
    remark: str = ""

    @property
    def training_fields(self):
        return [self.wav_path, self.speaker_name, self.language, self.text]


def _clean(value):
    return str(value or "").strip()


def parse_list_line(line):
    """解析一行 `.list`。列数 < 4 返回 None；emotion/remark 缺省为空串。"""
    parts = str(line or "").rstrip("\r\n").split("|")
    if len(parts) < 4:
        return None
    return ListLine(
        wav_path=_clean(parts[0]),
        speaker_name=_clean(parts[1]),
        language=_clean(parts[2]),
        text=_clean(parts[3]),
        emotion=_clean(parts[4]) if len(parts) >= 5 else "",
        remark=_clean(parts[5]) if len(parts) >= 6 else "",
    )


def format_list_line(item):
    """把 ListLine 格式化为 6 列 `.list` 行（emotion/remark 为空也保留分隔符）。

    注意：写出 6 列以保证与 proplus-hc-dev 分支互操作，主分支训练侧
    （1-get-text.py）已用 parse_list_line 只消费前 4 列，多余列不影响训练。
    """
    return "|".join(
        [
            str(item.wav_path or ""),
            str(item.speaker_name or ""),
            str(item.language or ""),
            str(item.text or ""),
            str(item.emotion or ""),
            str(item.remark or ""),
        ]
    )
