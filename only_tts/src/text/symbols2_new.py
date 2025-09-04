# punctuation = ['!', '?', '…', ",", ".","@"]#@는 SP정지
punctuation = ["!", "?", "…", ",", "."]  # @는 SP정지
punctuation.append("-")
pu_symbols = punctuation + ["SP", "SP2", "SP3", "UNK"]
pad = "_"

# 영어 ARPABET 음소 (영어 TTS에 필요)
arpa = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G", "AY0", "M", "AA0", "F",
    "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0", "EY1", "IH0", "K", "N", "W", "IY2", "T", "AA1",
    "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER", "AW0", "UW0",
    "R", "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P", "AY1", "EY0", "OY2", "TH", "HH",
    "D", "ER0", "CH", "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0", "L", "SH",
}

# 한국어 자모 (한국어 TTS에 필요)
ko_symbols = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ공정"

# 한국어/영어만 지원하는 심볼 집합
symbols = [pad] + pu_symbols + list(arpa) + list(ko_symbols)
symbols = sorted(set(symbols))

if __name__ == "__main__":
    print(f"Total symbols: {len(symbols)}")
    print(f"Korean symbols: {len(ko_symbols)}")
    print(f"English ARPA symbols: {len(arpa)}")
