def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)


class Text2SRT():
    def __init__(self, text_seg, ts_list, offset=0):
        self.token_list = [i for i in text_seg.split() if len(i)]
        self.ts_list = ts_list
        start, end = ts_list[0][0] - offset, ts_list[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)
    def text(self):
        res = ""
        for word in self.token_list:
            if '\u4e00' <= word <= '\u9fff':
                res += word
            else:
                res += " " + word
        return res
    def len(self):
        return len(self.token_list)
    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec+acc_ost*1000),
            time_convert(self.end_sec+acc_ost*1000), 
            self.text())
    def time(self, acc_ost=0.0):
        return (self.start_sec/1000+acc_ost, self.end_sec/1000+acc_ost)

def distribute_spk(sentence_list, sd_time_list):
    sd_sentence_list = []
    for d in sentence_list:
        sentence_start = d['ts_list'][0][0]
        sentence_end = d['ts_list'][-1][1]
        sentence_spk = 0
        max_overlap = 0
        for sd_time in sd_time_list:
            spk_st, spk_ed, spk = sd_time
            spk_st = spk_st*1000
            spk_ed = spk_ed*1000
            overlap = max(
                min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
            if overlap > max_overlap:
                max_overlap = overlap
                sentence_spk = spk
        d['spk'] = sentence_spk
        sd_sentence_list.append(d)
    return sd_sentence_list

def generate_srt(sentence_list):
    srt_total = ''
    for i, d in enumerate(sentence_list):
        t2s = Text2SRT(d['text_seg'], d['ts_list'])
        if 'spk' in d:
            srt_total += "{}  spk{}\n{}".format(i, d['spk'], t2s.srt())
        else:
            srt_total += "{}\n{}".format(i, t2s.srt())
    return srt_total

def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0):
    start, end = int(start * 1000), int(end * 1000)
    srt_total = ''
    cc = 1 + begin_index
    subs = []
    for i, d in enumerate(sentence_list):
        if d['ts_list'][-1][1] <= start:
            continue
        if d['ts_list'][0][0] >= end:
            break
        # parts in between
        if (d['ts_list'][-1][1] <= end and d['ts_list'][0][0] > start) or (d['ts_list'][-1][1] == end and d['ts_list'][0][0] == start):
            t2s = Text2SRT(d['text_seg'], d['ts_list'], offset=start)
            srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text()))
            cc += 1
            continue
        if d['ts_list'][0][0] <= start:
            if not d['ts_list'][-1][1] > end:
                for j, ts in enumerate(d['ts_list']):
                    if ts[1] > start:
                        break
                _text = " ".join(d['text_seg'].split()[j:])
                _ts = d['ts_list'][j:]
            else:
                for j, ts in enumerate(d['ts_list']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(d['ts_list']):
                    if ts[1] > end:
                        _end = j
                        break
                _text = " ".join(d['text_seg'].split()[_start:_end])
                _ts = d['ts_list'][_start:_end]
            if len(ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text()))
                cc += 1
            continue
        if d['ts_list'][-1][1] > end:
            for j, ts in enumerate(d['ts_list']):
                if ts[1] > end:
                    break
            _text = " ".join(d['text_seg'].split()[:j])
            _ts = d['ts_list'][:j]
            if len(_ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text())
                    )
                cc += 1
            continue
    return srt_total, subs, cc
