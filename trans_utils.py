PUNC_LIST = ['，', '。', '！', '？', '、']


def pre_proc(text):
    res = ''
    for i in range(len(text)):
        if text[i] in PUNC_LIST:
            continue
        if '\u4e00' <= text[i] <= '\u9fff':
            if len(res) and res[-1] != " ":
                res += ' ' + text[i]+' '
            else:
                res += text[i]+' '
        else:
            res += text[i]
    if res[-1] == ' ':
        res = res[:-1]
    return res

def proc(raw_text, timestamp, dest_text):
    # simple matching
    ld = len(dest_text.split())
    mi, ts = [], []
    offset = 0
    while True:
        fi = raw_text.find(dest_text, offset, len(raw_text))
        # import pdb; pdb.set_trace()
        ti = raw_text[:fi].count(' ')
        if fi == -1:
            break
        offset = fi + ld
        mi.append(fi)
        ts.append([timestamp[ti][0]*16, timestamp[ti+ld-1][1]*16])
        # import pdb; pdb.set_trace()
    return ts

def proc_spk(dest_spk, sd_sentences):
    ts = []
    for d in sd_sentences:
        d_start = d['ts_list'][0][0]
        d_end = d['ts_list'][-1][1]
        spkid=dest_spk[3:]
        if str(d['spk']) == spkid and d_end-d_start>999:
            ts.append([d['start']*16, d['end']*16])
    return ts

def generate_vad_data(data, sd_sentences, sr=16000):
    assert len(data.shape) == 1
    vad_data = []
    for d in sd_sentences:
        d_start = round(d['ts_list'][0][0]/1000, 2)
        d_end = round(d['ts_list'][-1][1]/1000, 2)
        vad_data.append([d_start, d_end, data[int(d_start * sr):int(d_end * sr)]])
    return vad_data

def write_state(output_dir, state):
    for key in ['/recog_res_raw', '/timestamp', '/sentences', '/sd_sentences']:
        with open(output_dir+key, 'w') as fout:
            fout.write(str(state[key[1:]]))
    if 'sd_sentences' in state:
        with open(output_dir+'/sd_sentences', 'w') as fout:
            fout.write(str(state['sd_sentences']))

import os
def load_state(output_dir):
    state = {}
    with open(output_dir+'/recog_res_raw') as fin:
        line = fin.read()
        state['recog_res_raw'] = line
    with open(output_dir+'/timestamp') as fin:
        line = fin.read()
        state['timestamp'] = eval(line)
    with open(output_dir+'/sentences') as fin:
        line = fin.read()
        state['sentences'] = eval(line)
    if os.path.exists(output_dir+'/sd_sentences'):
        with open(output_dir+'/sd_sentences') as fin:
            line = fin.read()
            state['sd_sentences'] = eval(line)
    return state
        
    