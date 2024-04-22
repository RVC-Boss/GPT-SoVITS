import sys
import copy
import librosa
import logging
import argparse
import numpy as np
import soundfile as sf
import moviepy.editor as mpy
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from subtitle_utils import generate_srt, generate_srt_clip, distribute_spk
from trans_utils import pre_proc, proc, write_state, load_state, proc_spk, generate_vad_data
from argparse_tools import ArgumentParser, get_commandline_args

from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip


class VideoClipper():
    def __init__(self, asr_pipeline, sd_pipeline=None):
        logging.warning("Initializing VideoClipper.")
        self.asr_pipeline = asr_pipeline
        self.sd_pipeline = sd_pipeline

    def recog(self, audio_input, sd_switch='no', state=None):
        if state is None:
            state = {}
        sr, data = audio_input
        assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
        if len(data.shape) == 2:  # multi-channel wav input
            logging.warning("Input wav shape: {}, only first channel reserved.").format(data.shape)
            data = data[:,0]
        state['audio_input'] = (sr, data)
        data = data.astype(np.float64)
        rec_result = self.asr_pipeline(audio_in=data)
        if sd_switch == 'yes':
            vad_data = generate_vad_data(data.astype(np.float32), rec_result['sentences'], sr)
            sd_result = self.sd_pipeline(audio=vad_data, batch_size=1)
            rec_result['sd_sentences'] = distribute_spk(rec_result['sentences'], sd_result['text'])
            res_srt = generate_srt(rec_result['sd_sentences'])
            state['sd_sentences'] = rec_result['sd_sentences']
        else:
            res_srt = generate_srt(rec_result['sentences'])
        state['recog_res_raw'] = rec_result['text_postprocessed']
        state['timestamp'] = rec_result['time_stamp']
        state['sentences'] = rec_result['sentences']
        res_text = rec_result['text']
        return res_text, res_srt, state

    def clip(self, dest_text, start_ost, end_ost, state, dest_spk=None):
        # get from state
        audio_input = state['audio_input']
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        sr, data = audio_input
        data = data.astype(np.float64)

        all_ts = []
        if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
            for _dest_text in dest_text.split('#'):
                _dest_text = pre_proc(_dest_text)
                ts = proc(recog_res_raw, timestamp, _dest_text)
                for _ts in ts: all_ts.append(_ts)
        else:
            for _dest_spk in dest_spk.split('#'):
                ts = proc_spk(_dest_spk, state['sd_sentences'])
                for _ts in ts: all_ts.append(_ts)
        ts = all_ts
        # ts.sort()
        srt_index = 0
        clip_srt = ""
        if len(ts):
            start, end = ts[0]
            start = min(max(0, start+start_ost*16), len(data))
            end = min(max(0, end+end_ost*16), len(data))
            res_audio = data[start:end]
            start_end_info = "from {} to {}".format(start/16000, end/16000)
            srt_clip, _, srt_index = generate_srt_clip(sentences, start/16000.0, end/16000.0, begin_index=srt_index)
            clip_srt += srt_clip
            for _ts in ts[1:]:  # multiple sentence input or multiple output matched
                start, end = _ts
                start = min(max(0, start+start_ost*16), len(data))
                end = min(max(0, end+end_ost*16), len(data))
                start_end_info += ", from {} to {}".format(start, end)
                res_audio = np.concatenate([res_audio, data[start+start_ost*16:end+end_ost*16]], -1)
                srt_clip, _, srt_index = generate_srt_clip(sentences, start/16000.0, end/16000.0, begin_index=srt_index-1)
                clip_srt += srt_clip
        if len(ts):
            message = "{} periods found in the speech: ".format(len(ts)) + start_end_info
        else:
            message = "No period found in the speech, return raw speech. You may check the recognition result and try other destination text."
            res_audio = data
        return (sr, res_audio), message, clip_srt

    def video_recog(self, vedio_filename, sd_switch='no'):
        vedio_filename = vedio_filename
        clip_video_file = vedio_filename[:-4] + '_clip.mp4'
        video = mpy.VideoFileClip(vedio_filename)
        audio_file = vedio_filename[:-3] + 'wav'
        video.audio.write_audiofile(audio_file)
        wav = librosa.load(audio_file, sr=16000)[0]
        state = {
            'vedio_filename': vedio_filename,
            'clip_video_file': clip_video_file,
            'video': video,
        }
        # res_text, res_srt = self.recog((16000, wav), state)
        return self.recog((16000, wav), sd_switch, state)

    def video_clip(self, dest_text, start_ost, end_ost, state, font_size=32, font_color='white', add_sub=False, dest_spk=None):
        # get from state
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        video = state['video']
        clip_video_file = state['clip_video_file']
        vedio_filename = state['vedio_filename']
        
        all_ts = []
        srt_index = 0
        if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
            for _dest_text in dest_text.split('#'):
                _dest_text = pre_proc(_dest_text)
                ts = proc(recog_res_raw, timestamp, _dest_text)
                for _ts in ts: all_ts.append(_ts)
        else:
            for _dest_spk in dest_spk.split('#'):
                ts = proc_spk(_dest_spk, state['sd_sentences'])
                for _ts in ts: all_ts.append(_ts)
        time_acc_ost = 0.0
        ts = all_ts
        # ts.sort()
        clip_srt = ""
        if len(ts):
            start, end = ts[0][0] / 16000, ts[0][1] / 16000
            srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index, time_acc_ost=time_acc_ost)
            start, end = start+start_ost/1000.0, end+end_ost/1000.0
            video_clip = video.subclip(start, end)
            start_end_info = "from {} to {}".format(start, end)
            clip_srt += srt_clip
            if add_sub:
                generator = lambda txt: TextClip(txt, font='./font/STHeitiMedium.ttc', fontsize=font_size, color=font_color)
                subtitles = SubtitlesClip(subs, generator)
                video_clip = CompositeVideoClip([video_clip, subtitles.set_pos(('center','bottom'))])
            concate_clip = [video_clip]
            time_acc_ost += end+end_ost/1000.0 - (start+start_ost/1000.0)
            for _ts in ts[1:]:
                start, end = _ts[0] / 16000, _ts[1] / 16000
                srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index-1, time_acc_ost=time_acc_ost)
                chi_subs = []
                sub_starts = subs[0][0][0]
                for sub in subs:
                    chi_subs.append(((sub[0][0]-sub_starts, sub[0][1]-sub_starts), sub[1]))
                start, end = start+start_ost/1000.0, end+end_ost/1000.0
                _video_clip = video.subclip(start, end)
                start_end_info += ", from {} to {}".format(start, end)
                clip_srt += srt_clip
                if add_sub:
                    generator = lambda txt: TextClip(txt, font='./font/STHeitiMedium.ttc', fontsize=font_size, color=font_color)
                    subtitles = SubtitlesClip(chi_subs, generator)
                    _video_clip = CompositeVideoClip([_video_clip, subtitles.set_pos(('center','bottom'))])
                    # _video_clip.write_videofile("debug.mp4", audio_codec="aac")
                concate_clip.append(copy.copy(_video_clip))
                time_acc_ost += end+end_ost/1000.0 - (start+start_ost/1000.0)
            message = "{} periods found in the audio: ".format(len(ts)) + start_end_info
            logging.warning("Concating...")
            if len(concate_clip) > 1:
                video_clip = concatenate_videoclips(concate_clip)
            video_clip.write_videofile(clip_video_file, audio_codec="aac")
        else:
            clip_video_file = vedio_filename
            message = "No period found in the audio, return raw speech. You may check the recognition result and try other destination text."
            srt_clip = ''
        return clip_video_file, message, clip_srt


def get_parser():
    parser = ArgumentParser(
        description="ClipVideo Argument",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        help="Stage, 0 for recognizing and 1 for clipping",
        required=True
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file path",
        required=True
    )
    parser.add_argument(
        "--sd_switch",
        type=str,
        choices=("no", "yes"),
        default="no",
        help="Trun on the speaker diarization or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Output files path",
    )
    parser.add_argument(
        "--dest_text",
        type=str,
        default=None,
        help="Destination text string for clipping",
    )
    parser.add_argument(
        "--dest_spk",
        type=str,
        default=None,
        help="Destination spk id for clipping",
    )
    parser.add_argument(
        "--start_ost",
        type=int,
        default=0,
        help="Offset time in ms at beginning for clipping"
    )
    parser.add_argument(
        "--end_ost",
        type=int,
        default=0,
        help="Offset time in ms at ending for clipping"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path"
    )
    return parser


def runner(stage, file, sd_switch, output_dir, dest_text, dest_spk, start_ost, end_ost, output_file, config=None):
    audio_suffixs = ['wav']
    video_suffixs = ['mp4']
    if file[-3:] in audio_suffixs:
        mode = 'audio'
    elif file[-3:] in video_suffixs:
        mode = 'video'
    else:
        logging.error("Unsupported file format: {}".format(file))
    while output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    if stage == 1:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        # initialize modelscope asr pipeline
        logging.warning("Initializing modelscope asr pipeline.")
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            output_dir=output_dir,
        )
        sd_pipeline = pipeline(
            task='speaker-diarization',
            model='damo/speech_campplus_speaker-diarization_common',
            model_revision='v1.0.0'
        )
        audio_clipper = VideoClipper(inference_pipeline, sd_pipeline)
        if mode == 'audio':
            logging.warning("Recognizing audio file: {}".format(file))
            wav, sr = librosa.load(file, sr=16000)
            res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch)
        if mode == 'video':
            logging.warning("Recognizing video file: {}".format(file))
            res_text, res_srt, state = audio_clipper.video_recog(file, sd_switch)
        total_srt_file = output_dir + '/total.srt'
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
            logging.warning("Write total subtitile to {}".format(total_srt_file))
        write_state(output_dir, state)
        logging.warning("Recognition successed. You can copy the text segment from below and use stage 2.")
        print(res_text)
    if stage == 2:
        audio_clipper = VideoClipper(None)
        if mode == 'audio':
            state = load_state(output_dir)
            wav, sr = librosa.load(file, sr=16000)
            state['audio_input'] = (sr, wav)
            (sr, audio), message, srt_clip = audio_clipper.clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            if output_file is None:
                output_file = output_dir + '/result.wav'
            clip_srt_file = output_file[:-3] + 'srt'
            logging.warning(message)
            sf.write(output_file, audio, 16000)
            assert output_file.endswith('.wav'), "output_file must ends with '.wav'"
            logging.warning("Save clipped wav file to {}".format(output_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitile to {}".format(clip_srt_file))
        if mode == 'video':
            state = load_state(output_dir)
            state['vedio_filename'] = file
            if output_file is None:
                state['clip_video_file'] = file[:-4] + '_clip.mp4'
            else:
                state['clip_video_file'] = output_file
            clip_srt_file = state['clip_video_file'][:-3] + 'srt'
            state['video'] = mpy.VideoFileClip(file)
            clip_video_file, message, srt_clip = audio_clipper.video_clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            logging.warning("Clipping Log: {}".format(message))
            logging.warning("Save clipped mp4 file to {}".format(clip_video_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitile to {}".format(clip_srt_file))


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    runner(**kwargs)


if __name__ == '__main__':
    main()