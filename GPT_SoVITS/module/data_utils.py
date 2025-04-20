import os
import random
import traceback
import torch
import torch.utils.data
from tqdm import tqdm

from module.mel_processing import spectrogram_torch, spec_to_mel_torch
from text import cleaned_text_to_sequence
import torch.nn.functional as F
from tools.my_utils import load_audio

version = os.environ.get("version", None)


# ZeroDivisionError fixed by Tybost (https://github.com/RVC-Boss/GPT-SoVITS/issues/79)
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, hparams, val=False):
        exp_dir = hparams.exp_dir
        self.path2 = "%s/2-name2text.txt" % exp_dir
        self.path4 = "%s/4-cnhubert" % exp_dir
        self.path5 = "%s/5-wav32k" % exp_dir
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path4)
        assert os.path.exists(self.path5)
        names4 = set([name[:-3] for name in list(os.listdir(self.path4))])  # 去除.pt后缀
        names5 = set(os.listdir(self.path5))
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1]]

        self.audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
        tmp = self.audiopaths_sid_text
        leng = len(tmp)
        min_num = 100
        if leng < min_num:
            self.audiopaths_sid_text = []
            for _ in range(max(2, int(min_num / leng))):
                self.audiopaths_sid_text += tmp
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        print("phoneme_data_len:", len(self.phoneme_data.keys()))
        print("wav_data_len:", len(self.audiopaths_sid_text))

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0
        for audiopath in tqdm(self.audiopaths_sid_text):
            try:
                phoneme = self.phoneme_data[audiopath][0]
                phoneme = phoneme.split(" ")
                phoneme_ids = cleaned_text_to_sequence(phoneme, version)
            except Exception:
                print(f"{audiopath} not in self.phoneme_data !")
                skipped_phone += 1
                continue

            size = os.path.getsize("%s/%s" % (self.path5, audiopath))
            duration = size / self.sampling_rate / 2

            if duration == 0:
                print(f"Zero duration for {audiopath}, skipping...")
                skipped_dur += 1
                continue

            if 54 > duration > 0.6 or self.val:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * self.hop_length))
            else:
                skipped_dur += 1
                continue

        print("skipped_phone: ", skipped_phone, ", skipped_dur: ", skipped_dur)
        print("total left: ", len(audiopaths_sid_text_new))
        assert len(audiopaths_sid_text_new) > 1  # 至少能凑够batch size，这里todo
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        try:
            spec, wav = self.get_audio("%s/%s" % (self.path5, audiopath))
            with torch.no_grad():
                ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
                if ssl.shape[-1] != spec.shape[-1]:
                    typee = ssl.dtype
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)
                ssl.requires_grad = False
        except:
            traceback.print_exc()
            spec = torch.zeros(1025, 100)
            wav = torch.zeros(1, 100 * self.hop_length)
            ssl = torch.zeros(1, 768, 100)
            text = text[-1:]
            print("load audio or ssl error!!!!!!", audiopath)
        return (ssl, spec, wav, text)

    def get_audio(self, filename):
        audio_array = load_audio(filename, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False
        )
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        # with torch.no_grad():
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def random_slice(self, ssl, wav, mel):
        assert abs(ssl.shape[-1] - wav.shape[-1] // self.hop_length) < 3, ("first", ssl.shape, wav.shape)

        len_mel = mel.shape[1]
        if self.val:
            reference_mel = mel[:, : len_mel // 3]
            return reference_mel, ssl, wav, mel
        dir = random.randint(0, 1)
        sep_point = random.randint(int(len_mel // 3), int(len_mel // 3 * 2))

        if dir == 0:
            reference_mel = mel[:, :sep_point]
            ssl = ssl[:, :, sep_point:]
            wav2 = wav[:, sep_point * self.hop_length :]
            mel = mel[:, sep_point:]
        else:
            reference_mel = mel[:, sep_point:]
            ssl = ssl[:, :, :sep_point]
            wav2 = wav[:, : sep_point * self.hop_length]
            mel = mel[:, :sep_point]

        assert abs(ssl.shape[-1] - wav2.shape[-1] // self.hop_length) < 3, (
            ssl.shape,
            wav.shape,
            wav2.shape,
            mel.shape,
            sep_point,
            self.hop_length,
            sep_point * self.hop_length,
            dir,
        )
        return reference_mel, ssl, wav2, mel


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)

        max_ssl_len = max([x[0].size(2) for x in batch])
        max_ssl_len = int(2 * ((max_ssl_len // 2) + 1))
        max_spec_len = max([x[1].size(1) for x in batch])
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([x[3].size(0) for x in batch])

        ssl_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        ssl_padded = torch.FloatTensor(len(batch), batch[0][0].size(1), max_ssl_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)

        spec_padded.zero_()
        wav_padded.zero_()
        ssl_padded.zero_()
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            ssl = row[0]
            ssl_padded[i, :, : ssl.size(2)] = ssl[0, :, :]
            ssl_lengths[i] = ssl.size(2)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            text = row[3]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

        return ssl_padded, ssl_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, text_padded, text_lengths


class TextAudioSpeakerLoaderV3(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, hparams, val=False):
        exp_dir = hparams.exp_dir
        self.path2 = "%s/2-name2text.txt" % exp_dir
        self.path4 = "%s/4-cnhubert" % exp_dir
        self.path5 = "%s/5-wav32k" % exp_dir
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path4)
        assert os.path.exists(self.path5)
        names4 = set([name[:-3] for name in list(os.listdir(self.path4))])  # 去除.pt后缀
        names5 = set(os.listdir(self.path5))
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1]]

        self.audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
        tmp = self.audiopaths_sid_text
        leng = len(tmp)
        min_num = 100
        if leng < min_num:
            self.audiopaths_sid_text = []
            for _ in range(max(2, int(min_num / leng))):
                self.audiopaths_sid_text += tmp
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        print("phoneme_data_len:", len(self.phoneme_data.keys()))
        print("wav_data_len:", len(self.audiopaths_sid_text))

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0
        for audiopath in tqdm(self.audiopaths_sid_text):
            try:
                phoneme = self.phoneme_data[audiopath][0]
                phoneme = phoneme.split(" ")
                phoneme_ids = cleaned_text_to_sequence(phoneme, version)
            except Exception:
                print(f"{audiopath} not in self.phoneme_data !")
                skipped_phone += 1
                continue

            size = os.path.getsize("%s/%s" % (self.path5, audiopath))
            duration = size / self.sampling_rate / 2

            if duration == 0:
                print(f"Zero duration for {audiopath}, skipping...")
                skipped_dur += 1
                continue

            if 54 > duration > 0.6 or self.val:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * self.hop_length))
            else:
                skipped_dur += 1
                continue

        print("skipped_phone: ", skipped_phone, ", skipped_dur: ", skipped_dur)
        print("total left: ", len(audiopaths_sid_text_new))
        assert len(audiopaths_sid_text_new) > 1  # 至少能凑够batch size，这里todo
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.spec_min = -12
        self.spec_max = 2

        self.filter_length_mel = self.win_length_mel = 1024
        self.hop_length_mel = 256
        self.n_mel_channels = 100
        self.sampling_rate_mel = 24000
        self.mel_fmin = 0
        self.mel_fmax = None

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        try:
            spec, mel = self.get_audio("%s/%s" % (self.path5, audiopath))
            with torch.no_grad():
                ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
                if ssl.shape[-1] != spec.shape[-1]:
                    typee = ssl.dtype
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)
                ssl.requires_grad = False
        except:
            traceback.print_exc()
            mel = torch.zeros(100, 180)
            # wav = torch.zeros(1, 96 * self.hop_length)
            spec = torch.zeros(1025, 96)
            ssl = torch.zeros(1, 768, 96)
            text = text[-1:]
            print("load audio or ssl error!!!!!!", audiopath)
        return (ssl, spec, mel, text)

    def get_audio(self, filename):
        audio_array = load_audio(filename, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        audio_array24 = load_audio(
            filename, 24000
        )  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768######这里可以用GPU重采样加速
        audio24 = torch.FloatTensor(audio_array24)  # /32768
        audio_norm24 = audio24
        audio_norm24 = audio_norm24.unsqueeze(0)

        spec = spectrogram_torch(
            audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False
        )
        spec = torch.squeeze(spec, 0)

        spec1 = spectrogram_torch(
            audio_norm24,
            self.filter_length_mel,
            self.sampling_rate_mel,
            self.hop_length_mel,
            self.win_length_mel,
            center=False,
        )
        mel = spec_to_mel_torch(
            spec1, self.filter_length_mel, self.n_mel_channels, self.sampling_rate_mel, self.mel_fmin, self.mel_fmax
        )
        mel = torch.squeeze(mel, 0)
        mel = self.norm_spec(mel)
        # print(1111111,spec.shape,mel.shape)
        return spec, mel

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        # with torch.no_grad():
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollateV3:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # ssl, spec, wav,mel, text
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
        # (ssl, spec,mel, text)
        max_ssl_len = max([x[0].size(2) for x in batch])

        max_ssl_len1 = int(8 * ((max_ssl_len // 8) + 1))
        max_ssl_len = int(2 * ((max_ssl_len // 2) + 1))

        # max_ssl_len = int(8 * ((max_ssl_len // 8) + 1))
        # max_ssl_len1=max_ssl_len

        max_spec_len = max([x[1].size(1) for x in batch])
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        # max_wav_len = max([x[2].size(1) for x in batch])

        max_text_len = max([x[3].size(0) for x in batch])
        max_mel_len = int(max_ssl_len1 * 1.25 * 1.5)  ###24000/256,32000/640=16000/320

        ssl_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))
        # wav_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_mel_len)
        ssl_padded = torch.FloatTensor(len(batch), batch[0][0].size(1), max_ssl_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        # wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        spec_padded.zero_()
        mel_padded.zero_()
        ssl_padded.zero_()
        text_padded.zero_()
        # wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            # ssl, spec, wav,mel, text
            ssl = row[0]
            ssl_padded[i, :, : ssl.size(2)] = ssl[0, :, :]
            ssl_lengths[i] = ssl.size(2)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            # wav = row[2]
            # wav_padded[i, :, :wav.size(1)] = wav
            # wav_lengths[i] = wav.size(1)

            mel = row[2]
            mel_padded[i, :, : mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            text = row[3]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

        # return ssl_padded, spec_padded,mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths, wav_padded, wav_lengths,mel_lengths
        return ssl_padded, spec_padded, mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths, mel_lengths

class TextAudioSpeakerLoaderV4(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, hparams, val=False):
        exp_dir = hparams.exp_dir
        self.path2 = "%s/2-name2text.txt" % exp_dir
        self.path4 = "%s/4-cnhubert" % exp_dir
        self.path5 = "%s/5-wav32k" % exp_dir
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path4)
        assert os.path.exists(self.path5)
        names4 = set([name[:-3] for name in list(os.listdir(self.path4))])  # 去除.pt后缀
        names5 = set(os.listdir(self.path5))
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1]]

        self.audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
        tmp = self.audiopaths_sid_text
        leng = len(tmp)
        min_num = 100
        if leng < min_num:
            self.audiopaths_sid_text = []
            for _ in range(max(2, int(min_num / leng))):
                self.audiopaths_sid_text += tmp
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        print("phoneme_data_len:", len(self.phoneme_data.keys()))
        print("wav_data_len:", len(self.audiopaths_sid_text))

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0
        for audiopath in tqdm(self.audiopaths_sid_text):
            try:
                phoneme = self.phoneme_data[audiopath][0]
                phoneme = phoneme.split(" ")
                phoneme_ids = cleaned_text_to_sequence(phoneme, version)
            except Exception:
                print(f"{audiopath} not in self.phoneme_data !")
                skipped_phone += 1
                continue

            size = os.path.getsize("%s/%s" % (self.path5, audiopath))
            duration = size / self.sampling_rate / 2

            if duration == 0:
                print(f"Zero duration for {audiopath}, skipping...")
                skipped_dur += 1
                continue

            if 54 > duration > 0.6 or self.val:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * self.hop_length))
            else:
                skipped_dur += 1
                continue

        print("skipped_phone: ", skipped_phone, ", skipped_dur: ", skipped_dur)
        print("total left: ", len(audiopaths_sid_text_new))
        assert len(audiopaths_sid_text_new) > 1  # 至少能凑够batch size，这里todo
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.spec_min = -12
        self.spec_max = 2

        self.filter_length_mel = self.win_length_mel = 1280
        self.hop_length_mel = 320
        self.n_mel_channels = 100
        self.sampling_rate_mel = 32000
        self.mel_fmin = 0
        self.mel_fmax = None

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        try:
            spec, mel = self.get_audio("%s/%s" % (self.path5, audiopath))
            with torch.no_grad():
                ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
                if ssl.shape[-1] != spec.shape[-1]:
                    typee = ssl.dtype
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)
                ssl.requires_grad = False
        except:
            traceback.print_exc()
            mel = torch.zeros(100, 192)
            # wav = torch.zeros(1, 96 * self.hop_length)
            spec = torch.zeros(1025, 96)
            ssl = torch.zeros(1, 768, 96)
            text = text[-1:]
            print("load audio or ssl error!!!!!!", audiopath)
        return (ssl, spec, mel, text)

    def get_audio(self, filename):
        audio_array = load_audio(filename, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False
        )
        spec = torch.squeeze(spec, 0)
        spec1 = spectrogram_torch(audio_norm, 1280,32000, 320, 1280,center=False)
        mel = spec_to_mel_torch(spec1, 1280, 100, 32000, 0, None)
        mel = self.norm_spec(torch.squeeze(mel, 0))
        return spec, mel

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        # with torch.no_grad():
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollateV4:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # ssl, spec, wav,mel, text
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
        # (ssl, spec,mel, text)
        max_ssl_len = max([x[0].size(2) for x in batch])
        max_ssl_len = int(2 * ((max_ssl_len // 2) + 1))
        max_spec_len = max([x[1].size(1) for x in batch])
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        # max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([x[3].size(0) for x in batch])

        ssl_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))
        # wav_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len*2)
        ssl_padded = torch.FloatTensor(len(batch), batch[0][0].size(1), max_ssl_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        # wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        spec_padded.zero_()
        mel_padded.zero_()
        ssl_padded.zero_()
        text_padded.zero_()
        # wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            # ssl, spec, wav,mel, text
            ssl = row[0]
            ssl_padded[i, :, : ssl.size(2)] = ssl[0, :, :]
            ssl_lengths[i] = ssl.size(2)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            # wav = row[2]
            # wav_padded[i, :, :wav.size(1)] = wav
            # wav_lengths[i] = wav.size(1)

            mel = row[2]
            mel_padded[i, :, : mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            text = row[3]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

        # return ssl_padded, spec_padded,mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths, wav_padded, wav_lengths,mel_lengths
        return ssl_padded, spec_padded, mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths, mel_lengths


class TextAudioSpeakerLoaderV3b(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, hparams, val=False):
        exp_dir = hparams.exp_dir
        self.path2 = "%s/2-name2text.txt" % exp_dir
        self.path4 = "%s/4-cnhubert" % exp_dir
        self.path5 = "%s/5-wav32k" % exp_dir
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path4)
        assert os.path.exists(self.path5)
        names4 = set([name[:-3] for name in list(os.listdir(self.path4))])  # 去除.pt后缀
        names5 = set(os.listdir(self.path5))
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1]]

        self.audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
        tmp = self.audiopaths_sid_text
        leng = len(tmp)
        min_num = 100
        if leng < min_num:
            self.audiopaths_sid_text = []
            for _ in range(max(2, int(min_num / leng))):
                self.audiopaths_sid_text += tmp
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        print("phoneme_data_len:", len(self.phoneme_data.keys()))
        print("wav_data_len:", len(self.audiopaths_sid_text))

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0
        for audiopath in tqdm(self.audiopaths_sid_text):
            try:
                phoneme = self.phoneme_data[audiopath][0]
                phoneme = phoneme.split(" ")
                phoneme_ids = cleaned_text_to_sequence(phoneme, version)
            except Exception:
                print(f"{audiopath} not in self.phoneme_data !")
                skipped_phone += 1
                continue

            size = os.path.getsize("%s/%s" % (self.path5, audiopath))
            duration = size / self.sampling_rate / 2

            if duration == 0:
                print(f"Zero duration for {audiopath}, skipping...")
                skipped_dur += 1
                continue

            if 54 > duration > 0.6 or self.val:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * self.hop_length))
            else:
                skipped_dur += 1
                continue

        print("skipped_phone: ", skipped_phone, ", skipped_dur: ", skipped_dur)
        print("total left: ", len(audiopaths_sid_text_new))
        assert len(audiopaths_sid_text_new) > 1  # 至少能凑够batch size，这里todo
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.spec_min = -12
        self.spec_max = 2

        self.filter_length_mel = self.win_length_mel = 1024
        self.hop_length_mel = 256
        self.n_mel_channels = 100
        self.sampling_rate_mel = 24000
        self.mel_fmin = 0
        self.mel_fmax = None

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        try:
            spec, mel, wav = self.get_audio("%s/%s" % (self.path5, audiopath))
            with torch.no_grad():
                ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
                if ssl.shape[-1] != spec.shape[-1]:
                    typee = ssl.dtype
                    ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)
                ssl.requires_grad = False
        except:
            traceback.print_exc()
            mel = torch.zeros(100, 180)
            wav = torch.zeros(1, 96 * self.hop_length)
            spec = torch.zeros(1025, 96)
            ssl = torch.zeros(1, 768, 96)
            text = text[-1:]
            print("load audio or ssl error!!!!!!", audiopath)
        return (ssl, spec, wav, mel, text)

    def get_audio(self, filename):
        audio_array = load_audio(filename, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
        audio = torch.FloatTensor(audio_array)  # /32768
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        audio_array24 = load_audio(
            filename, 24000
        )  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768######这里可以用GPU重采样加速
        audio24 = torch.FloatTensor(audio_array24)  # /32768
        audio_norm24 = audio24
        audio_norm24 = audio_norm24.unsqueeze(0)

        spec = spectrogram_torch(
            audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False
        )
        spec = torch.squeeze(spec, 0)

        spec1 = spectrogram_torch(
            audio_norm24,
            self.filter_length_mel,
            self.sampling_rate_mel,
            self.hop_length_mel,
            self.win_length_mel,
            center=False,
        )
        mel = spec_to_mel_torch(
            spec1, self.filter_length_mel, self.n_mel_channels, self.sampling_rate_mel, self.mel_fmin, self.mel_fmax
        )
        mel = torch.squeeze(mel, 0)
        mel = self.norm_spec(mel)
        # print(1111111,spec.shape,mel.shape)
        return spec, mel, audio_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        # with torch.no_grad():
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollateV3b:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # ssl, spec, wav,mel, text
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
        # (ssl, spec,mel, text)
        max_ssl_len = max([x[0].size(2) for x in batch])

        max_ssl_len1 = int(8 * ((max_ssl_len // 8) + 1))
        max_ssl_len = int(2 * ((max_ssl_len // 2) + 1))

        # max_ssl_len = int(8 * ((max_ssl_len // 8) + 1))
        # max_ssl_len1=max_ssl_len

        max_spec_len = max([x[1].size(1) for x in batch])
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([x[4].size(0) for x in batch])
        max_mel_len = int(max_ssl_len1 * 1.25 * 1.5)  ###24000/256,32000/640=16000/320

        ssl_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][3].size(0), max_mel_len)
        ssl_padded = torch.FloatTensor(len(batch), batch[0][0].size(1), max_ssl_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        spec_padded.zero_()
        mel_padded.zero_()
        ssl_padded.zero_()
        text_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            # ssl, spec, wav,mel, text
            ssl = row[0]
            ssl_padded[i, :, : ssl.size(2)] = ssl[0, :, :]
            ssl_lengths[i] = ssl.size(2)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            mel = row[3]
            mel_padded[i, :, : mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            text = row[4]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

        return (
            ssl_padded,
            spec_padded,
            mel_padded,
            ssl_lengths,
            spec_lengths,
            text_padded,
            text_lengths,
            wav_padded,
            wav_lengths,
            mel_lengths,
        )
        # return ssl_padded, spec_padded,mel_padded, ssl_lengths, spec_lengths, text_padded, text_lengths,mel_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]

            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
