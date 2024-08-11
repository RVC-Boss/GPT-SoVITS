<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
Güçlü Birkaç Örnekli Ses Dönüştürme ve Metinden Konuşmaya Web Arayüzü.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)
[![Discord](https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/dnrgs5GHfG)

[**English**](../../README.md) | [**中文简体**](../cn/README.md) | [**日本語**](../ja/README.md) | [**한국어**](../ko/README.md) | **Türkçe**

</div>

---

## Özellikler:

1. **Sıfır Örnekli Metinden Konuşmaya:** 5 saniyelik bir vokal örneği girin ve anında metinden konuşmaya dönüşümünü deneyimleyin.

2. **Birkaç Örnekli Metinden Konuşmaya:** Daha iyi ses benzerliği ve gerçekçiliği için modeli yalnızca 1 dakikalık eğitim verisiyle ince ayarlayın.

3. **Çapraz Dil Desteği:** Eğitim veri setinden farklı dillerde çıkarım, şu anda İngilizce, Japonca ve Çinceyi destekliyor.

4. **Web Arayüzü Araçları:** Entegre araçlar arasında vokal eşliğinde ayırma, otomatik eğitim seti segmentasyonu, Çince ASR ve metin etiketleme bulunur ve yeni başlayanların eğitim veri setleri ve GPT/SoVITS modelleri oluşturmalarına yardımcı olur.

**[Demo videomuzu](https://www.bilibili.com/video/BV12g4y1m7Uw) buradan izleyin!**

Görünmeyen konuşmacılar birkaç örnekli ince ayar demosu:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**Kullanıcı Kılavuzu: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## Kurulum

### Test Edilmiş Ortamlar

- Python 3.9, PyTorch 2.0.1, CUDA 11
- Python 3.10.13, PyTorch 2.1.2, CUDA 12.3
- Python 3.9, PyTorch 2.2.2, macOS 14.4.1 (Apple silikon)
- Python 3.9, PyTorch 2.2.2, CPU cihazları

_Not: numba==0.56.4, py<3.11 gerektirir_

### Windows

Eğer bir Windows kullanıcısıysanız (win>=10 ile test edilmiştir), [entegre paketi indirin](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true) ve _go-webui.bat_ dosyasına çift tıklayarak GPT-SoVITS-WebUI'yi başlatın.

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**Not: Mac'lerde GPU'larla eğitilen modeller, diğer cihazlarda eğitilenlere göre önemli ölçüde daha düşük kalitede sonuç verir, bu nedenle geçici olarak CPU'lar kullanıyoruz.**

1. `xcode-select --install` komutunu çalıştırarak Xcode komut satırı araçlarını yükleyin.
2. FFmpeg'i yüklemek için `brew install ffmpeg` komutunu çalıştırın.
3. Aşağıdaki komutları çalıştırarak programı yükleyin:

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
```

### El ile Yükleme

#### Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

#### FFmpeg'i Yükleme

##### Conda Kullanıcıları

```bash
conda install ffmpeg
```

##### Ubuntu/Debian Kullanıcıları

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### Windows Kullanıcıları

[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) ve [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) dosyalarını indirin ve GPT-SoVITS kök dizinine yerleştirin.

##### MacOS Kullanıcıları
```bash
brew install ffmpeg
```

### Docker Kullanarak

#### docker-compose.yaml yapılandırması

0. Görüntü etiketleri hakkında: Kod tabanındaki hızlı güncellemeler ve görüntüleri paketleme ve test etme işleminin yavaş olması nedeniyle, lütfen şu anda paketlenmiş en son görüntüleri kontrol etmek için [Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits) adresini kontrol edin ve durumunuza göre seçim yapın veya alternatif olarak, kendi ihtiyaçlarınıza göre bir Dockerfile kullanarak yerel olarak oluşturun.
1. Ortam Değişkenleri：

- is_half: Yarım hassasiyet/çift hassasiyeti kontrol eder. Bu genellikle "SSL çıkarma" adımı sırasında 4-cnhubert/5-wav32k dizinleri altındaki içeriğin doğru şekilde oluşturulmamasının nedenidir. Gerçek durumunuza göre True veya False olarak ayarlayın.

2. Birim Yapılandırması，Kapsayıcı içindeki uygulamanın kök dizini /workspace olarak ayarlanmıştır. Varsayılan docker-compose.yaml, içerik yükleme/indirme için bazı pratik örnekler listeler.
3. shm_size： Windows üzerinde Docker Desktop için varsayılan kullanılabilir bellek çok küçüktür, bu da anormal işlemlere neden olabilir. Kendi durumunuza göre ayarlayın.
4. Dağıtım bölümü altında, GPU ile ilgili ayarlar sisteminize ve gerçek koşullara göre dikkatlice ayarlanmalıdır.

#### docker compose ile çalıştırma

```
docker compose -f "docker-compose.yaml" up -d
```

#### docker komutu ile çalıştırma

Yukarıdaki gibi, ilgili parametreleri gerçek durumunuza göre değiştirin, ardından aşağıdaki komutu çalıştırın:

```
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## Önceden Eğitilmiş Modeller

Önceden eğitilmiş modelleri [GPT-SoVITS Modelleri](https://huggingface.co/lj1995/GPT-SoVITS) adresinden indirin ve `GPT_SoVITS/pretrained_models` dizinine yerleştirin.

UVR5 (Vokal/Eşlik Ayırma ve Yankı Giderme, ayrıca) için, modelleri [UVR5 Ağırlıkları](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) adresinden indirin ve `tools/uvr5/uvr5_weights` dizinine yerleştirin.

Çince ASR (ayrıca) için, modelleri [Damo ASR Modeli](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Modeli](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), ve [Damo Punc Modeli](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) adreslerinden indirin ve `tools/asr/models` dizinine yerleştirin.

İngilizce veya Japonca ASR (ayrıca) için, modelleri [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) adresinden indirin ve `tools/asr/models` dizinine yerleştirin. Ayrıca, [diğer modeller](https://huggingface.co/Systran) daha küçük disk alanı kaplamasıyla benzer etkiye sahip olabilir.

## Veri Seti Formatı

TTS açıklama .list dosya formatı:

```
vocal_path|speaker_name|language|text
```

Dil sözlüğü:

- 'zh': Çince
- 'ja': Japonca
- 'en': İngilizce

Örnek:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

## Yapılacaklar Listesi

- [ ] **Yüksek Öncelikli:**

  - [x] Japonca ve İngilizceye yerelleştirme.
  - [x] Kullanıcı kılavuzu.
  - [x] Japonca ve İngilizce veri seti ince ayar eğitimi.

- [ ] **Özellikler:**
  - [ ] Sıfır örnekli ses dönüştürme (5s) / birkaç örnekli ses dönüştürme (1dk).
  - [ ] Metinden konuşmaya konuşma hızı kontrolü.
  - [ ] Gelişmiş metinden konuşmaya duygu kontrolü.
  - [ ] SoVITS token girdilerini kelime dağarcığı olasılık dağılımına değiştirme denemesi.
  - [ ] İngilizce ve Japonca metin ön ucunu iyileştirme.
  - [ ] Küçük ve büyük boyutlu metinden konuşmaya modelleri geliştirme.
  - [x] Colab betikleri.
  - [ ] Eğitim veri setini genişletmeyi dene (2k saat -> 10k saat).
  - [ ] daha iyi sovits temel modeli (geliştirilmiş ses kalitesi)
  - [ ] model karışımı

## (Ekstra) Komut satırından çalıştırma yöntemi
UVR5 için Web Arayüzünü açmak için komut satırını kullanın
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
Bir tarayıcı açamıyorsanız, UVR işleme için aşağıdaki formatı izleyin,Bu ses işleme için mdxnet kullanıyor
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision 
```
Veri setinin ses segmentasyonu komut satırı kullanılarak bu şekilde yapılır
```
python audio_slicer.py \
    --input_path "<orijinal_ses_dosyası_veya_dizininin_yolu>" \
    --output_root "<alt_bölümlere_ayrılmış_ses_kliplerinin_kaydedileceği_dizin>" \
    --threshold <ses_eşiği> \
    --min_length <her_bir_alt_klibin_minimum_süresi> \
    --min_interval <bitişik_alt_klipler_arasındaki_en_kısa_zaman_aralığı> 
    --hop_size <ses_eğrisini_hesaplamak_için_adım_boyutu>
```
Veri seti ASR işleme komut satırı kullanılarak bu şekilde yapılır (Yalnızca Çince)
```
python tools/asr/funasr_asr.py -i <girdi> -o <çıktı>
```
ASR işleme Faster_Whisper aracılığıyla gerçekleştirilir (Çince dışındaki ASR işaretleme)

(İlerleme çubukları yok, GPU performansı zaman gecikmelerine neden olabilir)
```
python ./tools/asr/fasterwhisper_asr.py -i <girdi> -o <çıktı> -l <dil>
```
Özel bir liste kaydetme yolu etkinleştirildi

## Katkı Verenler

Özellikle aşağıdaki projelere ve katkıda bulunanlara teşekkür ederiz:

### Teorik Araştırma
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
### Önceden Eğitilmiş Modeller
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
### Tahmin İçin Metin Ön Ucu
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [LangSegment](https://github.com/juntaosun/LangSegment)
### WebUI Araçları
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
  
## Tüm katkıda bulunanlara çabaları için teşekkürler

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>