#

<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
Güçlü Birkaç Örnekli Ses Dönüştürme ve Metinden Konuşmaya Web Arayüzü.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<a href="https://trendshift.io/repositories/7033" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7033" alt="RVC-Boss%2FGPT-SoVITS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![GitHub release](https://img.shields.io/github/v/release/RVC-Boss/gpt-sovits?style=for-the-badge&logo=github)](https://github.com/RVC-Boss/gpt-sovits/releases)

[![Train In Colab](https://img.shields.io/badge/Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/Colab-WebUI.ipynb)
[![Huggingface](https://img.shields.io/badge/免费在线体验-free_online_demo-yellow.svg?style=for-the-badge&logo=huggingface)](https://lj1995-gpt-sovits-proplus.hf.space/)
[![Image Size](https://img.shields.io/docker/image-size/xxxxrt666/gpt-sovits/latest?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xxxxrt666/gpt-sovits)

[![简体中文](https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
[![English](https://img.shields.io/badge/English-Read%20Docs-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://rentry.co/GPT-SoVITS-guide#/)
[![Change Log](https://img.shields.io/badge/Change%20Log-View%20Updates-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/en/Changelog_EN.md)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)

[**English**](../../README.md) | [**中文简体**](../cn/README.md) | [**日本語**](../ja/README.md) | [**한국어**](../ko/README.md) | **Türkçe**

</div>

---

## Özellikler

1. **Sıfır Örnekli Metinden Konuşmaya:** 5 saniyelik bir vokal örneği girin ve anında metinden konuşmaya dönüşümünü deneyimleyin.

2. **Birkaç Örnekli Metinden Konuşmaya:** Daha iyi ses benzerliği ve gerçekçiliği için modeli yalnızca 1 dakikalık eğitim verisiyle ince ayarlayın.

3. **Çapraz Dil Desteği:** Eğitim veri setinden farklı dillerde çıkarım, şu anda İngilizce, Japonca, Çince, Kantonca ve Koreceyi destekliyor.

4. **Web Arayüzü Araçları:** Entegre araçlar arasında vokal eşliğinde ayırma, otomatik eğitim seti segmentasyonu, Çince ASR ve metin etiketleme bulunur ve yeni başlayanların eğitim veri setleri ve GPT/SoVITS modelleri oluşturmalarına yardımcı olur.

**[Demo videomuzu](https://www.bilibili.com/video/BV12g4y1m7Uw) buradan izleyin!**

Görünmeyen konuşmacılar birkaç örnekli ince ayar demosu:

<https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb>

**Kullanıcı Kılavuzu: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

<div align="center">

## Kurulum

### Test Edilmiş Ortamlar

| Python Version | PyTorch Version | Device        |
| -------------- | --------------- | ------------- |
| Python 3.10    | PyTorch 2.8.0   | CUDA 12.6     |
| Python 3.11    | PyTorch 2.9.0   | CUDA 12.6     |
| Python 3.11    | PyTorch 2.7.0   | CUDA 12.8     |
| Python 3.11    | PyTorch 2.10.0  | CUDA 12.8     |
| Python 3.10    | PyTorch 2.8.0   | Apple silicon |
| Python 3.11    | PyTorch 2.9.0   | Apple silicon |
| Python 3.12    | PyTorch 2.10.0  | Apple silicon |
| Python 3.10    | PyTorch 2.9.0   | CPU           |

</div>

### Windows

Eğer bir Windows kullanıcısıysanız (win>=10 ile test edilmiştir), [entegre paketi indirin](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main) ve `go-webui.bat` dosyasına çift tıklayarak GPT-SoVITS-WebUI'yi başlatın.

Aşağıdaki komutları çalıştırarak programı yükleyin:

```pwsh
conda create -n GPTSoVITS python=3.11
conda activate GPTSoVITS
pwsh -F install.ps1 --help
```

### Linux

Aşağıdaki komutları çalıştırarak programı yükleyin:

```bash
conda create -n GPTSoVITS python=3.11
conda activate GPTSoVITS
bash install.sh --help
```

### macOS

**Not: Mac'lerde GPU'larla eğitilen modeller, diğer cihazlarda eğitilenlere göre önemli ölçüde daha düşük kalitede sonuç verir, bu nedenle geçici olarak CPU'lar kullanıyoruz.**

Aşağıdaki komutları çalıştırarak programı yükleyin:

```bash
conda create -n GPTSoVITS python=3.11
conda activate GPTSoVITS
bash install.sh --help
```

### El ile Yükleme

Aşağıdaki komutları çalıştırarak programı yükleyin:

```bash
conda create -n GPTSoVITS python=3.11
conda activate GPTSoVITS
conda install uv ffmpeg -c conda-forge

uv export --extra main -o pylock.toml -q --extra [mlx|cu126|cu128|rocm|cpu]
uv pip sync pylock.toml --no-break-system-packages --preview-features pylock
uv pip install ".[flash-attn]"
```

### Önceden Eğitilmiş Modeller

**Eğer `install.sh` başarıyla çalıştırılırsa, No.1,2,3 adımını atlayabilirsiniz.**

1. [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) üzerinden önceden eğitilmiş modelleri indirip `GPT_SoVITS/pretrained_models` dizinine yerleştirin.

2. [G2PWModel.zip(HF)](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)| [G2PWModel.zip(ModelScope)](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) üzerinden modeli indirip sıkıştırmayı açın ve `G2PWModel` olarak yeniden adlandırın, ardından `GPT_SoVITS/text` dizinine yerleştirin. (Sadece Çince TTS için)

3. UVR5 (Vokal/Enstrümantal Ayrımı & Yankı Giderme) için, [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) üzerinden modelleri indirip `gsv_tools/uvr5/uvr5_weights` dizinine yerleştirin.
   - UVR5'te bs_roformer veya mel_band_roformer modellerini kullanıyorsanız, modeli ve ilgili yapılandırma dosyasını manuel olarak indirip `gsv_tools/UVR5/UVR5_weights` klasörüne yerleştirebilirsiniz. **Model dosyası ve yapılandırma dosyasının adı, uzantı dışında aynı olmalıdır**. Ayrıca, model ve yapılandırma dosyasının adlarında **"roformer"** kelimesi yer almalıdır, böylece roformer sınıfındaki bir model olarak tanınır.

   - Model adı ve yapılandırma dosyası adı içinde **doğrudan model tipini belirtmek önerilir**. Örneğin: mel_mand_roformer, bs_roformer. Belirtilmezse, yapılandırma dosyasından özellikler karşılaştırılarak model tipi belirlenir. Örneğin, `bs_roformer_ep_368_sdr_12.9628.ckpt` modeli ve karşılık gelen yapılandırma dosyası `bs_roformer_ep_368_sdr_12.9628.yaml` bir çifttir. Aynı şekilde, `kim_mel_band_roformer.ckpt` ve `kim_mel_band_roformer.yaml` da bir çifttir.

4. Çince ASR için, [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) ve [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) üzerinden modelleri indirip `gsv_tools/asr/models` dizinine yerleştirin.

5. İngilizce veya Japonca ASR için, [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) üzerinden modeli indirip `gsv_tools/asr/models` dizinine yerleştirin. Ayrıca, [diğer modeller](https://huggingface.co/Systran) benzer bir etki yaratabilir ve daha az disk alanı kaplayabilir.

<div align="center">

## GPT-SoVITS Çalıştırma (Docker Kullanarak)

</div>

### Docker İmajı Seçimi

Kod tabanı hızla geliştiği halde Docker imajları daha yavaş yayınlandığı için lütfen şu adımları izleyin:

- En güncel kullanılabilir imaj etiketlerini görmek için [Docker Hub](https://hub.docker.com/r/xxxxrt666/gpt-sovits) adresini kontrol edin
- Ortamınıza uygun bir imaj etiketi seçin
- `Lite`, Docker imajında ASR modelleri ve UVR5 modellerinin **bulunmadığı** anlamına gelir. UVR5 modellerini manuel olarak indirebilirsiniz; ASR modelleri ise gerektiğinde program tarafından otomatik olarak indirilir
- Docker Compose sırasında, uygun mimariye (amd64 veya arm64) ait imaj otomatik olarak indirilir
- Docker Compose, mevcut dizindeki **tüm dosyaları** bağlayacaktır. Docker imajını kullanmadan önce lütfen proje kök dizinine geçin ve **en son kodu çekin**
- Opsiyonel: En güncel değişiklikleri almak için, sağlanan Dockerfile ile yerel olarak imajı kendiniz oluşturabilirsiniz

### Ortam Değişkenleri

- `is_half`: Yarı hassasiyet (fp16) kullanımını kontrol eder. GPU’nuz destekliyorsa, belleği azaltmak için `true` olarak ayarlayın.

### Paylaşılan Bellek Yapılandırması

Windows (Docker Desktop) ortamında, varsayılan paylaşılan bellek boyutu düşüktür ve bu beklenmedik hatalara neden olabilir. Sistem belleğinize göre Docker Compose dosyasındaki `shm_size` değerini (örneğin `16g`) artırmanız önerilir.

### Servis Seçimi

`docker-compose.yaml` dosyasında iki tür servis tanımlanmıştır:

- `GPT-SoVITS-CU126` ve `GPT-SoVITS-CU128`: Tüm özellikleri içeren tam sürüm.
- `GPT-SoVITS-CU126-Lite` ve `GPT-SoVITS-CU128-Lite`: Daha az bağımlılığa ve sınırlı işlevselliğe sahip hafif sürüm.

Belirli bir servisi Docker Compose ile çalıştırmak için şu komutu kullanın:

```bash
docker compose run --service-ports <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128>
```

### Docker İmajını Yerel Olarak Oluşturma

Docker imajını kendiniz oluşturmak isterseniz şu komutu kullanın:

```bash
bash docker_build.sh --cuda <12.6|12.8> [--lite]
```

### Çalışan Konteynere Erişim (Bash Shell)

Konteyner arka planda çalışırken, aşağıdaki komutla içine girebilirsiniz:

```bash
docker exec -it <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128> bash
```

<div align="center">

## Veri Seti Formatı

</div>

TTS açıklama .list dosya formatı:

```text

vocal_path|speaker_name|language|text

```

Dil sözlüğü:

- 'zh': Çince
- 'ja': Japonca
- 'en': İngilizce
- 'ko': Korece
- 'yue': Kantonca

Örnek:

```text

D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.

```

<div align="center">

## İnce Ayar ve Çıkarım

</div>

### WebUI'yi Açın

#### Entegre Paket Kullanıcıları

`go-webui.bat` dosyasına çift tıklayın veya `go-webui.ps1` kullanın.

#### Diğerleri

```bash
python webui.py <dil(isteğe bağlı)>
```

### İnce Ayar

#### Yol Otomatik Doldurma artık destekleniyor

1. Ses yolunu doldurun
2. Sesi küçük parçalara ayırın
3. Gürültü azaltma (isteğe bağlı)
4. ASR
5. ASR transkripsiyonlarını düzeltin
6. Bir sonraki sekmeye geçin ve modeli ince ayar yapın

### Çıkarım WebUI'sini Açın

#### Entegre Paket Kullanıcıları

`go-webui.bat` dosyasına çift tıklayın veya `go-webui.ps1` kullanın, ardından çıkarım webui'sini `1-GPT-SoVITS-TTS/1C-inference` adresinde açın.

#### Diğerleri

```bash
python GPT_SoVITS/inference_webui.py <dil(isteğe bağlı)>
```

VEYA

```bash
python webui.py
```

ardından çıkarım webui'sini `1-GPT-SoVITS-TTS/1C-inference` adresinde açın.

<div align="center">

## V2 Sürüm Notları

</div>

Yeni Özellikler:

1. Korece ve Kantonca destekler

2. Optimize edilmiş metin ön yüzü

3. Önceden eğitilmiş model 2k saatten 5k saate kadar genişletildi

4. Düşük kaliteli referans sesler için geliştirilmiş sentez kalitesi

   [detaylar burada](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

<div align="center">

## V3 Sürüm Notları

</div>

Yeni Özellikler:

1. **Tını benzerliği** daha yüksek olup, hedef konuşmacıyı yakınsamak için daha az eğitim verisi gerekmektedir (tını benzerliği, base model doğrudan kullanılacak şekilde fine-tuning yapılmadan önemli ölçüde iyileştirilmiştir).

2. GPT modeli daha **kararlı** hale geldi, tekrarlar ve atlamalar azaldı ve **daha zengin duygusal ifadeler** ile konuşma üretmek daha kolay hale geldi.

   [daha fazla detay](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

<div align="center">

## V4 Sürüm Notları

</div>

Yeni Özellikler:

1. **V4, V3'te görülen non-integer upsample işleminden kaynaklanan metalik ses sorununu düzeltti ve sesin boğuklaşmasını önlemek için doğrudan 48kHz ses çıktısı sunar (V3 sadece 24kHz destekler)**. Yazar, V4'ün V3'ün yerine geçebileceğini belirtmiştir ancak daha fazla test yapılması gerekmektedir.
   [Daha fazla bilgi](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)>)

<div align="center">

## V2Pro Sürüm Notları

</div>

Yeni Özellikler:

1. **V2 ile karşılaştırıldığında biraz daha yüksek VRAM kullanımı sağlar ancak V4'ten daha iyi performans gösterir; aynı donanım maliyeti ve hız avantajını korur**.
   [Daha fazla bilgi](<https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7)>)

2. V1/V2 ve V2Pro serisi benzer özelliklere sahipken, V3/V4 de yakın işlevleri paylaşır. Ortalama kalite düşük olan eğitim setleriyle V1/V2/V2Pro iyi sonuçlar verebilir ama V3/V4 veremez. Ayrıca, V3/V4’ün ürettiği ses tonu genel eğitim setine değil, referans ses örneğine daha çok benzemektedir.

<div align="center">

## Yapılacaklar Listesi

</div>

- [x] **Yüksek Öncelikli:**
  - [x] Japonca ve İngilizceye yerelleştirme.
  - [x] Kullanıcı kılavuzu.
  - [x] Japonca ve İngilizce veri seti ince ayar eğitimi.

- [ ] **Özellikler:**
  - [x] Sıfır örnekli ses dönüştürme (5s) / birkaç örnekli ses dönüştürme (1dk).
  - [x] Metinden konuşmaya konuşma hızı kontrolü.
  - [ ] ~~Gelişmiş metinden konuşmaya duygu kontrolü.~~
  - [ ] SoVITS token girdilerini kelime dağarcığı olasılık dağılımına değiştirme denemesi.
  - [x] İngilizce ve Japonca metin ön ucunu iyileştirme.
  - [ ] Küçük ve büyük boyutlu metinden konuşmaya modelleri geliştirme.
  - [x] Colab betikleri.
  - [x] Eğitim veri setini genişletmeyi dene (2k saat -> 10k saat).
  - [x] daha iyi sovits temel modeli (geliştirilmiş ses kalitesi)
  - [ ] model karışımı

<div align="center">

## Katkı Verenler

</div>

Özellikle aşağıdaki projelere ve katkıda bulunanlara teşekkür ederiz:

### Teorik Araştırma

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/d3df50503b36314a964f66cac1af1e19e95bcfa3/fish_speech/models/text2semantic/inference.py#L81)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### Önceden Eğitilmiş Modeller

- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### Tahmin İçin Metin Ön Ucu

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### WebUI Araçları

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

@Naozumi520'ye Kantonca eğitim setini sağladığı ve Kantonca ile ilgili bilgiler konusunda rehberlik ettiği için minnettarım.

<div align="center">

## Tüm katkıda bulunanlara çabaları için teşekkürler

</div>

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" alt=""/>
</a>
