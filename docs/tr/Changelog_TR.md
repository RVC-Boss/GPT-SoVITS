# Güncelleme Günlüğü

## 20240121

1. `config`e `is_share` eklendi. Colab gibi senaryolarda, WebUI'yi halka açık ağa yönlendirmek için bu değeri `True` olarak ayarlayabilirsiniz.
2. WebUI'ye İngilizce sistem çeviri desteği eklendi.
3. `cmd-asr`, FunASR modelinin dahil olup olmadığını otomatik olarak tespit eder; eğer varsayılan dizinde bulunamazsa, ModelScope'dan indirilecektir.
4. [Issue 79](https://github.com/RVC-Boss/GPT-SoVITS/issues/79)de bildirilen SoVITS eğitimindeki ZeroDivisionError'u sıfır uzunlukta örnekleri filtreleyerek düzeltmeye çalıştık.
5. `TEMP` klasöründeki önbelleğe alınmış ses dosyaları ve diğer dosyaları temizledik.
6. Referans sesinin sonunu içeren sentezlenmiş ses sorununu önemli ölçüde azalttık.

## 20240122

1. Aşırı kısa çıktı dosyalarının referans sesini tekrarlamasına neden olan sorun giderildi.
2. İngilizce ve Japonca eğitim için yerel destek test edildi (Japonca eğitim için kök dizinin İngilizce olmayan özel karakterlerden arındırılmış olması gerekir).
3. Ses yolu denetimi iyileştirildi. Yanlış bir giriş yolundan okumaya çalışıldığında, ffmpeg hatası yerine yolun mevcut olmadığını bildirir.

## 20240123

1. Hubert çıkarımının NaN hatalarına neden olup SoVITS/GPT eğitiminde ZeroDivisionError'a yol açtığı sorun çözüldü.
2. İnferans WebUI'de hızlı model değiştirme desteği eklendi.
3. Model dosyası sıralama mantığı optimize edildi.
4. Çince kelime ayrımı için `jieba` `jieba_fast` ile değiştirildi.

## 20240126

1. Çince-İngilizce ve Japonca-İngilizce karışık çıktı metinleri için destek eklendi.
2. Çıktı için isteğe bağlı bir bölme modu eklendi.
3. UVR5'in dizinlerden otomatik olarak çıkmasına neden olan okuma sorununu düzelttik.
4. Çeşitli yeni satır sorunlarını düzelterek çıkarım hatalarını giderdik.
5. Çıkarım WebUI'deki gereksiz günlükleri kaldırdık.
6. Mac'te eğitim ve çıkarım desteği eklendi.
7. Yarım hassasiyeti desteklemeyen GPU'lar için otomatik olarak tek hassasiyet zorlandı; CPU çıkarımında tek hassasiyet uygulandı.

## 20240128

1. Sayıların Çince karakterlere dönüştürülmesiyle ilgili sorunu düzelttik.
2. Cümlelerin başındaki birkaç karakterin yutulması sorununu düzelttik.
3. Mantıksız referans ses uzunluklarını sınırlamalar koyarak hariç tuttuk.
4. GPT eğitiminin kontrol noktalarını kaydetmemesi sorununu düzelttik.
5. Dockerfile'da model indirme sürecini tamamladık.

## 20240129

1. Yarım hassasiyet eğitimi ile ilgili sorun yaşayan 16 serisi gibi GPU'lar için eğitim yapılandırmalarını tek hassasiyete değiştirdik.
2. Mevcut Colab sürümünü test ettik ve güncelledik.
3. Eski sürüm FunASR ile ModelScope FunASR deposunun git klonlanmasıyla oluşan arayüz hizalama hatalarını düzelttik.

## 20240130

1. Çift tırnaklarla yol kopyalama hatalarını önlemek için tüm yol ile ilgili girdilerden otomatik olarak çift tırnakları kaldırdık.
2. Çince ve İngilizce noktalama işaretlerini ayırma sorunlarını düzelttik ve cümlelerin başına ve sonuna noktalama işaretleri ekledik.
3. Noktalama işaretlerine göre ayırma özelliğini ekledik.

## 20240201

1. Ayrılma hatalarına neden olan UVR5 format okuma hatasını düzelttik.
2. Karışık Çince-Japonca-İngilizce metinler için otomatik segmentasyon ve dil tanıma desteği sağladık.

## 20240202

1. `/` ile biten bir ASR yolunun dosya adını kaydetme hatasına neden olma sorununu düzelttik.
2. [PR 377](https://github.com/RVC-Boss/GPT-SoVITS/pull/377) PaddleSpeech'in Normalizer'ını tanıtarak "xx.xx%" (yüzde sembolleri) ve "元/吨" ifadesinin "元吨" yerine "元每吨" olarak okunması gibi sorunları düzelttik ve alt çizgi hatalarını giderdik.

## 20240207

1. [Issue 391](https://github.com/RVC-Boss/GPT-SoVITS/issues/391)de bildirilen dil parametresi karışıklığının Çinçe çıkarım kalitesini düşürme sorununu düzelttik.
2. [PR 403](https://github.com/RVC-Boss/GPT-SoVITS/pull/403) ile UVR5'i daha yüksek versiyonlarda librosa'ya uyarladık.
3. [Commit 14a2851](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8) `is_half` parametresinin booleana dönüştürülmemesi nedeniyle sürekli yarım hassasiyet çıkarımı yaparak 16 serisi GPU'larda `inf` hatasına neden olan UVR5 inf hatasını düzelttik.
4. İngilizce metin önyüzünü optimize ettik.
5. Gradio bağımlılıklarını düzelttik.
6. Veri seti hazırlığı sırasında kök dizini boş bırakıldığında `.list` tam yollarının otomatik olarak okunmasını destekledik.
7. Japonca ve İngilizce için Faster Whisper ASR'yi entegre ettik.

## 20240208

1. [Commit 59f35ad](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b) ile Windows 10 1909'da ve [Issue 232](https://github.com/RVC-Boss/GPT-SoVITS/issues/232)de (Geleneksel Çince Sistem Dili) bildirilen GPT eğitim durma sorununu düzeltmeye çalıştık.

## 20240212

1. Faster Whisper ve FunASR için mantığı optimize ettik, Hugging Face bağlantı sorunlarını önlemek için Faster Whisper'ı ayna indirmelere yönlendirdik.
2. [PR 457](https://github.com/RVC-Boss/GPT-SoVITS/pull/457) GPT tekrarı ve eksik karakterleri azaltmak için eğitim sırasında negatif örnekler oluşturarak deneysel DPO Loss eğitim seçeneğini etkinleştirdi ve çıkarım WebUI'de çeşitli çıkarım parametrelerini kullanılabilir hale getirdi.

## 20240214

1. Eğitimde Çince deney adlarını destekledik (önceden hatalara neden oluyordu).
2. DPO eğitimini zorunlu yerine isteğe bağlı bir özellik yaptık. Seçilirse, parti boyutu otomatik olarak yarıya indirilir. Çıkarım WebUI'de yeni parametrelerin iletilmemesi sorunlarını düzelttik.

## 20240216

1. Referans metin olmadan girişi destekledik.
2. [Issue 475](https://github.com/RVC-Boss/GPT-SoVITS/issues/475) de bildirilen Çince önyüz hatalarını düzelttik.

## 20240221

1. Veri işleme sırasında bir gürültü azaltma seçeneği ekledik (gürültü azaltma sadece 16kHz örnekleme hızını bırakır; yalnızca arka plan gürültüsü önemliyse kullanın).
2. [PR 559](https://github.com/RVC-Boss/GPT-SoVITS/pull/559), [PR 556](https://github.com/RVC-Boss/GPT-SoVITS/pull/556), [PR 532](https://github.com/RVC-Boss/GPT-SoVITS/pull/532), [PR 507](https://github.com/RVC-Boss/GPT-SoVITS/pull/507), [PR 509](https://github.com/RVC-Boss/GPT-SoVITS/pull/509) ile Çince ve Japonca önyüz işlemesini optimize ettik.
3. Mac CPU çıkarımını daha hızlı performans için MPS yerine CPU kullanacak şekilde değiştirdik.
4. Colab genel URL sorununu düzelttik.

## 20240306

1. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) çıkarımı %50 hızlandırdı (RTX3090 + PyTorch 2.2.1 + CU11.8 + Win10 + Py39 üzerinde test edildi).
2. Faster Whisper'ın Çince olmayan ASR'sini kullanırken artık önce Çin FunASR modelini indirmeyi gerektirmiyor.
3. [PR 610](https://github.com/RVC-Boss/GPT-SoVITS/pull/610) UVR5 yankı giderme modelindeki ayarın tersine çevrildiği sorunu düzeltti.
4. [PR 675](https://github.com/RVC-Boss/GPT-SoVITS/pull/675) CUDA mevcut olmadığında Faster Whisper için otomatik CPU çıkarımını etkinleştirdi.
5. [PR 573](https://github.com/RVC-Boss/GPT-SoVITS/pull/573) Mac'te doğru CPU çıkarımı sağlamak için `is_half` kontrolünü değiştirdi.

## 202403/202404/202405 Güncellemeleri

### Küçük Düzeltmeler:

1. Referans metin olmayan mod ile ilgili sorunlar düzeltildi.
2. Çince ve İngilizce metin önyüzü optimize edildi.
3. API formatı iyileştirildi.
4. CMD format sorunları düzeltildi.
5. Eğitim verisi işleme sırasında desteklenmeyen diller için hata uyarıları eklendi.
6. Hubert çıkarımındaki hata düzeltildi.

### Büyük Düzeltmeler:

1. VQ'yu dondurmadan yapılan SoVITS eğitimi sorunu (bu kalite düşüşüne neden olabilir) düzeltildi.
2. Hızlı çıkarım dalı eklendi.

## 20240610

### Küçük Düzeltmeler:

1. [PR 1168](https://github.com/RVC-Boss/GPT-SoVITS/pull/1168) & [PR 1169](https://github.com/RVC-Boss/GPT-SoVITS/pull/1169) saf noktalama işareti ve çoklu noktalama işareti metin girdisi için mantığı geliştirdi.
2. [Commit 501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232) UVR5'teki MDXNet yankı giderme için CMD formatını düzeltti, boşluk içeren yolları destekledi.
3. [PR 1159](https://github.com/RVC-Boss/GPT-SoVITS/pull/1159) `s2_train.py` içindeki SoVITS eğitimi için ilerleme çubuğu mantığını düzeltti.

### Büyük Düzeltmeler:

4. [Commit 99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a) WebUI'nin GPT ince ayarının, Çince giriş metinlerinin BERT özelliğini okumaması sorununu düzeltti, bu da çıkarım ile tutarsızlığa ve potansiyel kalite düşüşüne neden oluyordu.
   **Dikkat: Daha önce büyük miktarda veri ile ince ayar yaptıysanız, modelin kalitesini artırmak için yeniden ayar yapmanız önerilir.**

## 20240706

### Küçük Düzeltmeler:

1. [Commit 1250670](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041) CPU çıkarımında varsayılan yığın boyutu ondalık sorununu düzeltti.
2. [PR 1258](https://github.com/RVC-Boss/GPT-SoVITS/pull/1258), [PR 1265](https://github.com/RVC-Boss/GPT-SoVITS/pull/1265), [PR 1267](https://github.com/RVC-Boss/GPT-SoVITS/pull/1267) gürültü giderme veya ASR ile ilgili istisnalarla karşılaşıldığında bekleyen tüm ses dosyalarının çıkış yapmasına neden olan sorunları düzeltti.
3. [PR 1253](https://github.com/RVC-Boss/GPT-SoVITS/pull/1253) noktalama işaretlerine göre ayrılırken ondalıkların bölünmesi sorununu düzeltti.
4. [Commit a208698](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca) çoklu GPU eğitimi için çoklu işlem kaydetme mantığını düzeltti.
5. [PR 1251](https://github.com/RVC-Boss/GPT-SoVITS/pull/1251) gereksiz `my_utils`'ı kaldırdı.

### Büyük Düzeltmeler:

6. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) hızlandırılmış çıkarım kodu doğrulandı ve ana dala birleştirildi, taban ile tutarlı çıkarım etkileri sağlandı.
   Ayrıca referans metni olmayan modda hızlandırılmış çıkarımı destekler.

**Gelecek güncellemeler, `fast_inference` dalındaki değişikliklerin tutarlılığını doğrulamaya devam edecek.**

## 20240727

### Küçük Düzeltmeler:

1. [PR 1298](https://github.com/RVC-Boss/GPT-SoVITS/pull/1298) gereksiz i18n kodlarını temizledi.
2. [PR 1299](https://github.com/RVC-Boss/GPT-SoVITS/pull/1299) kullanıcı dosya yollarındaki sonlandırma eğik çizgilerinin komut satırı hatalarına neden olduğu sorunları düzeltti.
3. [PR 756](https://github.com/RVC-Boss/GPT-SoVITS/pull/756) GPT eğitimindeki adım hesaplama mantığını düzeltti.

### Büyük Düzeltmeler:

4. [Commit 9588a3c](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2) sentez için konuşma hızı ayarlamasını destekledi.
   Konuşma hızını ayarlarken rastgeleliği dondurmayı etkinleştirdi.

- 2024.07.27 [PR#1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306), [PR#1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356): BS-RoFormer vokal eşlik ayırma modeli desteği eklendi.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: KamioRinn
- 2024.07.27 [PR#1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351): Çince metin ön işleme iyileştirildi.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: KamioRinn

## 202408 (V2 Sürümü)

- 2024.08.01 [PR#1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1355): WebUI'de dosya işlerken yolların otomatik doldurulması.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666
- 2024.08.01 [Commit#e62e9653](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c): BS-Roformer için FP16 çıkarım desteği etkinleştirildi.
  - Tür: Performans Optimizasyonu
  - Katkıda Bulunan: RVC-Boss
- 2024.08.01 [Commit#bce451a2](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit#4c8b7612](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78): GPU tanıma mantığı optimize edildi, kullanıcıların girdiği rastgele GPU indekslerini işlemek için kullanıcı dostu mantık eklendi.
  - Tür: Chore
  - Katkıda Bulunan: RVC-Boss
- 2024.08.02 [Commit#ff6c193f](https://github.com/RVC-Boss/GPT-SoVITS/commit/ff6c193f6fb99d44eea3648d82ebcee895860a22)~[Commit#de7ee7c7](https://github.com/RVC-Boss/GPT-SoVITS/commit/de7ee7c7c15a2ec137feb0693b4ff3db61fad758): **GPT-SoVITS V2 modeli eklendi.**
  - Tür: Yeni Özellik
  - Katkıda Bulunan: RVC-Boss
- 2024.08.03 [Commit#8a101474](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3): FunASR kullanarak Kantonca ASR desteği eklendi.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: RVC-Boss
- 2024.08.03 [PR#1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387), [PR#1388](https://github.com/RVC-Boss/GPT-SoVITS/pull/1388): UI ve zamanlama mantığı optimize edildi.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666
- 2024.08.06 [PR#1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404), [PR#987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987), [PR#488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488): Çok sesli karakter işleme mantığı optimize edildi (Yalnızca V2).
  - Tür: Düzeltme, Yeni Özellik
  - Katkıda Bulunan: KamioRinn, RVC-Boss
- 2024.08.13 [PR#1422](https://github.com/RVC-Boss/GPT-SoVITS/pull/1422): Yalnızca bir referans ses yüklenebilme hatası düzeltildi; eksik dosyalar için uyarı açılır pencereleriyle veri seti doğrulama eklendi.
  - Tür: Düzeltme, Chore
  - Katkıda Bulunan: XXXXRT666
- 2024.08.20 [Issue#1508](https://github.com/RVC-Boss/GPT-SoVITS/issues/1508): Yukarı akış LangSegment kütüphanesi artık SSML etiketleri kullanarak sayıları, telefon numaralarını, tarihleri ve saatleri optimize ediyor.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: juntaosun
- 2024.08.20 [PR#1503](https://github.com/RVC-Boss/GPT-SoVITS/pull/1503): API düzeltildi ve optimize edildi.
  - Tür: Düzeltme
  - Katkıda Bulunan: KamioRinn
- 2024.08.20 [PR#1490](https://github.com/RVC-Boss/GPT-SoVITS/pull/1490): `fast_inference` dalı ana dala birleştirildi.
  - Tür: Yeniden Yapılandırma
  - Katkıda Bulunan: ChasonJiang
- 2024.08.21 **GPT-SoVITS V2 sürümü resmi olarak yayınlandı.**

## 202502 (V3 Sürümü)

- 2025.02.11 [Commit#ed207c4b](https://github.com/RVC-Boss/GPT-SoVITS/commit/ed207c4b879d5296e9be3ae5f7b876729a2c43b8)~[Commit#6e2b4918](https://github.com/RVC-Boss/GPT-SoVITS/commit/6e2b49186c5b961f0de41ea485d398dffa9787b4): **İnce ayar için 14GB VRAM gerektiren GPT-SoVITS V3 modeli eklendi.**
  - Tür: Yeni Özellik ([Wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)) referans)
  - Katkıda Bulunan: RVC-Boss
- 2025.02.12 [PR#2032](https://github.com/RVC-Boss/GPT-SoVITS/pull/2032): Çok dilli proje dokümantasyonu güncellendi.
  - Tür: Dokümantasyon
  - Katkıda Bulunan: StaryLan
- 2025.02.12 [PR#2033](https://github.com/RVC-Boss/GPT-SoVITS/pull/2033): Japonca dokümantasyon güncellendi.
  - Tür: Dokümantasyon
  - Katkıda Bulunan: Fyphen
- 2025.02.12 [PR#2010](https://github.com/RVC-Boss/GPT-SoVITS/pull/2010): Dikkat hesaplama mantığı optimize edildi.
  - Tür: Performans Optimizasyonu
  - Katkıda Bulunan: wzy3650
- 2025.02.12 [PR#2040](https://github.com/RVC-Boss/GPT-SoVITS/pull/2040): İnce ayar için gradyan kontrol noktası desteği eklendi (12GB VRAM gerektirir).
  - Tür: Yeni Özellik
  - Katkıda Bulunan: Kakaru Hayate
- 2025.02.14 [PR#2047](https://github.com/RVC-Boss/GPT-SoVITS/pull/2047), [PR#2062](https://github.com/RVC-Boss/GPT-SoVITS/pull/2062), [PR#2073](https://github.com/RVC-Boss/GPT-SoVITS/pull/2073): Yeni dil bölümleme aracına geçildi, çok dilli karışık metin bölme stratejisi iyileştirildi, sayı ve İngilizce işleme mantığı optimize edildi.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: KamioRinn
- 2025.02.23 [Commit#56509a17](https://github.com/RVC-Boss/GPT-SoVITS/commit/56509a17c918c8d149c48413a672b8ddf437495b)~[Commit#514fb692](https://github.com/RVC-Boss/GPT-SoVITS/commit/514fb692db056a06ed012bc3a5bca2a5b455703e): **GPT-SoVITS V3 modeli artık LoRA eğitimini destekliyor (ince ayar için 8GB GPU Belleği gerektirir).**
  - Tür: Yeni Özellik
  - Katkıda Bulunan: RVC-Boss
- 2025.02.23 [PR#2078](https://github.com/RVC-Boss/GPT-SoVITS/pull/2078): Vokal ve enstrüman ayırma için Mel Band Roformer model desteği eklendi.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: Sucial
- 2025.02.26 [PR#2112](https://github.com/RVC-Boss/GPT-SoVITS/pull/2112), [PR#2114](https://github.com/RVC-Boss/GPT-SoVITS/pull/2114): Çince yollarda MeCab hatası düzeltildi (özel olarak Japonca/Korece veya çok dilli metin bölme için).
  - Tür: Düzeltme
  - Katkıda Bulunan: KamioRinn
- 2025.02.27 [Commit#92961c3f](https://github.com/RVC-Boss/GPT-SoVITS/commit/92961c3f68b96009ff2cd00ce614a11b6c4d026f)~[Commit#250b1c73](https://github.com/RVC-Boss/GPT-SoVITS/commit/250b1c73cba60db18148b21ec5fbce01fd9d19bc): V3 modeliyle 24K ses üretirken "boğuk" ses sorununu hafifletmek için **24kHz'den 48kHz'e ses süper çözünürlük modelleri eklendi**.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: RVC-Boss
  - İlgili: [Issue#2085](https://github.com/RVC-Boss/GPT-SoVITS/issues/2085), [Issue#2117](https://github.com/RVC-Boss/GPT-SoVITS/issues/2117)
- 2025.02.28 [PR#2123](https://github.com/RVC-Boss/GPT-SoVITS/pull/2123): Çok dilli proje dokümantasyonu güncellendi.
  - Tür: Dokümantasyon
  - Katkıda Bulunan: StaryLan
- 2025.02.28 [PR#2122](https://github.com/RVC-Boss/GPT-SoVITS/pull/2122): Model tanımlayamadığında kısa CJK karakterleri için kural tabanlı tespit uygulandı.
  - Tür: Düzeltme
  - Katkıda Bulunan: KamioRinn
  - İlgili: [Issue#2116](https://github.com/RVC-Boss/GPT-SoVITS/issues/2116)
- 2025.02.28 [Commit#c38b1690](https://github.com/RVC-Boss/GPT-SoVITS/commit/c38b16901978c1db79491e16905ea3a37a7cf686), [Commit#a32a2b89](https://github.com/RVC-Boss/GPT-SoVITS/commit/a32a2b893436fad56cc82409121c7fa36a1815d5): Sentez hızını kontrol etmek için konuşma hızı parametresi eklendi.
  - Tür: Düzeltme
  - Katkıda Bulunan: RVC-Boss
- 2025.02.28 **GPT-SoVITS V3 resmi olarak yayınlandı**.

## 202503

- 2025.03.31 [PR#2236](https://github.com/RVC-Boss/GPT-SoVITS/pull/2236): Bağımlılıkların yanlış sürümlerinden kaynaklanan sorunlar düzeltildi.
  - Tür: Düzeltme
  - Katkıda Bulunan: XXXXRT666
  - İlgili:
    - PyOpenJTalk: [Issue#1131](https://github.com/RVC-Boss/GPT-SoVITS/issues/1131), [Issue#2231](https://github.com/RVC-Boss/GPT-SoVITS/issues/2231), [Issue#2233](https://github.com/RVC-Boss/GPT-SoVITS/issues/2233).
    - ONNX: [Issue#492](https://github.com/RVC-Boss/GPT-SoVITS/issues/492), [Issue#671](https://github.com/RVC-Boss/GPT-SoVITS/issues/671), [Issue#1192](https://github.com/RVC-Boss/GPT-SoVITS/issues/1192), [Issue#1819](https://github.com/RVC-Boss/GPT-SoVITS/issues/1819), [Issue#1841](https://github.com/RVC-Boss/GPT-SoVITS/issues/1841).
    - Pydantic: [Issue#2230](https://github.com/RVC-Boss/GPT-SoVITS/issues/2230), [Issue#2239](https://github.com/RVC-Boss/GPT-SoVITS/issues/2239).
    - PyTorch-Lightning: [Issue#2174](https://github.com/RVC-Boss/GPT-SoVITS/issues/2174).
- 2025.03.31 [PR#2241](https://github.com/RVC-Boss/GPT-SoVITS/pull/2241): **SoVITS v3 için paralel çıkarım etkinleştirildi.**
  - Tür: Yeni Özellik
  - Katkıda Bulunan: ChasonJiang

- Diğer küçük hatalar düzeltildi.

- ONNX çalışma zamanı GPU çıkarım desteği için entegre paket düzeltmeleri:
  - Tür: Düzeltme
  - Detaylar:
    - G2PW içindeki ONNX modelleri CPU'dan GPU çıkarımına geçirildi, CPU darboğazı önemli ölçüde azaltıldı;
    - foxjoy yankı giderme modeli artık GPU çıkarımını destekliyor.

## 202504 (V4 Sürümü)

- 2025.04.01 [Commit#6a60e5ed](https://github.com/RVC-Boss/GPT-SoVITS/commit/6a60e5edb1817af4a61c7a5b196c0d0f1407668f): SoVITS v3 paralel çıkarımı kilit açıldı; asenkron model yükleme mantığı düzeltildi.
  - Tür: Düzeltme
  - Katkıda Bulunan: RVC-Boss
- 2025.04.07 [PR#2255](https://github.com/RVC-Boss/GPT-SoVITS/pull/2255): Ruff ile kod biçimlendirme; G2PW bağlantısı güncellendi.
  - Tür: Stil
  - Katkıda Bulunan: XXXXRT666
- 2025.04.15 [PR#2290](https://github.com/RVC-Boss/GPT-SoVITS/pull/2290): Dokümantasyon temizlendi; Python 3.11 desteği eklendi; yükleyiciler güncellendi.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666
- 2025.04.20 [PR#2300](https://github.com/RVC-Boss/GPT-SoVITS/pull/2300): Colab, kurulum dosyaları ve model indirmeleri güncellendi.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666
- 2025.04.20 [Commit#e0c452f0](https://github.com/RVC-Boss/GPT-SoVITS/commit/e0c452f0078e8f7eb560b79a54d75573fefa8355)~[Commit#9d481da6](https://github.com/RVC-Boss/GPT-SoVITS/commit/9d481da610aa4b0ef8abf5651fd62800d2b4e8bf): **GPT-SoVITS V4 modeli eklendi.**
  - Tür: Yeni Özellik
  - Katkıda Bulunan: RVC-Boss
- 2025.04.21 [Commit#8b394a15](https://github.com/RVC-Boss/GPT-SoVITS/commit/8b394a15bce8e1d85c0b11172442dbe7a6017ca2)~[Commit#bc2fe5ec](https://github.com/RVC-Boss/GPT-SoVITS/commit/bc2fe5ec86536c77bb3794b4be263ac87e4fdae6), [PR#2307](https://github.com/RVC-Boss/GPT-SoVITS/pull/2307): V4 için paralel çıkarım etkinleştirildi.
  - Tür: Yeni Özellik
  - Katkıda Bulunan: RVC-Boss, ChasonJiang
- 2025.04.22 [Commit#7405427a](https://github.com/RVC-Boss/GPT-SoVITS/commit/7405427a0ab2a43af63205df401fd6607a408d87)~[Commit#590c83d7](https://github.com/RVC-Boss/GPT-SoVITS/commit/590c83d7667c8d4908f5bdaf2f4c1ba8959d29ff), [PR#2309](https://github.com/RVC-Boss/GPT-SoVITS/pull/2309): Model sürümü parametre aktarımı düzeltildi.
  - Tür: Düzeltme
  - Katkıda Bulunan: RVC-Boss, ChasonJiang
- 2025.04.22 [Commit#fbdab94e](https://github.com/RVC-Boss/GPT-SoVITS/commit/fbdab94e17d605d85841af6f94f40a45976dd1d9), [PR#2310](https://github.com/RVC-Boss/GPT-SoVITS/pull/2310): Numpy ve Numba sürüm uyumsuzluğu sorunu düzeltildi; librosa sürümü güncellendi.
  - Tür: Düzeltme
  - Katkıda Bulunan: RVC-Boss, XXXXRT666
  - İlgili: [Issue#2308](https://github.com/RVC-Boss/GPT-SoVITS/issues/2308)
- **2025.04.22 GPT-SoVITS V4 resmi olarak yayınlandı**.
- 2025.04.22 [PR#2311](https://github.com/RVC-Boss/GPT-SoVITS/pull/2311): Gradio parametreleri güncellendi.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666
- 2025.04.25 [PR#2322](https://github.com/RVC-Boss/GPT-SoVITS/pull/2322): Colab/Kaggle notebook betikleri iyileştirildi.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666

## 202505

- 2025.05.26 [PR#2351](https://github.com/RVC-Boss/GPT-SoVITS/pull/2351): Docker ve Windows otomatik derleme betikleri iyileştirildi; ön işleme biçimlendirme eklendi.
  - Tür: Chore
  - Katkıda Bulunan: XXXXRT666
- 2025.05.26 [PR#2408](https://github.com/RVC-Boss/GPT-SoVITS/pull/2408): Çok dilli metin bölme ve tanıma mantığı optimize edildi.
  - Tür: Düzeltme
  - Katkıda Bulunan: KamioRinn
  - İlgili: [Issue#2404](https://github.com/RVC-Boss/GPT-SoVITS/issues/2404)
- 2025.05.26 [PR#2377](https://github.com/RVC-Boss/GPT-SoVITS/pull/2377): SoVITS V3/V4 çıkarım hızını %10 artırmak için önbellekleme stratejileri uygulandı.
  - Tür: Performans Optimizasyonu
  - Katkıda Bulunan: Kakaru Hayate
- 2025.05.26 [Commit#4d9d56b1](https://github.com/RVC-Boss/GPT-SoVITS/commit/4d9d56b19638dc434d6eefd9545e4d8639a3e072), [Commit#8c705784](https://github.com/RVC-Boss/GPT-SoVITS/commit/8c705784c50bf438c7b6d0be33a9e5e3cb90e6b2), [Commit#fafe4e7f](https://github.com/RVC-Boss/GPT-SoVITS/commit/fafe4e7f120fba56c5f053c6db30aa675d5951ba): Açıklama arayüzü uyarı ile güncellendi: her sayfa tamamlandıktan sonra "Metni Gönder"e tıklayın, aksi takdirde değişiklikler kaydedilmez.
  - Tür: Düzeltme
  - Katkıda Bulunan: RVC-Boss
- 2025.05.29 [Commit#1934fc1e](https://github.com/RVC-Boss/GPT-SoVITS/commit/1934fc1e1b22c4c162bba1bbe7d7ebb132944cdc): UVR5 ve ONNX yankı giderme modellerinde, FFmpeg'in orijinal yollarında boşluk bulunan MP3/M4A dosyalarını kodlarken oluşan hatalar düzeltildi.
  - Tür: Düzeltme
  - Katkıda Bulunan: RVC-Boss

**Önizleme: Ejderha Teknesi Festivali'nden sonra V2 sürümüne dayalı büyük optimizasyon güncellemesi gelecek!**