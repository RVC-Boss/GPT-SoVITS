### 20240121 Güncellemesi

1. `config`e `is_share` eklendi. Colab gibi senaryolarda, WebUI'yi halka açık ağa yönlendirmek için bu değeri `True` olarak ayarlayabilirsiniz.
2. WebUI'ye İngilizce sistem çeviri desteği eklendi.
3. `cmd-asr`, FunASR modelinin dahil olup olmadığını otomatik olarak tespit eder; eğer varsayılan dizinde bulunamazsa, ModelScope'dan indirilecektir.
4. [Issue 79](https://github.com/RVC-Boss/GPT-SoVITS/issues/79)de bildirilen SoVITS eğitimindeki ZeroDivisionError'u sıfır uzunlukta örnekleri filtreleyerek düzeltmeye çalıştık.
5. `TEMP` klasöründeki önbelleğe alınmış ses dosyaları ve diğer dosyaları temizledik.
6. Referans sesinin sonunu içeren sentezlenmiş ses sorununu önemli ölçüde azalttık.

### 20240122 Güncellemesi

1. Aşırı kısa çıktı dosyalarının referans sesini tekrarlamasına neden olan sorun giderildi.
2. İngilizce ve Japonca eğitim için yerel destek test edildi (Japonca eğitim için kök dizinin İngilizce olmayan özel karakterlerden arındırılmış olması gerekir).
3. Ses yolu denetimi iyileştirildi. Yanlış bir giriş yolundan okumaya çalışıldığında, ffmpeg hatası yerine yolun mevcut olmadığını bildirir.

### 20240123 Güncellemesi

1. Hubert çıkarımının NaN hatalarına neden olup SoVITS/GPT eğitiminde ZeroDivisionError'a yol açtığı sorun çözüldü.
2. İnferans WebUI'de hızlı model değiştirme desteği eklendi.
3. Model dosyası sıralama mantığı optimize edildi.
4. Çince kelime ayrımı için `jieba` `jieba_fast` ile değiştirildi.

### 20240126 Güncellemesi

1. Çince-İngilizce ve Japonca-İngilizce karışık çıktı metinleri için destek eklendi.
2. Çıktı için isteğe bağlı bir bölme modu eklendi.
3. UVR5'in dizinlerden otomatik olarak çıkmasına neden olan okuma sorununu düzelttik.
4. Çeşitli yeni satır sorunlarını düzelterek çıkarım hatalarını giderdik.
5. Çıkarım WebUI'deki gereksiz günlükleri kaldırdık.
6. Mac'te eğitim ve çıkarım desteği eklendi.
7. Yarım hassasiyeti desteklemeyen GPU'lar için otomatik olarak tek hassasiyet zorlandı; CPU çıkarımında tek hassasiyet uygulandı.

### 20240128 Güncellemesi

1. Sayıların Çince karakterlere dönüştürülmesiyle ilgili sorunu düzelttik.
2. Cümlelerin başındaki birkaç karakterin yutulması sorununu düzelttik.
3. Mantıksız referans ses uzunluklarını sınırlamalar koyarak hariç tuttuk.
4. GPT eğitiminin kontrol noktalarını kaydetmemesi sorununu düzelttik.
5. Dockerfile'da model indirme sürecini tamamladık.

### 20240129 Güncellemesi

1. Yarım hassasiyet eğitimi ile ilgili sorun yaşayan 16 serisi gibi GPU'lar için eğitim yapılandırmalarını tek hassasiyete değiştirdik.
2. Mevcut Colab sürümünü test ettik ve güncelledik.
3. Eski sürüm FunASR ile ModelScope FunASR deposunun git klonlanmasıyla oluşan arayüz hizalama hatalarını düzelttik.

### 20240130 Güncellemesi

1. Çift tırnaklarla yol kopyalama hatalarını önlemek için tüm yol ile ilgili girdilerden otomatik olarak çift tırnakları kaldırdık.
2. Çince ve İngilizce noktalama işaretlerini ayırma sorunlarını düzelttik ve cümlelerin başına ve sonuna noktalama işaretleri ekledik.
3. Noktalama işaretlerine göre ayırma özelliğini ekledik.

### 20240201 Güncellemesi

1. Ayrılma hatalarına neden olan UVR5 format okuma hatasını düzelttik.
2. Karışık Çince-Japonca-İngilizce metinler için otomatik segmentasyon ve dil tanıma desteği sağladık.

### 20240202 Güncellemesi

1. `/` ile biten bir ASR yolunun dosya adını kaydetme hatasına neden olma sorununu düzelttik.
2. [PR 377](https://github.com/RVC-Boss/GPT-SoVITS/pull/377) PaddleSpeech'in Normalizer'ını tanıtarak "xx.xx%" (yüzde sembolleri) ve "元/吨" ifadesinin "元吨" yerine "元每吨" olarak okunması gibi sorunları düzelttik ve alt çizgi hatalarını giderdik.

### 20240207 Güncellemesi

1. [Issue 391](https://github.com/RVC-Boss/GPT-SoVITS/issues/391)de bildirilen dil parametresi karışıklığının Çinçe çıkarım kalitesini düşürme sorununu düzelttik.
2. [PR 403](https://github.com/RVC-Boss/GPT-SoVITS/pull/403) ile UVR5'i daha yüksek versiyonlarda librosa'ya uyarladık.
3. [Commit 14a2851](https://github.com/RVC-Boss/GPT-SoVITS/commit/14a285109a521679f8846589c22da8f656a46ad8) `is_half` parametresinin booleana dönüştürülmemesi nedeniyle sürekli yarım hassasiyet çıkarımı yaparak 16 serisi GPU'larda `inf` hatasına neden olan UVR5 inf hatasını düzelttik.
4. İngilizce metin önyüzünü optimize ettik.
5. Gradio bağımlılıklarını düzelttik.
6. Veri seti hazırlığı sırasında kök dizini boş bırakıldığında `.list` tam yollarının otomatik olarak okunmasını destekledik.
7. Japonca ve İngilizce için Faster Whisper ASR'yi entegre ettik.

### 20240208 Güncellemesi

1. [Commit 59f35ad](https://github.com/RVC-Boss/GPT-SoVITS/commit/59f35adad85815df27e9c6b33d420f5ebfd8376b) ile Windows 10 1909'da ve [Issue 232](https://github.com/RVC-Boss/GPT-SoVITS/issues/232)de (Geleneksel Çince Sistem Dili) bildirilen GPT eğitim durma sorununu düzeltmeye çalıştık.

### 20240212 Güncellemesi

1. Faster Whisper ve FunASR için mantığı optimize ettik, Hugging Face bağlantı sorunlarını önlemek için Faster Whisper'ı ayna indirmelere yönlendirdik.
2. [PR 457](https://github.com/RVC-Boss/GPT-SoVITS/pull/457) GPT tekrarı ve eksik karakterleri azaltmak için eğitim sırasında negatif örnekler oluşturarak deneysel DPO Loss eğitim seçeneğini etkinleştirdi ve çıkarım WebUI'de çeşitli çıkarım parametrelerini kullanılabilir hale getirdi.

### 20240214 Güncellemesi

1. Eğitimde Çince deney adlarını destekledik (önceden hatalara neden oluyordu).
2. DPO eğitimini zorunlu yerine isteğe bağlı bir özellik yaptık. Seçilirse, parti boyutu otomatik olarak yarıya indirilir. Çıkarım WebUI'de yeni parametrelerin iletilmemesi sorunlarını düzelttik.

### 20240216 Güncellemesi

1. Referans metin olmadan girişi destekledik.
2. [Issue 475](https://github.com/RVC-Boss/GPT-SoVITS/issues/475) de bildirilen Çince önyüz hatalarını düzelttik.

### 20240221 Güncellemesi

1. Veri işleme sırasında bir gürültü azaltma seçeneği ekledik (gürültü azaltma sadece 16kHz örnekleme hızını bırakır; yalnızca arka plan gürültüsü önemliyse kullanın).
2. [PR 559](https://github.com/RVC-Boss/GPT-SoVITS/pull/559), [PR 556](https://github.com/RVC-Boss/GPT-SoVITS/pull/556), [PR 532](https://github.com/RVC-Boss/GPT-SoVITS/pull/532), [PR 507](https://github.com/RVC-Boss/GPT-SoVITS/pull/507), [PR 509](https://github.com/RVC-Boss/GPT-SoVITS/pull/509) ile Çince ve Japonca önyüz işlemesini optimize ettik.
3. Mac CPU çıkarımını daha hızlı performans için MPS yerine CPU kullanacak şekilde değiştirdik.
4. Colab genel URL sorununu düzelttik.

### 20240306 Güncellemesi

1. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) çıkarımı %50 hızlandırdı (RTX3090 + PyTorch 2.2.1 + CU11.8 + Win10 + Py39 üzerinde test edildi).
2. Faster Whisper'ın Çince olmayan ASR'sini kullanırken artık önce Çin FunASR modelini indirmeyi gerektirmiyor.
3. [PR 610](https://github.com/RVC-Boss/GPT-SoVITS/pull/610) UVR5 yankı giderme modelindeki ayarın tersine çevrildiği sorunu düzeltti.
4. [PR 675](https://github.com/RVC-Boss/GPT-SoVITS/pull/675) CUDA mevcut olmadığında Faster Whisper için otomatik CPU çıkarımını etkinleştirdi.
5. [PR 573](https://github.com/RVC-Boss/GPT-SoVITS/pull/573) Mac'te doğru CPU çıkarımı sağlamak için `is_half` kontrolünü değiştirdi.

### 202403/202404/202405 Güncellemeleri

#### Küçük Düzeltmeler:

1. Referans metin olmayan mod ile ilgili sorunlar düzeltildi.
2. Çince ve İngilizce metin önyüzü optimize edildi.
3. API formatı iyileştirildi.
4. CMD format sorunları düzeltildi.
5. Eğitim verisi işleme sırasında desteklenmeyen diller için hata uyarıları eklendi.
6. Hubert çıkarımındaki hata düzeltildi.

#### Büyük Düzeltmeler:

1. VQ'yu dondurmadan yapılan SoVITS eğitimi sorunu (bu kalite düşüşüne neden olabilir) düzeltildi.
2. Hızlı çıkarım dalı eklendi.

### 20240610 Güncellemesi

#### Küçük Düzeltmeler:

1. [PR 1168](https://github.com/RVC-Boss/GPT-SoVITS/pull/1168) & [PR 1169](https://github.com/RVC-Boss/GPT-SoVITS/pull/1169) saf noktalama işareti ve çoklu noktalama işareti metin girdisi için mantığı geliştirdi.
2. [Commit 501a74a](https://github.com/RVC-Boss/GPT-SoVITS/commit/501a74ae96789a26b48932babed5eb4e9483a232) UVR5'teki MDXNet yankı giderme için CMD formatını düzeltti, boşluk içeren yolları destekledi.
3. [PR 1159](https://github.com/RVC-Boss/GPT-SoVITS/pull/1159) `s2_train.py` içindeki SoVITS eğitimi için ilerleme çubuğu mantığını düzeltti.

#### Büyük Düzeltmeler:

4. [Commit 99f09c8](https://github.com/RVC-Boss/GPT-SoVITS/commit/99f09c8bdc155c1f4272b511940717705509582a) WebUI'nin GPT ince ayarının, Çince giriş metinlerinin BERT özelliğini okumaması sorununu düzeltti, bu da çıkarım ile tutarsızlığa ve potansiyel kalite düşüşüne neden oluyordu. 
   **Dikkat: Daha önce büyük miktarda veri ile ince ayar yaptıysanız, modelin kalitesini artırmak için yeniden ayar yapmanız önerilir.**

### 20240706 Güncellemesi

#### Küçük Düzeltmeler:

1. [Commit 1250670](https://github.com/RVC-Boss/GPT-SoVITS/commit/db50670598f0236613eefa6f2d5a23a271d82041) CPU çıkarımında varsayılan yığın boyutu ondalık sorununu düzeltti.
2. [PR 1258](https://github.com/RVC-Boss/GPT-SoVITS/pull/1258), [PR 1265](https://github.com/RVC-Boss/GPT-SoVITS/pull/1265), [PR 1267](https://github.com/RVC-Boss/GPT-SoVITS/pull/1267) gürültü giderme veya ASR ile ilgili istisnalarla karşılaşıldığında bekleyen tüm ses dosyalarının çıkış yapmasına neden olan sorunları düzeltti.
3. [PR 1253](https://github.com/RVC-Boss/GPT-SoVITS/pull/1253) noktalama işaretlerine göre ayrılırken ondalıkların bölünmesi sorununu düzeltti.
4. [Commit a208698](https://github.com/RVC-Boss/GPT-SoVITS/commit/a208698e775155efc95b187b746d153d0f2847ca) çoklu GPU eğitimi için çoklu işlem kaydetme mantığını düzeltti.
5. [PR 1251](https://github.com/RVC-Boss/GPT-SoVITS/pull/1251) gereksiz `my_utils`'ı kaldırdı.

#### Büyük Düzeltmeler:

6. [PR 672](https://github.com/RVC-Boss/GPT-SoVITS/pull/672) hızlandırılmış çıkarım kodu doğrulandı ve ana dala birleştirildi, taban ile tutarlı çıkarım etkileri sağlandı.
   Ayrıca referans metni olmayan modda hızlandırılmış çıkarımı destekler.

**Gelecek güncellemeler, `fast_inference` dalındaki değişikliklerin tutarlılığını doğrulamaya devam edecek.**

### 20240727 Güncellemesi

#### Küçük Düzeltmeler:

1. [PR 1298](https://github.com/RVC-Boss/GPT-SoVITS/pull/1298) gereksiz i18n kodlarını temizledi.
2. [PR 1299](https://github.com/RVC-Boss/GPT-SoVITS/pull/1299) kullanıcı dosya yollarındaki sonlandırma eğik çizgilerinin komut satırı hatalarına neden olduğu sorunları düzeltti.
3. [PR 756](https://github.com/RVC-Boss/GPT-SoVITS/pull/756) GPT eğitimindeki adım hesaplama mantığını düzeltti.

#### Büyük Düzeltmeler:

4. [Commit 9588a3c](https://github.com/RVC-Boss/GPT-SoVITS/commit/9588a3c52d9ebdb20b3c5d74f647d12e7c1171c2) sentez için konuşma hızı ayarlamasını destekledi. 
   Konuşma hızını ayarlarken rastgeleliği dondurmayı etkinleştirdi.

### 20240806 Güncellemesi

1. [PR 1306](https://github.com/RVC-Boss/GPT-SoVITS/pull/1306), [PR 1356](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356) BS RoFormer vokal eşlik ayırma modelini desteklemeye başladı. [Commit e62e965](https://github.com/RVC-Boss/GPT-SoVITS/commit/e62e965323a60a76a025bcaa45268c1ddcbcf05c) FP16 çıkarımı etkinleştirdi.
2. Çince metin ön yüzünü geliştirdi.
   - [PR 488](https://github.com/RVC-Boss/GPT-SoVITS/pull/488) çoklu heceli karakterler için destek ekledi (v2 sadece);
   - [PR 987](https://github.com/RVC-Boss/GPT-SoVITS/pull/987) sayı belirleyici ekledi;
   - [PR 1351](https://github.com/RVC-Boss/GPT-SoVITS/pull/1351) aritmetik ve temel matematik formüllerini destekler;
   - [PR 1404](https://github.com/RVC-Boss/GPT-SoVITS/pull/1404) karışık metin hatalarını düzeltti.
3. [PR 1355](https://github.com/RVC-Boss/GPT-SoVITS/pull/1356) WebUI'de ses işlenirken yolları otomatik olarak doldurdu.
4. [Commit bce451a](https://github.com/RVC-Boss/GPT-SoVITS/commit/bce451a2d1641e581e200297d01f219aeaaf7299), [Commit 4c8b761](https://github.com/RVC-Boss/GPT-SoVITS/commit/4c8b7612206536b8b4435997acb69b25d93acb78) GPU tanıma mantığını optimize etti.
5. [Commit 8a10147](https://github.com/RVC-Boss/GPT-SoVITS/commit/8a101474b5a4f913b4c94fca2e3ca87d0771bae3) Kantonca ASR desteği ekledi.
6. GPT-SoVITS v2 desteği eklendi.
7. [PR 1387](https://github.com/RVC-Boss/GPT-SoVITS/pull/1387) zamanlama mantığını optimize etti.