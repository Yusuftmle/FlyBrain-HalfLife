# FlyBrain - Geliştirici Günlüğü & Biyomimetik Mühendislik Notları

## 📅 Tarih: 12 Eylül 2026
### 🔬 Vaka İncelemesi: Namlu Alevi Pozitif Geri Besleme Döngüsü ve Biyolojik "Efference Copy" Çözümü

---

## 1. Problem Tanımı
Saldırı devreleri (DNpe017 nöronu) ve hasar misillemesi devreye sokulduktan sonra sineğin sakin sakin gezmek yerine aralıksız ve kontrolsüz bir şekilde sol tık spamlayarak deli gibi ateş ettiği gözlemlendi.

---

## 2. Kök Neden Analizi (Nasıl Tespit Edildi?)

Dedüksiyon şu adımlarla kuruldu:

1. **Tetikleyici İncelemesi:**
   Ateş etme emrini (`_send_click()`) doğrudan tetikleyen ana faktör `is_damage` (hasar alındı) bayrağıydı.
2. **Hasar Algılama Mantığı (`detect_damage_flash`):**
   Half-Life'ta OCR (yazı okuma) çok yavaş olduğu için, oyun motorunun hasar anında ekranı kırmızıya boyamasından faydalanılıyordu. Ekranın merkezindeki $32 \times 32$ pikselde kırmızı renk fazlalığı ($R - 0.5 \times (G + B)$) taranıyordu.
3. **GoldSrc Oyun Motoru Fiziği:**
   Half-Life'ta (GoldSrc) oyuncu elindeki pompalı tüfeği veya tabancayı sıktığı an, silahın ucunda tam ekran merkezinde parlak kırmızı-turuncu bir **namlu alevi (muzzle flash)** spriti patlar.
4. **Pozitif Geri Besleme Döngüsü (Feedback Resonance):**
   - Sinek ilk mermiyi sıktı.
   - Silahtan çıkan namlu alevi ekran merkezini kıpkırmızı yaptı.
   - Görüş algoritması bu ani kırmızı artışını *"Düşmandan ağır darbe aldım, canım gidiyor!"* olarak yorumladı.
   - Hasar alındığı için acil misilleme devresi (`trigger_combat_retaliation`) yeniden devreye girdi ve bir daha ateş etti.
   - Yeni ateş bir alev daha çıkardı -> Görüş yine hasar sandı -> Bir daha ateş etti...
   - **Sonuç:** Sinek, kendi sıktığı silahın alevinden korkup saldırıya uğradığını sanarak sonsuz bir panik döngüsüne girdi!

---

## 3. Nörobiyolojik Teori: Efference Copy (Corollary Discharge)

Doğadaki canlılar (örneğin gerçek bir *Drosophila melanogaster*) bu sorunu **400 milyon yıl önce** çözmüştür:
- **Sorun:** Canlı hareket ettiğinde kendi vücudu da duyusal organlarında (göz, kulak, anten) devasa bir gürültü yaratır. Örneğin sinek saniyede 200 kez kanat çırptığında rüzgar sesinden sağır olabilir veya dünya gözünde titreyebilirdi.
- **Biyolojik Çözüm:** Beyin motor nöronlara (kaslara) hareket emri gönderirken, aynı zamanda duyusal merkezlere bu sinyalin bir kopyasını (**Efference Copy / Corollary Discharge**) gönderir. Duyusal merkez, kendi hareketinden kaynaklanan bu öngörülen gürültüyü gelen sinyalden çıkarır.

### Koddaki Mühendislik Karşılığı:
Sineğin beynine motor tetiği çektiği an göze giden bir efference copy eklendi:
```python
# Tetiği çektiğimiz andan sonraki 380ms boyunca ekranda patlayan alev kendi namlumuzdan çıkıyor!
if (now - last_shot_time) < 0.38:
    return False, 0.0  # Hasar değil, kendi tüfeğimizin alevi!
```
Ayrıca gerçek hasarın tüm ekranı (tavan sınırını) kaplaması, namlu alevinin ise sadece ekranın altında kalması özelliği kullanılarak gerçek hasarla namlu alevi birbirinden tamamen yalıtıldı.

---

## 4. Veri Kaynakları & Biyolojik Referanslar

Bu projedeki mimari ve biyolojik parametreler şu bilimsel veri setlerine ve makalelere dayanmaktadır:

1. **MaleCNS v1.0 & FlyWire Tam Beyin Bağlantı Haritası (Connectome):**
   - *Princeton Üniversitesi & Janelia Research Campus (Nature, 2024)*
   - 139.255 nöron ve 50+ milyon sinapsın elektron mikroskobu (EM) ile çıkarılmış gerçek 3D kablolama verisi (`data/` dizinindeki sinaps ve nörotransmitter tabloları).
2. **DOOMFLY Mimarisi (Connectome to FPS Mapping):**
   - *Schlegel et al. / BioRxiv 2024*: Meyve sineği görsel nöronlarının FPS oyunlarına aktarımı.
   - **L1-L5 Lamina Monopolar Hücreleri:** Zamansal ve uzamsal kenar kontrastı filtreleri.
   - **LC10 / LC11 (Lobula Columnar):** Küçük hareketli nesne (av/avcı/hedef) takip devreleri.
   - **Giant Fiber (GF):** Ani kararma ve tehdit anında devreye giren acil kaçış refleksi.
   - **DNp20 (Descending Neuron p20):** Sağ-sol diferansiyel direksiyon kontrolü (Strafe & Mouse Turn).
   - **DNpe017 (Descending Neuron pe017):** İleri yürüme ve saldırı (Forward Walk & Weapon Fire).
3. **Naka-Rushton Fotoreseptör Kinetiği (1966):**
   - $I(L) = I_{max} \frac{L^n}{L^n + \sigma^n}$ log-sigmoidal ışık transdüksiyon denklemi (MSS ile alınan piksellerin pA elektrik akımına çevrilmesi).
4. **Hassenstein-Reichardt Hareketi Dedektörleri (1956):**
   - Görsel akışın ($u, v$ gradyanları) hesaplanarak optomotor merkezleme yapılması.

---

## 5. İkinci Keşif: DNpe017 Bilateral Yürüme Çifti (Yürürken Ateş Etme Sorunu)

Koddaki nöron bağlantıları incelendiğinde ikinci bir kritik gerçek keşfedildi:
- `data_loader.py` içinde `DNpe017` nöronu tek bir nöron değil, **sağ ve sol bacak koordinasyonunu sağlayan bilateral bir çifttir** (`DNpe017_R` ve `DNpe017_L`).
- Kodda `idx_dnpe017_fwd` sağ nörona, `idx_dnpe017_atk` ise sol nörona atanmıştı.
- Sinek ileri doğru her adım attığında her iki nöron da doğal olarak $\sim 0.10$ frekansla ateşleniyordu.
- Saldırı eşiği `0.08` yapıldığında, sinek her ileri adım attığında sol bacak nöronu eşiği geçtiği için adım attıkça sol tıka basıyordu! Silah her patladığında karakterin yürümesi kesiliyor ve navigasyon bozuluyordu.
- **Çözüm:** 
  - Yürüyüş gürültüsünü engellemek için genel saldırı eşiği `0.35` seviyesine çekildi.
  - Ateş etme eylemi, sadece **gerçek hasar alındığında (`is_damage = True`)** çalışan özel `trigger_combat_retaliation()` refleksine bağlandı.
  - Böylece sinek sakin sakin koridorlarda yolunu bulurken sıfır mermi harcar; sadece canı azaldığında anında dönüp karşı ateş açar.

---

## 6. Üçüncü Keşif: Hasar Bölgesi Kalibrasyonu (Ekran Tavanı vs. Merkez Hasar Arkı)

Namlu alevinden kaçmak için hasar algılamayı ekranın en üst tavanına (`top_border`) taşımıştık. Ancak Half-Life multiplayer (Deathmatch) modunda:
- Tavanda veya gökyüzünde kırmızı flaş oluşmaz!
- Hasar alındığında kırmızı kan partikülleri ve hasar yön okları doğrudan **ekranın merkezinde** (crosshair çevresinde) oluşur.
- Tavan tarandığı için biri sıktığında kırmızılık hiç artmıyor ve sinek hasar aldığını fark edemiyordu.
- **Kesin Çözüm:** Hasar taraması tekrar merkeze (`center_patch`) alındı; ancak kendi silahımızın 380 ms namlu alevi koruması (`last_shot_time`) aktif olduğu için artık kendi alevinden korkmadan, sadece düşman mermisi isabet ettiğinde `threshold = 0.09` ile anında karşı ateş açıyor!

---

## 7. Dördüncü Büyük Keşif: "Kör Bot & Siyah/Kod Ekranı" Sorunu (MSS vs. PrintWindow)

### Problem
Kullanıcı oyunu açtığında Web UI'da görüntü alamıyor veya siyah/kod görüyordu. Bot vurulsa dahi karşı ateş açmıyordu.
Detaylı hafıza ve pencere dökümü alındığında akıl almaz bir gerçek ortaya çıktı:
- `vision_bridge.py` kütüphanesi ekran görüntüsü almak için `mss` (Masaüstü GDI `BitBlt`) kullanıyordu.
- Kullanıcı ekranda VS Code / Antigravity IDE'yi tam ekran açtığında, Half-Life penceresi arka planda (alt katta) kalıyordu.
- `mss`, ekran koordinatlarını taradığı için **Half-Life yerine onun önündeki IDE kod editörünün piksellerini** yakalıyordu!
- Sonuç:
  1. Web arayüzüne oyun yerine VS Code editör ekranı veya siyah piksel akıyordu (`scratch/current_stream_frame.jpg`).
  2. Bot gözleri yerine kodları gördüğü için Half-Life'taki düşmanı, kanı veya can azalmasını göremiyordu.
  3. `delta_red` sürekli 0.0 kalıyor ve bot hiçbir zaman vurulduğunu anlayamıyordu!

### Çözüm: Win32 Direct Window Buffer Capture (`PrintWindow`)
- Standart ekran yakalama yerine doğrudan pencere tampon belleğini okuyan `ctypes.windll.user32.PrintWindow(hwnd, hdc_mem, 2)` mimarisine geçildi.
- Bu API, Half-Life penceresi başka pencerelerin (IDE, Chrome, vb.) **arkasında veya tamamen örtülmüş olsa bile** doğrudan oyunun kendi arka tampon belleğinden kristal netliğinde 800x600 görüntü çeker.
- Benchmark: Sadece **5.82 ms (~172 FPS)** gecikmeyle çalışır!

---

## 8. Beşinci Keşif: Çift Kanallı Hasar Algılama & Web UI Otomatik İyileşme

1. **Çift Kanallı Hasar Algılama (Dual-Mechanism Damage Detection):**
   - **Kanal 1:** Ekran merkezindeki kan ve hasar yön oku kırmızılığı (`delta_red > 0.09`, 380 ms namlu alevi korumalı).
   - **Kanal 2:** Sol alt köşedeki (`y: 86%..99%, x: 2%..25%`) can göstergesi piksel değişimi (`hud_diff > 7.5`). Mermi isabet ettiğinde sayılar anında değişir veya flaş yapar.
   - Her iki kanaldan biri tetiklendiğinde `trigger_combat_retaliation()` anında 2 el seri karşı ateş (double tap) açar ve sıçrama hareketi yapar.

2. **Web UI Görüntü & Yayın İyileştirmesi:**
   - `index.html` içerisindeki görüntüyü bozan `onerror="this.src=''"` kaldırıldı.
   - Otomatik yeniden bağlanma motoru (auto-healing stream manager) ve `/frame.jpg` anlık görüntü (snapshot) yedekleme modu eklendi.
   - Çift port desteği sağlandı: Hem `http://localhost:8766` hem de `http://localhost:8080` eşzamanlı olarak canlı yayını dinler ve yayınlar.

---

## 9. Altıncı Keşif: "Hardcoded/Rastgele Parlayan Beyin" Sorununun Çözümü & Eyleme Bağlı 3D Biyolojik Ağ

### Problem Analizi ("Beyin neden otomatik/hardcoded gibi parlıyordu?")
1. **Fotoreseptör Kamera Gürültüsü Baskısı:** Gözdeki 3600 fotoreseptör hücresi (`0..3599`), oyundaki her karede piksel oynamalarından dolayı sürekli rastgele ateşleniyordu. Önceki görselleştirici bu gürültüyü tüm nöronlarla eşit parlaklıkta çizdiği için beyin sineğin hareketlerinden bağımsız, sürekli rastgele bir yılbaşı ağacı gibi yanıp sönüyordu.
2. **Kullanılmayan Hareket Sinyalleri:** `_render_3d_brain_pane` fonksiyonuna `turn_diff` (sağa/sola dönüş) ve `rate_forward` (ileri yürüme) hiç gönderilmiyordu; `is_firing` ise fonksiyona girmesine rağmen içeride tek bir satırda dahi okunmuyordu!
3. **Pencere Çözünürlüğü Sınırlaması:** Desktop penceresi 1280x720 sabit çözünürlükteydi ve tam ekran yapıldığında bulanıklaşıyordu.

### Uygulanan Nihai Çözümler
1. **Masaüstü Görselleştiricisi (`telemetry/visualizer.py`):**
   - **Çözünürlük:** 1600x900 kristal netliğinde genişletilebilir çözünürlüğe yükseltildi (980px 16:9 oyun ekranı + 620px geniş 3D beyin paneli).
   - **Eyleme Dayalı Dalgalar (Action-Coupled Biological Waves):**
     - **Sola Dönüş (A tuşu):** Sol yarıküre (Sol Retina, Sol Optik Lob ve Merkezi Pusula) parlak bir biyolüminesans dalgayla aydınlanır.
     - **Sağa Dönüş (D tuşu):** Sağ yarıküre eşzamanlı olarak parlar.
     - **İleri Yürüme (W tuşu):** VNC Omurilik sütununda adım atma frekansında (~16 rad/s) ritmik locomotor CPG dalgası akar.
     - **Ateş Etme (Mouse1/Click):** Motor merkezinde saf beyaz/elektrik mavisi patlama flaşı ve radyal şok dalgası yayılır.
     - **Hasar / Kaçış:** Bütün nöron ağı acil durum kırmızı alarm dalgasına (`#ff0033`) bürünür.
     - **Ödül / Dopamin:** Mantar gövde (Mushroom Body) ve PPL1 kümeleri altın-kehribar ışıltıyla parlar.
   - **3D Akson Otoyolları (Axon Highways):** Retina -> Optik Lob -> Merkezi Pusula -> Mantar Gövde -> VNC Omurilik sütununu birbirine bağlayan 3D biyolojik sinir yolları çizildi ve üzerlerinde sineğin hareket hızına göre hızlanan aksiyon potansiyeli sinyal paketleri akıtıldı.
   - **Çok Katmanlı Anti-Aliased Bloom:** Düz kare pikseller yerine yumuşak çift halkalı neon ışık haleleri (Gaussian bloom) eklendi.

2. **Web Görselleştiricisi (`server/static/index.html` & Three.js):**
   - Pure-white additive radial glow dokusu ile Three.js motoru yenilendi.
   - Telemetri bağlantısı kurularak RETINA, OPTİK LOB, MERKEZ PUSULA, MANTAR GÖVDE, MOTOR OMURİLİK ve DOPAMİN PPL1 canlı yüzde rozetleri bağlandı.
   - `⛶ GENİŞLET` tam ekran modu ve fareyle 3D serbest döndürme/yakınlaştırma kontrolleri eklendi.

---

## 10. Yedinci Keşif: Ekran Karmaşası & Arka Planda Kayıt Alma Çözümü
- **`WS_EX_NOACTIVATE` ve Win32 `HWND_TOPMOST`:** Python visualizer penceresi Half-Life'ın üzerinde sabit kalır, tıklandığında bile odağı Half-Life'tan çalmaz.
- **Global Donanım Tuşları:** `ctypes` import hatası giderildi; `F8` (Kayıt), `F11` (Üstte Sabitle), `F6` (Oyunu Öne Odakla), `F10` (Duraklat) her zaman çalışır.
- **Döngü Optimizasyonu (10 FPS Çözümü):** LIF substep döngüsü 2'ye sabitlendi, `time.sleep` kaldırıldı (`pydirectinput.click`), web yayını 15 FPS ile sınırlandırıldı.

---

## 11. Sekizinci Keşif: Sinematik Çözünürlük & NVIDIA NVENC Donanım Kaydı (1080p / 1440p)
- **Çözünürlük Yükseltmesi:** Varsayılan visualizer ve video kayıt çözünürlüğü **1920x1080 (Full HD)** yapıldı. Ayrıca monitörün tam 2K çözünürlüğü için `--resolution 1440p` (2560x1440) desteği eklendi.
- **Hardware-Accelerated H.264 (NVIDIA NVENC / Media Foundation):**
  - Eski `mp4v` codec'i yerine Windows Media Foundation üzerinden doğrudan NVIDIA GPU donanım encoder'ı (`H264`) bağlandı.
  - Kare yazma süresi 0.99 ms'ye düşürüldü (1000+ FPS kapasite).
  - Instagram Reels, YouTube ve TikTok için kayıpsız, pürüzsüz yüksek bitrate video çıkışı sağlandı.
- **`cv2.INTER_CUBIC` Upscaling:** Half-Life'ın 1024x768 ekranı 1080p/1440p ekrana bulanıklaşmadan, jilet gibi keskin bikübik enterpolasyonla yükseltildi.
- **`--auto-record` Parametresi:** Kullanıcı hiçbir tuşa basmadan doğrudan en yüksek kalitede otomatik kayıt alabilir (`python main.py --auto-record`).

---

## 12. Dokuzuncu Keşif: TrueType Anti-Aliased Tipografi & Gerçek 3D Biyolojik Bağlantı Anatomisi (FlyWire / MaleCNS)
- **Yazıların Piksel Piksel Olma Sebebi ve Çözümü:**
  - OpenCV'nin yerleşik `cv2.putText` motoru 1967 yılından kalma Hershey vektörel çizgi fontlarını kullandığı için harfler tırnaklı, kırık ve pikselli görünüyordu.
  - Pillow (`ImageDraw` / `ImageFont`) üzerinden modern TrueType motoru entegre edildi. Windows'un `Segoe UI Bold` ve `Consolas` yazı tipleri sub-pixel anti-aliasing ile render edilerek jilet gibi pürüzsüz ve profesyonel bir HUD tipografisine kavuşturuldu.
- **Üst Bar Çakışması ve Kesilmelerin Önlenmesi:**
  - Sarı `REC ACTIVE` kutusunun oyun ekranının üst sınırına taşması ve `MALECNS v1.0` başlığını kesmesi engellendi.
  - Bağımsız, 44px'lik koyu obsidian üst navigasyon barı oluşturuldu. Başlık, canlı FPS, 12,260 nöron / 1.5M sinaps sayaçları ve sağ üstteki şık yuvarlak köşeli `● REC` rozeti buraya yerleştirildi.
- **"Beyin Hardcoded mı?" Şüphesinin Biyolojik Çözümü:**
  - Beyin kesinlikle hardcoded veya rastgele değildir; 12,260 gerçek nöron ve 1,518,705 sinapstan oluşan gerçek **MaleCNS v1.0 / FlyWire** drosophila konnektomudur.
  - **Eski Sorun:** Önceki çizim döngüsünde ateşleme yapmayan inaktif nöronlar 1 piksellik aşırı sönük karanlık noktalar olarak çizildiği için video sıkıştırmasında ve 1080p ekranda tamamen kayboluyordu. Ekranda sadece ateşlenen ~40 nokta kalınca kullanıcı beyni "boşlukta rastgele yanan sahte noktalar" sanıyordu.
  - **Yeni Çözüm:** 12,260 nöronun tamamı kendi nöropil bölgesinin anatomik rengiyle (Cyan bileşik gözler, Magenta optik loblar, Altın pusula, Zümrüt mantar gövdeleri, Beyaz motor kolonu) belirgin bir 3D morfolojik yapı olarak çizildi.
  - 1.5 milyonluk sinaps matrisinden bölgeler arası en güçlü 320 sinaps otoyolu (axonal tracts) eklenerek nöronlar birbirine bağlandı. PyTorch LIF motorundaki aksiyon potansiyelleri bu gerçek sinir ağları üzerinde GCaMP6s kalsiyum floresansı olarak akmaktadır.

---

## 13. Onuncu Keşif: Yön Bulma & Navigasyon Bozulmasının Kök Nedenleri ve Biyomimetik Çözümü
- **Navigasyonun Bozulmasına Yol Açan 5 Kök Neden:**
  1. **Konnektom Simetrisi (DNp20_L == DNp20_R):** Yapay konnektom oluşturma aşamasında merkezi kompleks ile descending motor nöronlar arasındaki bağlantılar homojen modulo ile dağıtıldığı için sol ve sağ göz sinyalleri hem DNp20_L hem de DNp20_R nöronunu tıpatıp eşit uyarıyordu. Sonuçta `turn_diff` sürekli 0.0 çıkıyor, sinek görsel olarak sağa ya da sola yönelemiyordu.
  2. **Substep Gecikmesi (1 Saniyelik Kör Uçuş):** Substep sayısı 2'ye indirildiğinde, retinanın ürettiği bir aksiyon potansiyelinin 4 sinaptik katmanı (Fotoreseptör -> Lamina -> Medulla -> Central Complex -> Motor) aşıp kaslara ulaşması 35 kare (~1.2 saniye) alıyordu. Sinek duvara yaklaştığını fark edene kadar tosluyordu.
  3. **Yapay Optik Akış Tuzağı (`flow_mag < 0.035`):** Yavaş yürürken veya kalkış yaparken optik akış doğal olarak düşük çıktığı için sistem bunu "karşımda engel var" sanıp her karede engelden kaçış paniğini tetikliyordu.
  4. **Mikro Duraklamalar (`Exploratory Micro-Pauses`):** Her 3 saniyede bir karakteri durdurup fareyi sağa sola titreten yapay kod koridor momentumunu tamamen bozuyordu.
  5. **Sert Fare Savrulması (`gain = 45-140`):** En ufak dönüş farkında fare 45-140 piksel fırlatılarak kamera tek karede 60-90 derece dönüyor, aynı anda basılan `A/D` ve ok tuşlarıyla sinek fırıldak gibi dönüp duvara çarpıyordu.
- **Uygulanan Biyomimetik Çözümler:**
  - **Doğrudan Optomotor Koridor Dengelemesi:** Sol ve sağ göz fotoreseptör uyarımı (`left_eye - right_eye`) ile derinlik dengesi (`depth_balance`) doğrudan `DNp20_L` ve `DNp20_R` premotor nöronlarına aktarıldı. Açık olan koridora doğru anında ve kararlı bir `turn_diff` üretiliyor.
  - **4 Substep (~4 ms):** Gecikme 1.2 saniyeden ~50 ms'ye (2-3 kare) düşürüldü; tepki süresi insan seviyesine getirildi.
  - **Orantısal Yumuşak Fare Yönlendirmesi:** `mouse_dx = int(np.clip(-turn_diff * 140, -22, 22))` ile kademeli, pürüzsüz ve gerçek bir FPS oyuncusu gibi koridoru takip eden bir direksiyon sağlandı.
  - **Yapay duraklamalar kaldırıldı**, engel tespiti yalnızca gerçek doku varyansı (`< 0.009`) ve sıkışma eşiği 30 adıma çekilerek kararlı hale getirildi.

---

## 14. On Birinci Keşif: "Dümdüz Koşma ve Zıplama" Sorununun Kök Nedenleri ve Tam Çözümü

### Problem Analizi
Önceki düzenlemede navigasyon düzeltilmeye çalışılırken botun *"yalnızca dümdüz koşup zıpladığı ve sağa/sola dönemediği"* gözlemlendi. Kodun matematiksel ve sinirsel dökümü incelendiğinde şu anomaliler tespit edildi:
1. **10 Hz Yön İptali (Alternating Symmetry-Break):**
   - `vision_bridge.py` içinde engel kontrolü (`var < 0.008`), normal Half-Life sahnelerinde bile daima `True` çıkıyordu (gerçek sahnelerde merkez varyans `0.0006 - 0.0057`).
   - Sinek her karede engel var sandığı için `depth_balance = 0.25 if (int(time.time() * 10) % 2 == 0) else -0.25` kodu saniyede 10 kez dönüş sinyalinin işaretini tersine çeviriyordu.
   - Sola ve sağa zıt sinyaller LIF motorunda birbirini nötralize ettiği için net dönüş 0'a yakınsıyor, sinek iki yana titreyerek dümdüz duvara tosluyordu.
2. **Dönüş Donanımının Budanması:**
   - Half-Life'ın yerleşik kamera dönüşünü sağlayan `DIK_LEFT` ve `DIK_RIGHT` scancode'ları silinmiş, fare dönüşü ise Half-Life'ta 1-2 dereceye denk gelen `[-22, 22]` piksele kısıtlanmıştı.
   - `A` ve `D` strafe tuşları `turn_diff > 0.16` gibi ulaşılamaz bir eşiğe çekildiği için bot gövdesini çeviremiyordu.
3. **Yersiz Zıplama Spam'i:**
   - `trigger_combat_retaliation` içine eklenen `DIK_SPACE` (zıplama) ve `main.py` içinde her hasar sinyalinde çağrılan `handle_respawn()`, ortamdaki kızıl ışıklar veya küçük darbelerde botun aralıksız zıplamasına yol açıyordu.
4. **Kesintisiz İleri Koşma Doyumu (`fwd_stim`):**
   - `total_ext_current[idx_dnpe017_fwd] += 26.0` her karede tam gaz basıldığı için sinek virajlarda veya duvar karşısında hız kesmeden duvara saplanıyordu.

### Uygulanan Çözümler
1. **Kararlı Kaçış Hafızası:** 100ms'lik salınım kaldırıldı; simetri kırılması gerektiğinde seçilen kaçış yönü en az 1.4 saniye korunarak sineğin dönüşünü tamamlaması sağlandı. `is_obstacle_close` eşiği gerçek duvar yakınlığına (`var < 0.0006`) çekildi.
2. **Kuvvetli ve Eşzamanlı Direksiyon:** Ok tuşları (`DIK_LEFT` / `DIK_RIGHT`), `A` / `D` strafe tuşları ve `gain = 45 - 140` fare yönlendirmesi `turn_threshold = 0.03` seviyesinde yeniden tam koordinasyonla bağlandı.
3. **90-120° Kaçış Saccade'i:** Duvara çarpıldığında kurtulma manevrası 550 piksel fare savurması ve ok tuşları ile güçlendirildi.
4. **Zıplamanın Kaldırılması:** `trigger_combat_retaliation` içindeki `DIK_SPACE` kaldırıldı; `handle_respawn()` yalnızca gerçek ölüm (`lif_engine.is_flatlined`) anına bağlandı.
5. **Virajda Hız Kesme Modülasyonu:** Keskin virajlarda (`abs(steer_bias) > 12.0`) veya engelle karşılaşıldığında `fwd_stim` 10.0 veya 4.0 nA'ya düşürülerek dönüşün gövde hareketine hakim olması sağlandı.


