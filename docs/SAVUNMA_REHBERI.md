# Savunma Rehberi — Sunumda Neyi, Nasıl Anlatmalı?

Bu belge, raporu **akademik sunumda savunabilmeniz** için hazırlanmıştır. Rakamların kaynağı
`results/tables/*.csv` dosyalarıdır; rapordaki her sayı bu dosyalardan otomatik okunur.

> **Teslimden önce:** (1) `reports/sections_static.py` içindeki `STUDENT_INFO` alanını doldurun ve
> `python main.py --stage report` ile raporu yeniden üretin; (2) raporu **kendiniz okuyun** —
> savunmada her cümleyi açıklayabilmelisiniz; (3) kaynakçadaki künyeleri son kez kontrol edin
> (5 künye geliştirme sırasında web üzerinden doğrulandı; diğerleri klasik kaynaklardır).

---

## 1. İki dakikalık özet (açılış konuşması)

> "100 ürünlü 0/1 Knapsack örneğinde (kapasite = toplam ağırlığın %70'i = 1906.8, yuvarlama yok) GA, ACO,
> ikili PSO ve SA'yı **aynı veri, aynı 30 tohum ve neredeyse aynı değerlendirme bütçesiyle** (≈8000 nesne
> fonksiyonu değerlendirmesi) karşılaştırdım. Gerçek optimumu dinamik programlama ile hesapladım (4560) ve
> bağımsız bir MILP çözücüsüyle doğruladım. ACO 30 çalıştırmanın tamamında optimumu buldu; GA ortalamada
> %0.04 geride; SA ve özellikle ikili PSO daha zayıf. Ama asıl ilginç bulgular şunlar: (1) ikili PSO'nun
> kötü sonucu algoritmadan değil, ödevde verilen atalet aralığından geliyor — w=1.0 ile GA düzeyine çıkıyor;
> (2) kısıt yönetimi (onarım) algoritma seçiminden daha belirleyici — doldurmalı onarımla SA ve PSO da
> optimuma yaklaşıyor; (3) kendi geliştirdiğim GA-SA hibriti bu kolay örnekte GA'dan ayırt edilemedi ve
> bunu ablasyonla açıkladım."

## 2. Problem ve veri — söylenmesi gerekenler

| Bilgi | Değer |
|---|---|
| Ürün sayısı / toplam ağırlık / toplam değer | 100 / 2724 / 5121 |
| Kapasite | 0.70 × 2724 = **1906.8** (yuvarlanmadı; tamsayı ağırlıklarda uygun küme ⌊C⌋ = 1906 ile aynı) |
| Gerçek optimum (DP) | **4560** (ağırlık 1904, 78 ürün) — MILP (HiGHS) ile aynı |
| Oran-açgözlü sezgisel | 4552 (%0.18 geride) |
| Doğrusal gevşetme üst sınırı | 4565.38 |
| Temizleme | Eksik/tekrar/geçersiz kayıt **yok**; 2 ürün aynı (w, v) çiftini paylaşır, korundu |

**Mutlaka söyleyin:** örnek *kolay*dır (açgözlü bile %0.18 yakın). DP bu örneği milisaniyelerde kesin çözer;
meta-sezgiseller burada bir **karşılaştırma vasıtasıdır**, pratik çözüm yöntemi değil.

## 3. Algoritmalar nasıl çalışır (jüri diliyle)

**Ortak yapı.** Hepsi aynı değerlendiriciyi kullanır (her nesne fonksiyonu çağrısı sayılır, kapasiteyi aşan çözüm
değerlendirilemez). Kapasiteyi aşan çözüm *ortak onarımla* düzeltilir: seçili ürünlerden **değer/ağırlık oranı en düşük**
olanlar, kapasite sağlanana dek çıkarılır.

- **GA.** İkili kromozom (x_i = 1: ürün çantada). Turnuva seçimi (k=3) → tekdüze çaprazlama (p_c=0.9) → bit-flip mutasyon
  (p_m) → onarım → elitizm (2). *Mutasyon keşif–sömürü dengesini belirler:* çocuk başına beklenen değişen gen sayısı n·p_m.
  Küçükse popülasyon hızla homojenleşir (sömürü, erken yakınsama); büyükse iyi yapılar bozulur (keşif, kararsızlık).
- **ACO.** TSP'den farkı: feromon *ürünlere* konur (çözüm bir küme, tur değil). Karınca boş çantadan başlar, her adımda
  *kalan kapasiteye sığan* adaylardan olasılıkla bir ürün ekler: p_i ∝ τ_i^α · η_i^β, η_i = değer/ağırlık oranı. Aday kalmayınca
  durur → çözüm **uygun ve maksimaldir**. Feromon: buharlaşma ρ + en iyi çözüme pekiştirme + MAX–MIN sınırları.
- **İkili PSO.** Hız denklemi sürekli PSO ile aynı; ama konum `x = 1 eğer rand < sigmoid(v)` ile örneklenir. Hız bitin 1 olma
  *olasılığını* belirler. **Kritik ayrıntı:** v≈0 iken sigmoid=0.5, yani yakınsarken hız sönümlenirse bitler yazı-tura gibi çevrilir;
  küçük atalet bu gürültüyü artırır. Bu yüzden *büyük atalet daha iyidir* (sürekli PSO sezgisinin tersi).
- **SA.** Tek çözüm; komşu = swap (çıkar-ekle) veya bit-flip; Metropolis: P = min(1, e^{Δf/T}). Yüksek T'de kötü hamle kabulü
  keşfi sağlar, T düştükçe açgözlü aramaya dönüşür. Geometrik soğutma T_k = T_0·α^k.
- **GA-SA hibriti (bonus).** Her nesilde 28 çocuk GA ile üretilir; en iyi 2 birey kısa SA zinciriyle (6'şar komşu) iyileştirilir,
  zincirin en iyisi bireyin yerine geçer. Nesil başı 28 + 12 = 40 değerlendirme (bütçe uyumlu).

## 4. Deney tasarımı — neden adil? (savunma noktaları)

1. **Aynı veri, kapasite, 30 önceden tanımlı tohum** (2000–2029); koşular tohumlar arası serpiştirilir, süre ölçümü için ısınma turu yapılır.
2. **Aynı iterasyon ≠ aynı maliyet:** her değerlendirme sayıldı: GA 7640, ACO 8000, BPSO 8040, SA 8001, GA-SA 8030 (nominal 8040).
   *Tüm algoritmalar tam 8040 FE ile yeniden çalıştırıldığında sonuçlar ≤0.03 birim değişti* → bütçe farkı sonucu çarpıtmıyor.
3. **Ana parametre seçimi tarafsız:** her algoritmanın 3 değerlik ızgarası, raporlanmayan 10 *pilot* tohumla denendi; en iyisi
   seçildi (her algoritmaya eşit ayar çabası). Hibrit 9 adaydan seçildi → hibrit lehine yanlılık (raporda belirtildi).
4. **Süre ölçümü sağlam:** her (algoritma, tohum) çifti 3 kez koşturulup *medyan* alındı; üç tekrarın aynı çözümü verdiği otomatik denetlendi.
5. **İstatistik:** normallik reddedildiği için parametrik olmayan testler (Friedman, Kruskal–Wallis; Wilcoxon, Mann–Whitney; Holm düzeltmesi; A12, Cohen d).
6. **Tekrarlanabilirlik:** ana deneyi sıfırdan koşup kayıtlı sonuçlarla karşılaştırdım: 150 çalıştırmanın çözüm vektörleri bit düzeyinde aynı.
7. **Test:** 133 birim testi (brute-force ile DP/MILP, tüm algoritmalar, operatörler…). Testlerin gücünü *mutasyon testiyle* ölçtüm: kasıtlı eklenen 11 farklı hatanın tamamı yakalanıyor (ilk turda 3'ü kaçmıştı; bunlar için test eklendi).

## 5. Bulguların anlamı (tablo)

| Algoritma | Ortalama ± std | Optimumu bulma | Yorum |
|---|---|---|---|
| ACO | 4560.00 ± 0.00 | %100 | Sezgisel bilgi (değer/ağırlık) + maksimal inşa; en güvenilir, en pahalı |
| GA | 4558.10 ± 2.44 | %53 | En iyi hız/kalite dengesi |
| SA | 4554.57 ± 7.01 | %6.7 | Soğutma çizelgesi bütçeye göre erken soğuyor (≈63. iterasyonda kötü hamle kabulü bitiyor) |
| Binary PSO (w=0.9) | 4418.17 ± 37.78 | %0 | Sigmoid gürültüsü; **w=1.0 ile 4558.17 ± 2.00** |
| GA-SA (bonus) | 4557.33 ± 3.16 | %37 | GA'dan ayırt edilemedi (p_Holm ≈ 0.29); SA payı ablasyonu düz |

**İstatistiksel ≠ pratik anlamlılık (mutlaka vurgulayın):** ACO–GA farkı anlamlıdır (Holm sonrası p≈0.002) ama pratikte 1.9 birimdir
(%0.04); asıl fark **güvenilirliktedir** (%100 vs %53).

## 6. Jüri soruları — hazır cevaplar

**S: Neden ACO en iyi?**
A: İki bileşen: değer/ağırlık sezgisi ve her zaman maksimal çözüm inşası. Kanıt: β=0 (sezgisiz) ACO 4554.03'e düşer (%17 optimum);
doldurmalı onarımla SA 4560.00 (%100), PSO 4559.83 olur. Yani üstünlük kısmen *bilgi* avantajıdır, algoritma doğası değil.

**S: İkili PSO'da hata mı var?**
A: Hayır; birim testleriyle (sigmoid, hız sınırı, uygunluk, FE sayısı) ve w ablasyonuyla doğrulandı. w=0.4/0.7/0.9/1.0 için ortalama
3943/4139/4418/4558: w büyüdükçe iyileşiyor. Mekanizma: v→0 iken sigmoid 0.5 → bitler rastgele çevrilir. Özgün Kennedy–Eberhart PSO'su w=1 kullanır.

**S: Eşit bütçeyi nasıl sağladın?**
A: Her değerlendirme sayıldı (başlangıç dâhil). Fark ≤%5; ayrıca tüm algoritmalar tam 8040 FE'ye zorlandığında ortalamalar ≤0.03 birim değişti.

**S: Onarım yöntemi bazı algoritmalara haksız avantaj mı?**
A: Evet, ve bunu ölçtüm (Bölüm 6.3): rastgele-çıkar → oran-çıkar → oran-çıkar+doldur geçişinde GA 4550.7→4558.1→4559.6, PSO 4289→4418→4559.8,
SA 4551.2→4554.6→4560.0. Ana deneyde herkes aynı (oran-çıkaran) onarımı kullanır; ACO ek olarak maksimal inşa ve η sezgisinden yararlanır.

**S: Hibrit neden işe yaramadı?**
A: (i) GA zaten optimuma ~2 birim yakın; kalan fark çok-ürünlü değişim ister. (ii) SA payı ablasyonu 0/6/12/20 için düz (4557.0–4557.5); 30'da kötüleşiyor.
(iii) Pilotta seçilen konfigürasyonun üstünlüğü (4559.0) raporlanan tohumlarda kayboldu (4557.3) — seçim yanlılığı. (iv) Kapasite oranı analizinde %30–%50'de GA'dan kötü.
Dürüst sonuç: değeri bu örnekte kanıtlanamadı; daha zor örneklerde sınanmalı.

**S: Neden 10 değil 30 çalıştırma?**
A: Ödev alt sınır koyuyor; koşular ucuz, 30 örnek istatistiksel gücü artırıyor. İlk 10 tohum ayrıca incelenebilir.

**S: Neden hem Wilcoxon hem Mann–Whitney?**
A: Tohum eşleştirmesi *nominaldir* (algoritmalar farklı rastgele akış tüketir). Eşleştirilmiş ve eşleştirilmemiş testler aynı sonucu verdi → bulgu sağlam.

**S: Gerçek optimumu nasıl doğruladın?**
A: DP (kesin), MILP (HiGHS) ile çapraz doğrulama ve küçük örneklerde brute-force testleri; üçü de aynı değeri veriyor.

**S: Süreler güvenilir mi?**
A: Mutlak süreler makineye/yüke bağlı; bu yüzden medyan-of-3 protokolü kullandım ve ilk (tek geçişli) ölçümle farkı raporda şeffafça belirttim.
Sıralama (ACO en yavaş; GA ve PSO en hızlı) tutarlı. Ayrıca süreler vektörize NumPy uygulamasına bağlı.

**S: Sonuçlar başka problemlere genellenir mi?**
A: Hayır. Tek, kolay örnek. Kapasite oranı analizi bile sıralamanın değiştiğini gösteriyor (ACO %30'da %60 isabet, GA %77).

**S: DP varken neden meta-sezgisel?**
A: Bu çalışmada amaç algoritma davranışını ve deney metodolojisini incelemek. Pratikte bu veri için DP önerilir; meta-sezgiseller DP'nin uygulanamadığı varyantlar içindir.

## 7. Dürüstçe söyleyebileceğiniz sınırlılıklar

- Tek, kolay örnek; açgözlü sezgisel bile %0.18 yakın.
- Her algoritmada yalnızca bir parametre değiştirildi; c1, c2, T0, β, turnuva boyutu sabit.
- Binary PSO için post-hoc (w=1.0) düzey eklendi — ödev dışı, açıkça "ek" işaretli.
- Süreler tek makinede ve vektörize uygulamayla ölçüldü; mutlak değerler genellenemez.
- Hibrit tek tasarımla denendi; seçim yanlılığı mevcut.

## 8. Önerilen sunum akışı (≈10 dk)

1. Problem + veri + referans optimum (DP/MILP) — *1 slayt*
2. Dört algoritmanın tek cümlelik sezgisi + probleme özgü uyarlamalar — *2 slayt*
3. Adil deney tasarımı (FE bütçesi, tohumlar, pilot seçim) — *1 slayt*
4. Ana sonuçlar: yakınsama (fig02), kutu grafiği (fig04), tablo — *2 slayt*
5. Parametre duyarlılığı (fig09) ve PSO sürprizi — *1 slayt*
6. Onarım/bilgi ablasyonu (fig15) — *1 slayt* (en özgün bulgu)
7. Hibrit: tasarım + dürüst sonuç (fig12) — *1 slayt*
8. Sınırlılıklar + öneri (ACO güvenilirlik, GA hız; pratikte DP) — *1 slayt*

## 9. Dosya haritası (neyi nerede bulursunuz)

| İstek | Dosya |
|---|---|
| Ana sonuç tablosu | `results/tables/summary_main.csv` |
| İstatistik testleri | `results/tables/pairwise_main.csv`, `omnibus_main.csv`, `normality_main.csv` |
| Ham koşular (çözüm vektörü dâhil) | `results/raw/main_runs.csv` |
| Parametre duyarlılığı | `results/tables/summary_sensitivity_all.csv` |
| Onarım ablasyonu / ACO β / hibrit payı / kapasite | `results/tables/summary_repair_*.csv`, `summary_aco_beta.csv`, `summary_hybrid_share.csv`, `summary_capacity.csv` |
| Grafikler (PNG 300 dpi + PDF) | `results/figures/` |
| Rapor | `reports/final_report.pdf` |
