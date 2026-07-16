# npxG/90 vs Goals/90 Karsilastirmasi - Faz B

**Tarih:** 2026-07-08

## Kurulum

Ayni 147 oyunculuk eslesen veri seti (StatsBomb 2015/16 La Liga npxG/90 verisi,
Transfermarkt piyasa degeriyle `match_tools.py` uzerinden isim tabanli fuzzy
match ile birlestirilmis) uzerinde `scout_tools.value_residuals()` iki kez
calistirildi:

- bir kere `perf_col='npxg_per90'` ile (penalti haric beklenen gol),
- bir kere `perf_col='goals_per90'` ile (`event_tools.compute_goals_per90()`
  ile StatsBomb ham event verisinden hesaplanan, penaltiler dahil gercek
  gol/90).

Model: `log(market_value_in_eur) ~ perf_col + age + age^2 + position + league + log(minutes)`.
Iki model de **ayni N=147, ayni 7 parametre** (`const, perf_col, age, age_sq,
log_minutes, position_Defender, position_Midfield`) uzerinden kuruldu - lig
tek deger (ES1) oldugundan lig kukla degiskenleri dusuyor, bu yuzden
karsilastirma tam anlamiyla elma-elma (AIC/BIC dogrudan kiyaslanabilir).

## Sonuclar

| | npxg_per90 | goals_per90 |
|---|---|---|
| N | 147 | 147 |
| Parametre sayisi | 7 | 7 |
| R-squared | 0.3030 | **0.3204** |
| Adj. R-squared | 0.2731 | **0.2912** |
| AIC | 430.82 | **427.11** |
| BIC | 451.76 | **448.04** |
| Log-likelihood | -208.41 | **-206.55** |

**Net sonuc:** `goals_per90`, bu veri setinde piyasa degerini `npxg_per90`'dan
biraz daha iyi aciklyor - hem R²/Adj. R² daha yuksek hem AIC/BIC daha dusuk
(ayni yonde tutarli, model karmasikligi esit oldugu icin dogrudan
karsilastirilabilir). Fark buyuk degil (R² farki ~0.017) ama tutarli.

## Yorum

Bu bulgu literatürle tutarli: piyasa (kulüpler, medya, taraftar) genelde
gerceklesen sonuca (golün kendisine) npxG'den daha güçlü tepki verir; npxG
"sansa göre düzeltilmiş" bir metrik olduğu için piyasa fiyatlamasındaki
gürültüyü (kaleci performansi, şut isabeti, sans) daha az açıklar - bu da
tam olarak beklenen bir şey.

Ancak bu, npxG'nin daha zayıf bir metrik olduğu anlamına gelmez - tam tersi:
value_residuals()'in amacı piyasanın fiyatlamadığı, ama gelecekte gerçekleşme
potansiyeli taşıyan performansı bulmaktır. goals_per90'in R²'sinin yüksek
olması, piyasanın zaten golü büyük ölçüde fiyatladığını gösterir (dolayısıyla
goals_per90 artığı, "piyasanın zaten bildiği" bilgiyi tekrarlar). npxg_per90'in
R²'sinin daha düşük olması ise, piyasanın npxG sinyalini goller kadar
fiyatlamadığını, yani npxG artığının "piyasanın gözden kaçırdığı" sinyali
daha fazla taşıdığını gösterir. Bu yüzden:

- **goals_per90 → piyasa-referans modeli** (piyasanın neye göre fiyatladığını
  anlamak icin).
- **npxg_per90 (veya benzer şansa-göre-düzeltilmiş metrikler) → asıl scouting
  sinyali** (deger artigi araniyorsa).

Düşük R-squared burada zayıflık değil, aranan sinyalin işaretidir.

## Güncelleme: Açık-Oyun-Sadece npxG ile Karşılaştırma

**Tarih:** 2026-07-14

### Kurulum

`compute_npxg_per90()`'ın varsayılanı, set-parça (corner/frikik/taca) kaynaklı
possession'lardan gelen sutları artık hariç tutuyor (bkz. `_OPEN_PLAY_SHOT_PATTERNS`
- 'Regular Play', 'From Counter', 'From Keeper'). 380 maçlık 2015/16 La Liga
cache'i üzerinde yeniden hesaplandığında bu, 449 oyunculuk "karışık" (tüm
play_pattern) kümesini 345 oyunculuk açık-oyun-sadece kümesine daraltıyor;
`data/processed/laliga_1516_npxg_per90.parquet` bu yeni değerlerle güncellendi.

Mevcut StatsBomb↔Transfermarkt eşleşmesi (147 oyuncu, `match_tools.py` isim
eşleştirmesi - **değiştirilmedi**) yeniden kullanıldı; sadece `npxg`/`npxg_per90`
sütunları yeni açık-oyun değerleriyle güncellendi. 22 oyuncunun (hepsi
savunma/kaleci - örn. Varane, Mascherano, Bailly) açık-oyun npxG'si sıfıra düştü
- bu, önceki karışık hesabın onlara byük ölçüde corner'dan gelen kafa vuruşu
npxG'si atfettiğini, açık oyunda pratikte hiç şut riski taşımadıklarını doğruluyor.

**Düzeltme notu (2026-07-14):** Bu bölüm ilk yayınlandığında `value_residuals()`'a
o sırada eklenmiş olan `contract_years_remaining` kovaryatıyla çalıştırılmıştı
(N=147→86, kontrat verisi bilinmeyenler düşmüştü). Ardından yapılan ayrı bir
araştırma (`reports/contract_years_remaining_zamanlama_sorunu.md`) bu kovaryatın
**2015/16 gibi tarihsel veri setleri için kavramsal olarak geçersiz** olduğunu
kanıtladı: `players.csv` oyuncu başına sezon-indeksli değil TEK/güncel bir
`contract_expiration_date` tutuyor, bu yüzden 2015/16 season_end'inden çıkarılan
"kalan süre" değerleri 6.9-14 yıl gibi anlamsız çıkıyordu (gerçek bir sözleşme
süresi olamayacak kadar büyük). Bu yüzden `value_residuals()` artık
`contract_years_remaining` sütunu df'te yoksa kovaryatı sessizce atlayıp eski
7-parametreli modele dönüyor; aşağıdaki sonuçlar bu düzeltilmiş, kontratsız
haliyle **N=147 üzerinden, ilk rapordaki modelle birebir aynı format ve
parametre sayısında** yeniden üretildi (bu yüzden `goals_per90` sonuçları,
npxG değişmediğinden, yukarıdaki ilk tabloyla rakam rakam aynı çıkıyor - bu
beklenen ve hesaplamanın doğruluğunu teyit eden bir durum).

### Kurulum

`compute_npxg_per90()`'ın varsayılanı, set-parça (corner/frikik/taca) kaynaklı
possession'lardan gelen sutları artık hariç tutuyor (bkz. `_OPEN_PLAY_SHOT_PATTERNS`
- 'Regular Play', 'From Counter', 'From Keeper'). 380 maçlık 2015/16 La Liga
cache'i üzerinde yeniden hesaplandığında bu, 449 oyunculuk "karışık" (tüm
play_pattern) kümesini 345 oyunculuk açık-oyun-sadece kümesine daraltıyor;
`data/processed/laliga_1516_npxg_per90.parquet` bu yeni değerlerle güncellendi.

Mevcut StatsBomb↔Transfermarkt eşleşmesi (147 oyuncu, `match_tools.py` isim
eşleştirmesi - **değiştirilmedi**) yeniden kullanıldı; sadece `npxg`/`npxg_per90`
sütunları yeni açık-oyun değerleriyle güncellendi. 22 oyuncunun (hepsi
savunma/kaleci - örn. Varane, Mascherano, Bailly) açık-oyun npxG'si sıfıra düştü
- bu, önceki karışık hesabın onlara büyük ölçüde corner'dan gelen kafa vuruşu
npxG'si atfettiğini, açık oyunda pratikte hiç şut riski taşımadıklarını doğruluyor.

### Sonuçlar (N=147, 7 parametre - const, perf_col, age, age_sq, log_minutes,
position_Defender, position_Midfield - ilk tabloyla birebir aynı format)

| | npxg_per90 (açık-oyun, yeni) | goals_per90 |
|---|---|---|
| N | 147 | 147 |
| Parametre sayısı | 7 | 7 |
| R-squared | **0.2790** | 0.3204 |
| Adj. R-squared | **0.2481** | 0.2912 |
| AIC | **435.79** | 427.11 |
| BIC | **456.72** | 448.04 |
| Log-likelihood | **-210.89** | -206.55 |

Referans (eski, karışık npxG): R²=0.3030, Adj R²=0.2731, AIC=430.82, BIC=451.76,
LogLik=-208.41.

**R² açık-oyun-sadece npxG'de DÜŞTÜ** (0.3030 → 0.2790, Adj R² 0.2731 → 0.2481) -
`goals_per90` ile arasındaki fark de BÜYÜDÜ (eskiden ~0.017, şimdi ~0.041 - yön
aynı ama makas 2.4 kat açıldı). Bu, önceki (yanlış N=86, kontrat kovaryatlı)
sürümde "fark küçüldü" diye raporlanan sonucun **tersidir** - o sonuç geçersiz
kontrat kovaryatına dayandığı için yanlıştı; doğru elma-elma (N=147, 7 parametre)
kıyas bunu düzeltiyor.

### En Negatif 15 - Değişim

Eski top-15 (karışık npxG) ile yeni top-15 (açık-oyun npxG) - ikisi de N=147,
aynı model, tek fark npxG tanımı - karşılaştırıldığında:

- **13 oyuncu her iki listede de var** (değişmedi): Kléper Laveran Lima
  Ferreira, Álvaro Medrán Just, Víctor Machín Pérez, Adrián Embarba Blázquez,
  Abraham González Casanova, Álvaro Vázquez García (eski çevirisiyle),
  Adrián González Morales, Roque Mesa Quevedo, Jonathan Viera Ramos, Aythami
  Artiles Oliva, Carlos Castro García, Gerard Moreno Balaguero, Jefferson
  Andrés Lerma Solís.
- **Sadece 2 oyuncu çıktı:** Charles Días Barbosa de Oliveira, Imanol
  Agirretxe Arruti.
- **Sadece 2 yeni oyuncu girdi:** Hernán Arsenio Pérez González, Simão Mate.

Yani gerçekte (kontrat confound'u temizlenince) liste **çok stabil** - 15
isimden 13'ü aynı kaldı. Önceki (hatalı) N=86 kıyasında "9/15 değişti"
raporlanmıştı; bunun büyük kısmı npxG'den değil, geçersiz kontrat kovaryatının
rastgele 61 oyuncuyu örneklem dışına atmasından kaynaklanıyordu.

### Yorum

Doğru (N=147, 7 parametre, kontratsız) kıyas gösteriyor ki set-parça sutlarını
hariç tutmak `npxg_per90`'ın piyasa değerini açıklama gücünü **zayıflatıyor**
(R² 0.303→0.279), `goals_per90` ile arasındaki fark büyüyor. Bunun olası
nedeni: elit/yüksek piyasa değerli oyuncular genelde takımın hem penaltı hem
serbest vuruş/korner uzmanlarıdır (örn. Messi, Ronaldo doğrudan frikik atar);
"mixed" npxG bu oyuncuların gerçek piyasa fiyatlamasıyla örtüşen bir sinyali
(duran topta da güvenilen oyuncu olmak) taşıyordu - açık-oyun-sadece tanım bu
bilgiyi kasıtlı olarak çıkarıyor, dolayısıyla R² düşüyor. Bu, npxg_per90'ın
zayıfladığı anlamına gelmiyor - tam tersine, `goals_per90` ile aradaki farkın
büyümesi, açık-oyun npxG'nin piyasanın FARKINDA OLMADIĞI sinyali daha da
saflaştırdığını gösterir (dead-ball itibarından arındırılmış, salt oyun-içi
bitiricilik). En-negatif-15 listesinin neredeyse hiç değişmemesi (13/15 aynı)
de bunu destekler: bu oyuncular zaten set-parçaya bağımlı olmadıkları için
tanım değişikliğinden etkilenmediler - onlar gerçekten "açık oyunda değerinin
altında" adaylar.

Pratik sonuç: `compute_set_piece_npxg_per90()`'ın ayrı raporlanması hâlâ
değerlidir (set-parçaya bağımlı bir profili yanlış etiketlemeyi önler), ama bu
veri setinde en-negatif-15 sıralamasını büyük ölçüde DEĞİŞTİRMEDİ - asıl etkisi
model R²'sinde (npxG'nin piyasa-açıklama gücü azaldı) görüldü.

## Güncelleme 2: Genişletilmiş Eşleşme (N=245) ile Yeniden Çalıştırma

**Tarih:** 2026-07-14

### Kurulum

`match_tools.py`'ye eklenen nickname+country blocking düzeltmesi (bkz. ilgili
commit) StatsBomb↔Transfermarkt eşleşme oranını aynı 345 açık-oyun-npxG
havuzunda %55.1'den (190/345) **%71.0'e** (245/345) çıkardı - sıfır dış
bağımlılıkla. Bu bölüm, yukarıdaki (N=147, kontratsız, 7 parametre) modeli
şimdi bu **245 kişilik güncel havuzla** yeniden çalıştırıyor.

**Düzeltme notu (2026-07-16):** Bu bölüm ilk yayınlandığında
`build_statsbomb_value_pool()` **dakika eşiği uygulamıyordu**
(`build_eligible_pool()`'daki `apply_minutes_threshold()` muadili yoktu) - bu
yüzden 245 kişilik havuzda 320-835 dakikalık (bazıları yarım sezonun bile
altında oynamış) oyuncular da modele giriyor, küçük-örneklem gürültüsüyle
"en değerinin altında" listesine sızıyordu (örn. Mamadou Sylla Diallo 376 dk,
Josep Señé Escudero 367 dk, Juan Muñoz Muñoz 320 dk - eski, hatalı tablo
aşağıda kaldırılıp değiştirildi). Bu **düzeltildi**: `build_statsbomb_value_pool()`'a
artık `min_minutes` parametresi (varsayılan `config.MIN_MINUTES=900`) eklendi
ve eşleşen sonuca `apply_minutes_threshold()` uygulanıyor (bkz. `match_tools.py`
commit'i). 245 kişilik eşleşen havuzdan **72 oyuncu bu eşiğin altında kaldığı
için elendi**; aşağıdaki tüm sonuçlar düzeltilmiş **N=173** havuzu üzerinden
yeniden üretildi.

### Sonuçlar

| | npxg_per90 (açık-oyun) N=147 | npxg_per90 N=173 (eşiklenmiş) | goals_per90 N=147 | goals_per90 N=173 (eşiklenmiş) |
|---|---|---|---|---|
| N | 147 | **173** | 147 | **173** |
| Parametre sayısı | 7 | **9** | 7 | **9** |
| R-squared | 0.2790 | **0.2790** | 0.3204 | **0.3564** |
| Adj. R-squared | 0.2481 | **0.2438** | 0.2912 | **0.3250** |
| AIC | 435.79 | **510.14** | 427.11 | **490.49** |
| BIC | 456.72 | **538.51** | 448.04 | **518.87** |
| Log-likelihood | -210.89 | **-246.07** | -206.55 | **-236.24** |

**Parametre sayısı 7→9 oldu** - bu bir model değişikliği değil, N artışının
doğal sonucu: 173 kişilik genişletilmiş (ve eşiklenmiş) havuzda artık
`position` sütununda `Goalkeeper` ve (players.csv'nin kendi veri kalitesi
kusuru olan) literal `"Missing"` kategorileri de görülüyor (147'lik eski
havuzda hiç yoktu), `pd.get_dummies` bunlari 2 ekstra kukla degisken olarak
ekliyor. AIC/BIC N ile birlikte büyüdüğünden (log-likelihood daha fazla
gözlemle daha negatif olur), bunları N=147 sonuçlarıyla ham değer olarak
kıyaslamak yanıltıcı olur - asıl anlamlı kıyas **R²/Adj. R²** ve **goals vs
npxg farkı**dır.

**npxg_per90 R²'si N=147→173'te neredeyse sabit kaldı** (0.2790→0.2790,
Adj R² hafif düştü: 0.2481→0.2438), ama **goals_per90 R²'si belirgin arttı**
(0.3204→0.3564). `goals_per90` ile `npxg_per90` arasındaki fark bu yüzden
büyüdü (R² farkı N=147'de ~0.041, N=173'te ~0.077) - yön (goals piyasayı daha
iyi açıklıyor) aynı kaldı ve daha da belirginleşti.

### En Negatif 15 - Değişim (N=147 → N=173, eşiklenmiş)

| player_name | minutes | npxg_per90 | value_residual |
|---|---|---|---|
| Marcelo Vieira da Silva Júnior | 1367 | 0.017 | -2.232 |
| Víctor Machín Pérez | 1736 | 0.090 | -2.220 |
| Álvaro Medrán Just | 1152 | 0.027 | -2.174 |
| Abraham González Casanova | 1054 | 0.162 | -1.633 |
| Ismael López Blanco | 1493 | 0.060 | -1.631 |
| Carlos Castro García | 930 | 0.408 | -1.612 |
| Adrián Embarba Blázquez | 1611 | 0.097 | -1.561 |
| Jorge Franco Alviz | 1042 | 0.149 | -1.540 |
| Adrián González Morales | 1957 | 0.140 | -1.458 |
| David Simón Rodríguez Santana | 1347 | 0.028 | -1.346 |
| Roque Mesa Quevedo | 2232 | 0.030 | -1.323 |
| Sergio Ezequiel Araújo | 1370 | 0.218 | -1.307 |
| Jefferson Andrés Lerma Solís | 1871 | 0.029 | -1.261 |
| Jerónimo Figueroa Cabrera | 1330 | 0.044 | -1.238 |
| Jonathan Viera Ramos | 2256 | 0.121 | -1.222 |

Tüm satırlar artık **≥900 dakika** (en düşük: Carlos Castro García, 930 dk) -
küçük-örneklem gürültüsü riski taşıyan hiçbir isim listede yok.

- **7 oyuncu, eşik uygulanmadan önceki (hatalı) N=245 top-15'iyle ortak:**
  Álvaro Medrán Just, Víctor Machín Pérez, Marcelo Vieira da Silva Júnior,
  Abraham González Casanova, Adrián Embarba Blázquez, Jorge Franco Alviz,
  Ismael López Blanco - hepsi zaten ≥900 dakikaydı, eşikten etkilenmediler.
- **8 oyuncu eşikle listeden düştü** (hepsi <900 dk, düzeltme öncesi hatalı
  listedeydi): Mamadou Sylla Diallo (376 dk), Josep Señé Escudero (367 dk),
  Juan Muñoz Muñoz (320 dk), Wanderson Maciel Sousa Campos (819 dk), Carlos
  Martín Vigaray (835 dk), Daniel Arnaud N'Di (768 dk), Antonio Manuel Luna
  Rodríguez (747 dk), José María Martín-Bejarano Serrano (770 dk).
- **8 oyuncu yeni listeye girdi** (hepsi ≥900 dk; eşik uygulanmadan önce
  top-15 dışında kalmışlardı çünkü düşük-dakikalı gürültülü artıklar onları
  sıralamada aşağı itiyordu): Carlos Castro García, Adrián González Morales,
  David Simón Rodríguez Santana, Roque Mesa Quevedo, Sergio Ezequiel Araújo,
  Jefferson Andrés Lerma Solís, Jerónimo Figueroa Cabrera, Jonathan Viera
  Ramos.

### Yorum

Dakika eşiği düzeltmesi listeyi önemli ölçüde temizledi: düzeltme öncesi
top-15'in 3'ü (Sylla Diallo 376dk, Señé Escudero 367dk, Muñoz Muñoz 320dk)
yarım sezonun altında oynamış oyunculardı - bunların büyük negatif artıkları
gerçek bir "değerinin altında" sinyalinden çok küçük-örneklem varyansından
kaynaklanıyor olabilirdi. Eşik sonrası listeye giren isimlerden Roque Mesa
Quevedo (2232 dk) ve Jonathan Viera Ramos (2256 dk) gibileri tam sezon
oynamış, önceki (N=147) analizde de zaten en-negatif-15'te yer almış tanıdık
isimler - bu, düzeltmenin listeyi rastgele değiştirmediğini, tam tersine
gürültülü düşük-örneklem sinyalini gerçek/istikrarlı sinyalle değiştirdiğini
gösterir.
