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
