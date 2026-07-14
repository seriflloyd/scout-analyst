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

**Önemli metodolojik not (karşılaştırılabilirliği sınırlayan bir etken):**
Faz C'de `value_residuals()`'a `contract_years_remaining` zorunlu kovaryat
olarak eklendi (satırı olmayanlar otomatik düşüyor). Bu 147 oyunculuk La Liga
setinde sadece 86'sının (`players.csv`'den) sözleşme bitiş tarihi biliniyor -
61 satır **sırf bu yüzden** düştü. Aşağıdaki `npxg_per90` ve `goals_per90`
modelleri aynı N=86 üzerinden koşuldu (yani ikisi arasındaki kıyas hâlâ
elma-elma), ama yukarıdaki eski tablo (N=147, 7 parametre, kontrat kovaryatı
yok) ile **doğrudan** kıyaslanamaz - N ve parametre sayısındaki fark npxG
tanımından değil, aradan geçen zamanda eklenen kontrat kovaryatından kaynaklanıyor.

### Sonuçlar (N=86, 8 parametre - const, perf_col, age, age_sq, log_minutes,
contract_years_remaining, position_Defender, position_Midfield)

| | npxg_per90 (açık-oyun, yeni) | goals_per90 |
|---|---|---|
| N | 86 | 86 |
| Parametre sayısı | 8 | 8 |
| R-squared | 0.3559 | **0.3709** |
| Adj. R-squared | 0.2981 | **0.3144** |
| AIC | 259.44 | **257.42** |
| BIC | 279.08 | **277.05** |
| Log-likelihood | -121.72 | **-120.71** |

Referans (eski, N=147, 7 parametre, karışık npxG): R²=0.3030/0.3204,
Adj R²=0.2731/0.2912, AIC=430.82/427.11 (npxg/goals). R² farkı (goals - npxg)
eskiden ~0.017 idi, yenide (N=86, kontrat kovaryatlı) ~0.015 - fark küçüldü
ama yön aynı: `goals_per90` bu veri setinde piyasayı hâlâ biraz daha iyi
açıklıyor. N ve parametre sayısı farklı olduğundan bu daralmanın ne kadarı
açık-oyun ayrımından, ne kadarı örneklem küçülmesinden geldiği bu kurulumla
kesin ayrıştırılamıyor - ama yön tersine dönmedi, yani temel sonuç
(`goals_per90` → piyasa-referans, `npxg_per90` → asıl scouting sinyali) geçerliliğini koruyor.

### En Negatif 15 - Değişim

Eski top-15 (karışık npxG, N=147) ile yeni top-15 (açık-oyun npxG, N=86)
karşılaştırıldığında:

- **6 oyuncu her iki listede de var:** Klepper Laveran Lima Ferreira, Álvaro
  Medrán Just, Abraham González Casanova, Adrián Embarba Blázquez, Gerard
  Moreno Balaguero, Charles Dias Barbosa de Oliveira.
- **9 oyuncu eski top-15'ten çıktı** - ama bunların **6'sı sadece kontrat
  verisi eksikliği yüzünden** yeni N=86 kümesine hiç girmedi (npxG'yle
  ilgisi yok - Faz C confound'u): Víctor Machón Pérez, Imanol Agirretxe
  Arruti, Carlos Castro García, Adrián González Morales, Roque Mesa Quevedo,
  Aythami Artiles Oliva. Geriye kalan **3'ü gerçekten npxG tanım
  değişikliğinden dolayı** listeden çıktı: Álvaro Vázquez García (npxg_per90
  0.436→0.185, sıra 25'e düştü), Jonathan Viera Ramos (0.239→0.121, sıra 17),
  Jefferson Andrés Lerma Solís (0.104→0.029, sıra 16) - açık-oyun npxG'leri
  ciddi düştüğü için model artık onlardan zaten düşük performans bekliyor,
  dolayısıyla "beklenenden düşük fiyatlanma" sinyalleri zayıfladı (value_residual
  daha az negatife kaydı).
- **9 yeni oyuncu listeye girdi:** Iago Aspas Juncal, Pablo Fornals Malla,
  Rubén Castro Martín, José Luis Morales Nogales, Damián Nicolás Suárez
  Suárez, Luis Hernández Rodríguez, Diego Javier Llorente Ríos, Celso Borges
  Mora, Cristiano Biraghi - açık-oyun ayrımı bu oyuncuların gerçek açık-oyun
  bitiricilik profilini karışık hesaba göre daha net (ve bazılarında daha
  düşük) yakaladığı için modelin beklentisiyle piyasa fiyatı arasındaki fark
  büyüdü.

### Yorum

Açık-oyun ayrımı sinyali **ne kesin güçlendirdi ne zayıflattı** - yön aynı
kaldı (goals piyasayı biraz daha iyi açıklıyor, npxg'nin düşük R²'si hâlâ
"piyasanın gözden kaçırdığı sinyal" olarak yorumlanabilir) ama listenin
içeriği belirgin şekilde değişti: 380 maçlık gerçek veride 9/15 isim döndü,
bunun 6'sı salt kontrat-verisi confound'undan, 3'ü gerçek npxG yeniden
tanımından kaynaklanıyor. Pratik sonuç: set-parça sutlarını dışlamak,
kafa-vuruşu/duran-top uzmanı olmayan ama yüksek "mixed" npxG'ye sahip görünen
oyuncuları (örn. Vázquez, Viera, Lerma) listeden çıkarıp, gerçek açık-oyun
bitiricilik profiline sahip oyuncuları (Aspas, Fornals, Rubén Castro, Morales
gibi) öne çıkarıyor - bu, `compute_set_piece_npxg_per90()`'ın ayrı raporlanma
amacına tam uyuyor: set-parçaya bağımlı bir profili "açık oyunda değerinin
altında" diye yanlış etiketlemeyi önlüyor.
