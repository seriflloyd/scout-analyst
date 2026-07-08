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
