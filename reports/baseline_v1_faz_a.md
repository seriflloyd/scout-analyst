# Baseline v1 - Faz A

**Tarih:** 2026-07-07

## Yontem

```
log(market_value_in_eur) ~ goals_per90 + age + age^2 + position + league + log(minutes)
```

- `age`: `date_of_birth` ile oyuncu-sezonun bitis tarihi (season_end) arasindaki farktan
  yil cinsinden hesaplanir (look-ahead bias'siz).
- `position` ve `league` kukla degiskenler olarak modele girer.
- Regresyon: statsmodels OLS (`src/tools/scout_tools.py::value_residuals`).
- Veri: `data/processed/eligible_pool.parquet` (top-5 lig, son 3 sezon, dakika esigi
  ve taban-deger filtresi uygulanmis oyuncu-sezon havuzu).

## Model istatistikleri

| Istatistik | Deger |
|---|---|
| R-squared | 0.5508 |
| N (gozlem sayisi) | 4206 |

## En negatif 15 artik (deger residual'i en dusuk 15 oyuncu-sezon)

| name | position | age | goals_per90 | market_value_in_eur | value_residual | dusuk_sinyal_guvenilirligi |
|---|---|---|---|---|---|---|
| Oliver Arblaster | Midfield | 21.0 | 0.000 | 450000 | -3.879 | True |
| Pierre Ekwah | Midfield | 24.0 | 0.060 | 500000 | -3.015 | False |
| Pierre Ekwah | Midfield | 23.0 | 0.000 | 500000 | -2.698 | True |
| Karl Darlow | Goalkeeper | 36.2 | 0.000 | 200000 | -2.646 | True |
| Alex Palmer | Goalkeeper | 29.4 | 0.000 | 900000 | -2.436 | True |
| Yellu Santiago | Midfield | 20.6 | 0.095 | 1000000 | -2.368 | False |
| Timon Weiner | Goalkeeper | 27.0 | 0.000 | 500000 | -2.289 | True |
| Finn Porath | Midfield | 28.9 | 0.196 | 600000 | -2.271 | False |
| Arthur Atta | Midfield | 22.0 | 0.093 | 1000000 | -2.239 | False |
| Sam Morsy | Midfield | 34.3 | 0.000 | 400000 | -2.238 | True |
| Alaa Bellaarouch | Goalkeeper | 22.9 | 0.000 | 1000000 | -2.231 | True |
| Lucas Mincarelli | Defender | 21.0 | 0.098 | 1000000 | -2.226 | False |
| Michael Svoboda | Defender | 26.2 | 0.068 | 900000 | -2.189 | False |
| Jordan Clark | Midfield | 31.3 | 0.073 | 1000000 | -2.157 | False |
| Jack Robinson | Defender | 31.3 | 0.073 | 1000000 | -2.101 | False |

## Not

Bu, StatsBomb event verisi eklenmeden once tek-metrikli (goals_per90) baseline'dir -
Faz B sonrasi karsilastirma icin referans noktasi.
