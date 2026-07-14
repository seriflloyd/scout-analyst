# contract_years_remaining Zamanlama Sorunu - Araştırma

**Tarih:** 2026-07-14

## Sorunun tespiti

`eligible_pool.parquet`'te `contract_years_remaining` kapsamı %92.9 iken, La Liga
2015/16 eşleşen alt-kümesinde (147 oyuncu, `laliga_1516_matched_open_play.parquet`)
sadece %58.5 (86/147) idi. Bu fark araştırıldı.

## Bulgular

**1) NaN kalan 61 oyuncu (La Liga 2015/16 alt-kümesi):** Piqué, Kroos, Rakitić,
Varane, Bale, Mascherano, Agirretxe, Aduriz, Gameiro, Lucas Pérez, Negredo ve
diğerleri (tam liste kod/analiz çıktısında).

**2) NaN'ların gerçek nedeni:** Başlangıç hipotezi ("season_end'den önceki bir
tarih 0'a değil NaN'a çevriliyor, bu yanlış") **doğrulanmadı** - 61 oyuncunun
hepsinde `players.csv`'deki `contract_expiration_date` doğrudan **NaT**
(tamamen eksik), negatif bir fark değil. `last_season` sütunuyla çapraz kontrol:

| last_season aralığı | n | contract_expiration_date eksik oranı |
|---|---|---|
| ≤2015 | 1846 | %73.8 |
| 2016-2018 | 2515 | %54.9 |
| 2019-2021 | 3502 | %33.4 |
| 2022-2023 | 2753 | %33.0 |
| 2024-2025 | 21132 | %20.0 |

Eksiklik oranı oyuncunun güncelliğiyle ters orantılı - alan sadece "hâlâ takip
edilen" oyuncular için tutarlı biçimde dolu.

**3) Daha ciddi sorun - kalan 86 satırda:** Bu satırların `contract_years_remaining`
değerleri **6.9 ile 14.0 yıl arasında** (ortalama 9.77 yıl) - gerçek bir futbol
sözleşmesi süresi olamayacak kadar büyük. Bunun nedeni: `players.csv` oyuncu
başına yalnızca **tek, güncel (veri setinin son güncellendiği andaki) bir
`contract_expiration_date`** tutuyor - sezona özgü sözleşme geçmişi YOK. Örneğin
Messi için kayıtlı tarih `2028-12-31` - bu 2015/16'daki değil, verinin en son
güncellendiği zamandaki sözleşmesi. Bunu `season_end=2016-06-30` ile çıkarınca
anlamsız "~12.5 yıl kaldı" çıkıyor.

**4) Sistemik kapsam doğrulandı:** Bu, sadece La Liga 2015/16'ya özgü değil -
`eligible_pool`'un kendi 3 sezonu (2024/2025/2026) içinde de aynı desen var:

| sezon | n | NaN oranı | ortalama yıl | max yıl |
|---|---|---|---|---|
| 2024 | 1546 | %4.3 | 3.33 | 10.49 |
| 2025 | 1546 | %2.5 | 2.57 | 9.49 |
| 2026 | 815 | %18.8 | 2.07 | 7.50 |

Aynı oyuncu için değer, sezon ne kadar geriye giderse o kadar büyüyor - çünkü
hep AYNI (tek/güncel) bitiş tarihinden çıkarılıyor. `n_seasons` penceresi
geçmişe (2015/16 gibi 10 yıl öncesine) gittikçe bu tamamen anlamsızlaşıyor.

## Sonuç ve düzeltme

`players.csv`'de sezon bazlı sözleşme geçmişi olmadığından, `contract_years_remaining`
**sadece havuzdaki en güncel sezon için anlamlıdır** - `n_seasons` penceresindeki
daha eski sezonlar için hesaplanan değer, oyuncunun O ZAMAN henüz imzalamadığı
(gelecekte imzalayacağı) bir sözleşmeyi kullanan bir look-ahead bias'tır.

`build_eligible_pool()` düzeltildi: `contract_years_remaining`/`is_free_agent_soon`
artık sadece `season == max(kept seasons)` olan satırlarda hesaplanıyor; havuzun
daha eski sezonlarında NaN/False bırakılıyor (bkz. `tests/test_data_tools.py`,
`test_build_eligible_pool_contract_years_remaining_only_for_latest_season`).

**Açık kalan sonuç:** La Liga 2015/16 gibi tamamen tarihsel bir veri setinde
(havuzun "en güncel sezonu" kavramı yok, tek sezonun kendisi 10 yıl eski)
`contract_years_remaining` düzeltmeyle bile hesaplanamaz/anlamsız kalır - bu
tür geçmiş-sezon analizlerinde bu kovaryat modelden tamamen çıkarılmalıdır.
Bu, önceki Faz C güncellemesindeki (`reports/npxg_vs_goals_faz_b.md`,
"Güncelleme: Açık-Oyun-Sadece npxG ile Karşılaştırma" bölümü) N=86, 8-parametreli
modelin `contract_years_remaining` kovaryatının geçersiz veriye dayandığı, o
karşılaştırmanın yeniden (kovaryat olmadan) yapılması gerektiği anlamına gelir.
