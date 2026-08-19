# UrbanVerse final PoC sonuç özeti

Bu belge final GitHub README'sindeki ana sonuçların kısa Türkçe karşılığıdır.

## Ana sonuç

Final PoC, “foundation model her durumda daha iyi” gibi geniş bir iddiayı desteklemiyor. Daha savunulabilir sonuç şu:

> **TimesFM, çalıştırılan saatlik örneklemde güçlü 24-saat Seasonal Naive baseline'a karşı tutarlı ek değer gösteriyor. Moirai'nin saatlik avantajı daha pencere-bağımlı. 5-dakikalık çözünürlükte daha uzun geçmiş hata ve bazı shape metriklerini iyileştiriyor fakat yüksek frekanslı adım-adım hareket takibini tamamen çözmüyor.**

## Saatlik güçlü baseline

| Yöntem | Mean MAE | Seasonal Naive'a karşı |
|---|---:|---:|
| Persistence | 12.563 | — |
| Seasonal Naive (24h) | 2.985 | baseline |
| TimesFM 2.5 | **1.567** | **8/9** |
| Moirai 2.0 | 2.218 | 4/9 |

Persistence'ın 24-saat horizon'da zayıf olduğu görülüyor: Seasonal Naive mean MAE'yi 12.563'ten 2.985'e düşürüyor. Buna rağmen TimesFM Seasonal Naive'dan yaklaşık **%47.5 daha düşük mean MAE** elde ediyor ve 8/9 pencerede kazanıyor. Moirai'nin mean MAE'si de düşük fakat 4/9 pencere galibiyeti avantajın daha değişken olduğunu gösteriyor.

TimesFM'in en büyük kazançları Seasonal Naive'ın hata yaptığı pencerelerde ortaya çıkıyor. Önceki günün aynı saatinin zaten iyi tahmin verdiği pencerelerde TimesFM'in ek avantajı küçülüyor. Bu betimleyici bir örüntüdür; “atipik gün” gibi nedensel bir açıklama kanıtlanmış değildir.

## Context-length kontrolü

5-dakikalık veri ve 2-saat horizon sabit tutulup yalnız geçmiş uzunluğu değiştirildi: 8 saat, 24 saat, 7 gün.

| Model | 8 h MAE | 24 h MAE | 7 d MAE |
|---|---:|---:|---:|
| TimesFM | 3.385 | 2.913 | **2.560** |
| Moirai | 3.137 | **2.412** | 2.461 |

TimesFM 8 saatten 7 güne geçince yaklaşık **%24.4** iyileşiyor. Moirai'de en iyi MAE 24 saatte geliyor; 7 gün ek fayda sağlamıyor. En büyük kazancın bir tam günlük döngü görünür olduğunda ortaya çıkması, 8-saat context'in gerçek bir confound olduğunu gösteriyor; ancak 5-dakikalık shape tracking hâlâ saatlik seviyeye yaklaşmıyor.

## Sınırlamalar

- Final ana deneyler tek bir METR-LA sensörüne dayanıyor (`773062`).
- 5-minute ve hourly örnekler weekday-only.
- Hourly 24-saat horizonlar örtüşüyor ve 25–26 Haziran 2012'de yoğunlaşıyor.
- Context ablation kalite filtresinden sonra tek günün beş zaman rejimini kullanıyor (`2012-06-27`).
- Weekly sadece iki gelecek noktası içerdiği için visual exploratory.
- Sonuçlar betimleyici; istatistiksel genelleme iddiası yapılmıyor.
