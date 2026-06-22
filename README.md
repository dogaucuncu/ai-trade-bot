<p align="center">
 <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
 <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
 <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
 <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
 <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status">
</p>

<h1 align="center">AI Trade Bot</h1>

<p align="center">
 <strong>Yapay Zeka Destekli Kripto & Hisse Senedi Mikro-Sermaye Trading Botu</strong>
</p>

<p align="center">
 LSTM derin öğrenme, ensemble strateji motoru, gerçek zamanlı dashboard ve<br>
 çok katmanlı risk yönetimi ile donatılmış otonom trading sistemi.
</p>

<p align="center">
 <a href="#özellikler">Özellikler</a> •
 <a href="#mimari">Mimari</a> •
 <a href="#kurulum">Kurulum</a> •
 <a href="#kullanım">Kullanım</a> •
 <a href="#stratejiler">Stratejiler</a> •
 <a href="#backtesting">Backtesting</a> •
 <a href="#katkıda-bulunma">Katkıda Bulunma</a>
</p>

---

## İçindekiler

- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Konfigürasyon](#konfigürasyon)
- [Kullanım](#kullanım)
- [Stratejiler](#stratejiler)
- [Makine Öğrenmesi](#makine-öğrenmesi)
- [Risk Yönetimi](#risk-yönetimi)
- [Dashboard](#dashboard)
- [Dürüst Backtesting & Edge Durumu](#dürüst-backtesting--edge-durumu)
- [API Referansı](#api-referansı)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [Sorumluluk Reddi / Disclaimer](#sorumluluk-reddi)

---

## Özellikler

### Yapay Zeka & Makine Öğrenmesi
- **LSTM Derin Öğrenme Modeli** — Küçük LSTM ağı (hidden 48, 1 katman) ile fiyat yönü tahmini (UP / DOWN / SIDEWAYS)
- **Durağan (stationary) özellikler** — Ham fiyat seviyeleri yerine getiri/oran-bazlı 15 özellik; lookahead-bias'sız scaler; sınıf ağırlıkları
- **Duygu Analizi & Haber Risk Kapısı** — Keyword-bazlı sentiment + Crypto Fear & Greed Index, ücretsiz/keyless haber kaynaklarıyla (RSS + Alpaca News) beslenir. Emir öncesi **muhafazakâr risk kapısı** olarak işler (kötü habere karşı işlem açmayı engeller). **Tamamen Claude/LLM'siz** — runtime'da token harcamaz. Varsayılan kapalı, opt-in (bkz. `.env`)

### Strateji Motoru (kanıt-temelli)
- **VWAP Reversion** (15m) — Hacim-ağırlıklı ortalamaya dönüş. **Aktif** — `strategy_eval`'de tüm 5 coinde PF>1 (en güçlü in-sample edge). _OOS marjinal; paper'da forward-test şart._
- **Mean Reversion** (15m) — Z-score ortalamaya dönüş. **Aktif** — kural-bazlı edge (AVAX/DOGE)
- **ML Strategy** (15m) — LSTM tahminlerine dayalı; yalnızca eğitilmiş, uyumlu modeli olan coinlerde (ör. DOGE) çalışır
- **Breakout** (1h) — Donchian kanal kopuşu. Uygulanmış ama **KAPALI** (PF<1 her coinde)
- **Scalping & Momentum** — Uygulanmış ama **varsayılan olarak KAPALI** (dürüst backtestlerde komisyon sonrası para kaybediyorlardı — bkz. [Dürüst Backtesting](#dürüst-backtesting--edge-durumu))
- **Ensemble** — Ağırlıklı oylama altyapısı (opsiyonel)
- **Parametre Optimizasyonu** — `backtest/optimize.py` ile in-sample/out-of-sample bölmeli grid-search + overfit uyarısı

### Dürüst Backtesting Altyapısı
- **Leak-free walk-forward** — Model her fold'da geçmişte eğitilip yalnızca görülmemiş gelecekte test edilir
- **Gerçekçi maliyet** — Komisyon + slippage her işlemde; **buy & hold kıyaslaması**
- **Timeframe-bilinçli metrikler** — Doğru yıllıklaştırılmış Sharpe/Sortino, max drawdown, profit factor
- **Paper performans takibi** — Trade/equity SQLite'a yazılır; `paper_report.py` ile dürüst rapor

### Çok Katmanlı Risk Yönetimi
- **Circuit Breaker** — 4 durumlu (NORMAL → CAUTIOUS → HALTED → EMERGENCY) otomatik koruma sistemi
- **Position Sizing** — Kelly Criterion bazlı optimal pozisyon boyutlandırma
- **Risk Manager** — Günlük kayıp limiti, max drawdown, portföy maruziyeti kontrolleri
- **Trailing Stop** — Dinamik stop-loss güncelleme mekanizması

### Çoklu Borsa Desteği
- **Binance** — Kripto para çiftleri (SOL, AVAX, XRP, ADA, DOGE — `CRYPTO_SYMBOLS` ile yapılandırılabilir)
- **Alpaca Markets** — ABD hisse senetleri (AAPL, MSFT, TSLA, NVDA, AMD, META) + fractional shares. **Hisse veri feed'i artık canlı** (IEX feed, ücretsiz plan) — piyasa-saatleri kontrolüyle yalnızca borsa açıkken işler
- **Paper & Live Mode** — Testnet üzerinde paper trading veya gerçek hesapla canlı trading
- **TLS araya girme uyumu** — `truststore` ile OS sertifika mağazası kullanılır (kurumsal proxy/AV SSL inspection olan ağlarda HTTPS'i, doğrulamayı kapatmadan çözer)

### Gerçek Zamanlı Dashboard
- **WebSocket** ile canlı veri akışı
- **TradingView Charts** — Interaktif mum grafikleri
- **Pozisyon & İşlem Takibi** — Açık pozisyonlar, trade geçmişi, sinyal monitörü
- **Bot Kontrolleri** — Start / Stop / Emergency Stop butonları
- **Performans İstatistikleri** — Win rate, Sharpe ratio, equity curve

### Bildirim Sistemi
- **E-posta Bildirimleri** — SMTP üzerinden; engine'e bağlı: pozisyon açılış/kapanış, acil durdurma (circuit breaker) ve günlük P&L özeti
- **WebSocket Push** — Dashboard üzerinden anlık bildirimler (toast)

---

## Mimari

```
┌──────────────────────────────────────────────────────────┐
│                      main.py (Entry Point)                │
│                    CLI args + config loader                │
└────────────────────────┬─────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    BotEngine        │
              │  (Orkestratör)      │
              └──┬───┬───┬───┬─────┘
                 │   │   │   │
    ┌────────────┘   │   │   └─────────────┐
    ▼                ▼   ▼                 ▼
┌────────┐   ┌──────────────┐   ┌──────────────┐
│Strategy│   │ Risk Manager │   │  Executors   │
│Ensemble│   │ + Circuit    │   │ Binance /    │
│        │   │   Breaker    │   │ Alpaca       │
└───┬────┘   └──────────────┘   └──────────────┘
    │
    ├── VWAP Reversion (15m)      [aktif]
    ├── Mean Reversion (15m)      [aktif]
    ├── ML Strategy (15m, LSTM)   [model varsa]
    ├── Breakout (1h)             [kapali]
    ├── Scalping (1m)             [kapali]
    └── Momentum (1h)             [kapali]

┌───────────────────────────────────────────┐
│           Dashboard (FastAPI)              │
│  REST API + WebSocket + Jinja2 Templates   │
└───────────────────────────────────────────┘
```

---

## Proje Yapısı

```
AI-Trade-Bot/
│
├── main.py                     # Ana giriş noktası (CLI) — paper/live engine + dashboard
├── run_bot.py                  # Bot başlatma scripti
├── train_model.py              # ML model eğitim scripti (coin başına, 15m, küçük model)
├── paper_report.py             # Paper/live performans raporu (DB'den, dürüst metrikler)
├── run_backtest.py             # (eski) tekli strateji backtest scripti
├── requirements.txt            # Python bağımlılıkları
│
├── config/                     # Konfigürasyon
│   ├── __init__.py
│   ├── settings.py             # Merkezi ayarlar (dataclass + .env)
│   ├── .env.example            # Ortam değişkenleri şablonu
│   └── .env                    # Gerçek anahtarlar (git'e dahil DEĞİL)
│
├── src/                        # Ana kaynak kodu
│   ├── __init__.py
│   │
│   ├── bot/                    # Bot motoru
│   │   ├── engine.py           # Ana orkestratör (trading loop)
│   │   └── scheduler.py        # Görev zamanlayıcı
│   │
│   ├── data/                   # Veri toplama & depolama
│   │   ├── collector.py        # OHLCV veri toplayıcı (Binance + Alpaca/IEX)
│   │   ├── news_feed.py        # Haber başlıkları (keyless RSS + Alpaca News)
│   │   ├── storage.py          # SQLite async depolama (SQLAlchemy)
│   │   └── websocket_feed.py   # Binance WebSocket canlı veri akışı
│   │
│   ├── indicators/             # Teknik göstergeler
│   │   └── technical.py        # RSI, MACD, Bollinger, EMA, ATR, vb.
│   │
│   ├── strategy/               # Trading stratejileri
│   │   ├── base.py             # Abstract base class + Signal/Position
│   │   ├── vwap_reversion.py   # VWAP ortalamaya dönüş (15m) — AKTİF
│   │   ├── mean_reversion.py   # Z-score ortalamaya dönüş (15m) — AKTİF
│   │   ├── ml_strategy.py      # LSTM bazlı ML stratejisi (15m) — model varsa AKTİF
│   │   ├── breakout.py         # Donchian kopuş (1h) — varsayılan KAPALI (edge yok)
│   │   ├── scalping.py         # Scalping (1m) — varsayılan KAPALI (kârsız)
│   │   ├── momentum.py         # Trend takip (1h) — varsayılan KAPALI (kârsız)
│   │   └── ensemble.py         # Ağırlıklı oylama ensemble (opsiyonel)
│   │
│   ├── ml/                     # Makine öğrenmesi modülleri
│   │   ├── lstm_model.py       # LSTM model + predictor wrapper
│   │   ├── predictor.py        # Tahmin pipeline
│   │   ├── trainer.py          # Model eğitim yöneticisi
│   │   └── sentiment.py        # Duygu analizi (keyword + Fear&Greed)
│   │
│   ├── risk/                   # Risk yönetimi
│   │   ├── manager.py          # Risk kontrol merkezi
│   │   ├── circuit_breaker.py  # Circuit breaker (4 durum)
│   │   ├── sentiment_gate.py   # Haber/sentiment risk kapısı (Claude'suz, opt-in)
│   │   └── position_sizer.py   # Kelly Criterion pozisyon boyutlandırma
│   │
│   └── execution/              # Emir yürütme
│       ├── order_manager.py    # Emir yönetim sistemi
│       ├── binance_exec.py     # Binance executor (CCXT)
│       └── alpaca_exec.py      # Alpaca executor
│
├── dashboard/                  # Web dashboard
│   ├── __init__.py
│   ├── app.py                  # FastAPI uygulaması + REST API + WebSocket
│   ├── notifications.py        # E-posta & push bildirim sistemi
│   ├── templates/
│   │   └── index.html          # Ana dashboard HTML (TradingView Charts)
│   └── static/                 # CSS, JS, assets
│
├── backtest/                   # Backtesting & değerlendirme
│   ├── __init__.py
│   ├── metrics.py              # Timeframe-bilinçli risk metrikleri + buy&hold
│   ├── backtester.py           # Strateji backtester (Plotly raporları)
│   ├── walkforward_ml.py       # Dürüst leak-free walk-forward ML backtest (--all)
│   ├── strategy_eval.py        # Kural-bazlı stratejileri coinlerde dürüst ölç
│   ├── optimize.py             # Parametre grid-search (IS/OOS + overfit uyarısı)
│   ├── robustness.py           # Seed + eşik sağlamlık testi (edge gerçek mi?)
│   └── results/                # Sonuç dosyaları (git'e dahil DEĞİL)
│
├── models/                     # Eğitilmiş ML modelleri (train_model.py üretir)
│   └── <COIN>_<TF>/            # ör. DOGE_USDT_15m/ (ağırlıklar git'e dahil DEĞİL)
│
├── data/                       # Veritabanı
│   └── tradebot.db             # SQLite — candles, trades, equity, açık pozisyonlar (git'e dahil DEĞİL)
│
├── logs/                       # Log dosyaları (git'e dahil DEĞİL)
│
└── tests/                      # Test dosyaları
    ├── test_api.py             # API bağlantı testleri
    └── test_strategy.py        # Strateji testleri
```

---

## Kurulum

### Gereksinimler

| Araç | Versiyon | Zorunlu mu? | İndirme |
|------|---------|:-----------:|--------|
| **Python** | 3.11+ | Evet | [python.org](https://www.python.org/downloads/) |
| **pip** | Son sürüm | Evet | Python ile birlikte gelir |
| **Git** | 2.40+ | Evet | [git-scm.com](https://git-scm.com/download/win) |
| **Redis** | 7.0+ | Opsiyonel | [redis.io](https://redis.io/downloads/) |
| **NVIDIA CUDA** | 11.8+ | Opsiyonel | [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) |

### 1. Repoyu Klonlayın

```bash
git clone https://github.com/dogaucuncu/ai-trade-bot.git
cd ai-trade-bot
```

### 2. Sanal Ortam Oluşturun

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

> **PyTorch GPU Desteği (Opsiyonel):**
> Eğer NVIDIA GPU'nuz varsa, LSTM model eğitimi çok daha hızlı olur.
> [pytorch.org](https://pytorch.org/get-started/locally/) adresinden CUDA sürümünüze uygun komutu kullanın.
> ```bash
> # Örnek: CUDA 11.8 için
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```
> GPU yoksa PyTorch otomatik olarak CPU modunda çalışır.

### 4. Redis Kurulumu (Opsiyonel)

Redis, caching ve pub-sub için kullanılır. Bot Redis olmadan da çalışır.

```bash
# Windows (WSL veya Docker ile)
docker run -d --name redis -p 6379:6379 redis:latest

# Linux
sudo apt install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis
```

### 5. Ortam Değişkenlerini Ayarlayın

```bash
# .env.example dosyasını kopyalayın
cp config/.env.example config/.env

# Editörünüzle açıp API anahtarlarınızı girin
notepad config/.env    # Windows
nano config/.env       # Linux/macOS
```

---

## Konfigürasyon

Tüm ayarlar `config/.env` dosyasından yüklenir. Önemli parametreler:

### Genel Ayarlar

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `TRADING_MODE` | `paper` | `paper` (simülasyon) veya `live` (gerçek) |
| `INITIAL_CAPITAL` | `50.0` | Başlangıç sermayesi (USD) |
| `CRYPTO_SYMBOLS` | `SOL/USDT,AVAX/USDT,XRP/USDT,ADA/USDT,DOGE/USDT` | İşlenecek coinler (virgülle) |
| `CRYPTO_ALLOCATION` | `0.75` | Kripto'ya ayrılan oran (%75) |
| `STOCK_ALLOCATION` | `0.25` | Hisse senedine ayrılan oran (%25) |
| `LOG_LEVEL` | `INFO` | Log seviyesi (DEBUG/INFO/WARNING/ERROR) |
| `DASHBOARD_PORT` | `8000` | Dashboard port numarası |

> **Coin seçimi:** Varsayılan 5 likit coin. `$1000+` sermayeye çıkınca `BTC/USDT,ETH/USDT` eklemek mantıklı (en yüksek likidite). Tek satır: `CRYPTO_SYMBOLS=AVAX/USDT,SOL/USDT,DOGE/USDT`

### API Anahtarları (Zorunlu)

#### Binance API Key Alma

| Parametre | Açıklama |
|-----------|----------|
| `BINANCE_API_KEY` | Binance API anahtarı |
| `BINANCE_SECRET_KEY` | Binance gizli anahtar |
| `BINANCE_TESTNET` | `true` = testnet (önerilen), `false` = gerçek hesap |

**Adımlar:**
1. [binance.com](https://www.binance.com/en/register) adresinden hesap oluşturun
2. Kimlik doğrulaması (KYC) yapın
3. **Testnet için (önerilen):** [testnet.binance.vision](https://testnet.binance.vision/) → GitHub ile giriş → API key oluşturun
4. **Gerçek hesap için:** Binance → Hesap → API Yönetimi → API Oluştur
5. API izinlerinde sadece **"Spot Trading"** ve **"Read"** izinlerini açın
6. **"Withdrawal"** iznini asla açmayın!

#### Alpaca API Key Alma

| Parametre | Açıklama |
|-----------|----------|
| `ALPACA_API_KEY` | Alpaca API anahtarı |
| `ALPACA_SECRET_KEY` | Alpaca gizli anahtar |
| `ALPACA_PAPER` | `true` = paper trading (önerilen), `false` = gerçek |

**Adımlar:**
1. [alpaca.markets](https://alpaca.markets/docs/trading/getting_started/) adresinden ücretsiz hesap oluşturun
2. Dashboard'a giriş yapın → Sol menüde **"Paper Trading"** seçin
3. **API Keys** bölümünden **"Generate New Key"** tıklayın
4. `API Key ID` ve `Secret Key` değerlerini kopyalayın
5. Secret key sadece bir kez gösterilir, kaydetmeyi unutmayın!

> **Not:** Alpaca yalnızca ABD hisse senetlerini destekler. ABD dışından kullanım için bazı kısıtlamalar olabilir.

### SMTP E-posta Bildirimleri (Opsiyonel)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `SMTP_HOST` | `smtp.gmail.com` | SMTP sunucu adresi |
| `SMTP_PORT` | `587` | SMTP port (TLS) |
| `SMTP_USERNAME` | — | E-posta adresiniz |
| `SMTP_PASSWORD` | — | Uygulama şifresi (Gmail App Password) |
| `SMTP_FROM` | — | Gönderici e-posta adresi |
| `SMTP_TO` | — | Alıcı e-posta adresi |

**Gmail Uygulama Şifresi Alma:**
1. [myaccount.google.com](https://myaccount.google.com/) → Güvenlik
2. **"2 Adımlı Doğrulama"** aktif olmalı
3. Güvenlik → **"Uygulama Şifreleri"** → Uygulama seçin: "Posta" → Cihaz: "Diğer" → "AI Trade Bot" yazın
4. Oluşturulan 16 haneli şifreyi `SMTP_PASSWORD` olarak girin
5. Normal Gmail şifrenizi değil, uygulama şifresini kullanın!

### Sentiment / Haber Risk Kapısı (Opsiyonel, Claude'suz)

Emir öncesi muhafazakâr bir haber/sentiment filtresi. Yalnızca **kötü habere
karşı işlem açmayı engeller**; işlem başlatmaz ve **hiçbir LLM/Claude çağrısı
yapmaz** (keyword + Fear&Greed + ücretsiz RSS/Alpaca News). Önce gözlem modunda
(`GATE=false`) loglardan skorları izleyin, güvendikten sonra `GATE=true` yapın.

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `SENTIMENT_ENABLED` | `false` | Kapı için ana anahtar |
| `SENTIMENT_GATE` | `false` | `true` = veto uygula; `false` = sadece gözlemle/logla |
| `SENTIMENT_THRESHOLD` | `0.5` | "Güçlü" sayılan \|skor\| eşiği |
| `SENTIMENT_CACHE_TTL` | `1800` | Sembol başına skor önbelleği (saniye) |
| `CRYPTOPANIC_TOKEN` | — | Opsiyonel ücretsiz CryptoPanic token'ı (yoksa RSS kullanılır) |

### Risk Parametreleri (Hardcoded — `config/settings.py`)

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `max_risk_per_trade` | %2 | Trade başına max risk (~$1) |
| `daily_loss_limit` | %3 | Günlük max kayıp |
| `max_open_positions` | 3 | Eş zamanlı max pozisyon |
| `max_drawdown` | %15 | Maksimum drawdown (Emergency trigger) |
| `max_portfolio_exposure` | %30 | Piyasaya max maruziyet |

---

## Kullanım

### Botu Başlatma

```bash
# Paper mode (varsayılan) — gerçek para riski yok
python main.py

# Verbose logging ile
python main.py -v

# Live mode (gerçek para!)
python main.py --mode live
```

### Paper Performansını Görme

```bash
# Çalışan/biten paper oturumunun dürüst performans raporu (DB'den)
python paper_report.py
```

### Dürüst Backtesting & Değerlendirme

```bash
# Leak-free walk-forward ML backtest (tek coin)
python -m backtest.walkforward_ml --symbol DOGE/USDT --tf 15m

# Tüm coinlerde ML edge taraması + karşılaştırma tablosu
python -m backtest.walkforward_ml --all --tf 15m --candles 12000 --train 6000 --test 2000

# Kural-bazlı stratejileri tüm coinlerde dürüstçe ölç (PF = edge)
python -m backtest.strategy_eval

# Strateji parametrelerini tara (in-sample/out-of-sample + overfit uyarısı)
python -m backtest.optimize --strategy vwap_reversion --symbol DOGE/USDT --tf 15m

# Bir coin'in edge'i gerçek mi, şans mı? (seed + eşik sağlamlık testi)
python -m backtest.robustness --symbol DOGE/USDT --tf 15m
```

### ML Modelini Eğitme

```bash
# Tüm config coinleri (15m, ~20k bar, küçük model)
python train_model.py

# Tek coin (ör. DOGE-ML'i aktive etmek için)
python train_model.py --symbol DOGE/USDT --tf 15m
```

> **Önemli:** Eğitim doğruluğu (val_acc) **kâr kanıtı DEĞİLDİR.** Bir modele güvenmeden önce mutlaka `walkforward_ml` ile out-of-sample edge testinden geçirin.

### Dashboard'a Erişim

Bot çalışırken tarayıcınızda:
```
http://127.0.0.1:8000
```

---

## Stratejiler

> Stratejiler **dürüst out-of-sample backtestten geçirildi** ve canlı motorda yalnızca
> edge gösterenler aktif. Sonuçlar için [Dürüst Backtesting](#dürüst-backtesting--edge-durumu).

### VWAP Reversion (15m) — AKTİF
- **Sinyal:** Fiyatın rolling VWAP'tan sapması band'ı (varsayılan %1.5) aşınca ortalamaya dönüş alış/satış + hacim onayı; VWAP'a dönünce erken çıkış
- **SL/TP:** %1.5 stop / %1.5 hedef
- **Durum:** `strategy_eval`'de **5/5 coinde PF>1** (DOGE 1.78, SOL 1.56, AVAX 1.27, XRP 1.23, ADA 1.03) — pakedeki en güçlü in-sample edge. **Dürüstlük notu:** 70/30 IS/OOS bölmesinde OOS edge'i marjinal (SOL ~0.95, AVAX ~0.87); `mean_reversion` ile aynı çıtada olduğu için **paper forward-test** amacıyla aktif. Kripto + hisselerde çalışır.

### Mean Reversion (15m) — AKTİF
- **Sinyal:** Close fiyatının EMA(21)'e göre Z-score'u < −2 (alış) / > +2 (satış) + hacim onayı
- **SL/TP:** %1.5 stop / %1.5 hedef
- **Durum:** Kural-bazlı edge (AVAX PF 1.14, DOGE PF 1.11). Kripto + hisselerde çalışır.

### ML Strategy (15m) — model varsa AKTİF
- **Sinyal:** LSTM yön tahmini (UP/DOWN), güven > %40
- **SL/TP:** %1.5 stop / %3 hedef
- **Durum:** Yalnızca eğitilmiş + uyumlu modeli olan coinde çalışır (backtestte sadece DOGE edge gösterdi). Kripto-only.

### Breakout (1h) — varsayılan KAPALI
- Donchian kanal kopuşu + hacim onayı, ATR-tabanlı stop. **Tüm coinlerde PF<1 (0.73–0.91)** → edge yok.

### Scalping (1m) — varsayılan KAPALI
- RSI + Bollinger + hacim. **Tüm coinlerde PF 0.2–0.4** → %0.4 hedef, %0.2 komisyonu kaldıramıyor. Yapısal olarak kârsız.

### Momentum (1h) — varsayılan KAPALI
- MACD + EMA crossover. **Tüm coinlerde PF 0.45–0.88** → komisyon sonrası kaybediyor.

> Stratejileri `src/bot/engine.py` içindeki `enabled_strategies` ile aç/kapat.
> Yeni bir stratejinin parametrelerini taramak için: `python -m backtest.optimize --strategy <ad> --symbol <coin> --tf <tf>`

---

## Makine Öğrenmesi

### LSTM Modeli

| Özellik | Değer |
|---------|-------|
| Mimari | 1 katmanlı LSTM + FC head (küçük model — overfit önleme) |
| Hidden Size | 48 |
| Dropout | 0.2 |
| Giriş Özellikleri | 15 **durağan** (getiri/oran-bazlı) |
| Çıkış | 3 sınıf (UP / DOWN / SIDEWAYS) |
| Lookback | 60 bar |
| Eğitim verisi | 15m, ~20k bar (~7 ay) — coin başına |

**Neden küçük model?** ~200k parametreli büyük model (eski hidden=128/2-katman) bu veri ölçeğinde ezberler (overfit). Küçük model + bol veri daha sağlıklı.

### Giriş Özellikleri (durağan)
Ham fiyat seviyeleri (trend halinde scaler aralığını taşar → modeli kör eder) yerine
getiri ve oran-bazlı özellikler kullanılır:
```
Getiriler        : log_ret_1, log_ret_3
Mum geometrisi   : hl_range, co_ret, upper_wick, lower_wick
RSI (bounded)    : rsi_norm
MACD (norm.)     : macd_norm, macd_hist_norm
Bollinger        : bb_pband, bb_bandwidth
EMA oranları     : close_ema9_ratio, ema9_ema21_ratio
Volatilite       : atr_pct
Hacim            : volume_ratio
```

### ML Doğruluk İlkeleri
- **Lookahead-bias yok** — scaler yalnızca eğitim bölümüne fit edilir
- **Sınıf ağırlıkları** — SIDEWAYS baskınlığına karşı ters-frekans ağırlıkları
- **Leak-free walk-forward** — gerçek out-of-sample değerlendirme (`backtest/walkforward_ml.py`)

### Sentiment Analizi
- **Keyword-based** — 25 pozitif + 28 negatif borsa terimlerini tarar (LLM yok)
- **Fear & Greed Index** — alternative.me API'den otomatik çekim
- **Kombine skor** — Başlıklar %60 + FGI %40 ağırlıklı harmanlama
- **Haber kaynağı** — `src/data/news_feed.py`: ücretsiz/keyless RSS (kripto) + Alpaca News (hisse)
- **Risk kapısı** — `src/risk/sentiment_gate.py` ile emir öncesi muhafazakâr veto (opt-in; bkz. [Sentiment config](#sentiment--haber-risk-kapısı-opsiyonel-claudesuz))

---

## Risk Yönetimi

### Circuit Breaker Durumları

```
NORMAL ──→ CAUTIOUS ──→ HALTED ──→ EMERGENCY
  ↑           │            │           │
  └───────────┴────────────┴───────────┘
                  (reset)
```

| Durum | Tetikleyici | Eylem |
|-------|------------|-------|
| `NORMAL` | — | Tüm işlemler aktif |
| `CAUTIOUS` | Flash crash (>%5 / 1dk), stale data (>30s) | Yeni giriş duraklatılır |
| `HALTED` | Günlük kayıp limiti, API bağlantı kopması | Trading durdurulur |
| `EMERGENCY` | Max drawdown (%15) aşılması | Tüm pozisyonlar kapatılır |

### Position Sizing (Kelly Criterion)
```python
# Optimal pozisyon boyutu hesaplama
kelly_fraction = win_rate - (1 - win_rate) / payoff_ratio
position_size = kelly_fraction * capital * kelly_multiplier
# Minimum $5, maksimum sermayenin %10'u ile sınırlı
```

---

## Dashboard

Dashboard `http://127.0.0.1:8000` adresinde çalışır ve şunları sunar:

- **Canlı Mum Grafikleri** — TradingView Lightweight Charts
- **Hesap Özeti** — Bakiye, P&L, sermaye dağılımı
- **Açık Pozisyonlar** — Gerçek zamanlı unrealized P&L
- **İşlem Geçmişi** — Tüm kapatılmış işlemler
- **Strateji Sinyalleri** — Son alınan sinyaller ve güven skorları
- **Performans Metrikleri** — Win rate, Sharpe ratio, profit factor
- **Bot Kontrolleri** — Start, Stop, Emergency Stop
- **WebSocket** — 2 saniyelik canlı güncelleme döngüsü

---

## Dürüst Backtesting & Edge Durumu

> Bu projenin en önemli kısmı: **dürüst, kendini kandırmayan ölçüm.** Eski backtestler
> üç ölümcül hata içeriyordu (in-sample test, lookahead-bias, komisyon churn'ü) ve
> sahte-parlak sonuçlar üretiyordu. Yeni altyapı bunları giderir.

### Komutlar

```bash
# Leak-free walk-forward ML backtest (tek coin / tüm coinler)
python -m backtest.walkforward_ml --symbol DOGE/USDT --tf 15m
python -m backtest.walkforward_ml --all --tf 15m --candles 12000 --train 6000 --test 2000

# Kural-bazlı stratejileri tüm coinlerde ölç (PF = sizing-bağımsız edge)
python -m backtest.strategy_eval

# Sağlamlık testi: edge gerçek mi, şans mı? (seed + eşik taraması)
python -m backtest.robustness --symbol DOGE/USDT --tf 15m
```

### Özellikler
- **Leak-free walk-forward** — model geçmişte eğitilir, yalnızca görülmemiş gelecekte test edilir
- Gerçekçi komisyon (%0.1) + slippage (%0.05) her işlemde
- **Buy & hold kıyaslaması** — "excess vs hold" raporlanır
- Timeframe-bilinçli Sharpe/Sortino, max drawdown, profit factor (`backtest/metrics.py`)
- JSON sonuç dosyaları (`backtest/results/`)

### Ölçülen Edge Durumu (dürüst)

**ML (15m, out-of-sample, komisyon dahil):** 5 coinden yalnızca DOGE pozitif edge gösterdi.

| Coin | Net % | Buy&Hold % | PF | Sharpe | Durum |
|------|-------|-----------|-----|--------|-------|
| **DOGE** | +17.99 | −10.24 | 1.27 | 2.53 | Evet |
| AVAX | −7.43 | −27.51 | 0.92 | −0.74 | Hayir |
| SOL | −26.89 | −15.04 | 0.65 | −3.85 | Hayir |
| XRP | −29.85 | −14.24 | 0.64 | −4.73 | Hayir |
| ADA | −51.72 | −31.88 | 0.52 | −8.14 | Hayir |

**Kural-bazlı (PF = işlem başına edge, komisyon sonrası):** 15 kombinasyondan 2'si PF>1.

| Strateji | En iyi sonuç |
|----------|--------------|
| mean_reversion | **AVAX PF 1.24**, SOL PF 1.03 (diğerleri <1) |
| scalping | hepsinde 0.2–0.4 (kârsız) |
| momentum | hepsinde 0.45–0.88 (kârsız) |

> **Dürüst sonuç:** Henüz hiçbiri kanıtlanmış bir "para makinesi" değil. En güçlü sinyaller
> mean-reversion @ AVAX/SOL ve DOGE-ML. Büyük "excess vs hold" sayıları çoğunlukla
> stratejinin düşen piyasada **flat kalmasından** kaynaklanır (beceri değil). Gerçek kanıt =
> uzun süreli paper ileri-test. Test dönemi tek rejimdi (düşüş); farklı rejimde sonuçlar değişebilir.

---

## API Referansı

Dashboard REST API endpoints:

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/api/status` | Bot durumu (running, mode, uptime) |
| `GET` | `/api/account` | Hesap bilgileri (bakiye, P&L) |
| `GET` | `/api/positions` | Açık pozisyonlar listesi |
| `GET` | `/api/trades` | Son 50 işlem geçmişi |
| `GET` | `/api/signals` | Son 50 strateji sinyali |
| `GET` | `/api/performance` | Performans istatistikleri |
| `GET` | `/api/chart_data?symbol=DOGE/USDT&timeframe=1h` | OHLCV grafik verisi |
| `POST` | `/api/bot/start` | Botu başlat |
| `POST` | `/api/bot/stop` | Botu durdur |
| `POST` | `/api/bot/emergency-stop` | Acil durdurma (tüm pozisyonları kapat) |
| `WS` | `/ws` | WebSocket canlı veri akışı |

---

## Testler

```bash
# Tüm testleri çalıştır
pytest

# API bağlantı testleri
python test_api.py

# Strateji testleri (canlı veri)
python test_strategy.py

# Yeni geliştirmelerin birim testleri (offline, ağ mock'lu)
pytest test_enhancements.py -q
```

---

## Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Lütfen aşağıdaki adımları takip edin:

1. **Fork** yapın — Bu repoyu kendi GitHub hesabınıza fork'layın
2. **Branch** oluşturun — `git checkout -b feature/harika-ozellik`
3. **Commit** yapın — `git commit -m 'feat: harika özellik eklendi'`
4. **Push** edin — `git push origin feature/harika-ozellik`
5. **Pull Request** açın — Değişikliklerinizi açıklayan bir PR oluşturun

### Commit Mesaj Formatı

```
feat: yeni özellik eklendi
fix: hata düzeltildi
docs: dokümantasyon güncellendi
refactor: kod yeniden düzenlendi
test: test eklendi
chore: bakım görevi
```

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

## Sorumluluk Reddi

### 🇹🇷 Türkçe

> **Bu yazılım yalnızca eğitim ve araştırma amaçlıdır. Yatırım, finans, hukuk veya
> vergi tavsiyesi DEĞİLDİR ve bu şekilde yorumlanamaz.**
>
> **Garanti Verilmez (Hiçbir Şekilde).** Bu yazılım, açık veya zımni hiçbir garanti
> olmaksızın **"OLDUĞU GİBİ" (AS IS)** ve **"MEVCUT HALİYLE" (AS AVAILABLE)** sunulur.
> Geliştirici(ler); ticarete elverişlilik, belirli bir amaca uygunluk, ihlal etmeme,
> kesintisiz/hatasız/güvenli çalışma, verilerin doğruluğu ya da herhangi bir kâr,
> getiri veya kazanç dahil ancak bunlarla sınırlı olmamak üzere **hiçbir açık veya
> zımni garanti, taahhüt ya da beyanda bulunmaz.**
>
> **Hiçbir Yasal/Mali Sorumluluk Kabul Edilmez.** Geliştirici(ler) ve katkıda
> bulunanlar; bu yazılımın kullanımından, kullanılamamasından, hatalarından,
> kesintilerinden, alınan işlem kararlarından veya bunların sonuçlarından doğan
> **hiçbir doğrudan, dolaylı, arızi, özel, cezai veya sonuç niteliğindeki zarardan
> (kâr kaybı, sermaye kaybı, veri kaybı dahil) yürürlükteki yasaların izin verdiği
> azami ölçüde sorumlu tutulamaz.** Tüm risk ve sorumluluk tamamen kullanıcıya aittir.
>
> - **"Garantili getiri" diye bir şey yoktur.** Bu botun edge'i dürüstçe ölçüldü ve şu an
>   **marjinal/kanıtlanmamış** (bkz. [Edge Durumu](#ölçülen-edge-durumu-dürüst)). Hiçbir kâr vaadi yoktur.
> - **Geçmiş performans ve backtest sonuçları gelecekteki sonuçların göstergesi değildir.**
> - **Önce uzun süre paper modda çalıştırın.** Gerçek paraya ancak paper'da tutarlı,
>   pozitif risk-ayarlı performans gördükten sonra geçin.
> - Bu bot ile yapılan işlemler finansal kayıplara yol açabilir; kripto ve hisse
>   piyasaları yüksek risk içerir.
> - **Kaybetmeyi göze alamayacağınız parayı asla yatırmayın.**
> - Yatırım kararlarınızdan, tüm işlem, vergi ve yasal yükümlülüklerden ve bulunduğunuz
>   ülkenin mevzuatına uyumdan **yalnızca siz sorumlusunuz.**
> - Üçüncü taraf hizmetler (Binance, Alpaca vb.) bağımsızdır; bu proje onlarla bağlantılı
>   değildir ve onların hizmetlerinden/kesintilerinden sorumlu değildir. Onların kullanım
>   koşulları geçerlidir.
> - **Bu yazılımı kullanarak bu şartların tamamını kabul etmiş sayılırsınız.**

### 🇬🇧 English

> **This software is for educational and research purposes only. It is NOT investment,
> financial, legal, or tax advice and must not be construed as such.**
>
> **No Warranty (Whatsoever).** This software is provided **"AS IS"** and
> **"AS AVAILABLE"**, without warranty of any kind, express or implied. The author(s)
> make **no warranties, guarantees, or representations whatsoever**, including but not
> limited to the implied warranties of merchantability, fitness for a particular
> purpose, non-infringement, uninterrupted/error-free/secure operation, accuracy of
> data, or any profit, return, or gain.
>
> **No Legal or Financial Liability.** To the maximum extent permitted by applicable
> law, the author(s) and contributors **shall not be liable for any direct, indirect,
> incidental, special, punitive, or consequential damages** (including loss of profits,
> loss of capital, or loss of data) arising out of or in connection with the use of,
> or inability to use, this software, its errors or interruptions, any trading
> decisions made, or their outcomes. **All risk and responsibility rest solely with
> the user.**
>
> - **There is no such thing as a "guaranteed return".** This bot's edge has been
>   honestly measured and is currently **marginal/unproven** (see [Edge Status](#ölçülen-edge-durumu-dürüst)).
>   No profit is promised.
> - **Past performance and backtest results are not indicative of future results.**
> - **Run it in paper mode for a long time first.** Only move to real money after you
>   have seen consistent, positive risk-adjusted performance in paper trading.
> - Trading with this bot can lead to financial losses; crypto and equity markets are
>   high-risk.
> - **Never invest money you cannot afford to lose.**
> - **You alone are responsible** for your investment decisions, for all trading, tax,
>   and legal obligations, and for compliance with the laws of your jurisdiction.
> - Third-party services (Binance, Alpaca, etc.) are independent; this project is not
>   affiliated with them and is not responsible for their services or outages. Their
>   terms of use apply.
> - **By using this software, you accept these terms in full.**

---

<p align="center">
 Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
</p>

<p align="center">
 Made with and AI
</p>
