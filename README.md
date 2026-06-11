<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status">
</p>

<h1 align="center">🤖 AI Trade Bot</h1>

<p align="center">
  <strong>Yapay Zeka Destekli Kripto & Hisse Senedi Mikro-Sermaye Trading Botu</strong>
</p>

<p align="center">
  LSTM derin öğrenme, ensemble strateji motoru, gerçek zamanlı dashboard ve<br>
  çok katmanlı risk yönetimi ile donatılmış otonom trading sistemi.
</p>

<p align="center">
  <a href="#-özellikler">Özellikler</a> •
  <a href="#-mimari">Mimari</a> •
  <a href="#-kurulum">Kurulum</a> •
  <a href="#-kullanım">Kullanım</a> •
  <a href="#-stratejiler">Stratejiler</a> •
  <a href="#-backtesting">Backtesting</a> •
  <a href="#-katkıda-bulunma">Katkıda Bulunma</a>
</p>

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Mimari](#-mimari)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Konfigürasyon](#-konfigürasyon)
- [Kullanım](#-kullanım)
- [Stratejiler](#-stratejiler)
- [Makine Öğrenmesi](#-makine-öğrenmesi)
- [Risk Yönetimi](#-risk-yönetimi)
- [Dashboard](#-dashboard)
- [Backtesting](#-backtesting)
- [API Referansı](#-api-referansı)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)
- [Sorumluluk Reddi](#-sorumluluk-reddi)

---

## ✨ Özellikler

### 🧠 Yapay Zeka & Makine Öğrenmesi
- **LSTM Derin Öğrenme Modeli** — İki katmanlı LSTM ağı ile fiyat yönü tahmini (UP / DOWN / SIDEWAYS)
- **Duygu Analizi** — Haber başlıkları için keyword-bazlı sentiment analizi + Crypto Fear & Greed Index entegrasyonu
- **FinBERT Hazırlığı** — HuggingFace Transformers ile gelişmiş NLP sentiment analizi altyapısı (stub)

### 📊 Çoklu Strateji Motoru
- **Scalping** — 1 dakikalık mumlarla hızlı alım-satım, RSI + Bollinger Bands + hacim analizi
- **Mean Reversion** — 15 dakikalık zaman diliminde ortalamaya dönüş stratejisi
- **Momentum** — 1 saatlik trend takip stratejisi, ADX + EMA crossover
- **ML Strategy** — LSTM model tahminlerine dayalı yapay zeka stratejisi
- **Ensemble** — Ağırlıklı oylama ile tüm stratejilerin konsensüs sinyali, dinamik ağırlık ayarı

### 🛡️ Çok Katmanlı Risk Yönetimi
- **Circuit Breaker** — 4 durumlu (NORMAL → CAUTIOUS → HALTED → EMERGENCY) otomatik koruma sistemi
- **Position Sizing** — Kelly Criterion bazlı optimal pozisyon boyutlandırma
- **Risk Manager** — Günlük kayıp limiti, max drawdown, portföy maruziyeti kontrolleri
- **Trailing Stop** — Dinamik stop-loss güncelleme mekanizması

### 🔗 Çoklu Borsa Desteği
- **Binance** — Kripto para çiftleri (DOGE, SHIB, PEPE, XRP, ADA, SOL, AVAX, MATIC)
- **Alpaca Markets** — ABD hisse senetleri (AAPL, MSFT, TSLA, NVDA, AMD, META) + fractional shares
- **Paper & Live Mode** — Testnet üzerinde paper trading veya gerçek hesapla canlı trading

### 📡 Gerçek Zamanlı Dashboard
- **WebSocket** ile canlı veri akışı
- **TradingView Charts** — Interaktif mum grafikleri
- **Pozisyon & İşlem Takibi** — Açık pozisyonlar, trade geçmişi, sinyal monitörü
- **Bot Kontrolleri** — Start / Stop / Emergency Stop butonları
- **Performans İstatistikleri** — Win rate, Sharpe ratio, equity curve

### 🔔 Bildirim Sistemi
- **E-posta Bildirimleri** — SMTP üzerinden trade ve risk uyarıları
- **WebSocket Push** — Dashboard üzerinden anlık bildirimler

---

## 🏗 Mimari

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
    ├── Scalping (1m)
    ├── Mean Reversion (15m)
    ├── Momentum (1h)
    └── ML Strategy (1h, LSTM)

┌───────────────────────────────────────────┐
│           Dashboard (FastAPI)              │
│  REST API + WebSocket + Jinja2 Templates   │
└───────────────────────────────────────────┘
```

---

## 📁 Proje Yapısı

```
AI-Trade-Bot/
│
├── main.py                     # Ana giriş noktası (CLI)
├── run_bot.py                  # Bot başlatma scripti
├── run_backtest.py             # Backtest çalıştırma scripti
├── train_model.py              # ML model eğitim scripti
├── requirements.txt            # Python bağımlılıkları
│
├── config/                     # Konfigürasyon
│   ├── __init__.py
│   ├── settings.py             # Merkezi ayarlar (dataclass + .env)
│   ├── .env.example            # Ortam değişkenleri şablonu
│   └── .env                    # 🔒 Gerçek anahtarlar (git'e dahil DEĞİL)
│
├── src/                        # Ana kaynak kodu
│   ├── __init__.py
│   │
│   ├── bot/                    # Bot motoru
│   │   ├── engine.py           # Ana orkestratör (trading loop)
│   │   └── scheduler.py        # Görev zamanlayıcı
│   │
│   ├── data/                   # Veri toplama & depolama
│   │   ├── collector.py        # OHLCV veri toplayıcı
│   │   ├── storage.py          # SQLite async depolama (SQLAlchemy)
│   │   └── websocket_feed.py   # Binance WebSocket canlı veri akışı
│   │
│   ├── indicators/             # Teknik göstergeler
│   │   └── technical.py        # RSI, MACD, Bollinger, EMA, ATR, vb.
│   │
│   ├── strategy/               # Trading stratejileri
│   │   ├── base.py             # Abstract base class + Signal/Position
│   │   ├── scalping.py         # Scalping stratejisi (1m)
│   │   ├── mean_reversion.py   # Ortalamaya dönüş (15m)
│   │   ├── momentum.py         # Trend takip (1h)
│   │   ├── ml_strategy.py      # LSTM bazlı ML stratejisi
│   │   └── ensemble.py         # Ağırlıklı oylama ensemble
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
├── backtest/                   # Backtesting motoru
│   ├── __init__.py
│   ├── backtester.py           # Strateji backtester (Plotly raporları)
│   └── results/                # Backtest sonuç dosyaları
│
├── models/                     # Eğitilmiş ML modelleri
│   ├── DOGE_USDT_15m/          # DOGE/USDT 15m LSTM modeli
│   └── DOGE_USDT_1h/           # DOGE/USDT 1h LSTM modeli
│
├── data/                       # Veritabanı
│   └── tradebot.db             # SQLite veritabanı (git'e dahil DEĞİL)
│
├── logs/                       # Log dosyaları (git'e dahil DEĞİL)
│
└── tests/                      # Test dosyaları
    ├── test_api.py             # API bağlantı testleri
    └── test_strategy.py        # Strateji testleri
```

---

## 🚀 Kurulum

### Gereksinimler

- **Python** 3.11 veya üzeri
- **pip** paket yöneticisi
- **Git** versiyon kontrol sistemi
- **Redis** (opsiyonel — caching/pub-sub için)

### 1. Repoyu Klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/ai-trade-bot.git
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

> **Not:** PyTorch GPU desteği için [pytorch.org](https://pytorch.org/get-started/locally/) adresinden CUDA sürümünüze uygun komutu kullanın.

### 4. Ortam Değişkenlerini Ayarlayın

```bash
# .env.example dosyasını kopyalayın
cp config/.env.example config/.env

# Editörünüzle açıp API anahtarlarınızı girin
notepad config/.env    # Windows
nano config/.env       # Linux/macOS
```

---

## ⚙️ Konfigürasyon

Tüm ayarlar `config/.env` dosyasından yüklenir. Önemli parametreler:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `TRADING_MODE` | `paper` | `paper` (simülasyon) veya `live` (gerçek) |
| `INITIAL_CAPITAL` | `50.0` | Başlangıç sermayesi (USD) |
| `CRYPTO_ALLOCATION` | `0.75` | Kripto'ya ayrılan oran (%75) |
| `STOCK_ALLOCATION` | `0.25` | Hisse senedine ayrılan oran (%25) |
| `BINANCE_API_KEY` | — | Binance API anahtarı |
| `BINANCE_SECRET_KEY` | — | Binance gizli anahtar |
| `BINANCE_TESTNET` | `true` | Testnet kullanımı |
| `ALPACA_API_KEY` | — | Alpaca API anahtarı |
| `ALPACA_SECRET_KEY` | — | Alpaca gizli anahtar |
| `ALPACA_PAPER` | `true` | Paper trading modu |
| `LOG_LEVEL` | `INFO` | Log seviyesi (DEBUG/INFO/WARNING/ERROR) |
| `DASHBOARD_PORT` | `8000` | Dashboard port numarası |

### Risk Parametreleri (Hardcoded — `config/settings.py`)

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `max_risk_per_trade` | %2 | Trade başına max risk (~$1) |
| `daily_loss_limit` | %3 | Günlük max kayıp |
| `max_open_positions` | 3 | Eş zamanlı max pozisyon |
| `max_drawdown` | %15 | Maksimum drawdown (Emergency trigger) |
| `max_portfolio_exposure` | %30 | Piyasaya max maruziyet |

---

## 💻 Kullanım

### Botu Başlatma

```bash
# Paper mode (varsayılan) — gerçek para riski yok
python main.py

# Verbose logging ile
python main.py -v

# Live mode (⚠️ gerçek para!)
python main.py --mode live
```

### ML Modelini Eğitme

```bash
python train_model.py
```

### Backtest Çalıştırma

```bash
python run_backtest.py
```

### Dashboard'a Erişim

Bot çalışırken tarayıcınızda:
```
http://127.0.0.1:8000
```

---

## 📈 Stratejiler

### Scalping (1 dakika)
- **Sinyal:** RSI oversold/overbought + Bollinger Band squeeze + hacim spike
- **Hedef:** Küçük ama sık kazançlar
- **Stop Loss:** ATR bazlı dinamik SL

### Mean Reversion (15 dakika)
- **Sinyal:** Fiyatın Bollinger alt bandına temas + RSI divergence
- **Hedef:** Ortalamaya dönüş hareketi
- **Stop Loss:** Band dışı kapanış

### Momentum (1 saat)
- **Sinyal:** EMA 9/21 crossover + ADX > 25 + hacim onayı
- **Hedef:** Trend yönünde uzun süreli pozisyon
- **Trailing Stop:** ATR bazlı dinamik takip

### Ensemble (Konsensüs)
- **Ağırlıklar:** Scalping %30, Mean Reversion %30, Momentum %40
- **Minimum güven:** %60 konsensüs eşiği
- **Giriş koşulu:** En az 2 stratejinin aynı yönde anlaşması
- **Dinamik ağırlık:** Performansa dayalı Sharpe-benzeri otomatik ayar

---

## 🧠 Makine Öğrenmesi

### LSTM Modeli

| Özellik | Değer |
|---------|-------|
| Mimari | 2 katmanlı LSTM + FC head |
| Hidden Size | 128 |
| Dropout | 0.2 |
| Giriş Özellikleri | 17 (OHLCV + teknik göstergeler) |
| Çıkış | 3 sınıf (UP / DOWN / SIDEWAYS) |
| Lookback | 60 bar |
| Eşikler | UP > +0.3%, DOWN < -0.3% |

### Giriş Özellikleri
```
OHLCV (5)       : open, high, low, close, volume
RSI (1)          : rsi_14
MACD (3)         : macd_line, macd_signal, macd_histogram
Bollinger (4)    : bb_upper, bb_middle, bb_lower, bb_bandwidth
EMA (2)          : ema_9, ema_21
ATR (1)          : atr_14
Volume Ratio (1) : volume_ratio
```

### Sentiment Analizi
- **Keyword-based** — 25 pozitif + 29 negatif borsa terimlerini tarar
- **Fear & Greed Index** — alternative.me API'den otomatik çekim
- **Kombine skor** — Başlıklar %60 + FGI %40 ağırlıklı harmanlama

---

## 🛡️ Risk Yönetimi

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

## 📊 Dashboard

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

## 🔬 Backtesting

### Tekli Strateji Backtesti

```python
from backtest.backtester import Backtester
from src.strategy.momentum import MomentumStrategy

bt = Backtester(initial_capital=50.0)
result = await bt.run(
    strategy=MomentumStrategy(),
    symbol="SOL/USDT",
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-06-01",
)
print(result.summary())
```

### Çoklu Strateji Karşılaştırma

```python
comparison = await bt.compare_strategies(
    strategies=[ScalpingStrategy(), MeanReversionStrategy(), MomentumStrategy()],
    symbol="SOL/USDT",
    timeframe="1h",
)
```

### Walk-Forward Testi

```python
results = await bt.walk_forward(
    strategy=MomentumStrategy(),
    symbol="SOL/USDT",
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-12-01",
    train_pct=0.70,
    n_splits=3,
)
```

### Backtesting Özellikleri
- ✅ Gerçekçi fee simülasyonu (Binance %0.1)
- ✅ Slippage modelleme (%0.05)
- ✅ Position sizing entegrasyonu
- ✅ Risk yönetimi kuralları uygulanır
- ✅ Plotly ile interaktif HTML raporlar
- ✅ JSON sonuç dosyaları (`backtest/results/`)

---

## 🔌 API Referansı

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

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest

# API bağlantı testleri
python test_api.py

# Strateji testleri
python test_strategy.py
```

---

## 🤝 Katkıda Bulunma

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

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

## ⚠️ Sorumluluk Reddi

> **Bu yazılım yalnızca eğitim ve araştırma amaçlıdır.**
>
> - Bu bot ile yapılan işlemler finansal kayıplara yol açabilir
> - Yazılım "olduğu gibi" sunulur, hiçbir garanti verilmez
> - Gerçek para ile kullanmadan önce kapsamlı backtesting yapın
> - Kaybetmeyi göze alamayacağınız parayı asla yatırmayın
> - Kripto para ve hisse senedi piyasaları yüksek risk içerir
> - Yatırım kararlarınızdan yalnızca siz sorumlusunuz
> - Bu yazılımın geliştiricileri hiçbir mali sorumluluk kabul etmez

---

<p align="center">
  ⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
</p>

<p align="center">
  Made with ❤️ and 🤖 AI
</p>
