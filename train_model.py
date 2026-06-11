import asyncio
from loguru import logger
from src.ml.trainer import ModelTrainer

async def main():
    trainer = ModelTrainer()
    symbol = "DOGE/USDT"
    try:
        logger.info("Starting model training for {}...", symbol)
        history = await trainer.train_lstm(
            symbol=symbol,
            timeframe="15m",
            lookback_days=60,
            epochs=100,
            lookback=60,
            batch_size=64,
        )
        logger.info("Final val_accuracy: {}", history.val_accuracies[-1] if history.val_accuracies else 'N/A')
    except Exception as e:
        logger.exception("Failed: {}", e)
    finally:
        await trainer.close()

if __name__ == "__main__":
    asyncio.run(main())
