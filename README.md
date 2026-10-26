# Stock Price Prediction Using Stacked LSTM Networks

A college case study (B.E. CSE – AI & ML, CBIT Hyderabad) that predicts the **next trading day's closing price** of a stock with a stacked LSTM network, using 60 days of prices and technical indicators.

> Educational project only. Not financial advice.

## How it works

1. **Data:** daily OHLCV prices from Yahoo Finance (`yfinance`), split- and dividend-adjusted
2. **Features:** Close, Volume, 10-day and 30-day moving averages, RSI (14), MACD, daily return
3. **Preprocessing:** chronological 70 / 15 / 15 train / validation / test split; Min–Max scalers fitted on training data only, so there is no data leakage
4. **Sequences:** a 60-day sliding window (60 × 7 features) predicts the next day's close
5. **Model:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(25, ReLU) → Dense(1), 31,699 parameters
6. **Training:** Adam (lr 0.001), MSE loss, early stopping and learning-rate reduction
7. **Evaluation:** RMSE, MAE, MAPE, R² and directional accuracy, compared with a naive "tomorrow = today" baseline

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: Apple (AAPL) from 2015
python stock_lstm_predictor.py

# Indian stocks (NSE)
python stock_lstm_predictor.py --ticker RELIANCE.NS --start 2015-01-01
python stock_lstm_predictor.py --ticker TCS.NS --window 60 --epochs 50
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--ticker` | `AAPL` | Yahoo Finance symbol |
| `--start` | `2015-01-01` | First date of data |
| `--end` | latest | Last date of data |
| `--window` | `60` | Look-back days per sample |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `32` | Batch size |
| `--units` | `64` | Units in the first LSTM layer |

## Outputs

Everything is saved in `outputs/`:

- `01_price_history.png`: price with moving averages
- `02_training_loss.png`: training and validation loss
- `03_actual_vs_predicted.png`: predictions on the test set
- `04_error_distribution.png`: histogram of prediction errors
- `metrics.csv`: test-set metrics for the LSTM and the naive baseline
- `lstm_<ticker>.keras`: the trained model

The script also prints a forecast for the next trading day.

## Results

<!-- Replace with your own numbers from outputs/metrics.csv -->

| Model | RMSE | MAE | MAPE (%) | R² | Directional Acc. (%) |
| --- | --- | --- | --- | --- | --- |
| Stacked LSTM | – | – | – | – | – |
| Naive baseline | – | – | – | – | – |

## Authors

- P. Akshar (160124748125)
- Prajeeth Reddy (160124748127)

Guide: Panigrahi Srikanth, Assistant Professor, Department of CSE (AI & ML), CBIT Hyderabad
