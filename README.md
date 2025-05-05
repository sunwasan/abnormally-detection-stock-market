# Financial Anomaly Detection using Variational Autoencoder (VAE)

This project implements a deep learning-based anomaly detection system for financial time series data using a Variational Autoencoder (VAE) with LSTM architecture. The system is designed to identify unusual trading patterns in stock market data that might indicate significant market events or information asymmetry.

## Project Overview

The system analyzes daily trading data from the CPALL stock, building a model that can detect anomalous trading patterns based on various features including price, volume, and trade counts. By learning the normal patterns in financial data, the VAE can identify when market behavior deviates from expected patterns.

## Key Features

- **LSTM-VAE Architecture**: Combination of LSTM networks with a Variational Autoencoder for sequence-based anomaly detection
- **Daily Trading Aggregations**: Processing of tick-level trading data into meaningful daily aggregations
- **Anomaly Visualization**: Visual representation of detected anomalies alongside price movements
- **Quantitative Analysis**: Statistical evaluation of detected anomalies and their relationship with price volatility

## Methodology

1. **Data Preparation**: 
   - Load and process trading data from parquet files
   - Extract daily price series and create daily trading aggregations
   - Split data into training (70%) and testing (30%) sets

2. **Model Architecture**:
   - Encoder: LSTM layers that compress the input sequence into a latent representation
   - Sampling: Probabilistic sampling from the latent space distribution
   - Decoder: LSTM layers that reconstruct the original sequence from the latent representation

3. **Anomaly Detection**:
   - Calculate reconstruction loss for each test sequence
   - Identify anomalies based on a threshold (mean + 0.5 * standard deviation)
   - Visualize anomalies against price movements

## Key Findings

- The model effectively identifies days with abnormal trading patterns based on volume and price features
- Anomalous trading days often coincide with higher price volatility (approximately 3-4x higher)
- The VAE reconstruction loss provides a continuous measure of abnormality, allowing for threshold adjustment
- Significant market events are often captured as anomalies by the model

## Implementation Details

- **Technologies**: Python, TensorFlow, Pandas
- **Input Data**: Time series of daily trading data with features including price, volume, and trade counts
- **Output**: Anomaly scores with visual indicators of unusual market behavior

## Future Work

1. Incorporate additional features such as market indices and sector performance
2. Test the model on different time frames (hourly, weekly) for multi-scale anomaly detection
3. Develop an early warning system by predicting future anomalies
4. Optimize hyperparameters for improved anomaly detection accuracy

## Usage

The main analysis and model implementation can be found in `report.ipynb`, which contains the complete workflow from data preparation to anomaly visualization.

### Data Loading Note

For reproducibility, it's recommended to use the processed daily data from saved CSV files rather than reprocessing raw data:

```python
tm_daily = pd.read_csv(local_data_dir / f"{symbol}_daily.csv", index_col=0, parse_dates=True)
print(f"Loaded {len(tm_daily):,} records for {symbol} daily data")
```

## References

- *Anomaly Detection in Stock Market Transactions: A Comparison of Deep Learning Methods* (2023). Research Gate. https://www.researchgate.net/publication/382496033_Anomaly_Detection_in_Stock_Market_Transactions_A_Comparison_of_Deep_Learning_Methods
- Foster, D. (2019). *Generative Deep Learning*. O'Reilly Media.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
