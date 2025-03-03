# Oil Price Volatility Classification and Prediction

## Project Overview
This repository contains the implementation of machine learning models to classify and predict oil price volatility using sentiment analysis, financial news headlines, Google Trends data, and energy market metrics. The project aims to help stakeholders in the energy sector make more informed decisions by accurately forecasting oil price volatility.

## Authors
- Kerem Çiftçi
- Eren Ergüzel 
- Abdullah Rüzgar
- Mazhar Mert Ulukaya

## Dataset
The project utilizes a comprehensive dataset spanning from June 2011 to September 2024, including:

1. **Financial News Headlines**: 15+ years of data (22,000+ headlines) from Oilprice.com
2. **Sentiment Analysis**: 
   - FinBERT scores for headline sentiment analysis
   - VADER sentiment analysis metrics
3. **Google Trends Data**: 17 features including search rates for terms like "WTI", "crude", "oil price", "OPEC"
4. **Energy Metrics**: 30 features from U.S. Energy Information Administration (EIA) including:
   - Crude oil production metrics
   - Petroleum consumption data
   - Inventory withdrawal data
5. **Market Variables**:
   - VIX (market volatility index)
   - DXY (U.S. Dollar Index) 
   - Geopolitical Risk (GPR) Index
6. **Target Variable**: 
   - Daily 30-day volatility (sourced from Bloomberg Terminal)
   - Monthly average of daily volatility for classification

## Model Implementation

### Classification Models
Four models to classify volatility as high (>32) or low (≤32):

1. **Artificial Neural Network (ANN)**
   - Input layer: 16 neurons
   - Hidden layers: 64 neurons and 32 neurons
   - Output layer: 1 neuron with sigmoid activation
   - ReLU activation in hidden layers
   - Early stopping for regularization

2. **Random Forest Classifier**
   - Ensemble approach for robustness
   - Effective for handling a large number of features

3. **Gradient Boosting**
   - Models complex and non-linear relationships
   - Built-in regularization to reduce overfitting

4. **Logistic Regression**
   - Binary classification with probability output

### Prediction Models
Four models to predict the actual volatility values:

1. **Artificial Neural Network**
   - Similar architecture to classification ANN
   - Single output neuron without activation function
   - Early stopping for regularization

2. **Random Forest Regression**
   - Ensemble method for robust predictions
   - Feature dimensionality reduction

3. **Gradient Boosting**
   - Captures complex relationships
   - Effective for handling outliers

4. **Elastic Net Regression**
   - Combines Lasso and Ridge regression
   - Feature selection with regularization
   - Optimized hyperparameters


## Project Structure

├───data
│   ├───processed
│   └───raw
├───reports
├───results
│   ├───images
│   └───logs
└───src


The project source code is contained in the following notebook:

    `src/models.ipynb`


## Getting Started

### Prerequisites
- Python3
- Jupyter Notebook or JupyterLab
- Required packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - keras
  - matplotlib
  - seaborn


### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/oil-price-volatility.git
cd oil-price-volatility
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:
```bash
jupyter notebook
```

### Data Collection
- Financial news headlines require web scraping from Oilprice.com
- Google Trends data can be collected using the pytrends API
- EIA data is available through their public API
- Market variables (VIX, DXY) can be obtained from financial data providers
- Note: Some data sources may require API keys or subscriptions

## Results

### Classification Performance
- **Random Forest**: Highest ROC curve with strong generalization
- **Gradient Boosting**: Higher variance with greater accuracy
- **Logistic Regression**: Surprisingly effective, indicating linear relationships
- **Neural Network**: Showed bias toward the lower variance class

### Prediction Performance
- **Elastic Net**: Best performer with Mean R² Score of 0.6817
- **Random Forest**: Strong generalization with Mean R² Score of 0.6549
- **Gradient Boosting**: Good at capturing outliers with Mean R² Score of 0.6588
- **Neural Network**: Lower performance with Mean R² Score of 0.5720

### Sentiment Analysis Impact
- Including sentiment features improved prediction accuracy
- LSTM model with sentiment features reduced MSE from 272 to 241
- Sentiment data plays a complementary rather than dominant role


## Acknowledgments
- Data sources: Oilprice.com, Google Trends, EIA, Bloomberg Terminal
- Academic references are listed in the full paper