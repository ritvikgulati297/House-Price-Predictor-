# House Price Prediction

A regression project that predicts house prices from property features : size, room counts, amenities, and location indicators  and identifies which of those features actually move the price the most.

## Overview

Real estate buyers and sellers often price homes off gut feeling or rough comparisons. This project builds and compares two regression models on a real housing dataset to turn that guesswork into a data-backed estimate, and digs into *why* the model predicts what it does.

## Dataset

[Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) : 545 houses, 13 columns:

- `price` — target variable
- `area`, `bedrooms`, `bathrooms`, `stories`, `parking` — numeric features
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` , yes/no amenity & location flags
- `furnishingstatus` — furnished / semi-furnished / unfurnished

The dataset arrived clean — no missing values and no duplicate rows.

## Methodology

1. **Exploration** : loaded the data, checked shape, dtypes, and missing values.
2. **Cleaning** : verified no nulls/duplicates, one-hot encoded all categorical and yes/no columns.
3. **Modeling** : 80/20 train-test split; trained Linear Regression and Random Forest Regressor; evaluated both with MAE, RMSE, and R², then validated the comparison with 5-fold cross-validation and multiple random splits rather than trusting a single split.
4. **Visualization** : price distribution histogram, feature correlation heatmap, and a Random Forest feature-importance chart.
5. **Insights** : interpreted what drives price, how accurate the model really is, and what it implies for a real estate business.

## Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | ₹970,043 | ₹1,324,507 | 0.65 |
| Random Forest | ₹1,021,546 | ₹1,400,566 | 0.61 |

Linear Regression edges out Random Forest on this dataset, and the edge holds up across cross-validation, not just one lucky split — likely because price scales fairly linearly with the strongest features and 545 rows isn't much for a tree ensemble to exploit.

**Biggest price drivers:** `area` and `bathrooms`, by a wide margin, followed by `airconditioning`, `stories`, and `parking`.

**Accuracy in plain terms:** the model explains about 65% of why prices vary (R² = 0.65) and is typically off by about ₹9.7 lakh, or roughly 21% of a house's price — useful as a quick ballpark, not a substitute for an actual appraisal.

## Repository Structure

```
.
├── analysis.ipynb       # full notebook: all 5 tasks, executed with outputs
├── Housing.csv           # dataset used
├── charts/               # exported chart images
│   ├── chart1_price_distribution.png
│   ├── chart2_correlation_heatmap.png
│   └── chart3_feature_importance.png
├── summary.pdf            # 1-page written summary
└── README.md
```

## Tools & Libraries

Python 3, Jupyter Notebook, pandas, scikit-learn, matplotlib, seaborn.

## Running It Locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook analysis.ipynb
```

## Ritvik Gulati

[Your Name]
