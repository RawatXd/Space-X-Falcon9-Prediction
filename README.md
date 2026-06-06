# Predicting Falcon 9 Landing Success & Launch Cost Impact

## Project Overview

This project uses machine learning to predict whether a SpaceX Falcon 9 first stage will successfully land and be reused. Successful recovery dramatically reduces launch cost (approx. $165M down to $62M), so forecasting landing outcomes provides financial insight for satellite operators planning missions.

## Objectives
* Collect and prepare Falcon 9 launch data from SpaceX API
* Engineer features related to flight history, payload, site, and booster config
* Train and tune ML models to classify landing success
* Estimate expected launch cost based on predicted reuse probability
* Provide risk guidance for mission planning

## Methods
* **Data**: SpaceX v4 API (90 Falcon 9 launches through 2020)
* **Models**: Logistic Regression, SVM, Decision Tree, KNN
* **Techniques**: One-hot encoding, standardization, GridSearchCV, confusion matrices
* **Metrics**: Cross-validation accuracy, test accuracy

## Key Results
* Best models: Logistic Regression & SVM (~0.83 test accuracy)
* Demonstrated cost-impact calculation using landing probability
* Example inference for a new mission:
    * Landing success probability: ~85%
    * Estimated launch cost: ~$77M
    * Risk level: Low


## Strategic Insights
* Booster landing probability drives launch-cost expectation
* Customers can use probability-based pricing logic to:
    * Negotiate contracts when risk is higher
    * Compare SpaceX vs. traditional launch pricing
    * Allocate insurance and contingency spending
    * More historical data will improve model robustness, especially across orbit classes and booster configurations


## Future Work
* Incorporate telemetry & weather data
* Deploy as interactive dashboard
* Add calibration + ROC-AUC reporting

