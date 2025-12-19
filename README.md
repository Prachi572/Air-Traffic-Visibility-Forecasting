# Air Traffic Visibility Forecasting

## Overview
Machine Learning pipeline for predicting visibility for Air Traffic Control using environmental data. Achieves R² = 0.86+ using ensemble methods.

## Tech Stack
- **Python 3.x**
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Matplotlib, Seaborn
- **Encoding**: Category Encoders

## Project Structure
- `Predicting visibility for Air Traffic Control (1).ipynb` - Main notebook with complete ML pipeline
- `PROJECT_SUMMARY.txt` - Comprehensive project documentation and interview guide
- `IndianWeatherRepository.csv` - Dataset (not included due to size)

## Key Features
✅ Feature engineering with 3 encoding techniques (Target, Binary, Cyclic)  
✅ PCA dimensionality reduction (46 → 32 features)  
✅ 6+ models compared (Decision Tree, Random Forest, KNN, SVR, XGBoost, LightGBM)  
✅ Hyperparameter tuning with GridSearchCV  
✅ Overfitting mitigation (pruning, regularization, CV)  

## Results
- **Best Model**: Random Forest / Gradient Boosting
- **Test R²**: 0.86+
- **Test RMSE**: 0.78 km
- **Target**: visibility_km (2.5-10 km range)

## Installation
```bash
pip install numpy pandas scikit-learn xgboost lightgbm category-encoders matplotlib seaborn
```

## Usage
Open and run the Jupyter notebook:
```bash
jupyter notebook "Predicting visibility for Air Traffic Control (1).ipynb"
```

## Model Pipeline
1. **Data Loading** - 540 records, 46 features
2. **Feature Engineering** - Correlation analysis, encoding, scaling
3. **PCA** - Dimensionality reduction (99% variance with 32 components)
4. **Model Training** - Multiple algorithms with cross-validation
5. **Hyperparameter Tuning** - GridSearchCV optimization
6. **Evaluation** - R², RMSE, MAE, learning curves

## Author
Prachi

## License
MIT License
