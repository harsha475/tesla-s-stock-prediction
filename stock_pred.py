import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


tesla = yf.Ticker("TSLA")
stocks = tesla.history(interval="1d", start="2020-01-01", end="2023-01-01")

# Drop unnecessary columns
stocks.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

# Add tomorrow's closing price and target variable
stocks["Tomorrow"] = stocks["Close"].shift(-1)
stocks["Target"] = (stocks["Tomorrow"] > stocks["Close"]).astype(int)


stocks["MA_10"] = stocks["Close"].rolling(window=10).mean()  # 10-day moving average
stocks["MA_50"] = stocks["Close"].rolling(window=50).mean()  # 50-day moving average
stocks["RSI"] = 100 - (100 / (1 + stocks["Close"].pct_change().apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                                  stocks["Close"].pct_change().apply(lambda x: abs(x)).rolling(window=14).mean()))


stocks.dropna(inplace=True)

# Define predictors (add new features)
predictors = ["Close", "Volume", "Open", "High", "Low", "MA_10", "MA_50", "RSI"]

# Train-test split
train = stocks.iloc[:-100]
test = stocks.iloc[-100:]


scaler = StandardScaler()
train_scaled = train.copy()
test_scaled = test.copy()
train_scaled[predictors] = scaler.fit_transform(train[predictors])
test_scaled[predictors] = scaler.transform(test[predictors])

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [10, 50, 100],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(train_scaled[predictors], train_scaled["Target"])

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(train_scaled[predictors], train_scaled["Target"])

# Make predictions
preds = best_model.predict(test_scaled[predictors])

# Evaluate the model
print("Predictions:", preds)
print("Precision Score:", precision_score(test["Target"], preds))
print("Accuracy Score:", accuracy_score(test["Target"], preds))

# Feature importance visualization
feature_importances = pd.Series(best_model.feature_importances_, index=predictors).sort_values(ascending=False)
feature_importances.plot(kind="bar", title="Feature Importance")
plt.show()
