import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import load_clean_data

df = load_clean_data()

# Encode category
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category_main"])

# Features
X = df[["actual_price", "rating", "rating_count", "category_encoded"]]

# Target variable
y = df["discount_percent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

print("Random Forest Metrics:")
print("MAE:", mean_absolute_error(y_test, rf_y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_y_pred)))
print("R2 Score:", r2_score(y_test, rf_y_pred))


print("Training Gradient Boosting")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,          
    learning_rate=0.1,    
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)

print("Gradient Boosting Metrics:")
print("MAE:", mean_absolute_error(y_test, gb_y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, gb_y_pred)))
print("R2 Score:", r2_score(y_test, gb_y_pred))

#joblib.dump(rf_model, "model.pkl") 
joblib.dump(gb_model, "model.pkl") 

joblib.dump(le, "label_encoder.pkl")

# Drift Analysis Stats
training_stats = {
    "actual_price_mean": float(X_train["actual_price"].mean()),
    "actual_price_std": float(X_train["actual_price"].std()),
    "rating_mean": float(X_train["rating"].mean()),
    "rating_std": float(X_train["rating"].std()),
    "rating_count_mean": float(X_train["rating_count"].mean()),
    "rating_count_std": float(X_train["rating_count"].std())
}

with open("training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=4)

print("Training completed")