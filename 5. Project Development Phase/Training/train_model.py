import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
data = pd.read_excel("flood dataset.xlsx")
X = data[["Temp", "Humidity", "Cloud Cover"]]
y = data["flood"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
print(data["flood"].value_counts())
print("Model trained successfully!")