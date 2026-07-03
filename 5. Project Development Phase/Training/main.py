import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
print("Libraries imported successfully!")
dataset = pd.read_excel("flood dataset.xlsx")
dataset.columns=dataset.columns.str.strip()
print(dataset.columns.tolist())
print("Dataset loaded successfully!")
print(dataset.head())
print("Shape:", dataset.shape)
dataset.info()
print(dataset.describe())
num_cols = dataset.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    dataset[col] = np.where(dataset[col] < lower, lower, dataset[col])
    dataset[col] = np.where(dataset[col] > upper, upper, dataset[col])
print("Outlier handling completed")
print(dataset.dtypes)
cat_cols = dataset.select_dtypes(include=['object']).columns
print("Categorical Columns:", list(cat_cols))
if len(cat_cols) == 0:
    print("No categorical columns found. Encoding not required.")
print(dataset.isnull().any())
print(dataset.isnull().sum())
plt.figure(figsize=(8,5))
sns.histplot(dataset.iloc[:,0], kde=True)
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(y=dataset.iloc[:,0])
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("Dataset loaded successfully!")
print(dataset.head())
X = dataset[['Temp',
             'Humidity',
             'Cloud Cover',
             'ANNUAL',
             'Jan-Feb',
             'Mar-May',
             'Jun-Sep',
             'Oct-Dec',
             'avgjune',
             'sub']]
print(dataset['flood'].unique())
print(dataset['flood'].value_counts())
y = dataset['flood']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
dtree = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=99/16,
    random_state=42
)
dtree.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
xgb.fit(X_train, y_train)
dt_pred = dtree.predict(X_test)
rf_pred = rf.predict(X_test)
knn_pred = knn.predict(X_test)
xgb_pred = xgb.predict(X_test)
print("Decision Tree:", accuracy_score(y_test, dt_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("KNN:", accuracy_score(y_test, knn_pred))
print("XGBoost:", accuracy_score(y_test, xgb_pred))
print("Decision Tree:", accuracy_score(y_test, dt_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("KNN:", accuracy_score(y_test, knn_pred))
print("XGBoost:", accuracy_score(y_test, xgb_pred))
cm = confusion_matrix(y_test, xgb_pred)
print(cm)
print(classification_report(y_test, xgb_pred))
joblib.dump(xgb, "model.pkl")
joblib.dump(scaler, "transform.save")
print("Model saved successfully!")
print(cm)