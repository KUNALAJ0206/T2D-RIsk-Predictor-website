import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('Dataset of Diabetes .csv')

# 2. Clean data columns
df['CLASS'] = df['CLASS'].str.strip()
df['Target'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

# 3. Feature Engineering (Formulas)
df['TG_HDL_Ratio'] = df['TG'] / df['HDL']
df['Non_HDL'] = df['Chol'] - df['HDL']

# 4. Define Features & Target
features = ['HDL', 'LDL', 'TG', 'Chol', 'VLDL', 'TG_HDL_Ratio', 'Non_HDL', 'AGE', 'BMI']
X = df[features]
y = df['Target']

# 5. Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test_scaled)
print("--- Model Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']))