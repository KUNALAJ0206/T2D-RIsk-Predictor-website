import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- STEP 1: LOAD AND TRAIN MODEL BEHIND THE SCENES ---
print("Training model using dataset...")
df = pd.read_csv('Dataset of Diabetes .csv')
df['CLASS'] = df['CLASS'].str.strip()
df['Target'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

# Feature engineering formulas
df['TG_HDL_Ratio'] = df['TG'] / df['HDL']
df['Non_HDL'] = df['Chol'] - df['HDL']

features = ['HDL', 'LDL', 'TG', 'Chol', 'VLDL', 'TG_HDL_Ratio', 'Non_HDL', 'AGE', 'BMI']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete!\n")

# --- STEP 2: FUNCTION FOR WEBSITE (FLASK BACKEND) ---
def predict_patient(hdl, ldl, tg, chol, vldl, age, bmi):
    tg_hdl_ratio = tg / hdl if hdl > 0 else 0
    non_hdl = chol - hdl

    input_data = pd.DataFrame([[
        hdl, ldl, tg, chol, vldl, tg_hdl_ratio, non_hdl, age, bmi
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    pred_class = model.predict(input_scaled)[0]
    pred_probs = model.predict_proba(input_scaled)[0]

    return pred_class, pred_probs, tg_hdl_ratio, non_hdl

# --- STEP 3: FUNCTION TO ACCEPT YOUR CUSTOM INPUTS (TERMINAL) ---
def predict_custom_patient():
    print("=== ENTER YOUR CUSTOM PATIENT DETAILS ===")
    try:
        age = float(input("Enter Age (years): "))
        bmi = float(input("Enter BMI (kg/m²): "))
        hdl = float(input("Enter HDL Cholesterol (mmol/L): "))
        ldl = float(input("Enter LDL Cholesterol (mmol/L): "))
        tg = float(input("Enter Triglycerides (TG) (mmol/L): "))
        chol = float(input("Enter Total Cholesterol (Chol) (mmol/L): "))
        vldl = float(input("Enter VLDL (mmol/L): "))
    except ValueError:
        print("Invalid input! Please enter numerical values only.")
        return

    pred_class, pred_probs, tg_hdl_ratio, non_hdl = predict_patient(hdl, ldl, tg, chol, vldl, age, bmi)

    labels = {
        0: "Non-Diabetic (Low Risk)", 
        1: "Pre-Diabetic (Moderate Risk)", 
        2: "Diabetic (High Risk)"
    }

    # --- DISPLAY THE MAPPED RESULTS ---
    print("\n==================================================")
    print("                 PREDICTION REPORT                ")
    print("==================================================")
    print(f" Derived TG/HDL Ratio : {tg_hdl_ratio:.2f}")
    print(f" Derived Non-HDL      : {non_hdl:.2f} mmol/L")
    print("--------------------------------------------------")
    print(f" Mapped Risk Category : {labels[pred_class]}")
    print("--------------------------------------------------")
    print(" Probability Breakdown:")
    print(f"   • Non-Diabetic     : {pred_probs[0]*100:.1f}%")
    print(f"   • Pre-Diabetic     : {pred_probs[1]*100:.1f}%")
    print(f"   • Diabetic         : {pred_probs[2]*100:.1f}%")
    print("==================================================\n")

# Run the prompt if executed directly
if __name__ == "__main__":
    predict_custom_patient()