import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score

# === Load test data ===
test_data = pd.read_csv("parkinsons_disease_data_cls_test.csv")  # ← Replace with your test file

# === Load preprocessing artifacts ===
with open("preprocessing_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

mode = artifacts["mode_education_level"]
drop_cols = artifacts["columns_to_drop"]
label_mappings = artifacts["label_mappings"]
numeric_cols = artifacts["numeric_cols"]

# === Fill missing 'EducationLevel' with mode ===
test_data["EducationLevel"] = test_data["EducationLevel"].fillna(mode)

# === Convert WeeklyPhysicalActivity from H:MM to float hours ===
for i in range(len(test_data)):
    value = str(test_data.loc[i, 'WeeklyPhysicalActivity (hr)'])
    if ':' in value:
        h, m = map(float, value.split(':'))
        test_data.loc[i, 'WeeklyPhysicalActivity (hr)'] = h + m / 60
    else:
        test_data.loc[i, 'WeeklyPhysicalActivity (hr)'] = float(value)

# === Drop irrelevant columns if they exist ===
for col in ['DoctorInCharge', 'PatientID']:
    if col in test_data.columns:
        test_data.drop(columns=col, inplace=True)

# === Parse 'MedicalHistory' and 'Symptoms' ===
def parse_dict_column(df, colname):
    if colname not in df.columns:
        print(f"Skipping parsing '{colname}' — column not found.")
        return
    for i in range(len(df)):
        raw = str(df.loc[i, colname])
        if raw.lower() == 'nan':
            continue
        items = raw.split(', ')
        for item in items:
            if ': ' in item:
                k, v = item.split(': ')
                k = k.strip("{'")
                v = v.strip("'}")
                df.loc[i, k] = 1 if v.strip().lower() == 'yes' else 0

# Use it safely:
parse_dict_column(test_data, "MedicalHistory")
parse_dict_column(test_data, "Symptoms")

parse_dict_column(test_data, "MedicalHistory")
parse_dict_column(test_data, "Symptoms")

test_data.drop(columns=["MedicalHistory", "Symptoms"], inplace=True, errors="ignore")

# === Feature engineering ===
test_data["HealthyLifestyleScore"] = (
    test_data["DietQuality"] + test_data["SleepQuality"] + test_data["WeeklyPhysicalActivity (hr)"]
) / 3
test_data["Disease Symptoms"] = (
    test_data["Tremor"] + test_data["Bradykinesia"] + test_data["SleepDisorders"] + test_data["PosturalInstability"]
) / 4
test_data["ChronicDiseasesScore"] = (test_data["Hypertension"] + test_data["Diabetes"]) / 2

# === Yes/No mapping ===
yes_no_cols = ['Tremor', 'Bradykinesia', 'SleepDisorders', 'PosturalInstability',
               'Hypertension', 'Diabetes', 'Stroke', 'Constipation', 'SpeechProblems',
               'TraumaticBrainInjury', 'Rigidity']

for col in yes_no_cols:
    if col in test_data.columns and test_data[col].dtype == "object":
        test_data[col] = test_data[col].map({'Yes': 1, 'No': 0})

# === Drop training-time columns ===
test_data.drop(columns=[col for col in drop_cols if col in test_data.columns], inplace=True)

# === Apply label encoding using training-time mappings ===
for col, mapping in label_mappings.items():
    if col in test_data.columns:
        test_data[col] = test_data[col].map(mapping)

# === Fill remaining NaNs ===
test_data.fillna(0, inplace=True)

# === Scale numeric features ===
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

# === Separate features and label ===
X_test = test_data.drop("Diagnosis", axis=1)
y_test = test_data["Diagnosis"]

# === Load the model ===
with open("random_forest_model.pkl", "rb") as f:  # ← Replace with your chosen model file
    model = pickle.load(f)

# === Predict and evaluate ===
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
