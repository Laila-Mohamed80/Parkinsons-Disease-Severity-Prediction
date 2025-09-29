import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("parkinsons_disease_data_cls.csv")  # Use your updated dataset with 'Diagnosis'

# Handle missing values
mode = data['EducationLevel'].mode()[0]
data['EducationLevel'] = data['EducationLevel'].fillna(mode)

# Convert WeeklyPhysicalActivity from H:MM to float hours
for i in range(len(data)):
    value = str(data.loc[i, 'WeeklyPhysicalActivity (hr)'])
    if ':' in value:
        h, m = map(float, value.split(':'))
        data.loc[i, 'WeeklyPhysicalActivity (hr)'] = h + m / 60
    else:
        data.loc[i, 'WeeklyPhysicalActivity (hr)'] = float(value)

# Drop irrelevant columns
data.drop(columns=['DoctorInCharge', 'PatientID'], inplace=True)

# Parse 'MedicalHistory'
for i in range(len(data)):
    mh_items = str(data.loc[i, 'MedicalHistory']).split(', ')
    for item in mh_items:
        if ': ' in item:
            k, v = item.split(': ')
            k = k.strip("{'")
            v = v.strip("'}")
            data.loc[i, k] = 1 if v.strip().lower() == 'yes' else 0

# Parse 'Symptoms'
for i in range(len(data)):
    s_items = str(data.loc[i, 'Symptoms']).split(', ')
    for item in s_items:
        if ': ' in item:
            k, v = item.split(': ')
            k = k.strip("{'")
            v = v.strip("'}")
            data.loc[i, k] = 1 if v.strip().lower() == 'yes' else 0

data.drop(columns=['MedicalHistory', 'Symptoms'], inplace=True)

# Feature engineering
data['HealthyLifestyleScore'] = (
    data['DietQuality'] + data['SleepQuality'] + data['WeeklyPhysicalActivity (hr)']
) / 3
data['Disease Symptoms'] = (
    data['Tremor'] + data['Bradykinesia'] + data['SleepDisorders'] + data['PosturalInstability']
) / 4
data['ChronicDiseasesScore'] = (data['Hypertension'] + data['Diabetes']) / 2

# Convert all Yes/No values to 1/0 after parsing
yes_no_cols = ['Tremor', 'Bradykinesia', 'SleepDisorders', 'PosturalInstability',
               'Hypertension', 'Diabetes', 'Stroke', 'Constipation', 'SpeechProblems',
               'TraumaticBrainInjury', 'Rigidity']

for col in yes_no_cols:
    if col in data.columns and data[col].dtype == 'object':
        data[col] = data[col].map({'Yes': 1, 'No': 0})

# Drop redundant columns
drop_cols = [
    'Smoking', 'Tremor', 'Rigidity', 'Bradykinesia', 'SpeechProblems',
    'SleepDisorders', 'PosturalInstability', 'Constipation', 'TraumaticBrainInjury',
    'Hypertension', 'Diabetes', 'DietQuality', 'SleepQuality', 'WeeklyPhysicalActivity (hr)',
    'Ethnicity', 'EducationLevel'
]
data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

# Fill any remaining missing values
data.fillna(0, inplace=True)

# Scale numeric features
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.drop('Diagnosis')
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Manual label mappings (to ensure consistency between train and test)
label_mappings = {
    "Gender": {"Male": 1, "Female": 0},
    "FamilyHistoryParkinsons": {"Yes": 1, "No": 0},
    "Stroke": {"Yes": 1, "No": 0}
}

for col, mapping in label_mappings.items():
    if col in data.columns:
        data[col] = data[col].map(mapping)

# Save preprocessing artifacts for test-time
preprocessing_artifacts = {
    "mode_education_level": mode,
    "columns_to_drop": drop_cols,
    "label_mappings": label_mappings,
    "numeric_cols": list(numeric_cols)
}
with open("preprocessing_artifacts.pkl", "wb") as f:
    pickle.dump(preprocessing_artifacts, f)

# Debug prints
print(f"Data shape after preprocessing: {data.shape}")
print(f"Any NaNs left? {data.isnull().any().any()}")

# Train/test split
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill NaNs in train and test separately (just in case)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
# Classifiers and hyperparameters
models_params = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        "C": [0.01, 0.1, 1.0]
    }),
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20]
    }),
    "SVM": (SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    })
}

results = []

# Train and evaluate models
for name, (model, param_grid) in models_params.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    start_train = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - start_train

    best_model = grid.best_estimator_
    start_test = time.time()
    y_pred = best_model.predict(X_test)
    test_time = time.time() - start_test
    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Train Time": train_time,
        "Test Time": test_time
    })

    with open(f"{name.replace(' ', '_').lower()}_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

# Plot results
results_df = pd.DataFrame(results)

plt.figure(figsize=(16, 5))
for i, metric in enumerate(["Accuracy", "Train Time", "Test Time"]):
    plt.subplot(1, 3, i+1)
    sns.barplot(x="Model", y=metric, data=results_df)
    plt.title(metric)
    plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("classification_results.png")
plt.show()
