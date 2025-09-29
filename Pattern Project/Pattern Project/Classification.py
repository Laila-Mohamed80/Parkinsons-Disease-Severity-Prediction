import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import pointbiserialr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from scipy.stats import chi2_contingency
import pandas as pd

data= pd.read_csv("C:/Users/Admin/Desktop/New folder (2)/Phase 2/parkinsons_disease_data_cls.csv")


print(data.isnull().sum())
# There is nulls in one colummn which is EducationLevel so we handle it with fill it with mode
# el educationLevel column kan fe nulls fa 8ayarnaha fel mode
mode = data['EducationLevel'].mode()[0]
data['EducationLevel'] = data['EducationLevel'].fillna(mode)
# print("Dublicated values:")
# print(data.duplicated().sum())
# #There is no duplicates

# #Transfer H:MM type into float type
for i in range(len(data)):
    oldValue = data.loc[i, 'WeeklyPhysicalActivity (hr)']
    if ':' in oldValue:
        hour,minute = oldValue.split(':')
        hour=float(hour)*60
        minute = float(minute)
        newValue = hour+minute
        newValue /=60
    else:
        newValue = float(oldValue)
    data.loc[i, 'WeeklyPhysicalActivity (hr)'] = newValue
#shelna DoctorInCharge 3ashan feeha nafs el vals beta3et el (DrXXXConfid) fel rows kolaha fa maloosh lazma
data.drop('DoctorInCharge', axis=1, inplace=True)

# shelna el patientID 3ashan maloosh lazma fel prediction
data.drop('PatientID', axis=1, inplace=True)

NumData = data.drop( columns=['Gender','Smoking','EducationLevel','Ethnicity','Symptoms','MedicalHistory','Diagnosis'])

scaler = MinMaxScaler()
NumData= pd.DataFrame(scaler.fit_transform(NumData), columns=NumData.columns)
with open('scalerClassifi.pk2', 'wb') as f:
    pickle.dump(scaler, f)
NumData['HealthyLifestyleScore'] = ((NumData['DietQuality'] + NumData['SleepQuality'] + pd.to_numeric(NumData['WeeklyPhysicalActivity (hr)'])) / 3)
NumData.drop('DietQuality', axis=1, inplace=True)
NumData.drop('SleepQuality', axis=1, inplace=True)


#print(NumData)
from scipy.stats import pointbiserialr
results = []
for feature in NumData.columns:
    correlation, p_value = pointbiserialr(data['Diagnosis'], NumData[feature])
    results.append({
        'Feature': feature, 
        'Correlation': correlation,
        'p_value': p_value
    })
results_df = pd.DataFrame(results)
top_10_significant = results_df.sort_values(by='p_value').head(10)
selected_columns = top_10_significant['Feature'].tolist()  
#print(NumData[selected_columns])
NumericDF =NumData[selected_columns]
train_means = NumericDF.mean()
with open('trainClass_means.pkl', 'wb') as f:
    pickle.dump(train_means, f)


############################################################################

# han2asem el 'MedicalHistory' 3ala kaza column w ba3den nesheelo
for i in range(len(data)):
    value = data.loc[i, 'MedicalHistory']
    items = value.split(', ')
    for item in items:
        key, val = item.split(': ')
        key=key.strip("{'")
        val=val.strip("'}")
        data.loc[i, key] = val
data.drop(columns=['MedicalHistory'], inplace=True)

# han2asem el 'symptom' 3ala kaza column w ba3den nesheelo
for i in range(len(data)):
    value = data.loc[i, 'Symptoms']
    items = value.split(', ')
    for item in items:
        key, val = item.split(': ')
        key=key.strip("{'")
        val=val.strip("'}")
        data.loc[i, key] = val
data.drop(columns=['Symptoms'], inplace=True)

from scipy.stats import chi2_contingency
import pandas as pd
categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking',
                       'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension',
                       'Diabetes', 'Depression', 'Stroke', 'Tremor', 'Rigidity',
                       'Bradykinesia', 'PosturalInstability', 'SpeechProblems',
                       'SleepDisorders', 'Constipation']
category_Data = data[categorical_columns]

label_encoders = {}
for column in category_Data:
    le = LabelEncoder()
    category_Data[column] = le.fit_transform(category_Data[column])
    label_encoders[column] = le  


with open('label_encodersClassifi.pk2', 'wb') as f:
    pickle.dump(label_encoders, f)

#print(category_Data)
results = []

for feature in category_Data:

    crosstab = pd.crosstab(category_Data[feature], data['Diagnosis'])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    results.append({
        'Feature': feature,
        'Chi2_statistic': chi2,
        'p_value': p
    })

chi2_df = pd.DataFrame(results)

chi2_df_sorted = chi2_df.sort_values(by='p_value').head(10)
selected_columns =chi2_df_sorted['Feature'].tolist()  
category_Data =category_Data[selected_columns]
print(category_Data)
X = pd.concat([category_Data, NumData], axis=1)

y = data['Diagnosis']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_results = []
C_values = [0.1, 1, 10] 
for C_val in C_values:
    svm = SVC(C=C_val)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    svm_results.append({
        'C': C_val,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    })

# 2. **Decision Tree Model** (اختيار معامل max_depth)
dt_results = []
max_depth_values = [3, 5, 10]  # ثلاث قيم مختلفة لـ max_depth
for max_depth_val in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=max_depth_val)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    dt_results.append({
        'max_depth': max_depth_val,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    })

# 3. **Gradient Boosting Model** (اختيار معامل learning_rate)
gb_results = []
learning_rate_values = [0.01, 0.1, 0.3]  # ثلاث قيم مختلفة لـ learning_rate
for lr in learning_rate_values:
    gb = GradientBoostingClassifier(learning_rate=lr)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    gb_results.append({
        'learning_rate': lr,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    })

# عرض النتائج
print("SVM Results:")
for result in svm_results:
    print(f"C={result['C']}, Accuracy={result['accuracy']}")
    print(f"Confusion Matrix:\n{result['confusion_matrix']}")

print("\nDecision Tree Results:")
for result in dt_results:
    print(f"max_depth={result['max_depth']}, Accuracy={result['accuracy']}")
    print(f"Confusion Matrix:\n{result['confusion_matrix']}")

print("\nGradient Boosting Results:")
for result in gb_results:
    print(f"learning_rate={result['learning_rate']}, Accuracy={result['accuracy']}")
    print(f"Confusion Matrix:\n{result['confusion_matrix']}")