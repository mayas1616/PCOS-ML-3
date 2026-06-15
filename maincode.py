import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


#Load/Inspect Data
try:
    # This loads your file from the Windows folder bc im chuddy asf and lowkenuinly hate uploading folders
    df = pd.read_csv('/mnt/c/Users/abhay/Downloads/archive/pcos_data.csv')
    print("Dataset successfully loaded!")
except FileNotFoundError:
    print("Error: Could not find the file at the specified Windows path.")
    exit()


print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")


# preprocess data and clean it bc it stupid asf
# Drop non-predictive columns if exist lol bc they dookie ngl
columns_to_drop = ['Sl. No', 'Patient File No.', 'Unnamed: 44']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])


# all columns are numeric, convert strings/spaces into NaN values
df = df.apply(pd.to_numeric, errors='coerce')


# Handle missing vals by puting column medians :p
df = df.fillna(df.median())


# Separate features (X) and target label (y)
# Adjust PCOS Y/N if ya nasty lol
target_col = 'PCOS (Y/N)'
if target_col not in df.columns:
    # Fallback to look for variations like 'PCOS_Diagnosis'
    target_col = [col for col in df.columns if 'pcos' in col.lower()][0]


X = df.drop(columns=[target_col])
y = df[target_col].astype(int)


print(f"Processed Feature Matrix Shape: {X.shape}")
print(f"Target Distribution: {np.bincount(y)}\n")




# Train-Test Split data so we can get model performance but data is lowk dookie so we're cooked its okay tho :(((
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




# Train Random Forest Classifier the goated part everything else lame fr
# Using class_weight='balanced'  account for slight target imbalance
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)


# Model eval(it did dookie im pissed but medical mysoginy can suck my booty)
y_pred = rf_model.predict(X_test)


print(" Evaluation Results ")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#  Feature Importance This was so fun actually but it took like an hour to figure out
#  helps see which symptoms or hormones drive prediction most
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)


print("\n Top 5 Most Diagnostic Features ")
print(feature_imp_df.head(5).to_string(index=False))

