import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


data = []

for _ in range(500):
    ssc = random.randint(40, 100)
    hsc = random.randint(40, 100)
    degree = random.randint(40, 100)
    coding = random.randint(1, 100)
    communication = random.randint(1, 100)

    # Logical placement rule
        
    if (degree > 60 and coding > 60 and communication > 50):
        placement = "Placed"
    else:
        placement = "Not Placed"

    data.append([ssc, hsc, degree, coding, communication, placement])

df = pd.DataFrame(data, columns=[
    "ssc", "hsc", "degree", "coding", "communication", "placement"
])

df.to_csv("placement_dataset.csv", index=False)

# -------------------------------
# STEP 2: Load Dataset
# -------------------------------

df = pd.read_csv("placement_dataset.csv")
print(df.head())

# Encode target
df["placement"] = df["placement"].astype("category")
placement_mapping = dict(enumerate(df["placement"].cat.categories))
df["placement"] = df["placement"].cat.codes

print("Placement Mapping:", placement_mapping)

# Split features & target
X = df.drop("placement", axis=1)
y = df["placement"]

# -------------------------------
# STEP 3: Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# STEP 4: Train Model (SVM)
# -------------------------------

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 0.01],
    "kernel": ["rbf", "linear"]
}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_

# -------------------------------
# STEP 5: Evaluation
# -------------------------------

y_pred = best_model.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 6: Save Model
# -------------------------------

pickle.dump(best_model, open("placement_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(placement_mapping, open("placement_mapping.pkl", "wb"))

print("Model Saved Successfully ✅")