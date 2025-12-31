import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "age": [45, 50, 60, 37, 48, 52, 41, 63, 55, 46, 38, 59, 42, 49, 57, 36, 51, 44, 62, 53],
    "sex": [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    "cholesterol": [210, 250, 190, 180, 220, 230, 200, 240, 195, 205, 185, 225, 215, 235, 245, 175, 210, 195, 240, 220],
    "blood_pressure": [130, 140, 120, 110, 135, 145, 125, 150, 118, 132, 112, 138, 128, 142, 148, 115, 136, 124, 149, 134],
    "exercise_induced_angina": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    "target": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
})

X = data[["age","sex","cholesterol","blood_pressure","exercise_induced_angina"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_patient = pd.DataFrame({
    "age": [47],
    "sex": [1],
    "cholesterol": [225],
    "blood_pressure": [133],
    "exercise_induced_angina": [0]
})
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)[0][1]  
print("Prediction for new patient:", prediction[0])
print("Probability of heart disease:", probability)

plt.figure(figsize=(10,6))
for target_class in [0,1]:
    subset = data[data["target"]==target_class]
    plt.scatter(subset["age"], subset["cholesterol"], label=f"Target={target_class}", s=100, alpha=0.7)

plt.scatter(new_patient["age"], new_patient["cholesterol"], color='red', label='New Patient', edgecolor='k', s=200)

plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Heart Disease Prediction Data")
plt.legend()
plt.grid(True)
plt.show()
