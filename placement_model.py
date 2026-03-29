# ---------------- IMPORT LIBRARIES ---------------- #

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------- DATASET (NO CSV NEEDED) ---------------- #

data = pd.DataFrame({
    'cgpa': [5,6,7,8,9,6,7,8,9,5,7,8,6,9,5],
    'skills': [2,3,4,5,5,3,4,5,5,2,4,5,3,5,2],
    'internships': [0,1,1,2,2,0,1,2,2,0,1,2,0,2,0],
    'placed': [0,0,1,1,1,0,1,1,1,0,1,1,0,1,0]
})

# ---------------- TRAIN MODEL ---------------- #

X = data[['cgpa','skills','internships']]
y = data['placed']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- USER INPUT ---------------- #

print("\n🎓 Student Placement Predictor\n")

cgpa = float(input("Enter CGPA (0–10): "))
skills = int(input("Enter Skills (1–5): "))
internships = int(input("Enter Internships (0–2): "))

# ---------------- PREDICTION ---------------- #

sample = pd.DataFrame([[cgpa, skills, internships]],
                      columns=['cgpa','skills','internships'])

result = model.predict(sample)

# ---------------- OUTPUT ---------------- #

if result[0] == 1:
    print("\n🎉 Result: Student will be PLACED")
else:
    print("\n❌ Result: Student NOT likely to be placed")