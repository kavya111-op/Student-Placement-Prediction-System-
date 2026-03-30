# ================= IMPORT LIBRARIES =================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# ================= DATASET =================

data = pd.DataFrame({
'cgpa': [5,6,7,8,9,6,7,8,9,5,7,8,6,9,5],
'skills': [2,3,4,5,5,3,4,5,5,2,4,5,3,5,2],
'internships': [0,1,1,2,2,0,1,2,2,0,1,2,0,2,0],
'placed': [0,0,1,1,1,0,1,1,1,0,1,1,0,1,0]
})

# ================= TRAIN MODEL =================

X = data[['cgpa','skills','internships']]
y = data['placed']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ================= UI WIDGETS =================

title = widgets.HTML("<h2 style='color:blue;'>🎓 Student Placement Predictor</h2>")

cgpa_slider = widgets.FloatSlider(
value=7, min=0, max=10, step=0.1, description='CGPA:',
style={'description_width': 'initial'}
)

skills_slider = widgets.IntSlider(
value=3, min=1, max=5, step=1, description='Skills:',
style={'description_width': 'initial'}
)

internship_slider = widgets.IntSlider(
value=1, min=0, max=2, step=1, description='Internships:',
style={'description_width': 'initial'}
)

button = widgets.Button(description="🔍 Predict", button_style='success')
output = widgets.Output()

# ================= PREDICTION FUNCTION =================

def predict(b):
    output.clear_output()   # clear outside context

    with output:
        try:
            sample = pd.DataFrame([[cgpa_slider.value, skills_slider.value, internship_slider.value]],
                                  columns=['cgpa','skills','internships'])

            result = model.predict(sample)
            prob = model.predict_proba(sample)[0]

            if result[0] == 1:
                display(HTML("<h3 style='color:green;'>🎉 Student will be PLACED</h3>"))
            else:
                display(HTML("<h3 style='color:red;'>❌ Student NOT likely to be placed</h3>"))

            display(HTML(f"<b>Placement Probability:</b> {prob[1]*100:.2f}%"))

        except Exception as e:
            print("Error:", e)

    # ================= GRAPH 1: Probability ================= #
    plt.figure()
    labels = ['Not Placed', 'Placed']
    values = [prob[0]*100, prob[1]*100]

    plt.bar(labels, values)
    plt.xlabel("Outcome")
    plt.ylabel("Probability (%)")
    plt.title("Placement Prediction Probability")
    plt.show()

    # ================= GRAPH 2: Feature Importance ================= #
    plt.figure()
    features = ['CGPA', 'Skills', 'Internships']
    importance = model.feature_importances_

    plt.bar(features, importance)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.show()


# Button click event

button.on_click(predict)

# ================= DISPLAY =================

display(title, cgpa_slider, skills_slider, internship_slider, button, output)
