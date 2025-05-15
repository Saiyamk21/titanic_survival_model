import gradio as gr
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "titanic_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_survival(pclass, age, sibsp, parch, fare, sex_str, embarked_str):
    if sex_str.lower() == "female":
        sex_encoded = 0
    elif sex_str.lower() == "male":
        sex_encoded = 1
    else:
        return "Error: Invalid sex input. Use 'male' or 'female'."

    embarked_map = {'S': 2, 'C': 0, 'Q': 1}
    if embarked_str.upper() in embarked_map:
        embarked_encoded = embarked_map[embarked_str.upper()]
    else:
        return "Error: Invalid embarked input. Use 'S', 'C', or 'Q'."

    features_df = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_encoded, embarked_encoded]],
                               columns=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"])

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]

    if prediction == 1:
        result_text = "Survived"
    else:
        result_text = "Did Not Survive"

    return f"Prediction: {result_text}\nProbability of Survival: {probability[1]:.2f}\nProbability of Not Surviving: {probability[0]:.2f}"

iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Pclass (Ticket class: 1, 2, or 3)", value=3),
        gr.Number(label="Age (in years)", value=30),
        gr.Number(label="SibSp (Number of siblings/spouses aboard)", value=0),
        gr.Number(label="Parch (Number of parents/children aboard)", value=0),
        gr.Number(label="Fare (Passenger fare)", value=10.0),
        gr.Radio(label="Sex", choices=["male", "female"], value="male"),
        gr.Radio(label="Embarked (Port of Embarkation: C, Q, S)", choices=["S", "C", "Q"], value="S")
    ],
    outputs=gr.Textbox(label="Survival Prediction"),
    title="Titanic Survival Predictor",
    description="Enter passenger details to predict their chance of survival. Based on a RandomForest model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
