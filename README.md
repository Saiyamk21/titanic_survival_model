# Titanic Survival Predictor

A machine learning model that predicts passenger survival on the Titanic based on various features like age, gender, ticket class, etc.

## Live Demo
Access the live model: [Titanic Survival Predictor on Hugging Face](https://huggingface.co/spaces/saiyamkkkalls/tittanicspace)

## Project Structure
- `training_model.ipynb`: Jupyter notebook with detailed model development process
- `titanic_model.py`: Python script to train the model
- `app.py`: Gradio web application for model inference
- `check.py`: Utility script for model testing
- `titanic_model.pkl`: Saved RandomForest model
- `requirements.txt`: Required Python dependencies
- `train.csv`: Training dataset
- `gender_submission.csv`: Sample submission file
- `titanic_ieee.pdf`: Documentation/paper related to the project
- `flutter/`: Mobile application implementation

## Features Used in the Model
- Passenger Class (Pclass)
- Age
- Number of Siblings/Spouses Aboard (SibSp)
- Number of Parents/Children Aboard (Parch)
- Fare
- Sex (male/female)
- Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Getting Started

### Prerequisites
- Python 3.6+
- Required packages listed in requirements.txt

### Installation
1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```

### Run the Web Application
```
python app.py
```
This will launch a Gradio web interface where you can input passenger details to predict survival.

### Train the Model
To retrain the model from scratch:
```
python titanic_model.py
```

## Flutter Mobile Application
The project includes a Flutter implementation for mobile devices. Navigate to the `flutter/titanic_survival_predictor_flutter-main/` directory for details.

## Model Details
- Algorithm: Random Forest Classifier
- Features: Passenger class, age, family relations, fare, gender, and port of embarkation
- Training data: Historical Titanic passenger records

## License
This project is open-source and available for educational and research purposes.
