from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

# Variables globales para almacenar el estado
dataset = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global dataset
    file = request.files['file']
    if file:
        dataset = pd.read_csv(file)
        return jsonify({
            'columns': dataset.columns.tolist(),
            'preview': dataset.head().to_html()
        })
    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/train_model', methods=['POST'])
def train_model():
    global dataset, model
    data = request.json
    target_column = data['target_column']
    feature_columns = data['feature_columns']
    
    X = dataset[feature_columns]
    y = dataset[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return jsonify({'accuracy': accuracy})

@app.route('/analyze', methods=['GET'])
def analyze():
    global dataset, model
    if model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    feature_importance = model.feature_importances_
    feature_names = model.feature_names_in_
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    
    return jsonify({'feature_importance_plot': img_base64})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    data = request.json
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)