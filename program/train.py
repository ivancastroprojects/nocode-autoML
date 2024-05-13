import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import serializer

# Función para entrenar los modelos
# "model_params" en json se vería así: "algorithms": {{"knn", "params": {"k": 3}
# deben pasárnoslos así: params = {'n_neighbors': 5, 'algorithm': 'auto'}
def train_models(X_train, y_train, model_params):
    trained_models = {}
    for model_name, params in model_params.items():
        if model_name in serializer.model_classes:
            model_class = serializer.model_classes[model_name]
            model = model_class.set_params(**params)  # Instantiate model with provided hyperparameters
            # model = serializer.deserialize_model(model_name)
            model.fit(X_train, y_train)  # Train the model
            trained_models.add(model)
            serializer.to_json(model, model_name)
        else:
            print(f"Model '{model_name}' not found. Skipping...")
    return trained_models


# Función para evaluar los modelos
def evaluate_models(trained_models, X_test, y_test):
    evaluation_results = {}
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        
        evaluation_results[model_name] = {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": cm}
    return evaluation_results