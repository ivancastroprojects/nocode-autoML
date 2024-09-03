#evaluation.py

import numpy as np
import pandas as pd

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_score
from data.visualizer import Visualizer

# Funci\u00f3n para evaluar modelos de clasificaci\u00f3n
class Evaluation:
    def __init__(self, problem_type, dataset_name):
        self.problem_type = problem_type
        self.dataset_name = dataset_name
        self.visualizer = Visualizer()
        self.visualizer.set_dataset_name(dataset_name)

    def evaluate(self, model, X_test, y_test, y_pred, selected_features):
        """
        Evalúa el modelo y genera visualizaciones.
        
        Args:
        model: El modelo entrenado
        X_test: Características de prueba
        y_test: Etiquetas reales de prueba
        y_pred: Predicciones del modelo
        selected_features: Lista de características seleccionadas
        """
        print(f"\nEvaluación del modelo {model.__class__.__name__}:")
        
        if self.problem_type == 'classification':
            self.evaluate_classification(y_test, y_pred)
        else:
            self.evaluate_regression(y_test, y_pred)
        
        self.visualizer.set_model_name(model.__class__.__name__)
        self.generate_visualizations(model, X_test, y_test, y_pred, selected_features)

    def evaluate_classification(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

    def evaluate_regression(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")

    def generate_visualizations(self, model, X_test, y_test, y_pred, selected_features):
        if self.problem_type == 'classification':
            self.visualizer.plot_confusion_matrix(y_test, y_pred, np.unique(y_test))
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                self.visualizer.plot_roc_curve(y_test, y_pred_proba)
        else:
            self.visualizer.plot_residuals(y_test, y_pred)
        
        if hasattr(model, 'feature_importances_'):
            self.visualizer.plot_feature_importance(model, selected_features)
        
        self.visualizer.plot_feature_distributions(X_test, selected_features)

    def evaluate_classification_models(self, models, X_test, y_test, target, feature_names):
        for model in models:
            y_pred = model.predict(X_test)
            
            # Intentar obtener probabilidades
            try:
                y_pred_proba = model.predict_proba(X_test)
            except AttributeError:
                # Si el modelo no tiene predict_proba, usar decision_function si está disponible
                try:
                    y_pred_proba = model.decision_function(X_test)
                except AttributeError:
                    # Si tampoco tiene decision_function, usar y_pred
                    y_pred_proba = y_pred
            
            # Si y_pred_proba es unidimensional, convertirlo a bidimensional
            if y_pred_proba.ndim == 1:
                y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))
            
            # ... (resto del código de evaluación)
            
            self.visualizer.plot_roc_curve(y_test, y_pred_proba)
            
            # ... (resto del código)

    # ... (otros métodos)
        # Asegurarse de que X_test sea un DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        evaluation_results = {}

        # Asegurarse de que features contiene los nombres correctos de las columnas
        if isinstance(X_test, pd.DataFrame):
            features = X_test.columns.tolist()
        else:
            features = [f'Feature_{i}' for i in range(X_test.shape[1])]

        for model in models:
            model_name = model.__class__.__name__
            self.visualizer.set_model_name(model_name)
            
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='r2'))

            evaluation_results[model_name] = {
                "r2_score": r2,
                "mean_squared_error": mse,
                "cross_validation_score": cv_score
            }

            # Visualizaciones
            self.visualizer.plot_residuals(y_test, y_pred)
            self.visualizer.plot_feature_importance(model, features)
            self.visualizer.plot_feature_distributions(X_test, features)
            
            # Crear un DataFrame con las características y el objetivo
            X_test_with_target = X_test.copy()
            X_test_with_target[target] = y_test

            # Crear grupos de características
            groups = [[feature] for feature in features]
            
            # Llamar a create_grouped_histograms con los grupos correctos
            self.visualizer.create_grouped_histograms(X_test_with_target, groups, target)

        return evaluation_results
    
    
    def compare_models(self, base_model, optimized_model, X_test, y_test):
        base_pred = base_model.predict(X_test)
        optimized_pred = optimized_model.predict(X_test)

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        print("Métrica    | Modelo Base | Modelo Optimizado | Diferencia")
        print("-----------+-------------+-------------------+-----------")
        
        for metric in metrics:
            if metric == 'accuracy':
                base_value = accuracy_score(y_test, base_pred)
                opt_value = accuracy_score(y_test, optimized_pred)
            elif metric == 'precision':
                base_value = precision_score(y_test, base_pred, average='weighted')
                opt_value = precision_score(y_test, optimized_pred, average='weighted')
            elif metric == 'recall':
                base_value = recall_score(y_test, base_pred, average='weighted')
                opt_value = recall_score(y_test, optimized_pred, average='weighted')
            elif metric == 'f1':
                base_value = f1_score(y_test, base_pred, average='weighted')
                opt_value = f1_score(y_test, optimized_pred, average='weighted')
            
            diff = opt_value - base_value
            print(f"{metric.capitalize():10} | {base_value:.4f}      | {opt_value:.4f}            | {diff:+.4f}")