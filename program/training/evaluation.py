#evaluation.py

import numpy as np
import pandas as pd

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_score
from program.data.visualizer import Visualizer
from program.training.scikitdb.serializer import model_classes
from program.utils.logger import logger

# Funci\u00f3n para evaluar modelos de clasificaci\u00f3n
class Evaluation:
    def __init__(self, problem_type, short_dataset_name, model_name=None):
        self.problem_type = problem_type
        self.visualizer = Visualizer()
        self.visualizer.set_dataset_name(short_dataset_name)
        if model_name:
            self.visualizer.set_model_name(model_name)

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
            self.evaluate_classification_metrics(y_test, y_pred)
        else:
            self.evaluate_regression_metrics(y_test, y_pred)
        
        self.visualizer.set_model_name(model.__class__.__name__)

    def evaluate_classification_metrics(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate_regression_metrics(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        return {
            "r2_score": r2,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae
        }

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
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names if feature_names else [f'feature_{i}' for i in range(X_test.shape[1])])

        evaluation_results = {}
        features = X_test.columns.tolist()

        for model in models:
            model_name = model.__class__.__name__
            self.visualizer.set_model_name(model_name)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            roc_auc = np.nan
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except ValueError as e:
                        print(f"Could not calculate ROC AUC for {model_name}: {e}")

            cv_scoring_metric = 'accuracy'
            cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring=cv_scoring_metric))

            evaluation_results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "cross_validation_score": cv_score
            }

            self.visualizer.plot_confusion_matrix(y_test, y_pred, np.unique(y_test))
            if hasattr(model, 'predict_proba'):
                self.visualizer.plot_roc_curve(y_test, y_pred_proba)
            
            if hasattr(model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(model, features)
            self.visualizer.plot_feature_distributions(X_test, features)

        return evaluation_results

    def evaluate_regression_models(self, models, X_test, y_test, target, feature_names):
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names if feature_names else [f'feature_{i}' for i in range(X_test.shape[1])])
        
        evaluation_results = {}
        features = X_test.columns.tolist()

        for model in models:
            model_name = model.__class__.__name__
            self.visualizer.set_model_name(model_name)
            
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            cv_scoring_metric = 'r2'
            cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring=cv_scoring_metric))

            evaluation_results[model_name] = {
                "r2_score": r2,
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "cross_validation_score": cv_score
            }

            self.visualizer.plot_residuals(y_test, y_pred)
            if hasattr(model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(model, features)
            self.visualizer.plot_feature_distributions(X_test, features)

        return evaluation_results
    
    def compare_models(self, base_model, optimized_model, X_test, y_test):
        base_pred = base_model.predict(X_test)
        optimized_pred = optimized_model.predict(X_test)

        print("\nComparación de modelos:")
        print("Métrica    | Modelo Base | Modelo Optimizado | Diferencia")
        print("-----------+-------------+-------------------+-----------")
        
        problem_type_lower = self.problem_type.lower()

        if 'clasificación' in problem_type_lower or 'clasificacion' in problem_type_lower:
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
            for metric_name in metrics_to_compare:
                if metric_name == 'accuracy':
                    ase_value = accuracy_score(y_test, base_pred)
                    opt_value = accuracy_score(y_test, optimized_pred)
                elif metric_name == 'precision':
                    base_value = precision_score(y_test, base_pred, average='weighted', zero_division=0)
                    opt_value = precision_score(y_test, optimized_pred, average='weighted', zero_division=0)
                elif metric_name == 'recall':
                    base_value = recall_score(y_test, base_pred, average='weighted', zero_division=0)
                    opt_value = recall_score(y_test, optimized_pred, average='weighted', zero_division=0)
                elif metric_name == 'f1':
                    base_value = f1_score(y_test, base_pred, average='weighted', zero_division=0)
                    opt_value = f1_score(y_test, optimized_pred, average='weighted', zero_division=0)
                
                diff = opt_value - base_value
                print(f"{metric_name.capitalize():10} | {base_value:.4f}      | {opt_value:.4f}            | {diff:+.4f}")
        
        elif 'regresión' in problem_type_lower or 'regresion' in problem_type_lower:
            metrics_to_compare = ['r2_score', 'mean_squared_error', 'mean_absolute_error'] # RMSE is derived from MSE
            for metric_name in metrics_to_compare:
                if metric_name == 'r2_score':
                    base_value = r2_score(y_test, base_pred)
                    opt_value = r2_score(y_test, optimized_pred)
                    diff_format = "{diff:+.4f}" # Higher R2 is better
                elif metric_name == 'mean_squared_error':
                    base_value = mean_squared_error(y_test, base_pred)
                    opt_value = mean_squared_error(y_test, optimized_pred)
                    diff_format = "{diff:+.4f}" # Lower MSE is better, so positive diff is worse
                elif metric_name == 'mean_absolute_error':
                    base_value = mean_absolute_error(y_test, base_pred)
                    opt_value = mean_absolute_error(y_test, optimized_pred)
                    diff_format = "{diff:+.4f}" # Lower MAE is better
            
            diff = opt_value - base_value
            # Adjust interpretation of diff for error metrics (lower is better)
            if metric_name in ['mean_squared_error', 'mean_absolute_error']:
                # For error metrics, a negative diff means the optimized model is better (error decreased)
                # A positive diff means optimized model is worse (error increased)
                print(f"{metric_name.replace('_', ' ').capitalize():10} | {base_value:.4f}      | {opt_value:.4f}            | {diff_format.format(diff=diff)}") 
            else: # For R2 score (higher is better)
                print(f"{metric_name.replace('_', ' ').capitalize():10} | {base_value:.4f}      | {opt_value:.4f}            | {diff_format.format(diff=diff)}")
        else:
            print("Tipo de problema no reconocido para la comparación de modelos.")

    def _get_model_instance(self, model_name, params=None):
        """
        Obtiene una instancia de un modelo a partir de su nombre y parámetros,
        añadiendo verbosidad para el seguimiento del usuario.
        """
        if params is None:
            params = {}
        
        instance_params = params.copy()
        model_class = model_classes.get(model_name)
        
        if not model_class:
            logger.error(f"Modelo '{model_name}' no encontrado en el diccionario de clases.")
            return None

        # Añadir verbosidad para dar feedback en modelos que lo soporten
        import inspect
        try:
            sig = inspect.signature(model_class.__init__)
            if 'verbose' in sig.parameters:
                instance_params.setdefault('verbose', 1)
        except Exception:
            # Si la inspección falla por alguna razón, no es crítico.
            pass

        try:
            return model_class(**instance_params)
        except Exception as e:
            logger.error(f"Error al instanciar {model_name} con {instance_params}: {e}")
            logger.warning(f"Intentando instanciar {model_name} con parámetros por defecto.")
            return model_class()

    def train_and_evaluate_model(self, model_name, X_train, y_train, X_test, y_test, features, params=None):
        """
        Función principal que entrena y evalúa un modelo.
        """
        model_instance = self._get_model_instance(model_name, params)
        
        # Entrenar el modelo
        model_instance.fit(X_train, y_train)
        
        # Realizar predicciones
        y_train_pred = model_instance.predict(X_train)
        y_test_pred = model_instance.predict(X_test)
        
        # Evaluar
        train_score, test_score, metrics = self.evaluate_model(
            y_train, y_train_pred, y_test, y_test_pred
        )
        
        # Generar y guardar visualizaciones, pasando el tipo de problema
        self.visualizer.generate_visualizations(model_instance, X_test, y_test, y_test_pred, self.problem_type)
        
        return model_instance, train_score, test_score, metrics

    def evaluate_model(self, y_train, y_train_pred, y_test, y_test_pred):
        """
        Calcula las métricas de evaluación para un modelo.
        """
        problem_type_lower = self.problem_type.lower()
        
        if 'clasificación' in problem_type_lower or 'clasificacion' in problem_type_lower:
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)
            metrics = self.evaluate_classification_metrics(y_test, y_test_pred)
        elif 'regresión' in problem_type_lower or 'regresion' in problem_type_lower:
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            metrics = self.evaluate_regression_metrics(y_test, y_test_pred)
            metrics['R-squared'] = test_score
        else:
            logger.error(f"Tipo de problema desconocido: {self.problem_type}")
            return None, None, None # Devuelve None si el tipo de problema no es reconocido
            
        return train_score, test_score, metrics