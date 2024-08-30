#evaluation.py

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, median_absolute_error, r2_score, mean_absolute_error, explained_variance_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from data.visualizer import Visualizer

# Funci\u00f3n para evaluar modelos de clasificaci\u00f3n
class Evaluation:
    def __init__(self, problem_type, dataset_name):
        self.problem_type = problem_type
        self.visualizer = Visualizer()
        self.visualizer.set_dataset_name(dataset_name)

    def evaluate_classification_models(self, models, X_test, y_test, target):
        evaluation_results = {}

        for model in models:
            model_name = model.__class__.__name__
            self.visualizer.set_model_name(model_name)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5))

            evaluation_results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "cross_validation_score": cv_score
            }

            # Visualizaciones
            self.visualizer.plot_confusion_matrix(y_test, y_pred, np.unique(y_test))
            if y_pred_proba is not None:
                self.visualizer.plot_roc_curve(y_test, y_pred_proba)
            if hasattr(model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(model, X_test.columns if isinstance(X_test, pd.DataFrame) else None)
            
            feature_names = X_test.columns if isinstance(X_test, pd.DataFrame) else [f'Feature_{i}' for i in range(X_test.shape[1])]
            self.visualizer.plot_feature_distributions(X_test, feature_names)

        return evaluation_results

    def evaluate_regression_models(self, models, X_test, y_test, target, features):
        evaluation_results = {}

        for model in models:
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            cv_score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='r2'))

            evaluation_results[model.__class__.__name__] = {
                "r2_score": r2,
                "mean_squared_error": mse,
                "cross_validation_score": cv_score
            }

            # Visualizaciones
            self.visualizer.plot_residuals(y_test, y_pred)
            self.visualizer.plot_feature_importance(model, features)
            self.visualizer.plot_feature_distributions(X_test, features)

        return evaluation_results