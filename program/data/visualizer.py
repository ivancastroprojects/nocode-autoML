import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

class Visualizer:
    def __init__(self, base_output_dir='program/almacen/visualizaciones'):
        self.base_output_dir = base_output_dir
        self.output_dir = base_output_dir
        self.model_name = None
        self.dataset_name = None
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Paleta de colores personalizada
        self.color_palette = sns.color_palette("husl", 8)
        sns.set_palette(self.color_palette)

    def set_model_name(self, model_name):
        self.model_name = re.sub(r'[\\/*?:"<>|]', "_", model_name)
        self.output_dir = os.path.join(self.base_output_dir, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    @staticmethod
    def are_names_related(col1, col2):
        """
        Comprueba si dos nombres de columnas están relacionados basándose en palabras comunes.
        """
        words1 = set(col1.lower().split('_'))
        words2 = set(col2.lower().split('_'))
        return len(words1.intersection(words2)) > 0

    @staticmethod
    def group_related_variables(df, columns):
        """
        Agrupa variables relacionadas basándose en la correlación y nombres similares.
        """
        corr_matrix = df[columns].corr()
        groups = []
        used_columns = set()

        for col in columns:
            if col in used_columns:
                continue
            
            group = [col]
            used_columns.add(col)
            
            for other_col in columns:
                if other_col != col and other_col not in used_columns:
                    if abs(corr_matrix.loc[col, other_col]) > 0.7 or Visualizer.are_names_related(col, other_col):
                        group.append(other_col)
                        used_columns.add(other_col)
            
            groups.append(group)
        
        return groups

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names)
        cm_display.plot(cmap='viridis')
        plt.title(f'Matriz de Confusión - {self.model_name}\nDataset: {self.dataset_name}')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color=self.color_palette[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {self.model_name}\nDataset: {self.dataset_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()

    def plot_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Importancia de Características - {self.model_name}\nDataset: {self.dataset_name}")
            plt.bar(range(len(importances)), importances[indices], color=self.color_palette)
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
        else:
            print(f"El modelo {self.model_name} no proporciona importancia de características.")

    def plot_feature_distributions(self, X, feature_names):
        n_features = len(feature_names)
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))
        fig.suptitle(f'Distribución de Características - {self.model_name}\nDataset: {self.dataset_name}', fontsize=16)
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names):
            if isinstance(X, pd.DataFrame):
                data = X[feature]
            else:  # Asumimos que es un array de numpy
                data = X[:, i]
            
            sns.histplot(data, kde=True, ax=axes[i], color=self.color_palette[i % len(self.color_palette)])
            axes[i].set_title(f'Distribución de {feature}')
        
        if n_features % 2 != 0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'))
        plt.close()

    def create_grouped_histograms(self, df, groups):
        for i, group in enumerate(groups):
            n = len(group)
            if n <= 1:  # Solo crear histogramas para grupos con más de una variable
                continue
            
            fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
            fig.suptitle(f'Histogramas Agrupados - Grupo {i+1}\nDataset: {self.dataset_name}', fontsize=16)
            if n == 1:
                axes = [axes]
            
            for j, (ax, col) in enumerate(zip(axes, group)):
                df[col].hist(ax=ax, color=self.color_palette[j % len(self.color_palette)])
                ax.set_title(f'Histograma de {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frecuencia')
            
            plt.tight_layout()
            
            clean_name = re.sub(r'[\\/*?:"<>|]', "_", f"grupo_{i}")
            plt.savefig(os.path.join(self.output_dir, f'histograma_{clean_name}.png'))
            plt.close()

    def create_bar_plots(self, df, categorical_cols):
        for i, col in enumerate(categorical_cols):
            plt.figure(figsize=(12, 6))
            df[col].value_counts().plot(kind='bar', color=self.color_palette)
            plt.title(f'Distribución de {col} - {self.model_name}\nDataset: {self.dataset_name}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            
            clean_col = re.sub(r'[\\/*?:"<>|]', "_", col)
            plt.savefig(os.path.join(self.output_dir, f'barplot_{clean_col}.png'))
            plt.close()

    def create_correlation_matrix(self, df, numeric_cols, dataset_name):
        if len(numeric_cols) > 1:
            plt.figure(figsize=(14, 12))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title(f'Matriz de Correlación - {self.model_name}\nDataset: {self.dataset_name}')

            clean_name = re.sub(r'[\\/*?:"<>|]', "_", dataset_name)
            plt.savefig(os.path.join(self.output_dir, f'correlation_matrix_{clean_name}.png'))
            plt.close()

