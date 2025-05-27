#visualizer.py
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.style as mplstyle
matplotlib.use('Agg') # Configurar el backend ANTES de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

from itertools import cycle
from scipy import stats
from training.scikitdb.serializer import clean_filename, ensure_directory_exists, get_safe_path
from io import StringIO
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, base_output_dir='program/almacen/visualizaciones'):
        self.base_output_dir = base_output_dir
        self.output_dir = base_output_dir
        self.model_name = None
        self.dataset_name = None
        ensure_directory_exists(self.base_output_dir)
        mplstyle.use('fast') # Apply fast style for performance
        
        # Paleta de colores personalizada
        self.color_palette = sns.color_palette("husl", 8)
        sns.set_palette(self.color_palette)

    def _sanitize_text(self, text):
        """Replaces $ with \$ to avoid mathtext parsing issues."""
        if isinstance(text, str):
            return text.replace('$', '\\$')
        return text # Return as is if not a string (e.g. numeric)

    def set_model_name(self, model_name):
        self.model_name = clean_filename(model_name)
        self.output_dir = get_safe_path(self.base_output_dir, self.model_name)
        ensure_directory_exists(self.output_dir)

    def set_dataset_name(self, dataset_name):
        self.dataset_name = clean_filename(dataset_name)
        self._ensure_plot_directory_exists()

    def _ensure_plot_directory_exists(self):
        if self.dataset_name:
            ensure_directory_exists(get_safe_path('program/almacen/plots', self.dataset_name))

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
    
    def generate_visualizations(self, model, X_test, y_test, y_pred):
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance(model, X_test.columns)
        self.plot_roc_curve(y_test, model.predict_proba(X_test))
        self.plot_learning_curve(model, X_test, y_test)

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        if class_names is None:
            class_names = np.unique(y_true)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(self._sanitize_text(f'Matriz de Confusión - {self.model_name}'))
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba):
        plt.figure(figsize=(10, 8))
        
        # Convertir y_true a formato binario si no lo está
        classes = np.unique(y_true)
        n_classes = len(classes)
        y_bin = label_binarize(y_true, classes=classes)
        
        if n_classes == 2:
            if y_pred_proba.ndim == 1:
                # Si y_pred_proba es unidimensional, usarlo directamente
                fpr, tpr, _ = roc_curve(y_bin, y_pred_proba)
            else:
                # Si y_pred_proba es bidimensional, usar la segunda columna
                fpr, tpr, _ = roc_curve(y_bin, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Caso multiclase
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(self._sanitize_text(f'Curva ROC - {self.model_name}'))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()
        
    def plot_learning_curve(self, model, X, y):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5))
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 8))
        plt.title(self._sanitize_text(f'Curva de Aprendizaje - {self.model_name}'))
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Puntuación")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Puntuación de entrenamiento")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Puntuación de validación cruzada")
        plt.legend(loc="best")
        plt.savefig(f'{self.output_dir}/learning_curve.png')
        plt.close()

    def plot_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title(self._sanitize_text(f'Importancia de Características - {self.model_name}'))
            plt.bar(range(len(importances)), importances[indices])
            # Sanitize feature names for x-ticks
            sanitized_feature_names = [self._sanitize_text(feature_names[i]) for i in indices]
            plt.xticks(range(len(importances)), sanitized_feature_names, rotation=90)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/feature_importance.png')
            plt.close()
    
    def plot_feature_distributions(self, X, feature_names):
        n_features = len(feature_names)
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))
        fig.suptitle(self._sanitize_text(f'Distribución de Características - {self.model_name}\nDataset: {self.dataset_name}'), fontsize=16)
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names):
            if isinstance(X, pd.DataFrame):
                data = X[feature]
            else:  # Asumimos que es un array de numpy
                data = X[:, i]
            
            sns.histplot(data, kde=True, ax=axes[i], color=self.color_palette[i % len(self.color_palette)])
            axes[i].set_title(self._sanitize_text(f'Distribución de {feature}'))
        
        # Eliminar subplots vacíos
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar el diseño respetando el título
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'))
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        plt.figure(figsize=(10, 8))

        # Binarizar y_true si es necesario (similar a ROC)
        classes = np.unique(y_true)
        n_classes = len(classes)
        y_bin = label_binarize(y_true, classes=classes)

        if n_classes == 2:
            # Para el caso binario, y_pred_proba puede ser (n_samples,) o (n_samples, 2)
            # Usaremos la probabilidad de la clase positiva (generalmente la segunda columna o la única si es 1D)
            if y_pred_proba.ndim == 1:
                precision, recall, _ = precision_recall_curve(y_bin, y_pred_proba)
                ap = average_precision_score(y_bin, y_pred_proba)
            else:
                precision, recall, _ = precision_recall_curve(y_bin, y_pred_proba[:, 1])
                ap = average_precision_score(y_bin, y_pred_proba[:, 1])
            
            plt.plot(recall, precision, color='blue', lw=2, 
                     label=f'Curva Precisión-Recall (AP = {ap:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precisión')
            plt.title(self._sanitize_text(f'Curva Precisión-Recall - {self.model_name}'))
        else:
            # Caso multiclase: graficar una curva por clase (micro-promedio o una por una)
            # Aquí un ejemplo de una por una, similar a ROC multiclase
            precision = dict()
            recall = dict()
            average_precision = dict()
            colors = cycle(self.color_palette) # Usar la paleta de la clase

            for i, color in zip(range(n_classes), colors):
                precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
                average_precision[i] = average_precision_score(y_bin[:, i], y_pred_proba[:, i])
                plt.plot(recall[i], precision[i], color=color, lw=2,
                         label=f'Clase {classes[i]} (AP = {average_precision[i]:.2f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precisión')
            plt.title(self._sanitize_text(f'Curva Precisión-Recall Multiclase - {self.model_name}'))

        plt.legend(loc="best")
        plt.grid(True)
        # Guardar el gráfico
        plot_filename = os.path.join(self.output_dir, f'precision_recall_curve.png')
        ensure_directory_exists(os.path.dirname(plot_filename)) # Asegurar que el directorio existe
        plt.savefig(plot_filename)
        plt.close()

    def plot_residuals(self, y_true, y_pred):
        """
        Genera un gráfico de residuos para evaluar el rendimiento del modelo de regresión.

        Args:
        y_true (array-like): Los valores reales de la variable objetivo.
        y_pred (array-like): Los valores predichos por el modelo.
        """
        if self.dataset_name is None:
            print("Advertencia: Nombre del dataset no establecido. Usando 'unknown_dataset'.")
            self.dataset_name = 'unknown_dataset'
        
        if self.model_name is None:
            print("Advertencia: Nombre del modelo no establecido. Usando 'unknown_model'.")
            self.model_name = 'unknown_model'

        self._ensure_plot_directory_exists()

        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        
        # Gráfico de dispersión de residuos
        plt.subplot(2, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title(self._sanitize_text(f'Residuos vs Valores predichos - {self.model_name}'))
        plt.axhline(y=0, color='r', linestyle='--')
        
        # Histograma de residuos
        plt.subplot(2, 2, 2)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuos')
        plt.title(self._sanitize_text('Distribución de residuos'))
        
        # Q-Q plot
        plt.subplot(2, 2, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(self._sanitize_text('Q-Q plot de residuos'))
        
        # Residuos vs orden
        plt.subplot(2, 2, 4)
        plt.plot(residuals)
        plt.xlabel('Orden de observaciones')
        plt.ylabel('Residuos')
        plt.title(self._sanitize_text('Residuos vs Orden'))
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.tight_layout()
        
        plot_path = get_safe_path('program/almacen/plots', self.dataset_name, f'{self.model_name}_residuals.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Gráfico de residuos guardado en: {plot_path}")

    def create_grouped_histograms(self, df, groups, target_column):
        if not groups:
            print("Advertencia: No hay grupos de características para crear histogramas.")
            return

        # Si groups es una lista de enteros, convertirla en una lista de listas
        if isinstance(groups[0], int):
            groups = [groups]

        for i, group in enumerate(groups):
            # Asegurarse de que todas las características en el grupo existen en el DataFrame
            valid_features = [f for f in group if f in df.columns]
            
            if not valid_features:
                print(f"Advertencia: Ninguna de las características en el grupo {i+1} existe en el DataFrame.")
                continue

            n_features = len(valid_features)
            n_rows = (n_features + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))
            fig.suptitle(self._sanitize_text(f'Distribución de Características - Grupo {i+1}\n{self.model_name}\nDataset: {self.dataset_name}'), fontsize=16)
            axes = axes.flatten()

            for j, feature in enumerate(valid_features):
                sns.histplot(df[feature], kde=True, ax=axes[j], color=self.color_palette[j % len(self.color_palette)])
                axes[j].set_title(self._sanitize_text(f'Distribución de {feature}'))
                
                if target_column in df.columns:
                    ax2 = axes[j].twinx()
                    sns.scatterplot(x=feature, y=target_column, data=df, ax=ax2, color='red', alpha=0.5)
                    ax2.set_ylabel(target_column, color='red')

            # Eliminar subplots vacíos
            for j in range(n_features, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(self.output_dir, f'grouped_histograms_{i+1}.png'))
            plt.close()

    def create_bar_plots(self, df, categorical_cols):
        if not categorical_cols.any(): # Check if the pandas Index/Series is empty
            print("No hay columnas categóricas para crear gráficos de barras.")
            return

        num_plots = len(categorical_cols)
        n_cols_fig = 3  # Número de columnas de subplots por figura
        n_rows_fig = (num_plots + n_cols_fig - 1) // n_cols_fig

        fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(5 * n_cols_fig, 4 * n_rows_fig))
        axes = axes.flatten() # Convertir a un array 1D para fácil iteración

        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            # Contar frecuencias y limitar el número de barras si hay demasiadas categorías
            counts = df[col].value_counts()
            if len(counts) > 20: # Limitar a las 20 categorías más frecuentes
                counts = counts.nlargest(20)
                plot_title = f'{self._sanitize_text(col)} (Top 20)'
            else:
                plot_title = self._sanitize_text(col)
            
            # Sanitize category names (index of counts Series)
            sanitized_index = [self._sanitize_text(str(idx)) for idx in counts.index]

            # sns.barplot(x=sanitized_index, y=counts.values, ax=ax, palette="viridis") # Old call
            # Addressing FutureWarning: Assign x to hue and set legend=False.
            # Since we are plotting value_counts, x is the unique categories (sanitized_index) and y is their counts.
            # For a simple bar plot of counts without a separate hue dimension, 
            # simply providing x, y, and palette should be fine. The warning might be overly broad
            # or apply more directly when hue is explicitly used for a different variable.
            # However, to be safe and potentially silence it if it's just about palette + no hue:
            sns.barplot(x=sanitized_index, y=counts.values, ax=ax, palette="viridis", hue=sanitized_index, legend=False)
            ax.set_title(plot_title)
            ax.set_ylabel('Frecuencia')
            # ax.tick_params(axis='x', rotation=45, ha='right') # Incorrect: ha is not for tick_params
            # Set rotation and alignment on the tick labels themselves
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Ocultar ejes no utilizados
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(self._sanitize_text(f'Gráficos de Barras para Columnas Categóricas\nDataset: {self.dataset_name}'), fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = os.path.join(self.output_dir, f'bar_plots_categorical.png')
        ensure_directory_exists(os.path.dirname(plot_filename))
        plt.savefig(plot_filename)
        plt.close()

    def create_correlation_matrix(self, df, numeric_cols, title):
        if not numeric_cols.any():
            print("No hay columnas numéricas para la matriz de correlación.")
            return
        
        logger.info(f"[Visualizer.create_correlation_matrix] Received numeric_cols: {numeric_cols.tolist()}")
        logger.info(f"[Visualizer.create_correlation_matrix] DataFrame info before filtering numeric_cols:")
        # Log df.info() to a string buffer to avoid large console output if df is huge
        buffer = StringIO()
        df.info(buf=buffer)
        logger.info(buffer.getvalue())
            
        plt.figure(figsize=(15, 12))
        # Asegurarse que solo se usen columnas numéricas que existan en el df
        valid_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if not valid_numeric_cols:
            print("No hay columnas numéricas válidas en el DataFrame para la matriz de correlación.")
            logger.warning(f"[Visualizer.create_correlation_matrix] No valid numeric columns found from input. Original numeric_cols: {numeric_cols.tolist()}. Df columns: {df.columns.tolist()[:20]}...")
            return

        logger.info(f"[Visualizer.create_correlation_matrix] Valid numeric columns for heatmap: {valid_numeric_cols}")
        correlation_matrix = df[valid_numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title(self._sanitize_text(title), fontsize=18)
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
        plt.close()

