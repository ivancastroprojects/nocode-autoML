#datasetprocessing.py
import pandas as pd
import numpy as np
import time
import scipy
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.decomposition import PCA
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.feature_extraction.text import TfidfVectorizer

from program.utils.logger import logger
from program.data.visualizer import Visualizer
from program.training.scikitdb.serializer import get_safe_path
    

def determine_problem_type(y, feature_names=None):
    """
    Determina si el problema es de clasificación o regresión basado en la variable objetivo.
    
    Args:
    y (array-like): La variable objetivo.
    feature_names (list, optional): Nombres de las características.
    
    Returns:
    str: 'Clasificación Binaria', 'Clasificación Multiclase', o 'Regresión'
    """
    unique_values = np.unique(y)
    n_unique = len(unique_values)

    # Verificar si y es numérico
    is_numeric = np.issubdtype(y.dtype, np.number)

    if n_unique == 2:
        problem_type = 'Clasificación Binaria'
    elif not is_numeric or (n_unique > 2 and n_unique <= 10):
        problem_type = 'Clasificación Multiclase'
    elif is_numeric and (n_unique > 10 or np.issubdtype(y.dtype, np.floating)):
        problem_type = 'Regresión'
    else:
        # Si no podemos determinar claramente, asumimos clasificación multiclase
        problem_type = 'Clasificación Multiclase'
        logger.warning("No se pudo determinar claramente el tipo de problema. Se asume Clasificación Multiclase.")

    logger.info(f"Tipo de problema determinado: {problem_type}")
    return problem_type

def determine_problem_type_from_dataset(dataset, target_column='target'):
    """
    Determina el tipo de problema basado en el dataset completo.

    Args:
    dataset (Dataset o DataFrame): Dataset completo.
    target_column (str): Nombre de la columna objetivo.

    Returns:
    str: Tipo de problema ('Clasificación Binaria', 'Clasificación Multiclase', o 'Regresión')
    """
    if hasattr(dataset, 'get_dataframe'):
        df = dataset.get_dataframe()
    elif isinstance(dataset, pd.DataFrame):
        df = dataset
    else:
        raise ValueError("El dataset debe ser un objeto Dataset o un DataFrame de pandas.")

    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está presente en el dataset.")

    y = df[target_column]
    feature_names = df.columns.drop(target_column).tolist()
    return determine_problem_type(y, feature_names)


def EDA_initial_info(self):
    """
    Imprime información inicial sobre el dataset.
    """
    print("Dataset antes de procesar:")
    print(self.df.head())
    print(self.df.describe())
    missing_data = self.df.isnull().sum()
    print(f"Valores faltantes por columna antes de procesar:")
    print(missing_data[missing_data > 0])

def EDA_processed_info(self):
    """
    Imprime información una vez procesado el dataset.
    Generamos visualizaciones personalizadas y agrupadas automáticamente según categorías similares
    """
    logger.info(f"[EDA_processed_info] Entry: self ID = {id(self)}, self.df.shape = {self.df.shape}") # Log de entrada
    
    print("Dataset despues de procesar:")
    print(self.df.head())
    
    # Analisis de valores faltantes después de procesar:
    missing_data = self.df.isnull().sum()
    print(f"Valores faltantes por columna después de procesar:")
    print(missing_data[missing_data > 0])
    
    # Definir columnas numéricas y categóricas una vez, basadas en self.df
    numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
    
    # Intento de identificar categóricas "originales" o significativas para bar plots
    MAX_CATEGORICAL_BAR_PLOTS = 15 # Hard limit for how many bar plots to generate in total
    
    candidate_categorical_cols = []
    for col_name in self.df.columns:
        # Prioritize columns that are object/category and NOT clearly from transformers/passthrough
        if self.df[col_name].dtype in ['object', 'category'] and not col_name.startswith('num__') and not col_name.startswith('remainder__'):
            if self.df[col_name].nunique() < 50: # Only if not excessively high unique values
                candidate_categorical_cols.append(col_name)
        # For remainder columns, be very selective (potential for passthrough text)
        elif col_name.startswith('remainder__'):
            if self.df[col_name].nunique() < 10: # Very few unique values might be okay
                candidate_categorical_cols.append(col_name)
    
    # Sort by nunique to prioritize less cardinal columns if we have to truncate
    candidate_categorical_cols.sort(key=lambda col: self.df[col].nunique())
    
    # Apply hard limit
    selected_categorical_for_plot = pd.Index(candidate_categorical_cols[:MAX_CATEGORICAL_BAR_PLOTS])

    # Protección contra demasiadas columnas para visualizaciones intensivas
    MAX_COLS_FOR_FULL_EDA_PLOTS = 100 # Umbral ajustable
    if self.df.shape[1] > MAX_COLS_FOR_FULL_EDA_PLOTS:
        logger.warning(f"El dataset procesado tiene {self.df.shape[1]} columnas, lo cual excede el umbral de {MAX_COLS_FOR_FULL_EDA_PLOTS} para algunas visualizaciones EDA intensivas (ej. histogramas agrupados, matriz de correlación completa). Estas se omitirán o simplificarán.")
        # Podríamos optar por mostrar solo un subconjunto o un tipo de gráfico diferente aquí.
        # Por ahora, simplemente omitiremos las más pesadas.
        
        visualizer = Visualizer()
        visualizer.set_model_name("EDA_processed_limited") 
        visualizer.set_dataset_name(self.dataset_name)

        if not selected_categorical_for_plot.empty:
            logger.info(f"Generando gráficos de barras para (hasta) {len(selected_categorical_for_plot)} columnas categóricas (modo limitado)...")
            visualizer.create_bar_plots(self.df, selected_categorical_for_plot) # Already limited by MAX_CATEGORICAL_BAR_PLOTS
        else:
            print("No hay columnas categóricas para crear gráficos de barras.")
        
        logger.info("EDA intensiva omitida debido al alto número de columnas.")
        return # Salir temprano de la función para evitar colgar

    # Si el número de columnas es manejable, proceder con la EDA completa:
    # numeric_cols y categorical_cols ya están definidas arriba.
    
    # Crear instancia de Visualizer
    visualizer = Visualizer()
    visualizer.set_model_name("EDA_processed")  # Establecemos un nombre para la carpeta de salida
    visualizer.set_dataset_name(self.dataset_name)  # Asumiendo que tienes un atributo dataset_name en tu clase
    
    # Crear grupos de variables numéricas relacionadas
    groups = Visualizer.group_related_variables(self.df, numeric_cols)
    
    # Identificar la columna objetivo
    target_column = 'target' if 'target' in self.df.columns else None
    
    # Crear histogramas agrupados
    visualizer.create_grouped_histograms(self.df, groups, target_column)
    
    # Gráficos de barras para variables categóricas
    if not selected_categorical_for_plot.empty:
        logger.info(f"Generando gráficos de barras para {len(selected_categorical_for_plot)} columnas categóricas seleccionadas (máx. 20 categorías por plot)...")
        visualizer.create_bar_plots(self.df, selected_categorical_for_plot)
    else:
        print("No hay columnas categóricas para crear gráficos de barras.")
    
    # Matriz de correlación
    logger.info(f"[EDA_processed_info] Numeric columns for correlation matrix: {numeric_cols.tolist()}")
    visualizer.create_correlation_matrix(self.df, numeric_cols, "Dataset procesado")

def basic_dfpreprocess(df, target_column=None, categorical_features=None, numeric_features=None, 
                        datetime_features=None, text_features=None, outlier_columns=None,
                        imputation_strategy='knn', scaling_strategy='robust', 
                        encoding_strategy='onehot', handle_outliers_strategy='clip'):
    """
    Realiza un preprocesamiento avanzado del dataset.
    
    Args:
    df (pd.DataFrame): El DataFrame original a preprocesar.
    target_column (str): Nombre de la columna objetivo.
    categorical_features (list): Lista de columnas categóricas.
    numeric_features (list): Lista de columnas numéricas.
    datetime_features (list): Lista de columnas de fecha/hora.
    text_features (list): Lista de columnas de texto.
    outlier_columns (list): Lista de columnas para detectar outliers.
    imputation_strategy (str): Estrategia de imputación ('simple', 'knn', 'iterative').
    scaling_strategy (str): Estrategia de escalado ('standard', 'robust', 'minmax').
    encoding_strategy (str): Estrategia de codificación para categóricas ('onehot', 'ordinal').
    handle_outliers_strategy (str): Estrategia para manejar outliers ('clip', 'remove', 'impute').
    
    Returns:
    tuple: (DataFrame preprocesado, objeto preprocessor)
    """
    print("\nIniciando procesado básico del dataset:")
    
    # Verificar si los datos ya están normalizados
    # --- INICIO MODIFICACIÓN TEMPORAL ---
    # if is_data_normalized(df):
    #     print("Los datos parecen estar ya normalizados. Se omitirá la normalización adicional.")
    #     return df, None
    # --- FIN MODIFICACIÓN TEMPORAL ---
    
    # Crear una copia del DataFrame para no modificar el original
    df_copy = df.copy()
    
    # Convertir nombres de columnas problemáticos a strings
    df_copy.columns = [str(col) for col in df_copy.columns]
    
    # Separar la variable objetivo
    if target_column and target_column in df_copy.columns: # Asegurarse que target_column existe
        y = df_copy[target_column]
        X = df_copy.drop(columns=[target_column])
    else:
        X = df_copy.copy() # Usar una copia si no hay target_column o no existe
        y = None

    # Eliminar columnas de texto libre/alta cardinalidad no deseadas antes de la identificación de tipos
    cols_to_drop_explicitly = ['Name', 'Ticket', 'Cabin', 'PassengerId'] 
    # Verificar si las columnas existen en X antes de intentar eliminarlas
    existing_cols_to_drop = [col for col in cols_to_drop_explicitly if col in X.columns]
    if existing_cols_to_drop:
        X = X.drop(columns=existing_cols_to_drop)
        logger.info(f"Columnas eliminadas explícitamente: {existing_cols_to_drop}")
    logger.info(f"Columnas en X después de eliminación explícita: {X.columns.tolist()}")

    # --- LÓGICA DE DETECCIÓN DE TIPOS REFORZADA ---
    # Identificar tipos de columnas desde el DataFrame X actual
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_candidates = X.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Filtrar categóricas con alta cardinalidad
    high_cardinality_threshold = 50
    categorical_features = []
    high_cardinality_features = []
    
    for col in categorical_candidates:
        try:
            if X[col].nunique() > high_cardinality_threshold:
                high_cardinality_features.append(col)
            else:
                categorical_features.append(col)
        except Exception:
            # Si nunique() falla, tratarla como de alta cardinalidad
            high_cardinality_features.append(col)
            
    if high_cardinality_features:
        logger.warning(f"Columnas con alta cardinalidad (> {high_cardinality_threshold} valores únicos) serán descartadas: {high_cardinality_features}")
        X.drop(columns=high_cardinality_features, inplace=True)

    # Las columnas de texto se manejarán por separado si es necesario. Por ahora, nos aseguramos de que no se procesen aquí.
    text_features = []
    # --- FIN DE LA LÓGICA REFORZADA ---

    print(f"Características numéricas finales para transformador: {numeric_features}")
    print(f"Características categóricas finales para transformador: {categorical_features}")
    print(f"Características de fecha/hora finales para transformador: {datetime_features}")
    # Columns in potential_high_card_cols will be handled by 'remainder=passthrough' if not in other lists.
    
    # Manejar datos duplicados
    initial_rows = df_copy.shape[0]
    df_copy = df_copy.drop_duplicates()
    rows_dropped = initial_rows - df_copy.shape[0]
    if rows_dropped > 0:
        print(f"Removed {rows_dropped} duplicate rows.")
        
    # Manejar outliers
    if outlier_columns:
        outliers = detect_outliers(df_copy, outlier_columns)
        df_copy = handle_outliers(df_copy, outliers, strategy=handle_outliers_strategy)
    
    # Procesamiento de caracteristicas numericas
    if imputation_strategy == 'simple':
        imputer = SimpleImputer(strategy='median')
    elif imputation_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif imputation_strategy == 'iterative':
        imputer = enable_iterative_imputer()
    
    if scaling_strategy == 'standard':
        scaler = StandardScaler()
    elif scaling_strategy == 'robust':
        scaler = RobustScaler()
    elif scaling_strategy == 'minmax':
        scaler = MinMaxScaler()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])

    # Procesamiento de caracteristicas categoricas
    if encoding_strategy == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore')
    elif encoding_strategy == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Procesamiento de caracteristicas de fecha
    def extract_date_features(df_param):
        for col in datetime_features:
            df_param[col + '_year'] = df_param[col].dt.year
            df_param[col + '_month'] = df_param[col].dt.month
            df_param[col + '_day'] = df_param[col].dt.day
            df_param[col + '_dayofweek'] = df_param[col].dt.dayofweek
        return df_param

    X = extract_date_features(X)

    datetime_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Aquí se podrían añadir más pasos, como extraer día, mes, año...
    ])

    text_vectorizer = Pipeline(steps=[
        # TfidfVectorizer es una opción, pero requiere que las columnas de texto
        # se manejen como una sola entrada de texto por fila.
        # Por ahora, un placeholder. La lógica de text_features debe ser mejorada.
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
    ])

    # Crear el preprocesador con ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('datetime', datetime_transformer, datetime_features)
        ],
        remainder='drop' # Asegurarse de que el resto se descarta
    )

    # Entrenar el preprocesador y transformar los datos
    print("Ajustando el preprocesador y transformando los datos...")
    X_processed = preprocessor.fit_transform(X)
    
    # Obtener los nombres de las características después del preprocesamiento
    # Usar get_feature_names_out() del preprocesador es más robusto para incluir passthrough columns
    try:
        processed_feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        logger.error(f"Error al obtener nombres de características del preprocesador: {e}")
        # Fallback manual (menos ideal, pero mejor que nada si get_feature_names_out falla por alguna razón)
        logger.info("Intentando fallback manual para nombres de características...")
        processed_feature_names = numeric_features.copy()
        if len(categorical_features) > 0 and encoding_strategy == 'onehot':
            try:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                processed_feature_names.extend(cat_feature_names)
            except Exception as cat_e:
                logger.error(f"Error obteniendo nombres de cat encoder (fallback): {cat_e}")
                processed_feature_names.extend(categorical_features) # Como último recurso
        else:
            processed_feature_names.extend(categorical_features) # Para ordinal o si no hay one-hot
        
        # Añadir columnas passthrough (esto es la parte más delicada del fallback manual)
        # Necesitamos identificar qué columnas de X original NO estaban en numeric_features ni en las auto_identified_categoricals (antes del filtro de cardinalidad)
        # Esto es propenso a errores si las listas originales no se guardaron bien.
        # Esta es una aproximación:
        original_X_cols = X.columns.tolist()
        processed_in_transformers = set(numeric_features + categorical_features) # Usar la lista ANTES del filtro de cardinalidad
        passthrough_cols_approx = [col for col in original_X_cols if col not in processed_in_transformers]
        processed_feature_names.extend(passthrough_cols_approx)
        logger.info(f"Nombres de características (fallback manual): {processed_feature_names[:15]}...") # Loguear solo una parte
    
    # Convertir el resultado de nuevo a DataFrame
    df_processed = pd.DataFrame(X_processed.toarray() if scipy.sparse.issparse(X_processed) else X_processed, 
                                columns=processed_feature_names, index=X.index)
    
    # Añadir la variable objetivo de vuelta
    if y is not None:
        df_processed[str(target_column)] = y.loc[df_processed.index] # Asegurar alineación de índices
    
    print("\nDataset después de procesar:")
    print(df_processed.head()) # Imprimir head para verificar
    return df_processed, preprocessor

def optimized_dfpreprocess(df, target_column, n_features_to_select=15, apply_pca=True, feature_selection_method='f_classif'):
    """
    Realiza un preprocesamiento optimizado del dataset, incluyendo selección de características.
    """
    logger.info("Iniciando preprocesamiento optimizado del dataset...")
    df_copy = df.copy()

    # Separar X e y
    if target_column in df_copy.columns:
        y = df_copy[target_column]
        X = df_copy.drop(columns=[target_column])
    else:
        raise ValueError(f"La columna objetivo '{target_column}' no se encontró en el DataFrame.")

    # --- NUEVA LÓGICA: Descartar columnas de alta cardinalidad ---
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_threshold = 50
    high_cardinality_features = []

    for col in categorical_features:
        if X[col].nunique() > high_cardinality_threshold:
            high_cardinality_features.append(col)
    
    if high_cardinality_features:
        logger.info(f"Descartando por alta cardinalidad en 'optimized_dfpreprocess': {high_cardinality_features}")
        X = X.drop(columns=high_cardinality_features)
    # --- FIN DE LA NUEVA LÓGICA ---

    # El resto del preprocesamiento continúa con el DataFrame X ya filtrado
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # Selección de características
    if feature_selection_method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
    elif feature_selection_method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
    elif feature_selection_method == 'mutual_info_classif':
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
    elif feature_selection_method == 'mutual_info_regression':
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features_to_select)
    
    X_selected = selector.fit_transform(X, y)
    
    # PCA
    if apply_pca:
        pca = PCA(n_components=0.95)
        X_optimized = pca.fit_transform(X_selected)
        feature_names = [f'PC_{i+1}' for i in range(X_optimized.shape[1])]
    else:
        X_optimized = X_selected
        feature_names = X.columns[selector.get_support()].tolist()
    
    # Convertir el resultado de nuevo a DataFrame
    df_optimized = pd.DataFrame(X_optimized, columns=feature_names, index=df.index)
    df_optimized[target_column] = y
    
    return df_optimized

def detect_outliers(df, columns=None, method='zscore', threshold=3):
    """
    Detecta outliers en las columnas especificadas.
    
    Args:
    df (pd.DataFrame): El DataFrame a analizar.
    columns (list): Lista de columnas para detectar outliers.
    method (str): Método para detectar outliers ('zscore' o 'iqr').
    threshold (float): Umbral para considerar un valor como outlier.
    
    Returns:
    pd.DataFrame: DataFrame con una columna booleana por cada columna analizada indicando si es un outlier.
    """

    outliers = {}
    
    # Si no se especifican columnas, usar todas las columnas numéricas
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col not in df.columns:
            print(f"Advertencia: La columna '{col}' no está en el DataFrame. Se omitirá.")
            continue
        
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"Advertencia: La columna '{col}' no es numérica. Se omitirá.")
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
            raise ValueError("Método no soportado. Use 'iqr' o 'zscore'.")
        
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
    
    return outliers

def handle_outliers(df, outliers, strategy='clip'):
    """
    Maneja los outliers detectados.
    
    Args:
    df (pd.DataFrame): El DataFrame original.
    outliers (pd.DataFrame): DataFrame con las columnas de outliers.
    strategy (str): Estrategia para manejar outliers ('clip', 'remove', or 'impute').
    
    Returns:
    pd.DataFrame: DataFrame con los outliers manejados.
    """
    df_cleaned = df.copy()
    
    for col, outlier_indices in outliers.items():
        if strategy == 'remove':
            df_cleaned = df_cleaned.drop(outlier_indices)
        elif strategy == 'clip':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned.loc[outlier_indices, col] = df_cleaned.loc[outlier_indices, col].clip(lower_bound, upper_bound)
        elif strategy == 'mean':
            mean_value = df_cleaned[col].mean()
            df_cleaned.loc[outlier_indices, col] = mean_value
        elif strategy == 'median':
            median_value = df_cleaned[col].median()
            df_cleaned.loc[outlier_indices, col] = median_value
        else:
            raise ValueError("Estrategia no soportada. Use 'remove', 'clip', 'mean' o 'median'.")
    
    return df_cleaned

def select_best_features(X, y, problem_type, feature_names, k=10, method='basic'):
    """
    Selecciona las k mejores características de un dataset.
    """
    if X is None or y is None:
        return [], X 

    # Asegurarse de que k no sea mayor que el número de features disponibles
    n_features = X.shape[1]
    if k > n_features:
        print(f"Advertencia: k={k} es mayor que el número de features ({n_features}). Se ajustará k a {n_features}.")
        k = n_features

    print(f"Seleccionando las mejores {k} características de {n_features} disponibles...")

    if method == 'basic':
        # Pasa el k ajustado a la función subyacente
        selected_features = select_features_basic(X, y, problem_type, feature_names, k)
    elif method == 'intermediate':
        selected_features = select_features_intermediate(X, y, problem_type, feature_names, k)
    elif method == 'advanced':
        selected_features = select_features_advanced(X, y, problem_type, feature_names, k)
    else:
        raise ValueError("Método de selección de características no válido")

    # Crear el nuevo conjunto de datos con las características seleccionadas
    if isinstance(X, pd.DataFrame):
        X_new = X[selected_features]
    else:
        # Usar una comprensión de lista con un try-except para manejar características no encontradas
        # Asegurarse que feature_names sea una lista para poder usar .index()
        if not isinstance(feature_names, list):
            feature_names_list = list(feature_names)
        else:
            feature_names_list = feature_names
        
        feature_indices = [feature_names_list.index(feature) for feature in selected_features if feature in feature_names_list]
        X_new = X[:, feature_indices]
        # selected_features ya debería ser la lista correcta de nombres de las características en X_new

    logger.info(f"Características seleccionadas por select_best_features: {selected_features}")
    return selected_features, X_new

def is_data_normalized(df):
    """
    Verifica si los datos ya están normalizados comprobando si están en el rango [-1, 1].
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    
    # Verificar si todos los valores están entre -1 y 1
    values_in_range = ((df[numeric_cols] >= -1) & (df[numeric_cols] <= 1)).all()
    
    # Verificar si al menos una columna tiene valores negativos (para distinguir entre normalización [-1, 1] y [0, 1])
    has_negative = (df[numeric_cols] < 0).any().any()
    
    is_normalized = values_in_range.all()
    normalization_type = "[-1, 1]" if has_negative else "[0, 1]"
    
    print("Resultados de la verificación de normalización:")
    print(f"¿Datos normalizados? {is_normalized} (Tipo de normalización: {normalization_type})")
    
    return is_normalized

def select_features_basic(X, y, problem_type, feature_names, k):
    """
    Selección de características básica usando SelectKBest o SelectFromModel.
    """
    # Lógica para seleccionar el estimador para SelectFromModel
    if problem_type in ["Clasificación Binaria", "Clasificación Multiclase"]:
        # Usar un clasificador como estimador
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    elif problem_type == "Regresión":
        # Usar un regresor como estimador
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        raise ValueError(f"Tipo de problema no reconocido para la selección de características: {problem_type}")

    # Asegurarse de que max_features (que es k) no exceda el número de features
    n_features = X.shape[1]
    max_features = min(k, n_features)

    selector = SelectFromModel(estimator, max_features=max_features)
    
    print(f"Ejecutando SelectFromModel con max_features={max_features}")

    selector.fit(X, y)
    
    selected_mask = selector.get_support()
    return [feature for feature, selected in zip(feature_names, selected_mask) if selected]

def select_features_intermediate(X, y, problem_type, feature_names, k):
    """
    Método intermedio de selección de características usando una combinación de técnicas.
    """
    if problem_type == 'classification':
        f_selector = SelectKBest(f_classif, k=k)
        mi_selector = SelectKBest(mutual_info_classif, k=k)
    else:
        f_selector = SelectKBest(f_regression, k=k)
        mi_selector = SelectKBest(mutual_info_regression, k=k)
    
    f_selector.fit(X, y)
    mi_selector.fit(X, y)
    
    f_support = f_selector.get_support()
    mi_support = mi_selector.get_support()
    
    f_selected = [feature for feature, selected in zip(feature_names, f_support) if selected]
    mi_selected = [feature for feature, selected in zip(feature_names, mi_support) if selected]
    
    combined_selected = list(set(f_selected).union(set(mi_selected)))
    
    return combined_selected[:k]

def select_features_advanced(X, y, problem_type, feature_names, k, time_limit):
    """
    Método avanzado de selección de características usando algoritmo genético optimizado.
    """
    if problem_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, verbose=0)
        scoring = make_scorer(accuracy_score)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, verbose=0)
        scoring = make_scorer(r2_score)

    ga_selector = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        scoring=scoring,
        population_size=20,
        generations=5,
        n_jobs=-1,
        verbose=0,
        max_features=k,
        elitism=2,
        crossover_probability=0.8,
        mutation_probability=0.1
    )

    start_time = time.time()
    ga_selector.fit(X, y)

    while time.time() - start_time < time_limit and not ga_selector.converged_:
        ga_selector.fit(X, y)

    selected_mask = ga_selector.support_
    return [feature for feature, selected in zip(feature_names, selected_mask) if selected]

def get_feature_importance(X, y, problem_type, feature_names):
    """
    Obtiene la importancia de las características usando diferentes métodos.
    """
    if problem_type == 'classification':
        f_test = f_classif(X, y)
        mi = mutual_info_classif(X, y)
    else:
        f_test = f_regression(X, y)
        mi = mutual_info_regression(X, y)

    importance = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_test[0],
        'mutual_info': mi
    })
    importance['combined_score'] = importance['f_score'] * importance['mutual_info']
    return importance.sort_values('combined_score', ascending=False)

def determine_target_column(df):
    potential_targets = ['target', 'TARGET', 'Target', 'class', 'CLASS', 'Class', 'label', 'LABEL', 'Label', 'output', 'OUTPUT', 'Output']
    for col in potential_targets:
        if col in df.columns:
            return col
    # Fallback logic if no common name is found
    # This could be improved, e.g., by looking at data types or number of unique values
    if 'y' in df.columns: return 'y'
    if df.shape[1] > 0: return df.columns[-1] # As a last resort, use the last column
    return None

def ensure_dataset_processed_and_saved(df_original: pd.DataFrame, dataset_name: str, target_column: str = None):
    """
    Ensures a dataset is processed and saved to disk, including its raw and processed versions.

    Args:
        df_original (pd.DataFrame): The original DataFrame.
        dataset_name (str): The name of the dataset (used for subfolder and filenames).
        target_column (str, optional): The name of the target variable. If None, it will be inferred.

    Returns:
        tuple: (pd.DataFrame, preprocessor) The processed DataFrame and the fitted preprocessor.
               Returns (None, None) if an error occurs.
    """
    logger.info(f"[ensure_dataset_processed_and_saved] Processing dataset: {dataset_name}")

    try:
        # 1. Determine save paths
        raw_file_path = get_safe_path('program/almacen/datasets', dataset_name, f"{dataset_name}.csv")
        processed_file_path = get_safe_path('program/almacen/datasets', dataset_name, f"{dataset_name}_processed.csv")

        # 2. Save the original DataFrame
        logger.info(f"Saving original dataset to: {raw_file_path}")
        df_original.to_csv(raw_file_path, index=False)
        logger.info(f"Original dataset saved successfully.")

        # 3. Determine target column if not provided
        if target_column is None:
            target_column = determine_target_column(df_original)
            if target_column is None:
                logger.error("Could not determine target column. Preprocessing cannot proceed without it.")
                # Fallback: Attempt to process without a target, but this might limit preprocessing
                # Or, decide to not process if target is crucial
                # For now, we'll log and continue, but some steps might fail or be skipped
                pass # allow processing to continue, basic_dfpreprocess might handle it or fail gracefully
            else:
                logger.info(f"Determined target column: {target_column}")

        # 4. Perform basic preprocessing
        logger.info(f"Starting basic preprocessing for {dataset_name}...")
        # Infer feature types for basic_dfpreprocess
        if target_column and target_column in df_original.columns:
            X_temp = df_original.drop(columns=[target_column])
        else:
            X_temp = df_original.copy()

        numeric_features = X_temp.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_temp.select_dtypes(include='object').columns.tolist()
        # datetime_features can be added if relevant, for now an empty list
        datetime_features = []
        # text_features can be added if relevant
        text_features = []

        df_processed, preprocessor = basic_dfpreprocess(
            df_original.copy(), # Use a copy to avoid modifying the original df in memory
            target_column=target_column,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            datetime_features=datetime_features,
            text_features=text_features
        )
        logger.info(f"Basic preprocessing completed for {dataset_name}. Processed df shape: {df_processed.shape}")

        # 5. Save the processed DataFrame
        if df_processed is not None:
            logger.info(f"Saving processed dataset to: {processed_file_path}")
            df_processed.to_csv(processed_file_path, index=False)
            logger.info(f"Processed dataset saved successfully.")
            return df_processed, preprocessor
        else:
            logger.error(f"Processed DataFrame is None for {dataset_name}. Cannot save.")
            return None, None

    except Exception as e:
        logger.error(f"Error in ensure_dataset_processed_and_saved for {dataset_name}: {e}", exc_info=True)
        return None, None