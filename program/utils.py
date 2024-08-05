import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def auto_preprocess(df, target_column=None):
    """
    Automatically preprocess a dataset based on its characteristics.
    
    Args:
    df (pd.DataFrame): Input dataframe
    target_column (str): Name of the target column, if any
    
    Returns:
    pd.DataFrame: Preprocessed dataframe
    """
    print(df.info())
    
    if target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
        y = None

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create main pipeline
    main_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01))
    ])
    
    # Fit and transform the data
    X_processed = main_pipeline.fit_transform(X)
    
    # Get feature names after preprocessing
    numeric_feature_names = [f"{col}'" for col in numeric_features]
    if len(categorical_features) > 0:
        onehot_encoder = main_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        categorical_feature_names = [f"{col.split('_')[0]}'" for col in onehot_encoder.get_feature_names_out(categorical_features)]
    else:
        categorical_feature_names = []
    
    all_feature_names = numeric_feature_names + categorical_feature_names
    
    # Create processed DataFrame
    df_processed = pd.DataFrame(X_processed, columns=all_feature_names)
    
    # Add target column back if it exists
    if y is not None:
        df_processed[target_column] = y
    
    # Apply PCA if there are many features
    if X_processed.shape[1] > 10:
        pca = PCA(n_components=0.95)  # Preserve 95% of variance
        X_pca = pca.fit_transform(X_processed)
        pca_feature_names = [f"{col}_PC" for col in df_processed.columns[:X_pca.shape[1]]]
        df_pca = pd.DataFrame(X_pca, columns=pca_feature_names)
        
        if y is not None:
            df_pca[target_column] = y
        
        print(f"Applied PCA: Reduced from {X_processed.shape[1]} to {X_pca.shape[1]} features")
        return df_pca
    
    return df_processed