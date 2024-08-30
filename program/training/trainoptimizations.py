#trainingparams.py
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
import pandas as pd

import training.scikitdb.serializer as serializer
from data.global_data import dataset
import training.train as train

#Función para seleccionar las características más importantes y entrenar el modelo con ellas. 
# Utiliza la importancia de las características para optimizar el conjunto de datos de entrenamiento.
def train_with_important_features(X_train, y_train, model, problem_type, feature_names, k=10):
    # Aseguramos que X_train sea un array de NumPy
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    # Ajustamos k al número de características disponibles
    k = min(k, X_train.shape[1])
    
    # Seleccionamos las características
    selector = SelectKBest(score_func=f_classif if problem_type == 'classification' else f_regression, k=k)
    X_new = selector.fit_transform(X_train, y_train)
    selected_features = selector.get_support(indices=True)
    
    # Usamos los feature_names pasados como argumento
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    # Obtenemos los nombres de las columnas seleccionadas
    selected_feature_names = [feature_names[i] for i in selected_features]
    print(f"Características seleccionadas: {selected_feature_names}")

    # Creamos un nuevo DataFrame con las características seleccionadas
    X_new = pd.DataFrame(X_new, columns=selected_feature_names)

    algorithm = [{'name': model.__class__.__name__, 'model': model}]
    recommended_params = train_recommendedparams(X_new, y_train, algorithm)
    print("Parámetros recomendados basados en características importantes:", recommended_params)
    
    for params in recommended_params:
        if params['name'] == model.__class__.__name__:
            model.set_params(**params['best_params'])
    
    # En lugar de llamar a split_and_train, devolvemos los datos procesados
    return X_new, y_train, selected_feature_names, model

# Función para seleccionar automáticamente las características más relevantes
def select_features(X, y, model, k=10):
    if isinstance(model, tuple(serializer.classification_models)):
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

# Función para encontrar el modelo óptimo para el problema actual
def recommend_best_model(self, X_train, y_train, X_test, y_test):
    best_score = float('-inf')
    best_model = None
    
    algorithms = [{'name': model.__name__, 'model': model()} for model in serializer.model_classes.values() 
                    if ((self.problem_type == 'classification' and model in serializer.classification_models) or 
                        (self.problem_type == 'regression' and model in serializer.regression_models))]
    
    recommended_params = train_recommendedparams(X_train, y_train, algorithms)
    
    for params in recommended_params:
        model_info = {'name': params['name'], 'params': params['best_params']}
        trained_models = train.train_model(X_train, y_train, model_info, self.problem_type, self.dataset_name, False)
        
        for model in trained_models:
            if self.problem_type == 'classification':
                score = model.score(X_test, y_test)
            else:
                y_pred = model.predict(X_test)
                score = mean_squared_error(y_test, y_pred)  # Usamos el negativo para maximizar
            
            if score > best_score:
                best_score = score
                best_model = model
    
    print(f"El mejor modelo recomendado es: {best_model.__class__.__name__}")
    return best_model

#Función para buscar los mejores hiperparámetros para los algoritmos proporcionados utilizando GridSearchCV
def train_recommendedparams(X_train, y_train, algorithms):
    recommended_params = []
    n_samples, n_features = X_train.shape
    n_iter = 25 if n_samples > 10000 or n_features > 50 else 50

    for algorithm in algorithms:
        if isinstance(algorithm, dict) and 'name' in algorithm:
            algorithm_name = algorithm['name']
            if algorithm_name in serializer.model_classes:
                model_class = serializer.model_classes[algorithm_name]
                model = model_class()
                param_distributions = algorithm.get('param_distributions', {})
                if param_distributions:
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_distributions,
                        n_iter=n_iter,
                        scoring=make_scorer(accuracy_score),
                        n_jobs=-1,
                        cv=StratifiedKFold(n_splits=5),
                        verbose=1
                    )
                    random_search.fit(X_train, y_train)
                    best_params = random_search.best_params_
                    recommended_params.append({
                        'name': algorithm_name,
                        'best_params': best_params
                    })
                    print("Optimized parameters for " + algorithm_name + ": " + str(best_params))
                else:
                    print("No parameter distributions provided for " + algorithm_name + ".")
            else:
                print("Algorithm " + algorithm_name + " not found in serializer model classes.")
        else:
            print("Invalid algorithm configuration.")

    return recommended_params