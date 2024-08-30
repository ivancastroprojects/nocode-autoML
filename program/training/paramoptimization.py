#paramoptimization.py

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn_genetic import GASearchCV
from scipy.stats import uniform, randint
import training.scikitdb.serializer as serializer


def get_optimized_params(model, X, y, optimization_method='random'):
    """
    Obtiene los parámetros optimizados para un modelo dado.
    """
    algorithm = [{'name': model.__class__.__name__, 'model': model}]
    recommended_params = train_recommendedparams(X, y, algorithm, optimization_method)
    
    if recommended_params and recommended_params[0]['best_params']:
        return recommended_params[0]['best_params']
    else:
        print(f"No se encontraron parámetros recomendados para {model.__class__.__name__}. Usando configuración por defecto.")
        return {}

def train_recommendedparams(X_train, y_train, algorithms, optimization_method='random'):
    """
    Encuentra los parámetros recomendados para una lista de algoritmos.
    """
    recommended_params = []
    n_samples, n_features = X_train.shape
    n_iter = 25 if n_samples > 10000 or n_features > 50 else 50

    for algorithm in algorithms:
        if isinstance(algorithm, dict) and 'name' in algorithm:
            algorithm_name = algorithm['name']
            if algorithm_name in serializer.model_classes:
                model_class = serializer.model_classes[algorithm_name]
                model = model_class()
                param_distributions = get_param_grid(model)
                if param_distributions:
                    try:
                        if optimization_method == 'random':
                            search = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=param_distributions,
                                n_iter=n_iter,
                                cv=5,
                                verbose=1,
                                n_jobs=-1,
                                random_state=42,
                                error_score='raise'
                            )
                        elif optimization_method == 'grid':
                            search = GridSearchCV(
                                estimator=model,
                                param_grid=param_distributions,
                                cv=5,
                                verbose=1,
                                n_jobs=-1,
                                error_score='raise'
                            )
                        elif optimization_method == 'genetic':
                            search = GASearchCV(
                                estimator=model,
                                param_grid=param_distributions,
                                n_iter=n_iter,
                                cv=5,
                                verbose=1,
                                n_jobs=-1,
                                error_score='raise'
                            )
                        else:
                            raise ValueError(f"Método de optimización no reconocido: {optimization_method}")
                        
                        search.fit(X_train, y_train)
                        best_params = search.best_params_
                        recommended_params.append({
                            'name': algorithm_name,
                            'best_params': best_params
                        })
                        print(f"Parámetros optimizados para {algorithm_name}: {best_params}")
                    except Exception as e:
                        print(f"Error durante la optimización de {algorithm_name}: {str(e)}")
                        print("Usando parámetros por defecto.")
                        recommended_params.append({
                            'name': algorithm_name,
                            'best_params': model.get_params()
                        })
                else:
                    print(f"No se proporcionaron distribuciones de parámetros para {algorithm_name}.")
            else:
                print(f"Algoritmo {algorithm_name} no encontrado en las clases de modelo del serializador.")
        else:
            print("Configuración de algoritmo inválida.")

    return recommended_params

def get_param_grid(model):
    """
    Devuelve el grid de parámetros para un modelo dado.
    """
    param_grid = {}
    for param_name, param_value in model.get_params().items():
        if param_name.endswith('__'):  # Ignorar parámetros privados
            continue
        if isinstance(param_value, bool):
            param_grid[param_name] = [True, False]
        elif isinstance(param_value, int):
            param_grid[param_name] = randint(max(1, param_value // 2), min(100, param_value * 2))
        elif isinstance(param_value, float):
            param_grid[param_name] = uniform(max(0.0001, param_value / 10), min(1, param_value * 10))
        elif isinstance(param_value, str):
            param_grid[param_name] = [param_value]
    return param_grid