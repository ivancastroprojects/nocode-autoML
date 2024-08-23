#trainingparams.py
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.metrics import make_scorer, accuracy_score
from typing import List

import training.serializer as serializer
import training.train as train
import training.training as training


def train_with_important_features(dataset, target, model, k=10):
    X = dataset.drop(columns=[target])
    y = dataset[target]

    # Ajustamos k al número de características disponibles
    k = min(10, X.shape[1])
    X_new, selected_features = train.select_features(X, y, model, k=k)
    print(f"Selected features: {X.columns[selected_features].tolist()}")

    algorithm = [{'name': model.__class__.__name__, 'model': model}]
    recommended_params = train_recommendedparams(X_new, y, algorithm)
    print("Recommended parameters based on important features:", recommended_params)
    
    for params in recommended_params:
        if params['name'] == model.__class__.__name__:
            model.set_params(**params['best_params'])
    
    training.Training.train_and_evaluate(X_new, X.columns[selected_features].tolist(), selected_features)


def train_recommendedparams(X_train, y_train, algorithms: List[dict]):
    recommended_params = []
    n_samples, n_features = X_train.shape
    n_iter = 25 if n_samples > 10000 or n_features > 50 else 50
    param_factor = 0.5 if n_samples > 10000 or n_features > 50 else 1.0

    for algorithm in algorithms:
        if isinstance(algorithm, dict) and 'name' in algorithm:
            algorithm_name = algorithm['name']
            if algorithm_name in serializer.model_classes:
                model_class = serializer.model_classes[algorithm_name]
                model = model_class()
                param_grid = algorithm.get('param_grid', {})
                if param_grid:
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        scoring=make_scorer(accuracy_score),
                        n_jobs=-1,
                        cv=StratifiedKFold(n_splits=5),
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    recommended_params.append({
                        'name': algorithm_name,
                        'best_params': best_params
                    })
                    print(f"Optimized parameters for {algorithm_name}: {best_params}")
                else:
                    print(f"No parameter grid provided for {algorithm_name}.")
            else:
                print(f"Algorithm {algorithm_name} not found in serializer model classes.")
        else:
            print("Invalid algorithm configuration.")

    return recommended_params