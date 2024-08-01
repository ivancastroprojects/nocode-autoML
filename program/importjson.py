import json
from sklearn.utils import all_estimators
import inspect

# Función para obtener el valor por defecto de un parámetro

def get_default_value(param):
    if param.default is not inspect.Parameter.empty:
        return param.default
    else:
        return None

all_estimators_info = {}

# Recopilar información de todos los estimadores
for name, estimator in all_estimators():
    estimator_info = {
        "parameters": {}
    }
    
    if hasattr(estimator, "__init__"):
        signature = inspect.signature(estimator.__init__)
        for param_name, param in signature.parameters.items():
            if param_name != 'self':
                default_value = get_default_value(param)
                # Asegurarse de que el valor por defecto sea serializable
                if isinstance(default_value, (int, float, str, bool)):  # Solo tipos simples
                    estimator_info["parameters"][param_name] = default_value
    
    all_estimators_info[name] = estimator_info

# Guardar en un archivo JSON
with open('scikit_learn_estimators.json', 'w') as f:
    json.dump(all_estimators_info, f, indent=4)

print("El archivo JSON ha sido creado exitosamente.")

# Mostrar las primeras líneas del archivo JSON creado
with open('scikit_learn_estimators.json', 'r') as f:
    print("Primeras líneas del archivo JSON:")
    print("\
".join(f.readlines()[:100]))