#global_data.py
from types import SimpleNamespace

# Importar las clases necesarias para la inicialización
from program.data.dataset import Dataset
from program.training.training import Training

class GlobalAppState:
    def __init__(self):
        self.dataset: Dataset = None
        self.training: Training = Training() # Inicializar training con una instancia
        self.auth_token: str = None
        self.is_training_active: bool = False
        # Puedes añadir otros estados globales aquí si es necesario

# Crear la instancia única que será importada por otros módulos
global_data = GlobalAppState()

# Las siguientes líneas son para mantener la compatibilidad con código que pudiera
# estar importando 'dataset' o 'training' directamente desde este módulo,
# aunque la práctica recomendada sería acceder a ellos a través de global_data.dataset
# y global_data.training.
# Considera refactorizar el código para usar solo global_data.
dataset = global_data.dataset # Esto será None inicialmente
training = global_data.training # Esto será una instancia de Training
auth_token = global_data.auth_token # Esto será None inicialmente
