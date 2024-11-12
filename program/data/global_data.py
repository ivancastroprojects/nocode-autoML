#global_data.py
from data.dataset import Dataset
from training.training import Training

dataset: Dataset = None
training: Training = None
auth_token: str = None  # Nuevo: Token de autenticación
