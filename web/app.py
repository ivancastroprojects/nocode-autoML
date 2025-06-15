import sys
import json
import os
import shutil
import logging
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_from_directory, Response
from flask_cors import CORS

# Add the project root directory to sys.path to allow for absolute imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Local application imports
from program.config import Config
from program.data.global_data import global_data
from program.data.dataset import Dataset
from program.training.training import Training
from program.api import mqtt
from program.data.datasetprocessing import determine_target_column, ensure_dataset_processed_and_saved
from program.training.modeloptimization import get_popular_algorithms
from program.training.scikitdb.serializer import get_dataset_path, get_model_path, get_model_details, get_safe_path

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Configuración de la carpeta de plantillas y archivos estáticos
# Estos paths son relativos a la ubicación de app.py, así que están bien
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Ruta base para almacenar los datasets (usando project_root)
# project_root ya es Path, DATASET_STORAGE_PATH se define usando project_root / 'program' / ...
PROGRAM_DIR = project_root / 'program'
DATASET_STORAGE_PATH = PROGRAM_DIR / 'almacen' / 'datasets'
MODELS_STORAGE_PATH = PROGRAM_DIR / 'almacen' / 'models' # Nueva definición para modelos

# Clave secreta para mensajes flash
app.secret_key = Config.SECRET_KEY

# Configuración del logging
# La ruta al log debe ser absoluta y apuntar a la raíz del proyecto.
project_root = os.path.abspath(os.path.dirname(__file__))
LOG_FILE = os.path.join(project_root, '..', 'app.log') # Subir un nivel desde 'web'

# Variable global para estado de entrenamiento
IS_TRAINING_ACTIVE = False

# Ya no definimos GlobalData aquí, usamos la importada de program.data.global_data
# Asegúrate que program.data.global_data.py define e instancia `global_data`
# y que global_data.training es una instancia de Training.
# Si global_data.training es None inicialmente en program.data.global_data.py,
# podría necesitar ser inicializado a Training() aquí o al inicio de la app,
# o asegurar que la lógica en program.data.global_data.py lo haga.
# Por simplicidad, asumimos que program.data.global_data.global_data.training ya es un objeto Training.

# ... (el resto del archivo app.py permanece igual, 
#      usando la instancia `global_data` importada) ...

# Ejemplo de cómo podría estar global_data.training si necesita inicialización explícita aquí:
# if not hasattr(global_data, 'training') or global_data.training is None:
#     global_data.training = Training()

def list_trained_models_for_dataset(dataset_name: str) -> list:
    """Escanea la carpeta de modelos y lista los modelos entrenados para un dataset específico."""
    trained_models = []
    if not dataset_name: # Si no hay nombre de dataset, no hay modelos que listar
        return trained_models

    dataset_models_path = MODELS_STORAGE_PATH / dataset_name
    if dataset_models_path.exists() and dataset_models_path.is_dir():
        for file_path in dataset_models_path.iterdir():
            # Asumimos que los modelos son archivos .pkl por ahora
            # Podríamos añadir más lógica para parsear metadatos si es necesario
            if file_path.is_file() and file_path.suffix.lower() == '.pkl':
                # Podríamos devolver más info, como la fecha de modificación, o parsear métricas del nombre
                trained_models.append({
                    'filename': file_path.name,
                    # 'last_modified': file_path.stat().st_mtime 
                })
    # Ordenar modelos, por ejemplo, por nombre o fecha (si se añade)
    trained_models.sort(key=lambda x: x['filename'])
    return trained_models

def list_existing_datasets():
    """Escanea la carpeta de almacenamiento y lista los datasets (carpetas)."""
    datasets = {}
    if not DATASET_STORAGE_PATH.exists():
        DATASET_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        return datasets
    
    for item_path in DATASET_STORAGE_PATH.iterdir():
        if item_path.is_dir():
            files_in_dir = [f.name for f in item_path.iterdir() if f.is_file()]
            if files_in_dir: # Solo considerar si la carpeta tiene archivos
                datasets[item_path.name] = {'files': files_in_dir, 'path': str(item_path)}
    return datasets

@app.route('/')
def index():
    """Página principal de la aplicación."""
    return render_template('index.html', title="Inicio")

@app.route('/datasets')
def datasets_page():
    """Página para gestionar datasets."""
    existing_datasets = list_existing_datasets()
    active_dataset_name = None
    active_dataset_shape = None
    active_dataset_head_html = None
    trained_models_for_active_dataset = [] # Inicializar lista vacía

    if global_data.dataset and global_data.dataset.dataset_name:
        active_dataset_name = global_data.dataset.dataset_name
        df = global_data.dataset.get_dataframe()
        if df is not None and not df.empty:
            active_dataset_shape = df.shape
            active_dataset_head_html = df.head().to_html(classes=["table", "table-responsive", "table-striped", "table-hover"], border=0, index=False)
        
        # Llamar a la nueva función para obtener los modelos del dataset activo
        trained_models_for_active_dataset = list_trained_models_for_dataset(active_dataset_name)
            
    return render_template('datasets.html', 
                           title="Gestión de Datasets", 
                           existing_datasets=existing_datasets,
                           active_dataset_name=active_dataset_name,
                           active_dataset_shape=active_dataset_shape,
                           active_dataset_head=active_dataset_head_html,
                           trained_models=trained_models_for_active_dataset) # Pasar la lista de modelos

@app.route('/set_active_dataset', methods=['POST'])
def set_active_dataset():
    """
    Carga un dataset existente, lo establece como activo y redirige
    a una página de destino o a la página de datasets por defecto.
    """
    dataset_name_to_activate = request.form.get('dataset_name')
    next_url = request.form.get('next_url') # Nuevo: obtener la URL de redirección

    if not dataset_name_to_activate:
        flash('No se proporcionó el nombre del dataset para activar.', 'danger')
        return redirect(url_for('datasets_page'))

    dataset_dir_path = DATASET_STORAGE_PATH / dataset_name_to_activate
    potential_csv_path = dataset_dir_path / f'{dataset_name_to_activate}.csv'
    
    if not potential_csv_path.exists():
        found_csv = False
        if dataset_dir_path.is_dir():
            for file_path in dataset_dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.csv':
                    potential_csv_path = file_path
                    found_csv = True
                    break
        if not found_csv:
            flash(f'No se encontró ningún archivo CSV en la carpeta del dataset "{dataset_name_to_activate}".', 'danger')
            return redirect(url_for('datasets_page'))

    try:
        df = pd.read_csv(potential_csv_path)
        global_data.dataset = Dataset(df)
        global_data.dataset.dataset_name = dataset_name_to_activate
        global_data.training.dataset_name = dataset_name_to_activate

        # Ensure the dataset is processed and saved
        processed_df, preprocessor = ensure_dataset_processed_and_saved(df, dataset_name_to_activate)
        if processed_df is not None:
            # Update global_data.dataset with the processed one if needed, or handle as appropriate
            # For now, we assume the primary goal is to have it on disk.
            # If EDA is needed here, it should be triggered after this.
            flash(f'Dataset "{dataset_name_to_activate}" (archivo "{potential_csv_path.name}") cargado, procesado y establecido como activo.', 'success')
        else:
            flash(f'Dataset "{dataset_name_to_activate}" cargado, pero hubo un error al procesarlo o guardarlo. Revisa los logs.', 'warning')
            # Decide if we should still set it as active or clear it
            # global_data.dataset = None # Option: clear if processing failed critically
            # global_data.training.dataset_name = None

        # El target se configurará en la página de entrenamiento
        # flash(f'Dataset "{dataset_name_to_activate}" (archivo "{potential_csv_path.name}") cargado y establecido como activo.', 'success') # Original message replaced
    except Exception as e:
        flash(f'Error al cargar el dataset "{dataset_name_to_activate}": {str(e)}', 'danger')
        if global_data.dataset and global_data.dataset.dataset_name == dataset_name_to_activate:
             global_data.dataset = None

    # Redirigir a next_url si se proporcionó, si no, a la página de datasets
    # Por seguridad, en una app real se validaría que next_url sea una ruta interna.
    if next_url:
        return redirect(next_url)
    return redirect(url_for('datasets_page'))

@app.route('/delete_dataset_folder', methods=['POST'])
def delete_dataset_folder():
    dataset_name_to_delete = request.form.get('dataset_name')
    if not dataset_name_to_delete:
        flash('No se proporcionó el nombre del dataset para eliminar.', 'danger')
        return redirect(url_for('datasets_page'))

    dataset_folder_path = DATASET_STORAGE_PATH / dataset_name_to_delete
    
    if dataset_folder_path.exists() and dataset_folder_path.is_dir():
        try:
            shutil.rmtree(dataset_folder_path)
            flash(f'Dataset "{dataset_name_to_delete}" y todos sus archivos han sido eliminados.', 'success')
            if global_data.dataset and global_data.dataset.dataset_name == dataset_name_to_delete:
                global_data.dataset = None
                global_data.training.dataset_name = None
        except Exception as e:
            flash(f'Error al eliminar la carpeta del dataset "{dataset_name_to_delete}": {str(e)}', 'danger')
    else:
        flash(f'La carpeta del dataset "{dataset_name_to_delete}" no fue encontrada.', 'warning')
        
    return redirect(url_for('datasets_page'))

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se encontró el archivo', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Guardar el archivo original
                dataset_name = os.path.splitext(file.filename)[0]
                raw_dir = get_safe_path(Config.DATASET_DIR, dataset_name)
                os.makedirs(raw_dir, exist_ok=True)
                raw_path = os.path.join(raw_dir, file.filename)
                file.save(raw_path)
                
                flash(f'Archivo "{file.filename}" subido con éxito.', 'success')

                # Procesar y guardar el dataset automáticamente
                df_original = pd.read_csv(raw_path)
                target_column = determine_target_column(df_original) # Intenta autodetectar
                
                if not target_column:
                     flash('No se pudo autodetectar la columna objetivo. Por favor, asegúrate de que se llame "target" o similar.', 'warning')

                ensure_dataset_processed_and_saved(df_original, dataset_name, target_column)

                flash(f'Dataset "{dataset_name}" procesado y guardado.', 'info')
                
                return redirect(url_for('datasets_page'))

            except Exception as e:
                flash(f'Error al procesar el archivo: {str(e)}', 'danger')
                logger.error(f"Error en upload_file: {traceback.format_exc()}")
                return redirect(request.url)
        else:
            flash('Formato de archivo no válido. Por favor, sube un archivo .csv.', 'danger')
            return redirect(request.url)
            
    return redirect(url_for('datasets_page'))

@app.route('/get_model_options', methods=['GET'])
def get_model_options():
    # Obtener opciones de modelo basadas en el dataset cargado
    options = global_data.training.trainingparams.get_model_options(global_data.training.dataset)
    return jsonify(options)

@app.route('/training/configure', methods=['GET'])
def training_configure_page():
    """Muestra la página para configurar el entrenamiento."""
    if not global_data.dataset or not global_data.dataset.dataset_name:
        flash("Por favor, carga y activa un dataset primero.", "warning")
        return redirect(url_for('datasets_page'))

    df = global_data.dataset.get_dataframe()
    df_columns = []
    if df is not None and not df.empty:
        df_columns = df.columns.tolist()
    else:
        flash("El dataset activo está vacío o no se pudo cargar.", "danger")
        return redirect(url_for('datasets_page'))

    # Intentar preseleccionar el target si ya está en global_data.training
    pre_selected_target = global_data.training.target if global_data.training else None

    # Obtener algoritmos disponibles (esto podría depender del tipo de problema más adelante)
    # Por ahora, asumimos clasificación y regresión como en tu main.py
    available_algorithms = get_popular_algorithms(problem_type=None) # None para todos por ahora
    
    # Convertir la lista de tuplas a un diccionario más simple para la plantilla si es necesario
    # o ajustar la plantilla para que itere sobre la estructura devuelta por get_available_algorithms
    # Ejemplo: { 'RandomForestClassifier': 'RandomForestClassifier', ... }
    # Si get_available_algorithms devuelve [(name, class), ...], la plantilla ya está preparada para ello si se pasa como dict.
    # Para esta iteración, si get_available_algorithms devuelve algo como [('LogisticRegression', LR_class), ...], lo adaptamos:
    processed_algorithms = {name: name for name, _ in available_algorithms} if isinstance(available_algorithms, list) else available_algorithms

    return render_template('training_configure.html',
                           title="Configurar Entrenamiento",
                           active_dataset_name=global_data.dataset.dataset_name,
                           df_columns=df_columns,
                           pre_selected_target=pre_selected_target,
                           available_algorithms=processed_algorithms) # Pasar los algoritmos procesados

@app.route('/training/start', methods=['POST'])
def start_training_web():
    """Recibe la configuración del entrenamiento y la procesa (simulando MQTT)."""
    if not global_data.dataset or not global_data.dataset.dataset_name:
        flash("No hay un dataset activo para iniciar el entrenamiento.", "danger")
        return redirect(url_for('training_configure_page'))

    try:
        global_data.is_training_active = True # Marcar inicio del entrenamiento
        form_data = request.form
        target_column = form_data.get('target_column')
        feature_columns = form_data.getlist('feature_columns')
        selected_algorithms = form_data.getlist('selected_algorithms')
        cross_validation_split = form_data.get('cross_validation_split', type=int, default=80)
        optimization_strategy = form_data.get('optimization_strategy', 'none')
        n_iter_random = form_data.get('n_iter_random', type=int, default=10)
        n_trials_optuna = form_data.get('n_trials_optuna', type=int, default=30)
        recommendations = 'recommendations' in form_data

        # Validaciones básicas (ya existentes)
        if not target_column:
            flash("Debes seleccionar una columna objetivo.", "danger")
            return redirect(url_for('training_configure_page'))
        if not feature_columns:
            flash("Debes seleccionar al menos una característica.", "danger")
            return redirect(url_for('training_configure_page'))
        if not selected_algorithms:
            flash("Debes seleccionar al menos un algoritmo.", "danger")
            return redirect(url_for('training_configure_page'))

        # Actualizar la instancia global de training con los datos del formulario
        # (train_model en mqtt.py también hace esto, pero es bueno tenerlo aquí explícito
        # para asegurar que global_data.training esté alineado antes de la llamada simulada)
        global_data.training.target = target_column
        global_data.training.dataset_name = global_data.dataset.dataset_name 
        global_data.training.algorithms = selected_algorithms
        global_data.training.crossvalidation = cross_validation_split
        global_data.training.recommendations = recommendations
        # La clase Training podría necesitar un método para setear explícitamente la estrategia de optimización y sus params
        # Por ahora, el payload las llevará y train_model en mqtt.py debería usarlas
        # O, la clase Training debe leerlos de self.optimization_strategy, etc.

        training_payload_data = {
            "dataset": global_data.dataset.dataset_name,
            "target": target_column,
            "features": feature_columns,
            "model": selected_algorithms, 
            "params": {}, # Se llenará si se personalizan los hiperparámetros
            "crossvalidation": cross_validation_split,
            "recommendations": recommendations,
            "optimization_strategy": optimization_strategy,
            "n_iter_random": n_iter_random,
            "n_trials_optuna": n_trials_optuna
        }
        
        app.logger.info("--- Iniciando Simulación de Entrenamiento (llamada directa a mqtt.train_model) ---")
        app.logger.info(f"Payload para train_model: {json.dumps(training_payload_data, indent=2)}")

        # Limpiar el archivo de log antes de iniciar el nuevo entrenamiento
        try:
            log_file_path = project_root / 'program' / 'almacen' / 'logs' / 'app.log'
            if log_file_path.exists():
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"{'-'*20} Nueva Sesión de Entrenamiento Iniciada a las {pd.Timestamp.now()} {'-'*20}\n")
            app.logger.info("Archivo de log anterior limpiado para la nueva sesión.")
        except Exception as log_clear_exc:
            app.logger.error(f"Error al limpiar el archivo de log: {log_clear_exc}")

        # SIMULACIÓN DE MQTT: Llamar directamente a la función train_model
        try:
            # Asegurar que global_data.dataset está listo ANTES de llamar a train_model
            dataset_name_for_training = training_payload_data["dataset"]
            target_for_training = training_payload_data["target"]

            if global_data.dataset is None or global_data.dataset.dataset_name != dataset_name_for_training:
                app.logger.warning(f"global_data.dataset no coincide ('{global_data.dataset.dataset_name if global_data.dataset else 'None'}') o es None. Se requiere '{dataset_name_for_training}'. Intentando cargar...")
                
                path_to_load_processed = project_root / 'program' / 'almacen' / 'datasets' / dataset_name_for_training / f'{dataset_name_for_training}_processed.csv'
                path_to_load_raw = project_root / 'program' / 'almacen' / 'datasets' / dataset_name_for_training / f'{dataset_name_for_training}.csv' 
                # Considerar si el nombre original del archivo raw podría ser diferente y estar almacenado en algún lugar.
                # Por ahora, asumimos que el raw se llama igual que la carpeta del dataset.

                loaded_df_for_global = None
                load_path_used = None

                if path_to_load_processed.exists():
                    try:
                        loaded_df_for_global = pd.read_csv(path_to_load_processed)
                        load_path_used = path_to_load_processed
                        app.logger.info(f"Dataset PROCESADO '{dataset_name_for_training}' cargado para global_data desde {load_path_used}")
                    except Exception as e_load_proc:
                        app.logger.error(f"Error cargando dataset PROCESADO {path_to_load_processed}: {e_load_proc}")
                
                if loaded_df_for_global is None and path_to_load_raw.exists(): 
                    try:
                        loaded_df_for_global = pd.read_csv(path_to_load_raw)
                        load_path_used = path_to_load_raw
                        app.logger.info(f"Dataset RAW '{dataset_name_for_training}' cargado para global_data desde {load_path_used}")
                    except Exception as e_load_raw:
                        app.logger.error(f"Error cargando dataset RAW {path_to_load_raw}: {e_load_raw}")
                elif loaded_df_for_global is None: # Si ni el procesado ni el raw existen o fallan
                    app.logger.warning(f"No se encontró ni el archivo procesado ({path_to_load_processed.name}) ni el raw ({path_to_load_raw.name}) para '{dataset_name_for_training}'.")

                if loaded_df_for_global is not None:
                    global_data.dataset = Dataset(loaded_df_for_global) # dataset.py espera un DataFrame
                    global_data.dataset.dataset_name = dataset_name_for_training
                    # Actualizar la instancia de training también, ya que train_model en mqtt.py lo espera
                    global_data.training.dataset = global_data.dataset
                    global_data.training.dataset_name = dataset_name_for_training
                    global_data.training.target = target_for_training # Asegurar que el target también se setea aquí
                    app.logger.info(f"global_data.dataset y global_data.training.dataset actualizados con '{dataset_name_for_training}'. Target: '{target_for_training}'.")
                else:
                    app.logger.error(f"CRÍTICO: No se pudo cargar el dataset '{dataset_name_for_training}' en global_data.dataset. No se puede iniciar el entrenamiento.")
                    flash(f"Error crítico: No se pudo cargar el dataset '{dataset_name_for_training}' para el entrenamiento. Verifica que los archivos .csv (original y procesado) existan en la carpeta del dataset.", "danger")
                    global_data.is_training_active = False
                    return redirect(url_for('training_configure_page'))
            else:
                # El dataset ya está cargado y es el correcto, pero asegurar que la instancia de training lo tiene
                global_data.training.dataset = global_data.dataset
                global_data.training.dataset_name = global_data.dataset.dataset_name
                global_data.training.target = target_for_training # Re-asegurar el target por si acaso
                app.logger.info(f"global_data.dataset ('{global_data.dataset.dataset_name}') ya estaba activo y correcto. Target re-asegurado: '{target_for_training}'.")

            # Llamada a train_model
            best_model_info = mqtt.train_model(training_payload_data)

            app.logger.info("--- Simulación de Entrenamiento Completada ---")
            
            if best_model_info and isinstance(best_model_info, dict) and \
               best_model_info.get("dataset_name") and best_model_info.get("model_filename"):
                flash(f"Entrenamiento completado para '{best_model_info['dataset_name']}'. Mejor modelo: '{best_model_info['model_filename']}'.", "success")
                # Redirigir a la página de HUB de predicciones con el dataset recién usado preseleccionado
                return redirect(url_for('predictions_hub',
                                        dataset_name=best_model_info['dataset_name']))
            else:
                flash(f"Entrenamiento (simulado) completado para '{global_data.dataset.dataset_name if global_data.dataset else 'desconocido'}', pero no se pudo determinar el mejor modelo para la redirección al hub.", "info")
                # Fallback a la página de configuración si no hay detalles del mejor modelo
                return redirect(url_for('training_configure_page'))

        except Exception as train_exc:
            app.logger.error(f"Error durante la simulación de train_model: {train_exc}", exc_info=True)
            flash(f"Error durante el entrenamiento (simulado): {str(train_exc)}", "danger")
        finally:
            global_data.is_training_active = False # Marcar fin del entrenamiento

        return redirect(url_for('training_configure_page')) 

    except Exception as e:
        flash(f"Error al procesar la configuración de entrenamiento: {str(e)}", "danger")
        app.logger.error(f"Error en start_training_web: {e}", exc_info=True)
        global_data.is_training_active = False # Asegurar que se marca como inactivo en caso de error temprano
        return redirect(url_for('training_configure_page'))

# Ruta para servir archivos de modelos (si es necesario, aunque no es lo ideal para producción)
@app.route('/models/<path:dataset_group_name>/<filename>')
def serve_model_file(dataset_group_name, filename):
    models_dir = PROGRAM_DIR / 'almacen' / 'models' / dataset_group_name
    return send_from_directory(models_dir, filename)

# Ruta para servir visualizaciones (gráficos)
@app.route('/visualizations/<path:dataset_name>/<path:model_name>/<filename>')
def serve_visualization_file(dataset_name, model_name, filename):
    viz_dir = PROGRAM_DIR / 'almacen' / 'visualizaciones' / dataset_name / model_name
    return send_from_directory(viz_dir, filename)

@app.route('/get_training_status')
def get_training_status():
    return jsonify({'is_training_active': global_data.is_training_active})

@app.route('/get_training_logs')
def get_training_logs():
    """Endpoint para obtener los logs de entrenamiento."""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = f.read()
            return Response(logs, mimetype='text/plain')
        else:
            return Response("El archivo de log no se ha creado todavía.", mimetype='text/plain')
    except Exception as e:
        return Response(f"Error al leer el archivo de log: {str(e)}", mimetype='text/plain')

@app.route('/predict')
def predict_page():
    """Renderiza la nueva página unificada de predicciones."""
    existing_datasets = list_existing_datasets()
    return render_template('predict.html', 
                           title="Centro de Predicciones",
                           datasets=existing_datasets)

@app.route('/api/models/<path:dataset_name>')
def get_models_for_dataset(dataset_name):
    """API endpoint para obtener los modelos de un dataset."""
    models = list_trained_models_for_dataset(dataset_name)
    return jsonify({'models': models})

@app.route('/models/details/<dataset_name>/<model_name>')
def model_details_page(dataset_name, model_name):
    """
    Devuelve un conjunto completo de datos para el dashboard de un modelo,
    incluyendo métricas, importancia de features y más.
    """
    try:
        model_file_path = MODELS_STORAGE_PATH / dataset_name / model_name
        if not model_file_path.exists():
            return jsonify({'error': 'Archivo del modelo no encontrado.'}), 404

        # Cargar el modelo para extraer sus propiedades
        model = joblib.load(model_file_path)

        # Cargar los metadatos JSON asociados
        metadata_path = model_file_path.with_suffix('.json')
        if not metadata_path.exists():
            return jsonify({'error': 'Archivo de metadatos (.json) no encontrado.'}), 404
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 1. Extraer Métricas y Matriz de Confusión
        metrics = metadata.get('metrics', {})
        confusion_matrix = metrics.pop('confusion_matrix', None) # Extraer y quitar de métricas

        # 2. Extraer Importancia de Features
        feature_importance_data = _get_feature_importance_data(model, dataset_name, model_file_path)

        # 3. Extraer Parámetros del Modelo
        model_params = model.get_params() if hasattr(model, 'get_params') else {}

        # Construir la respuesta final
        dashboard_data = {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix,
            'feature_importance': feature_importance_data,
            'parameters': model_params,
            'model_type': type(model).__name__
        }
        
        return jsonify(dashboard_data)

    except Exception as e:
        app.logger.error(f"Error en get_model_dashboard_data para {model_name}: {str(e)}")
        return jsonify({'error': f'Error al construir el dashboard del modelo: {str(e)}'}), 500

def _get_feature_importance_data(model, dataset_name, model_file_path):
    """Función helper para extraer la importancia de las features."""
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)
    
    if importance is None:
        return {'error': 'El modelo no expone feature_importances_ o coef_.'}

    # Obtener nombres de las features desde el preprocesador
    preprocessor_path = model_file_path.with_name('preprocessor.joblib')
    if not preprocessor_path.exists():
        return {'error': 'Archivo del preprocesador no encontrado.'}
    
    preprocessor = joblib.load(preprocessor_path)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback a las columnas originales si falla
        dataset_dir_path = DATASET_STORAGE_PATH / dataset_name
        original_csv_path = next(dataset_dir_path.glob('*.csv'), None)
        df = pd.read_csv(original_csv_path)
        target_column = determine_target_column(df)
        feature_names = df.columns.drop(target_column).tolist()

    if len(importance) != len(feature_names):
        app.logger.warning(f"Discrepancia en longitud de importancia y features para {model_file_path.name}")
        return {'error': 'No se pudo mapear la importancia a los nombres de las features.'}

    sorted_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_features]
    values = [float(item[1]) for item in sorted_features]

    return {'labels': labels, 'values': values}

if __name__ == '__main__':
    # Creación de carpetas necesarias al iniciar la app
    DATASET_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    # Crear la carpeta de predicciones si no existe
    Path(app.static_folder, 'predictions').mkdir(parents=True, exist_ok=True)
    
    app.run(debug=True, port=5001)