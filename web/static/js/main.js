// main.js - Archivo JavaScript principal para interactividad futura

document.addEventListener('DOMContentLoaded', function () {
    console.log("Tokiiai ML Frontend Listo!");
    // Aquí se añadirá la lógica para interactuar con el backend, manejar formularios, etc.

    const trainingForm = document.getElementById('training-config-form');
    const logContainer = document.getElementById('training-log-container');
    let logPollingInterval;

    async function fetchTrainingLogs() {
        if (!logContainer) return;
        try {
            // Fetch logs
            const logResponse = await fetch('/get_training_logs');
            if (logResponse.ok) {
                const logText = await logResponse.text(); // Tratar la respuesta como texto plano
                logContainer.textContent = logText; // Asignar el texto directamente
                logContainer.scrollTop = logContainer.scrollHeight;
            } else {
                logContainer.textContent += '\nError cargando logs del servidor.';
            }

            // Fetch training status
            const statusResponse = await fetch('/get_training_status');
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                if (!statusData.is_training_active) {
                    console.log("Entrenamiento finalizado (según el servidor), deteniendo polling de logs.");
                    stopLogPolling();
                    // Opcional: cambiar mensaje en logContainer, ej. "Entrenamiento completado."
                    // if (logContainer.textContent.startsWith('Iniciando entrenamiento')) {
                    //     logContainer.textContent += '\n\nEntrenamiento completado.';
                    // }
                }
            } else {
                // No detener el polling si no podemos obtener el estado, podría ser un error temporal
                console.warn("No se pudo obtener el estado del entrenamiento.");
            }

        } catch (error) {
            console.error('Error polling logs/status:', error);
            logContainer.textContent += '\nError de conexión al cargar logs/estado.';
            // Considerar si detener el polling en caso de error de red persistente
        }
    }

    function startLogPolling() {
        if (!logContainer) return;
        stopLogPolling(); // Detener cualquier polling anterior
        logContainer.textContent = 'Iniciando entrenamiento, cargando logs...';
        fetchTrainingLogs(); // Carga inicial inmediata
        logPollingInterval = setInterval(fetchTrainingLogs, 3000); // Polling cada 3 segundos
    }

    function stopLogPolling() {
        clearInterval(logPollingInterval);
    }

    if (trainingForm) {
        trainingForm.addEventListener('submit', function(event) {
            // No prevenimos el default, el formulario se enviará normalmente.
            // Iniciamos el polling asumiendo que el entrenamiento comenzará.
            console.log("Formulario de entrenamiento enviado, iniciando polling de logs.");
            startLogPolling();
        });
    }
    
    // Lógica para selectores de optimización (copiada de training_configure.html)
    const optimizationStrategySelect = document.getElementById('optimization_strategy');
    const nIterRandomGroup = document.getElementById('n_iter_random_group');
    const nTrialsOptunaGroup = document.getElementById('n_trials_optuna_group');

    function toggleOptimizationParams() {
        if (!optimizationStrategySelect) return;
        const strategy = optimizationStrategySelect.value;
        if (nIterRandomGroup) nIterRandomGroup.style.display = (strategy === 'random') ? 'block' : 'none';
        if (nTrialsOptunaGroup) nTrialsOptunaGroup.style.display = (strategy === 'optuna') ? 'block' : 'none';
    }

    if (optimizationStrategySelect) {
        optimizationStrategySelect.addEventListener('change', toggleOptimizationParams);
        toggleOptimizationParams(); // Initial call
    }

    // Lógica para deshabilitar target de features (copiada de training_configure.html)
    const targetSelect = document.getElementById('target_column');
    const featuresCheckboxesContainer = document.getElementById('features-checkboxes');
    const allFeatureCheckboxes = featuresCheckboxesContainer ? featuresCheckboxesContainer.querySelectorAll('input[name="feature_columns"]') : [];

    function updateFeatureSelection() {
        if (!targetSelect || !allFeatureCheckboxes.length) return;
        const selectedTarget = targetSelect.value;
        allFeatureCheckboxes.forEach(checkbox => {
            if (checkbox.value === selectedTarget) {
                checkbox.checked = false;
                checkbox.disabled = true;
            } else {
                checkbox.disabled = false;
            }
        });
    }

    if (targetSelect) {
        targetSelect.addEventListener('change', updateFeatureSelection);
        updateFeatureSelection(); // Initial call
    }
}); 