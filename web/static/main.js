document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const modelSelect = document.getElementById('model-select');
    const trainingForm = document.getElementById('training-form');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        const response = await fetch('/upload_dataset', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            document.getElementById('dataset-info').textContent = data.message;
            populateColumnsAndFeatures(data.columns);
            document.getElementById('model-section').style.display = 'block';
            loadModelOptions();
        } else {
            alert(data.message);
        }
    });

    async function loadModelOptions() {
        const response = await fetch('/get_model_options');
        const options = await response.json();
        modelSelect.innerHTML = '';
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.name;
            opt.textContent = option.name;
            modelSelect.appendChild(opt);
        });
        document.getElementById('training-section').style.display = 'block';
    }

    modelSelect.addEventListener('change', (e) => {
        // Aquí puedes cargar los parámetros específicos del modelo seleccionado
        // y mostrarlos en #model-params
    });

    trainingForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(trainingForm);
        const data = {
            model: modelSelect.value,
            target: formData.get('target'),
            features: Array.from(document.querySelectorAll('input[name="features"]:checked')).map(cb => cb.value),
            params: {} // Aquí deberías recoger los parámetros específicos del modelo
        };
        const response = await fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.success) {
            alert(result.message);
        } else {
            alert('Error al iniciar el entrenamiento');
        }
    });

    function populateColumnsAndFeatures(columns) {
        const targetSelect = document.getElementById('target');
        const featuresCheckboxes = document.getElementById('features-checkboxes');
        
        targetSelect.innerHTML = '';
        featuresCheckboxes.innerHTML = '';

        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            targetSelect.appendChild(option);

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'features';
            checkbox.value = column;
            checkbox.id = `feature-${column}`;

            const label = document.createElement('label');
            label.htmlFor = `feature-${column}`;
            label.textContent = column;

            featuresCheckboxes.appendChild(checkbox);
            featuresCheckboxes.appendChild(label);
            featuresCheckboxes.appendChild(document.createElement('br'));
        });
    }
});