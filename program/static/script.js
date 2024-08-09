document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    document.querySelectorAll('.tab-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            document.querySelector(this.getAttribute('href')).classList.add('active');
        });
    });

    // Dataset upload
    document.getElementById('datasetForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', document.getElementById('datasetFile').files[0]);
        
        fetch('/upload_dataset', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('datasetPreview').innerHTML = data.preview;
            populateColumnSelectors(data.columns);
        });
    });

    // Model training
    document.getElementById('modelForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const targetColumn = document.getElementById('targetColumn').value;
        const featureColumns = Array.from(document.querySelectorAll('#featureColumns input:checked')).map(input => input.value);
        
        fetch('/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_column: targetColumn,
                feature_columns: featureColumns
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('modelResults').innerHTML = `Accuracy: ${data.accuracy}`;
        });
    });

    // Analysis
    document.getElementById('analyzeButton').addEventListener('click', function() {
        fetch('/analyze')
        .then(response => response.json())
        .then(data => {
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + data.feature_importance_plot;
            document.getElementById('analysisResults').innerHTML = '';
            document.getElementById('analysisResults').appendChild(img);
        });
    });

    // Prediction
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('predictionResults').innerHTML = `Prediction: ${data.prediction}`;
        });
    });
});

function populateColumnSelectors(columns) {
    const targetSelect = document.getElementById('targetColumn');
    const featureDiv = document.getElementById('featureColumns');
    
    targetSelect.innerHTML = '';
    featureDiv.innerHTML = '';
    
    columns.forEach(column => {
        targetSelect.innerHTML += `<option value="${column}">${column}</option>`;
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `feature_${column}`;
        checkbox.name = 'feature';
        checkbox.value = column;
        
        const label = document.createElement('label');
        label.htmlFor = `feature_${column}`;
        label.textContent = column;
        
        featureDiv.appendChild(checkbox);
        featureDiv.appendChild(label);
        featureDiv.appendChild(document.createElement('br'));
    });

    // Populate prediction form
    const predictionForm = document.getElementById('predictionForm');
    predictionForm.innerHTML = '';
    columns.forEach(column => {
        const input = document.createElement('input');
        input.type = 'number';
        input.name = column;
        input.placeholder = column;
        input.required = true;
        
        const label = document.createElement('label');
        label.htmlFor = column;
        label.textContent = column;
        
        predictionForm.appendChild(label);
        predictionForm.appendChild(input);
        predictionForm.appendChild(document.createElement('br'));
    });
    
    const submitButton = document.createElement('button');
    submitButton.type = 'submit';
    submitButton.textContent = 'Predecir';
    predictionForm.appendChild(submitButton);
}