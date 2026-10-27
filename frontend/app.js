let importanceChart = null;

async function predictFraud() {
    const scanBtn = document.querySelector('.btn-primary');
    scanBtn.disabled = true;
    scanBtn.innerHTML = `<span>⏳ Processing Telemetry...</span>`;

    // قراءة كافة الحقول العشرة من الشاشة
    const formData = {
        TransactionAmt: parseFloat(document.getElementById('TransactionAmt').value) || 0,
        card4: document.getElementById('card4').value,
        card6: document.getElementById('card6').value,
        DeviceType: document.getElementById('DeviceType').value,
        P_emaildomain: document.getElementById('P_emaildomain').value,
        ProductCD: document.getElementById('ProductCD').value,
        addr1: parseInt(document.getElementById('addr1').value) || 1,
        hour_of_day: parseInt(document.getElementById('hour_of_day').value) || 0,
        day_of_week: parseInt(document.getElementById('day_of_week').value) || 0,
        is_night_transaction: parseInt(document.getElementById('is_night_transaction').value) || 0,
        dist1: 50 // Default
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.success) {
            document.getElementById('resultsWrapper').classList.remove('hidden');

            const fraudProb = (result.fraud_probability * 100).toFixed(2);
            const legitProb = ((1 - result.fraud_probability) * 100).toFixed(2);
            const confidence = (result.confidence * 100).toFixed(2);

            // 1. Update Main Banner
            const statusCard = document.getElementById('statusCard');
            const predTitle = document.getElementById('predictionTitle');
            const statusIcon = document.getElementById('statusIcon');

            predTitle.innerText = result.prediction;
            document.getElementById('confidenceBanner').innerText = `Confidence: ${confidence}%`;

            if (result.prediction === "FRAUD") {
                statusCard.className = "glass-card main-status-card fraud-theme";
                statusIcon.innerText = "🚨";
            } else {
                statusCard.className = "glass-card main-status-card safe-theme";
                statusIcon.innerText = "💎";
            }

            // 2. Update KPI Metrics
            document.getElementById('fraudProbVal').innerText = `${fraudProb}%`;
            document.getElementById('legitProbVal').innerText = `${legitProb}%`;
            document.getElementById('confidenceVal').innerText = `${confidence}%`;

            // 3. Populate Factors Table
            const factorsTbody = document.getElementById('factorsTableBody');
            factorsTbody.innerHTML = '';

            const factors = result.driving_factors || [
                { feature: "card6", score: "618846.26", value: formData.card6 },
                { feature: "TransactionAmt", score: "602009.75", value: formData.TransactionAmt },
                { feature: "addr1", score: "378779.79", value: formData.addr1 },
                { feature: "P_emaildomain", score: "139426.40", value: formData.P_emaildomain },
                { feature: "day_of_week", score: "91218.11", value: formData.day_of_week }
            ];

            const chartLabels = [];
            const chartData = [];

            factors.forEach(item => {
                chartLabels.push(item.feature);
                chartData.push(parseFloat(item.score) || 100000);

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${item.feature}</strong></td>
                    <td>${item.score || 'N/A'}</td>
                    <td>${item.value}</td>
                `;
                factorsTbody.appendChild(tr);
            });

            // 4. Render Chart.js Horizontal Bar Chart
            renderChart(chartLabels.reverse(), chartData.reverse());

            // 5. Populate Transaction Summary
            renderSummary(formData, result);
        }
    } catch (error) {
        alert("Unable to reach prediction engine. Ensure FastAPI server is running.");
    } finally {
        scanBtn.disabled = false;
        scanBtn.innerHTML = `<span>🚨 Scan Transaction</span>`;
    }
}

function renderChart(labels, data) {
    const ctx = document.getElementById('importanceChart').getContext('2d');

    if (importanceChart) {
        importanceChart.destroy();
    }

    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Score',
                data: data,
                backgroundColor: 'rgba(23, 192, 235, 0.7)',
                borderColor: '#17c0eb',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#8c98a9' } },
                y: { grid: { display: false }, ticks: { color: '#f5f6fa' } }
            }
        }
    });
}

function renderSummary(inputData, result) {
    const userTable = document.getElementById('userFeaturesTable');
    userTable.innerHTML = '';
    for (const [key, val] of Object.entries(inputData)) {
        userTable.innerHTML += `<tr><td><strong>${key}</strong></td><td>${val}</td></tr>`;
    }

    const metaTable = document.getElementById('metadataTable');
    metaTable.innerHTML = `
        <tr><td><strong>Prediction Type</strong></td><td>${result.prediction}</td></tr>
        <tr><td><strong>Fraud Probability</strong></td><td>${result.fraud_probability}</td></tr>
        <tr><td><strong>Model Confidence</strong></td><td>${result.confidence}</td></tr>
        <tr><td><strong>Timestamp</strong></td><td>${new Date().toLocaleString()}</td></tr>
        <tr><td><strong>Model Type</strong></td><td>LightGBM / XGBoost</td></tr>
    `;
}

function resetUI() {
    document.getElementById('resultsWrapper').classList.add('hidden');
}