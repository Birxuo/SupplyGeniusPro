// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
    login: '/token',
    analyzeSupplyChain: '/analyze-supply-chain',
    optimizeInventory: '/optimize-inventory',
    simulateRisk: '/simulate-risk',
    generatePO: '/generate-po',
    advancedAnalytics: '/advanced-analytics'
};

// Authentication
async function login(username, password) {
    try {
        const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.login}`, {
            username,
            password
        });
        localStorage.setItem('token', response.data.access_token);
        return true;
    } catch (error) {
        console.error('Login failed:', error);
        return false;
    }
}

// API Utilities
function getAuthHeaders() {
    const token = localStorage.getItem('token');
    return {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json'
    };
}

// Dashboard Functions
async function loadSupplyChainMetrics() {
    try {
        const response = await axios.post(
            `${API_BASE_URL}${API_ENDPOINTS.analyzeSupplyChain}`,
            {},
            { headers: getAuthHeaders() }
        );
        
        const metrics = response.data;
        const container = document.getElementById('metricsContainer');
        
        // Update metrics cards
        container.innerHTML = `
            <div class="grid grid-cols-3 gap-4 mb-6">
                <div class="p-4 bg-blue-50 rounded-lg hover:shadow-lg transition-shadow">
                    <p class="text-sm text-blue-600 font-medium">Inventory Turnover</p>
                    <p class="text-2xl font-bold text-blue-800">${metrics.inventory_turnover.toFixed(2)}</p>
                    <div class="mt-2 text-sm text-blue-500">Efficiency Score</div>
                </div>
                <div class="p-4 bg-green-50 rounded-lg hover:shadow-lg transition-shadow">
                    <p class="text-sm text-green-600 font-medium">Order Fulfillment Rate</p>
                    <p class="text-2xl font-bold text-green-800">${(metrics.order_fulfillment_rate * 100).toFixed(1)}%</p>
                    <div class="mt-2 text-sm text-green-500">Performance Metric</div>
                </div>
                <div class="p-4 bg-yellow-50 rounded-lg hover:shadow-lg transition-shadow">
                    <p class="text-sm text-yellow-600 font-medium">Risk Score</p>
                    <p class="text-2xl font-bold text-yellow-800">${metrics.risk_score.toFixed(2)}</p>
                    <div class="mt-2 text-sm text-yellow-500">Risk Assessment</div>
                </div>
            </div>
            <div class="grid grid-cols-2 gap-6">
                <div class="bg-white p-4 rounded-lg shadow-md">
                    <h3 class="text-lg font-semibold mb-4">Supply Chain Performance</h3>
                    <div id="performanceChart"></div>
                </div>
                <div class="bg-white p-4 rounded-lg shadow-md">
                    <h3 class="text-lg font-semibold mb-4">Risk Analysis</h3>
                    <div id="riskChart"></div>
                </div>
            </div>
        `;

        // Create performance chart
        const performanceData = [{
            values: [metrics.inventory_turnover, metrics.order_fulfillment_rate * 100],
            labels: ['Inventory Turnover', 'Order Fulfillment Rate'],
            type: 'pie',
            marker: {
                colors: ['rgb(59, 130, 246)', 'rgb(34, 197, 94)']
            }
        }];

        const performanceLayout = {
            height: 300,
            margin: { t: 0, b: 0, l: 0, r: 0 },
            showlegend: true,
            legend: { orientation: 'h', y: -0.1 }
        };

        Plotly.newPlot('performanceChart', performanceData, performanceLayout);

        // Create risk analysis chart
        const riskData = [{
            type: 'indicator',
            mode: 'gauge+number',
            value: metrics.risk_score,
            title: { text: 'Risk Level' },
            gauge: {
                axis: { range: [0, 10] },
                bar: { color: 'rgb(234, 179, 8)' },
                bgcolor: 'white',
                borderwidth: 2,
                bordercolor: 'gray',
                steps: [
                    { range: [0, 3], color: 'rgb(34, 197, 94)' },
                    { range: [3, 7], color: 'rgb(234, 179, 8)' },
                    { range: [7, 10], color: 'rgb(239, 68, 68)' }
                ]
            }
        }];

        const riskLayout = {
            height: 300,
            margin: { t: 0, b: 0, l: 0, r: 0 }
        };

        Plotly.newPlot('riskChart', riskData, riskLayout);
    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

// Inventory Optimization
async function optimizeInventory() {
    const holdingCost = parseFloat(document.getElementById('holdingCost').value);
    const stockoutCost = parseFloat(document.getElementById('stockoutCost').value);
    
    if (isNaN(holdingCost) || isNaN(stockoutCost)) {
        alert('Please enter valid costs');
        return;
    }
    
    try {
        const response = await axios.post(
            `${API_BASE_URL}${API_ENDPOINTS.optimizeInventory}`,
            {
                current_stock: { /* Add sample data */ },
                historical_demand: [],
                lead_times: {},
                holding_cost: holdingCost,
                stockout_cost: stockoutCost
            },
            { headers: getAuthHeaders() }
        );
        
        const results = document.getElementById('optimizationResults');
        results.innerHTML = `
            <h3 class="text-lg font-semibold mb-2">Optimization Results</h3>
            <pre class="bg-gray-100 p-4 rounded overflow-x-auto">
                ${JSON.stringify(response.data, null, 2)}
            </pre>
        `;
        results.classList.remove('hidden');
    } catch (error) {
        console.error('Optimization failed:', error);
    }
}

// Risk Simulation
function updateSimulation() {
    const disruption = document.getElementById('disruption').value;
    const demandSpike = document.getElementById('demand-spike').value;
    
    document.getElementById('disruptionValue').textContent = disruption;
    document.getElementById('demandSpikeValue').textContent = demandSpike;
    
    simulateRisk(disruption, demandSpike);
}

async function simulateRisk(disruption, demandSpike) {
    try {
        const response = await axios.post(
            `${API_BASE_URL}${API_ENDPOINTS.simulateRisk}`,
            {
                disruption: parseFloat(disruption),
                demand_spike: parseFloat(demandSpike)
            },
            { headers: getAuthHeaders() }
        );
        
        // Update mitigation strategies with enhanced styling
        const container = document.getElementById('mitigationStrategies');
        container.innerHTML = response.data.strategies
            .map((strategy, index) => `
                <div class="p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-all transform hover:-translate-y-1">
                    <div class="flex items-center">
                        <div class="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                            <span class="text-blue-600 font-semibold">${index + 1}</span>
                        </div>
                        <div class="ml-4">
                            <h4 class="text-lg font-semibold text-gray-800">${strategy}</h4>
                            <p class="text-sm text-gray-500 mt-1">Impact Level: ${calculateImpact(disruption, demandSpike)}%</p>
                        </div>
                    </div>
                </div>
            `)
            .join('');

        // Create impact visualization
        const impactData = [{
            type: 'scatter',
            mode: 'lines+markers',
            x: [0, parseFloat(disruption)],
            y: [0, parseFloat(demandSpike)],
            line: {
                color: 'rgb(59, 130, 246)',
                width: 2
            },
            marker: {
                color: ['rgb(34, 197, 94)', 'rgb(239, 68, 68)'],
                size: 10
            }
        }];

        const impactLayout = {
            title: 'Supply Chain Impact Analysis',
            xaxis: {
                title: 'Disruption Level',
                range: [0, 100]
            },
            yaxis: {
                title: 'Demand Spike',
                range: [0, 100]
            },
            height: 300,
            margin: { t: 50, b: 50, l: 50, r: 20 }
        };

        Plotly.newPlot('simulationChart', impactData, impactLayout);
        document.getElementById('simulationResults').classList.remove('hidden');
    } catch (error) {
        console.error('Simulation failed:', error);
        showError('Simulation failed. Please try again.');
    }
}

function calculateImpact(disruption, demandSpike) {
    return Math.round((parseFloat(disruption) + parseFloat(demandSpike)) / 2);
}

function showError(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <strong class="font-bold">Error!</strong>
        <span class="block sm:inline"> ${message}</span>
    `;
    document.getElementById('simulation').insertBefore(alertDiv, document.getElementById('simulation').firstChild);
    setTimeout(() => alertDiv.remove(), 5000);
}

// Purchase Order Generation
async function generatePO() {
    const supplier = document.getElementById('supplier').value;
    const itemsText = document.getElementById('poItems').value;
    const output = document.getElementById('poOutput');
    
    if (!supplier.trim()) {
        showError('Please enter a supplier name');
        return;
    }
    
    try {
        const items = JSON.parse(itemsText);
        const formData = new FormData();
        formData.append('supplier', supplier);
        formData.append('items', JSON.stringify(items));
        
        output.innerHTML = `
            <div class="flex items-center justify-center p-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span class="ml-2 text-blue-600">Generating PO...</span>
            </div>
        `;
        output.classList.remove('hidden');
        
        const response = await axios.post(
            `${API_BASE_URL}${API_ENDPOINTS.generatePO}`,
            formData,
            { 
                headers: {
                    ...getAuthHeaders(),
                    'Content-Type': 'multipart/form-data'
                }
            }
        );
        
        output.innerHTML = `
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-gray-800">Purchase Order Details</h3>
                    <button onclick="downloadPO()" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors">
                        Download PDF
                    </button>
                </div>
                <div class="border-t border-gray-200 pt-4">
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div>
                            <p class="text-sm text-gray-600">Supplier</p>
                            <p class="font-medium">${supplier}</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600">PO Number</p>
                            <p class="font-medium">${response.data.po_number || 'N/A'}</p>
                        </div>
                    </div>
                    <div class="mt-4">
                        <p class="text-sm text-gray-600 mb-2">Items</p>
                        <div class="bg-gray-50 rounded-lg p-4">
                            <pre class="whitespace-pre-wrap text-sm">${JSON.stringify(items, null, 2)}</pre>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('PO generation failed:', error);
        showError('Please ensure the items are in valid JSON format');
    }
}

function downloadPO() {
    // Implement PDF download functionality
    console.log('Downloading PO...');
}

// Navigation
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('hidden');
    });
    document.getElementById(sectionId).classList.remove('hidden');
    
    if (sectionId === 'dashboard') {
        loadSupplyChainMetrics();
    }
}

// WebSocket Connection for Real-time Alerts
function connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/ws/alerts');
    
    ws.onmessage = (event) => {
        const alerts = JSON.parse(event.data).alerts;
        // Implement alert display logic
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    return ws;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    showSection('dashboard');
    const ws = connectWebSocket();
});