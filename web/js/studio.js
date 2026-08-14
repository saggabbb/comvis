document.addEventListener('DOMContentLoaded', async () => {
    // UI Elements
    const camStatusDot = document.getElementById('cam-status');
    const aiStatusDot = document.getElementById('ai-status');
    const demoBadge = document.getElementById('demo-badge');
    
    const predictionLetter = document.getElementById('prediction-letter');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    
    const btnAdd = document.getElementById('btn-add');
    const btnDelete = document.getElementById('btn-delete');
    const btnClear = document.getElementById('btn-clear');
    
    let currentPrediction = '-';
    let currentConfidence = 0;
    let loopTimer = null;

    // 1. Init Modules
    SentenceModule.init('sentence-display');
    
    const camReady = await CameraModule.init('camera');
    updateStatusIndicators();
    
    // 2. Connect WS
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host || 'localhost:8000';
    WSModule.connect(`${wsProtocol}//${wsHost}/ws/predict`);
    
    // Check Demo Mode
    try {
        const response = await fetch('/api/status');
        if (response.ok) {
            const data = await response.json();
            if (data.demo_mode) {
                demoBadge.classList.remove('hidden');
            }
        }
    } catch (e) {
        console.log('Status API not available, assuming normal mode');
    }

    // 3. Setup WS callback
    WSModule.onPrediction((data) => {
        if (data.letter && data.confidence !== undefined) {
            // Check if letter changed
            if (currentPrediction !== data.letter) {
                predictionLetter.classList.remove('active');
                
                // Small delay to let animation remove trigger
                setTimeout(() => {
                    predictionLetter.textContent = data.letter;
                    predictionLetter.classList.add('active');
                }, 50);
            }
            
            currentPrediction = data.letter;
            currentConfidence = data.confidence;
            
            // Update confidence
            const confPct = Math.round(currentConfidence * 100);
            confidenceValue.textContent = `${confPct}%`;
            confidenceBar.style.width = `${confPct}%`;
        }
    });

    // 4. Start prediction loop
    function startPredictionLoop() {
        if (loopTimer) clearInterval(loopTimer);
        
        // 8fps = 125ms
        loopTimer = setInterval(async () => {
            updateStatusIndicators();
            
            if (CameraModule.getStatus() === 'connected' && WSModule.isConnected()) {
                const blob = await CameraModule.captureFrame();
                if (blob) {
                    WSModule.send(blob);
                }
            }
        }, 125);
    }
    
    startPredictionLoop();

    // Helper for status dots
    function updateStatusIndicators() {
        if (CameraModule.getStatus() === 'connected') {
            camStatusDot.classList.add('active');
        } else {
            camStatusDot.classList.remove('active');
        }
        
        if (WSModule.getStatus() === 'connected') {
            aiStatusDot.classList.add('active');
        } else {
            aiStatusDot.classList.remove('active');
        }
    }

    // 5. Button Listeners
    btnAdd.addEventListener('click', () => {
        if (currentPrediction !== '-') {
            SentenceModule.addLetter(currentPrediction);
            checkTrigger();
        }
    });
    
    btnDelete.addEventListener('click', () => {
        SentenceModule.deleteLast();
    });
    
    btnClear.addEventListener('click', () => {
        SentenceModule.clear();
    });

    // 6. Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            e.preventDefault();
            if (currentPrediction !== '-') {
                SentenceModule.addLetter(currentPrediction);
                checkTrigger();
            }
        } else if (e.code === 'Backspace') {
            SentenceModule.deleteLast();
        } else if (e.code === 'Enter') {
            checkTrigger();
            SentenceModule.clear();
        }
    });

    // Trigger check
    function checkTrigger() {
        if (SentenceModule.checkTrigger()) {
            window.location.href = 'secret.html';
        }
    }
});
