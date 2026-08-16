document.addEventListener('DOMContentLoaded', async () => {
    // ========================================================
    // UI ELEMENTS
    // ========================================================
    const camStatusDot = document.getElementById('cam-status');
    const camStatusText = document.getElementById('cam-status-text');

    const handStatusDot = document.getElementById('hand-status');
    const handStatusText = document.getElementById('hand-status-text');

    const aiStatusDot = document.getElementById('ai-status');
    const aiStatusText = document.getElementById('ai-status-text');

    const demoBadge = document.getElementById('demo-badge');

    const camOverlayMsg = document.getElementById('cam-overlay-msg');
    const camMsgText = document.getElementById('cam-msg-text');
    const camSpinner = document.getElementById('cam-spinner');

    const btnToggleCam = document.getElementById('btn-toggle-cam');
    const btnToggleCamText = document.getElementById('btn-toggle-cam-text');

    const predictionLetter = document.getElementById('prediction-letter');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');

    const stabilityBadge = document.getElementById('stability-badge');
    const stabilityText = document.getElementById('stability-text');

    const recentSignsContainer = document.getElementById('recent-signs-container');

    const btnAdd = document.getElementById('btn-add');
    const btnDelete = document.getElementById('btn-delete');
    const btnClear = document.getElementById('btn-clear');

    const toastContainer = document.getElementById('toast-container');

    // ========================================================
    // APP STATE (CENTRALIZED PREDICTION STATE)
    // ========================================================
    const predictionState = {
        status: 'no-camera', // 'no-camera' | 'no-hand' | 'disconnected' | 'warming-up' | 'stable'
        letter: '-',
        confidence: 0,
        isStable: false,
        warmingCount: 0
    };

    let lastStableLetterAdded = null;
    let recentSigns = [];
    const MAX_RECENT_SIGNS = 5;
    let loopTimer = null;


    // ========================================================
    // TOAST NOTIFICATIONS
    // ========================================================
    function showToast(message, type = 'info', duration = 3500) {
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icon = type === 'error' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️';
        toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;

        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }


    // ========================================================
    // 1. INIT MODULES & SYSTEM STATUS
    // ========================================================
    SentenceModule.init('sentence-display', 'char-count-badge');

    // Check system status & Demo mode
    try {
        const response = await fetch('/api/status');
        if (response.ok) {
            const data = await response.json();
            if (data.demo_mode) {
                demoBadge.classList.remove('hidden');
            }
        }
    } catch (e) {
        console.log('[Studio] Status API not available, assuming default mode.');
    }


    // ========================================================
    // 2. CENTRALIZED STATE SYNCHRONIZATION ENGINE
    // ========================================================

    /**
     * Re-evaluates all state priorities and updates UI components consistently.
     */
    function updatePredictionState(incomingData = null) {
        const camStatus = CameraModule.getStatus();
        const handDetected = CameraModule.isHandDetected();
        const wsStatus = WSModule.getStatus();

        // 1. Camera Offline / Stopped
        if (camStatus !== 'connected') {
            predictionState.status = 'no-camera';
            predictionState.letter = '-';
            predictionState.confidence = 0;
            predictionState.isStable = false;
        }
        // 2. No Hand Detected
        else if (!handDetected) {
            predictionState.status = 'no-hand';
            predictionState.letter = '-';
            predictionState.confidence = 0;
            predictionState.isStable = false;
        }
        // 3. WebSocket Disconnected
        else if (wsStatus !== 'connected') {
            predictionState.status = 'disconnected';
            predictionState.letter = '-';
            predictionState.confidence = 0;
            predictionState.isStable = false;
        }
        // 4. Handle Incoming Prediction Data from Backend
        else if (incomingData) {
            if (incomingData.is_stable === false || !incomingData.letter) {
                predictionState.status = 'warming-up';
                predictionState.letter = '-';
                predictionState.confidence = 0;
                predictionState.isStable = false;
                predictionState.warmingCount = incomingData.buffer_size || 0;
            } else {
                predictionState.status = 'stable';
                predictionState.letter = incomingData.letter;
                predictionState.confidence = incomingData.confidence || 0;
                predictionState.isStable = true;
            }
        }

        renderPredictionUI();
        updateStatusDots();
    }


    /**
     * Renders UI strictly based on the current predictionState object.
     */
    function renderPredictionUI() {
        const { status, letter, confidence, isStable, warmingCount } = predictionState;

        // --- STABILITY BADGE & CURRENT SIGN ---
        switch (status) {
            case 'no-camera':
                setStabilityBadge('offline', 'Camera Off');
                renderCurrentSign('-', 0, false);
                break;

            case 'no-hand':
                setStabilityBadge('offline', 'No hand detected');
                renderCurrentSign('-', 0, false);
                break;

            case 'disconnected':
                setStabilityBadge('offline', 'AI connection unavailable');
                renderCurrentSign('-', 0, false);
                break;

            case 'warming-up':
                const label = warmingCount > 0 ? `Analyzing (${warmingCount}/5)` : 'Analyzing...';
                setStabilityBadge('warming', label);
                renderCurrentSign('-', 0, false);
                break;

            case 'stable':
                setStabilityBadge('stable', 'Stable');
                renderCurrentSign(letter, confidence, true);
                
                // Add to Recent Signs if this is a newly arrived stable letter
                if (lastStableLetterAdded !== letter) {
                    addRecentSign(letter);
                    lastStableLetterAdded = letter;
                }
                break;
        }
    }


    /**
     * Renders Current Sign letter and Confidence Bar.
     */
    function renderCurrentSign(letter, confidence, isActive) {
        if (predictionLetter.textContent !== letter) {
            predictionLetter.classList.remove('active');
            
            if (letter !== '-') {
                setTimeout(() => {
                    predictionLetter.textContent = letter;
                    predictionLetter.classList.add('active');
                }, 40);
            } else {
                predictionLetter.textContent = '-';
            }
        } else if (!isActive) {
            predictionLetter.classList.remove('active');
        }

        const confPct = isActive ? Math.round(confidence * 100) : 0;
        confidenceValue.textContent = `${confPct}%`;
        confidenceBar.style.width = `${confPct}%`;
    }


    /**
     * Updates top bar status dot indicators and camera overlay message banner.
     */
    function updateStatusDots() {
        const camStatus = CameraModule.getStatus();
        const handDetected = CameraModule.isHandDetected();
        const wsStatus = WSModule.getStatus();

        // Camera Status Dot
        if (camStatus === 'connected') {
            camStatusDot.className = 'status-dot active';
            camStatusText.textContent = 'CAMERA';
        } else if (camStatus === 'initializing') {
            camStatusDot.className = 'status-dot connecting';
            camStatusText.textContent = 'CAM INITIALIZING';
        } else {
            camStatusDot.className = 'status-dot offline';
            camStatusText.textContent = 'CAM OFFLINE';
        }

        // Hand Detection Dot
        if (camStatus === 'connected' && handDetected) {
            handStatusDot.className = 'status-dot active';
            handStatusText.textContent = 'HAND DETECTED';
        } else {
            handStatusDot.className = 'status-dot offline';
            handStatusText.textContent = 'HAND';
        }

        // AI Connection Dot
        if (wsStatus === 'connected') {
            aiStatusDot.className = 'status-dot active';
            aiStatusText.textContent = 'AI CONNECTED';
        } else if (wsStatus === 'connecting' || wsStatus === 'reconnecting') {
            aiStatusDot.className = 'status-dot connecting';
            aiStatusText.textContent = 'AI CONNECTING';
        } else {
            aiStatusDot.className = 'status-dot offline';
            aiStatusText.textContent = 'AI OFFLINE';
        }

        // Camera Overlay Message Banner Lifecycle
        if (camStatus === 'connected') {
            hideCameraOverlayMsg();
        } else if (camStatus === 'initializing') {
            setCameraOverlayMsg('Initializing camera...', true);
        } else if (camStatus === 'denied') {
            setCameraOverlayMsg('Camera permission required. Please enable camera in browser settings.', false);
        } else if (camStatus === 'unavailable') {
            setCameraOverlayMsg('Camera unavailable.', false);
        } else if (camStatus === 'stopped') {
            setCameraOverlayMsg('Camera stopped. Click [START CAMERA] to resume.', false);
        }
    }


    function setStabilityBadge(stateClass, text) {
        if (!stabilityBadge || !stabilityText) return;
        stabilityBadge.className = `stability-badge ${stateClass}`;
        stabilityText.textContent = text;
    }

    function setCameraOverlayMsg(text, showSpinner) {
        if (!camOverlayMsg || !camMsgText || !camSpinner) return;
        camOverlayMsg.classList.remove('hidden');
        camOverlayMsg.style.display = 'flex';
        camMsgText.textContent = text;
        camSpinner.style.display = showSpinner ? 'block' : 'none';
    }

    function hideCameraOverlayMsg() {
        if (!camOverlayMsg) return;
        camOverlayMsg.classList.add('hidden');
        camOverlayMsg.style.display = 'none';
    }


    // ========================================================
    // 3. CAMERA CONTROLS & LIFECYCLE
    // ========================================================
    async function startCamera() {
        setCameraOverlayMsg('Initializing camera...', true);

        const camReady = await CameraModule.start('camera');

        if (camReady) {
            hideCameraOverlayMsg();
            btnToggleCamText.textContent = 'STOP CAMERA';
            btnToggleCam.classList.remove('stopped');
            showToast('Camera connected successfully.', 'success');
        } else {
            const status = CameraModule.getStatus();
            if (status === 'denied') {
                setCameraOverlayMsg('Camera permission required. Please enable camera in browser settings.', false);
                showToast('Camera permission denied.', 'error');
            } else {
                setCameraOverlayMsg('Camera unavailable or stopped.', false);
                showToast('Unable to access camera.', 'error');
            }
        }
        updatePredictionState();
    }

    function stopCameraApp() {
        CameraModule.stop();
        if (loopTimer) {
            clearInterval(loopTimer);
            loopTimer = null;
        }
        btnToggleCamText.textContent = 'START CAMERA';
        btnToggleCam.classList.add('stopped');
        setCameraOverlayMsg('Camera stopped. Click [START CAMERA] to resume.', false);
        lastStableLetterAdded = null;
        updatePredictionState();
        showToast('Camera stopped.', 'info');
    }

    async function startCameraApp() {
        btnToggleCamText.textContent = 'STOP CAMERA';
        btnToggleCam.classList.remove('stopped');
        lastStableLetterAdded = null;
        await startCamera();
        startPredictionLoop();
    }

    btnToggleCam.addEventListener('click', () => {
        const camStatus = CameraModule.getStatus();
        if (camStatus === 'connected' || camStatus === 'initializing' || CameraModule.stream) {
            stopCameraApp();
        } else {
            startCameraApp();
        }
    });

    CameraModule.onStatusChange(() => updatePredictionState());
    CameraModule.onHandDetectChange(() => updatePredictionState());


    // ========================================================
    // 4. WEBSOCKET SETUP & PREDICTION HANDLER
    // ========================================================
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host || 'localhost:8000';
    WSModule.connect(`${wsProtocol}//${wsHost}/ws/predict`);

    WSModule.onStatusChange((wsStatus) => {
        updatePredictionState();
        if (wsStatus === 'connected') {
            showToast('Connected to AI Backend.', 'success');
        } else if (wsStatus === 'reconnecting') {
            showToast('Reconnecting to AI Backend...', 'info');
        } else if (wsStatus === 'disconnected') {
            showToast('AI Backend disconnected.', 'error');
        }
    });

    // Handle prediction data from backend
    WSModule.onPrediction((data) => {
        updatePredictionState(data);
    });


    // ========================================================
    // 5. PREDICTION FRAME LOOP (8 FPS)
    // ========================================================
    function startPredictionLoop() {
        if (loopTimer) clearInterval(loopTimer);

        // 8 FPS pacing = 125ms interval
        loopTimer = setInterval(async () => {
            updatePredictionState();

            if (CameraModule.getStatus() === 'connected' && WSModule.isConnected() && CameraModule.isHandDetected()) {
                const blob = await CameraModule.captureFrame();
                if (blob) {
                    WSModule.send(blob);
                }
            }
        }, 125);
    }


    // ========================================================
    // 6. RECENT SIGNS HISTORY (PERSISTENT SESSION HISTORY)
    // ========================================================
    function addRecentSign(letter) {
        if (!letter || letter === '-' || letter === '?') return;

        recentSigns.unshift(letter);
        if (recentSigns.length > MAX_RECENT_SIGNS) {
            recentSigns.pop();
        }

        renderRecentSigns();
    }

    function renderRecentSigns() {
        if (!recentSignsContainer) return;
        recentSignsContainer.innerHTML = '';

        if (recentSigns.length === 0) {
            recentSignsContainer.innerHTML = '<span class="history-empty">No signs detected yet</span>';
            return;
        }

        recentSigns.forEach(letter => {
            const pill = document.createElement('span');
            pill.className = 'history-pill';
            pill.textContent = letter;
            recentSignsContainer.appendChild(pill);
        });
    }


    // ========================================================
    // 7. SENTENCE BUILDER LISTENERS & KEYBOARD SHORTCUTS
    // ========================================================
    btnAdd.addEventListener('click', () => {
        if (predictionState.isStable && predictionState.letter !== '-' && predictionState.letter !== '?') {
            SentenceModule.addLetter(predictionState.letter);
            checkTrigger();
        }
    });

    btnDelete.addEventListener('click', () => {
        SentenceModule.deleteLast();
    });

    btnClear.addEventListener('click', () => {
        SentenceModule.clear();
        showToast('Sentence cleared.', 'info');
    });

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.code === 'Space') {
            e.preventDefault();
            if (predictionState.isStable && predictionState.letter !== '-' && predictionState.letter !== '?') {
                SentenceModule.addLetter(predictionState.letter);
                checkTrigger();
            }
        } else if (e.code === 'Backspace') {
            SentenceModule.deleteLast();
        } else if (e.code === 'Enter') {
            checkTrigger();
            SentenceModule.clear();
        }
    });

    function checkTrigger() {
        if (SentenceModule.checkTrigger()) {
            showToast('✨ SECRET UNLOCKED! Redirecting...', 'success', 2000);
            setTimeout(() => {
                window.location.href = 'secret.html';
            }, 1000);
        }
    }


    // ========================================================
    // 8. STARTUP & CLEANUP
    // ========================================================
    await startCamera();
    startPredictionLoop();

    window.addEventListener('beforeunload', () => {
        if (loopTimer) clearInterval(loopTimer);
        CameraModule.stop();
        WSModule.disconnect();
    });
});
