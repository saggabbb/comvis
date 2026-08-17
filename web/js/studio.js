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
        status: 'no-camera', // 'no-camera' | 'no-hand' | 'disconnected' | 'warming-up' | 'stable' | 'uncertain'
        letter: '-',
        confidence: 0,
        isStable: false,
        isUncertain: false,
        warmingCount: 0
    };

    let systemState = {
        demoMode: false,
        modelReady: false
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
    async function fetchSystemStatus() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                systemState.demoMode = data.demo_mode;
                systemState.modelReady = data.model_ready;
                if (data.demo_mode) {
                    demoBadge.classList.remove('hidden');
                }
                updateStatusDots();
            }
        } catch (e) {
            console.log('[Studio] Status API not available, assuming default mode.');
        }
    }
    await fetchSystemStatus();


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
                predictionState.isUncertain = false;
                predictionState.warmingCount = incomingData.buffer_size || 0;
            } else if (incomingData.is_uncertain) {
                predictionState.status = 'uncertain';
                predictionState.letter = incomingData.letter;
                predictionState.confidence = incomingData.confidence || 0;
                predictionState.isStable = true;
                predictionState.isUncertain = true;
            } else {
                predictionState.status = 'stable';
                predictionState.letter = incomingData.letter;
                predictionState.confidence = incomingData.confidence || 0;
                predictionState.isStable = true;
                predictionState.isUncertain = false;
            }
        }

        renderPredictionUI();
        updateStatusDots();
    }


    /**
     * Renders UI strictly based on the current predictionState object.
     */
    function renderPredictionUI() {
        const { status, letter, confidence, isStable, isUncertain, warmingCount } = predictionState;

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

            case 'uncertain':
                setStabilityBadge('warming', 'Uncertain');
                renderCurrentSign(letter, confidence, true, true);
                break;

            case 'stable':
                setStabilityBadge('stable', 'Stable');
                renderCurrentSign(letter, confidence, true, false);
                
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
    function renderCurrentSign(letter, confidence, isActive, isUncertain = false) {
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

        if (isUncertain) {
            confidenceBar.style.background = 'var(--warning-color)';
        } else {
            confidenceBar.style.background = ''; // reset to default css gradient
        }
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
            camStatusDot.className = 'status-indicator active';
            camStatusText.textContent = 'CAMERA';
        } else if (camStatus === 'initializing') {
            camStatusDot.className = 'status-indicator connecting';
            camStatusText.textContent = 'CAM INIT';
        } else {
            camStatusDot.className = 'status-indicator offline';
            camStatusText.textContent = 'CAM OFF';
        }

        // Hand Detection Dot
        if (camStatus === 'connected' && handDetected) {
            handStatusDot.className = 'status-indicator active';
            handStatusText.textContent = 'HAND DETECTED';
        } else {
            handStatusDot.className = 'status-indicator offline';
            handStatusText.textContent = 'HAND';
        }

        // Model / AI Connection Dot
        if (wsStatus === 'connected') {
            const pingText = predictionState.lastPingMs ? ` (${predictionState.lastPingMs}ms)` : '';
            if (systemState.demoMode) {
                aiStatusDot.className = 'status-indicator connecting';
                aiStatusText.textContent = `DEMO MODE${pingText}`;
            } else if (systemState.modelReady) {
                aiStatusDot.className = 'status-indicator active';
                aiStatusText.textContent = `MODEL READY${pingText}`;
            } else {
                aiStatusDot.className = 'status-indicator connecting';
                aiStatusText.textContent = `MODEL LOADING${pingText}`;
            }
        } else if (wsStatus === 'connecting' || wsStatus === 'reconnecting') {
            aiStatusDot.className = 'status-indicator connecting';
            aiStatusText.textContent = 'CONNECTING';
        } else {
            aiStatusDot.className = 'status-indicator offline';
            aiStatusText.textContent = 'MODEL OFF';
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
        predictionState.isWaitingForResponse = false;
        if (predictionState.frameSendTime) {
            predictionState.lastPingMs = Date.now() - predictionState.frameSendTime;
        }
        updatePredictionState(data);
    });


    // ========================================================
    // 5. PREDICTION FRAME LOOP (Dynamic FPS)
    // ========================================================
    function startPredictionLoop() {
        if (loopTimer) clearInterval(loopTimer);
        predictionState.isWaitingForResponse = false;
        predictionState.frameSendTime = 0;
        predictionState.lastPingMs = 0;

        // Poll at ~30 FPS (33ms), but ONLY send if we aren't waiting for a response.
        // This acts as a natural backpressure system and eliminates queue-buildup lag.
        loopTimer = setInterval(async () => {
            updatePredictionState();

            if (predictionState.isWaitingForResponse) {
                return; // Wait for backend to finish processing the previous frame
            }

            if (CameraModule.getStatus() === 'connected' && WSModule.isConnected() && CameraModule.isHandDetected()) {
                predictionState.isWaitingForResponse = true;
                const blob = await CameraModule.captureFrame();
                if (blob) {
                    predictionState.frameSendTime = Date.now();
                    WSModule.send(blob);
                } else {
                    predictionState.isWaitingForResponse = false;
                }
            }
        }, 33);
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
