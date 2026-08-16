window.CameraModule = {

    // ========================================================
    // LANDMARK VISUAL CONFIGURATION (PHASE 1)
    // ========================================================

    LANDMARK_RADIUS: 3,
    CONNECTION_WIDTH: 1.5,
    BOUNDING_BOX_WIDTH: 1.5,
    LABEL_FONT_SIZE: 11,

    // Colors
    COLOR_CONNECTION: '#8B5CF6',
    COLOR_LANDMARK: '#EC4899',
    COLOR_BBOX: '#22C55E',


    // ========================================================
    // CAMERA STATE
    // ========================================================

    videoElement: null,
    stream: null,

    canvas: null,
    ctx: null,

    overlay: null,
    overlayCtx: null,

    hands: null,

    status: 'offline', // 'offline' | 'initializing' | 'connected' | 'denied' | 'unavailable' | 'stopped'
    handDetected: false,
    onStatusChangeCallback: null,
    onHandDetectChangeCallback: null,

    animationFrameId: null,


    // ========================================================
    // INITIALIZE & START CAMERA
    // ========================================================

    async init(videoElementId) {
        return this.start(videoElementId);
    },

    setStatus(newStatus) {
        if (this.status !== newStatus) {
            this.status = newStatus;
            console.log(`[Camera] Status changed to: ${newStatus}`);
            if (this.onStatusChangeCallback) {
                this.onStatusChangeCallback(newStatus);
            }
        }
    },

    onStatusChange(callback) {
        this.onStatusChangeCallback = callback;
    },

    onHandDetectChange(callback) {
        this.onHandDetectChangeCallback = callback;
    },

    async start(videoElementId) {
        if (this.status === 'connected' && this.stream) {
            return true;
        }

        const id = videoElementId || (this.videoElement ? this.videoElement.id : 'camera');
        this.videoElement = document.getElementById(id);
        this.overlay = document.getElementById('hand-overlay');

        this.setStatus('initializing');

        if (!this.videoElement || !this.overlay) {
            console.error('[Camera] Video or overlay element missing.');
            this.setStatus('unavailable');
            return false;
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });

            this.videoElement.srcObject = this.stream;

            try {
                await this.videoElement.play();
            } catch (playErr) {
                console.log('[Camera] Play handling:', playErr);
            }

            // Immediately mark camera as connected
            this.setStatus('connected');

            // Hidden canvas for 320x320 frame crops sent over WebSocket
            if (!this.canvas) {
                this.canvas = document.createElement('canvas');
                this.canvas.width = 320;
                this.canvas.height = 320;
                this.ctx = this.canvas.getContext('2d');
            }

            // Overlay context
            this.overlayCtx = this.overlay.getContext('2d');
            this.resizeOverlay();

            // Initialize MediaPipe for overlay rendering
            if (!this.hands) {
                this.initHands();
            } else {
                this.startHandTracking();
            }

            console.log('[Camera] Camera connected & ready.');
            return true;

        } catch (error) {
            console.error('[Camera] Camera access error:', error);
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                this.setStatus('denied');
            } else {
                this.setStatus('unavailable');
            }
            return false;
        }
    },


    // ========================================================
    // INITIALIZE MEDIAPIPE HANDS (VISUAL OVERLAY ONLY)
    // ========================================================

    initHands() {
        if (typeof Hands === 'undefined') {
            console.warn('[Camera] MediaPipe Hands library not loaded from CDN.');
            return;
        }

        this.hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        this.hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        this.hands.onResults(results => {
            this.drawHandResults(results);
        });

        console.log('[Camera] MediaPipe Hands initialized for visual overlay.');
        this.startHandTracking();
    },


    // ========================================================
    // HAND TRACKING LOOP
    // ========================================================

    startHandTracking() {
        if (!this.hands) return;

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        const processFrame = async () => {
            if (this.status !== 'connected') {
                return;
            }

            if (this.videoElement && this.videoElement.readyState >= 2) {
                try {
                    await this.hands.send({ image: this.videoElement });
                } catch (error) {
                    console.error('[Camera] Hand tracking error:', error);
                }
            }

            if (this.status === 'connected') {
                this.animationFrameId = requestAnimationFrame(processFrame);
            }
        };

        processFrame();
    },


    // ========================================================
    // DRAW HAND LANDMARKS (REFINED PHASE 1)
    // ========================================================

    drawHandResults(results) {
        if (!this.overlayCtx || !this.overlay || this.status !== 'connected') return;

        // Ensure overlay canvas dimension matches ClientRect
        if (this.overlay.width !== this.overlay.clientWidth || this.overlay.height !== this.overlay.clientHeight) {
            this.resizeOverlay();
        }

        // Clear previous frame
        this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);

        const hasHand = Boolean(results.multiHandLandmarks && results.multiHandLandmarks.length > 0);
        const stateChanged = (this.handDetected !== hasHand);
        this.handDetected = hasHand;

        if (stateChanged && this.onHandDetectChangeCallback) {
            this.onHandDetectChangeCallback(this.handDetected);
        }

        if (!hasHand) {
            return;
        }

        const landmarks = results.multiHandLandmarks[0];

        // 1. Draw Connections (Thinner, subtle lines)
        if (typeof drawConnectors !== 'undefined') {
            drawConnectors(this.overlayCtx, landmarks, HAND_CONNECTIONS, {
                color: this.COLOR_CONNECTION,
                lineWidth: this.CONNECTION_WIDTH
            });
        }

        // 2. Draw Landmark Points (Smaller, clean points)
        if (typeof drawLandmarks !== 'undefined') {
            drawLandmarks(this.overlayCtx, landmarks, {
                color: this.COLOR_LANDMARK,
                lineWidth: 1,
                radius: this.LANDMARK_RADIUS
            });
        }

        // 3. Calculate Bounding Box
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        for (const lm of landmarks) {
            minX = Math.min(minX, lm.x);
            minY = Math.min(minY, lm.y);
            maxX = Math.max(maxX, lm.x);
            maxY = Math.max(maxY, lm.y);
        }

        // Padding
        const padX = (maxX - minX) * 0.12;
        const padY = (maxY - minY) * 0.12;

        minX = Math.max(0, minX - padX);
        minY = Math.max(0, minY - padY);
        maxX = Math.min(1, maxX + padX);
        maxY = Math.min(1, maxY + padY);

        const boxX = minX * this.overlay.width;
        const boxY = minY * this.overlay.height;
        const boxW = (maxX - minX) * this.overlay.width;
        const boxH = (maxY - minY) * this.overlay.height;

        // Draw Bounding Box
        this.overlayCtx.strokeStyle = this.COLOR_BBOX;
        this.overlayCtx.lineWidth = this.BOUNDING_BOX_WIDTH;
        this.overlayCtx.strokeRect(boxX, boxY, boxW, boxH);

        // 4. Draw Label "HAND" (Un-mirrored so text reads normally!)
        this.overlayCtx.save();
        this.overlayCtx.font = `600 ${this.LABEL_FONT_SIZE}px Inter, sans-serif`;
        this.overlayCtx.fillStyle = this.COLOR_BBOX;

        const labelText = "HAND";
        const textY = Math.max(14, boxY - 5);
        this.overlayCtx.translate(boxX + 35, textY);
        this.overlayCtx.scale(-1, 1);
        this.overlayCtx.fillText(labelText, 0, 0);
        this.overlayCtx.restore();
    },


    // ========================================================
    // RESIZE OVERLAY
    // ========================================================

    resizeOverlay() {
        if (!this.overlay) return;
        const rect = this.overlay.getBoundingClientRect();
        this.overlay.width = rect.width;
        this.overlay.height = rect.height;
    },


    // ========================================================
    // CAPTURE FRAME (FOR WEBSOCKET)
    // ========================================================

    captureFrame() {
        if (this.status !== 'connected' || !this.videoElement || !this.ctx) {
            return Promise.resolve(null);
        }

        const vw = this.videoElement.videoWidth;
        const vh = this.videoElement.videoHeight;

        if (vw === 0 || vh === 0) {
            return Promise.resolve(null);
        }

        // Center Crop Square
        const size = Math.min(vw, vh);
        const sx = (vw - size) / 2;
        const sy = (vh - size) / 2;

        this.ctx.drawImage(this.videoElement, sx, sy, size, size, 0, 0, 320, 320);

        return new Promise(resolve => {
            this.canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.7);
        });
    },


    // ========================================================
    // STATUS GETTERS & CONTROLS
    // ========================================================

    getStatus() {
        return this.status;
    },

    isHandDetected() {
        return this.handDetected;
    },

    stop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                try { track.stop(); } catch (e) {}
            });
            this.stream = null;
        }

        if (this.videoElement) {
            try { this.videoElement.pause(); } catch (e) {}
            this.videoElement.srcObject = null;
        }

        if (this.overlayCtx && this.overlay) {
            this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        }

        this.handDetected = false;
        if (this.onHandDetectChangeCallback) {
            this.onHandDetectChangeCallback(false);
        }

        this.setStatus('stopped');
        console.log('[Camera] Camera stopped cleanly.');
    }
};