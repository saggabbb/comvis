window.CameraModule = {

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

    status: 'offline',
    handDetected: false,


    // ========================================================
    // INITIALIZE CAMERA
    // ========================================================

    async init(videoElementId) {

        this.videoElement =
            document.getElementById(videoElementId);

        this.overlay =
            document.getElementById('hand-overlay');

        this.status = 'requesting';


        // ----------------------------------------------------
        // CHECK ELEMENTS
        // ----------------------------------------------------

        if (!this.videoElement) {

            console.error(
                '[Camera] Video element not found.'
            );

            this.status = 'offline';

            return false;
        }


        if (!this.overlay) {

            console.error(
                '[Camera] Hand overlay canvas not found.'
            );

            this.status = 'offline';

            return false;
        }


        // ----------------------------------------------------
        // GET CAMERA
        // ----------------------------------------------------

        try {

            this.stream =
                await navigator.mediaDevices.getUserMedia({

                    video: {
                        facingMode: 'user',

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        }
                    },

                    audio: false
                });


            this.videoElement.srcObject =
                this.stream;


            // Tunggu metadata video
            await new Promise(resolve => {

                if (
                    this.videoElement.readyState
                    >= 1
                ) {

                    resolve();

                } else {

                    this.videoElement.onloadedmetadata =
                        () => resolve();

                }

            });


            this.status = 'connected';


            // ------------------------------------------------
            // HIDDEN CANVAS
            // ------------------------------------------------

            this.canvas =
                document.createElement('canvas');

            this.canvas.width = 320;
            this.canvas.height = 320;

            this.ctx =
                this.canvas.getContext('2d');


            // ------------------------------------------------
            // OVERLAY CANVAS
            // ------------------------------------------------

            this.overlayCtx =
                this.overlay.getContext('2d');


            this.resizeOverlay();


            // ------------------------------------------------
            // INITIALIZE MEDIAPIPE
            // ------------------------------------------------

            this.initHands();


            console.log(
                '[Camera] Camera connected.'
            );


            return true;


        } catch (error) {

            console.error(
                '[Camera] Camera access error:',
                error
            );

            this.status = 'offline';

            return false;
        }
    },


    // ========================================================
    // INITIALIZE MEDIAPIPE HANDS
    // ========================================================

    initHands() {

        if (
            typeof Hands === 'undefined'
        ) {

            console.error(
                '[Camera] MediaPipe Hands library not loaded.'
            );

            return;
        }


        this.hands = new Hands({

            locateFile: (
                file
            ) => {

                return (
                    'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' +
                    file
                );

            }

        });


        // ----------------------------------------------------
        // MEDIAPIPE SETTINGS
        // ----------------------------------------------------

        this.hands.setOptions({

            maxNumHands: 1,

            modelComplexity: 1,

            minDetectionConfidence: 0.5,

            minTrackingConfidence: 0.5

        });


        // ----------------------------------------------------
        // RESULTS CALLBACK
        // ----------------------------------------------------

        this.hands.onResults(
            results => {

                this.drawHandResults(
                    results
                );

            }
        );


        console.log(
            '[Camera] MediaPipe Hands initialized.'
        );


        // Mulai loop hand tracking
        this.startHandTracking();
    },


    // ========================================================
    // HAND TRACKING LOOP
    // ========================================================

    async startHandTracking() {

        if (!this.hands) {
            return;
        }


        const processFrame =
            async () => {

                if (
                    this.status !==
                    'connected'
                ) {

                    return;
                }


                if (
                    this.videoElement.readyState
                    >= 2
                ) {

                    try {

                        await this.hands.send({
                            image:
                                this.videoElement
                        });

                    } catch (error) {

                        console.error(
                            '[Camera] Hand tracking error:',
                            error
                        );

                    }
                }


                requestAnimationFrame(
                    processFrame
                );
            };


        processFrame();
    },


    // ========================================================
    // DRAW HAND LANDMARKS
    // ========================================================

    drawHandResults(results) {

        if (
            !this.overlayCtx ||
            !this.overlay
        ) {

            return;
        }


        const width =
            this.overlay.clientWidth;

        const height =
            this.overlay.clientHeight;


        if (
            width === 0 ||
            height === 0
        ) {

            return;
        }


        // ----------------------------------------------------
        // RESIZE CANVAS
        // ----------------------------------------------------

        if (
            this.overlay.width !==
            this.overlay.clientWidth ||

            this.overlay.height !==
            this.overlay.clientHeight
        ) {

            this.resizeOverlay();
        }


        // ----------------------------------------------------
        // CLEAR PREVIOUS FRAME
        // ----------------------------------------------------

        this.overlayCtx.clearRect(
            0,
            0,
            this.overlay.width,
            this.overlay.height
        );


        this.handDetected = false;


        // ----------------------------------------------------
        // CHECK HAND
        // ----------------------------------------------------

        if (
            !results.multiHandLandmarks ||
            results.multiHandLandmarks.length === 0
        ) {

            return;
        }


        // Hanya gunakan tangan pertama
        const landmarks =
            results.multiHandLandmarks[0];


        this.handDetected = true;


        // ----------------------------------------------------
        // DRAW CONNECTIONS
        // ----------------------------------------------------

        if (
            typeof drawConnectors !==
            'undefined'
        ) {

            drawConnectors(

                this.overlayCtx,

                landmarks,

                HAND_CONNECTIONS,

                {
                    color: '#8B5CF6',
                    lineWidth: 3
                }

            );
        }


        // ----------------------------------------------------
        // DRAW LANDMARK POINTS
        // ----------------------------------------------------

        if (
            typeof drawLandmarks !==
            'undefined'
        ) {

            drawLandmarks(

                this.overlayCtx,

                landmarks,

                {
                    color: '#EC4899',
                    lineWidth: 1,
                    radius: 4
                }

            );
        }


        // ----------------------------------------------------
        // CALCULATE BOUNDING BOX
        // ----------------------------------------------------

        let minX = 1;
        let minY = 1;

        let maxX = 0;
        let maxY = 0;


        for (
            const landmark
            of landmarks
        ) {

            minX =
                Math.min(
                    minX,
                    landmark.x
                );

            minY =
                Math.min(
                    minY,
                    landmark.y
                );

            maxX =
                Math.max(
                    maxX,
                    landmark.x
                );

            maxY =
                Math.max(
                    maxY,
                    landmark.y
                );
        }


        // ----------------------------------------------------
        // ADD PADDING
        // ----------------------------------------------------

        const paddingX =
            (maxX - minX) * 0.15;

        const paddingY =
            (maxY - minY) * 0.15;


        minX =
            Math.max(
                0,
                minX - paddingX
            );

        minY =
            Math.max(
                0,
                minY - paddingY
            );

        maxX =
            Math.min(
                1,
                maxX + paddingX
            );

        maxY =
            Math.min(
                1,
                maxY + paddingY
            );


        // ----------------------------------------------------
        // DRAW BOUNDING BOX
        // ----------------------------------------------------

        const boxX =
            minX * this.overlay.width;

        const boxY =
            minY * this.overlay.height;

        const boxWidth =
            (maxX - minX) *
            this.overlay.width;

        const boxHeight =
            (maxY - minY) *
            this.overlay.height;


        this.overlayCtx.strokeStyle =
            '#22C55E';

        this.overlayCtx.lineWidth = 2;

        this.overlayCtx.strokeRect(
            boxX,
            boxY,
            boxWidth,
            boxHeight
        );


        // ----------------------------------------------------
        // LABEL
        // ----------------------------------------------------

        this.overlayCtx.font =
            '600 14px Inter, sans-serif';

        this.overlayCtx.fillStyle =
            '#22C55E';

        this.overlayCtx.fillText(
            'HAND',
            boxX,
            Math.max(
                18,
                boxY - 6
            )
        );
    },


    // ========================================================
    // RESIZE OVERLAY
    // ========================================================

    resizeOverlay() {

        if (!this.overlay) {
            return;
        }


        const rect =
            this.overlay.getBoundingClientRect();


        this.overlay.width =
            rect.width;

        this.overlay.height =
            rect.height;
    },


    // ========================================================
    // CAPTURE FRAME
    // ========================================================

    captureFrame() {

        if (
            this.status !==
            'connected' ||

            !this.videoElement ||

            !this.ctx
        ) {

            return Promise.resolve(null);
        }


        const vw =
            this.videoElement.videoWidth;

        const vh =
            this.videoElement.videoHeight;


        if (
            vw === 0 ||
            vh === 0
        ) {

            return Promise.resolve(null);
        }


        // ----------------------------------------------------
        // CENTER CROP
        // ----------------------------------------------------

        const size =
            Math.min(
                vw,
                vh
            );


        const sx =
            (vw - size) / 2;

        const sy =
            (vh - size) / 2;


        this.ctx.drawImage(

            this.videoElement,

            sx,
            sy,

            size,
            size,

            0,
            0,

            320,
            320

        );


        // ----------------------------------------------------
        // JPEG
        // ----------------------------------------------------

        return new Promise(
            resolve => {

                this.canvas.toBlob(

                    blob => {

                        resolve(
                            blob
                        );

                    },

                    'image/jpeg',

                    0.7
                );

            }
        );
    },


    // ========================================================
    // GET STATUS
    // ========================================================

    getStatus() {

        return this.status;
    },


    // ========================================================
    // GET HAND STATUS
    // ========================================================

    isHandDetected() {

        return this.handDetected;
    },


    // ========================================================
    // STOP CAMERA
    // ========================================================

    stop() {

        if (this.stream) {

            this.stream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );

            this.stream = null;
        }


        // Bersihkan overlay

        if (this.overlayCtx) {

            this.overlayCtx.clearRect(
                0,
                0,
                this.overlay.width,
                this.overlay.height
            );
        }


        this.handDetected = false;

        this.status = 'offline';


        console.log(
            '[Camera] Camera stopped.'
        );
    }
};