window.CameraModule = {
    videoElement: null,
    stream: null,
    canvas: null,
    ctx: null,
    status: 'offline', // 'connected' | 'offline' | 'requesting'

    async init(videoElementId) {
        this.videoElement = document.getElementById(videoElementId);
        this.status = 'requesting';
        
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
            this.status = 'connected';
            
            // Create hidden canvas for capturing frames
            this.canvas = document.createElement('canvas');
            this.canvas.width = 320;
            this.canvas.height = 320;
            this.ctx = this.canvas.getContext('2d');
            
            return true;
        } catch (error) {
            console.error('Camera access denied or error:', error);
            this.status = 'offline';
            return false;
        }
    },

    captureFrame() {
        if (this.status !== 'connected' || !this.videoElement) {
            return Promise.resolve(null);
        }
        
        // Calculate crop to center the square
        const vw = this.videoElement.videoWidth;
        const vh = this.videoElement.videoHeight;
        
        if (vw === 0 || vh === 0) return Promise.resolve(null);
        
        const size = Math.min(vw, vh);
        const sx = (vw - size) / 2;
        const sy = (vh - size) / 2;
        
        this.ctx.drawImage(this.videoElement, sx, sy, size, size, 0, 0, 320, 320);
        
        return new Promise(resolve => {
            this.canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.7);
        });
    },

    getStatus() {
        return this.status;
    },

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.status = 'offline';
        }
    }
};
