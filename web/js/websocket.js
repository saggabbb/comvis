window.WSModule = {
    ws: null,
    status: 'disconnected', // 'connected' | 'disconnected' | 'connecting'
    url: '',
    reconnectTimer: null,
    onPredictionCallback: null,
    reconnectAttempts: 0,

    connect(url) {
        this.url = url;
        this.status = 'connecting';
        
        try {
            this.ws = new WebSocket(url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.status = 'connected';
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                if (this.onPredictionCallback) {
                    try {
                        const data = JSON.parse(event.data);
                        this.onPredictionCallback(data);
                    } catch (e) {
                        console.error('Failed to parse prediction:', e);
                    }
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.status = 'disconnected';
                this.scheduleReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.status = 'disconnected';
            };
        } catch (e) {
            console.error('Failed to create WebSocket:', e);
            this.status = 'disconnected';
            this.scheduleReconnect();
        }
    },

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        
        // Exponential backoff up to 5s
        const backoff = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 5000);
        this.reconnectAttempts++;
        
        console.log(`Reconnecting in ${Math.round(backoff)}ms...`);
        this.reconnectTimer = setTimeout(() => {
            if (this.status !== 'connected') {
                this.connect(this.url);
            }
        }, backoff);
    },

    send(blob) {
        if (this.status === 'connected' && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(blob);
            return true;
        }
        return false;
    },

    onPrediction(callback) {
        this.onPredictionCallback = callback;
    },

    getStatus() {
        return this.status;
    },

    isConnected() {
        return this.status === 'connected';
    }
};
