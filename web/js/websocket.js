window.WSModule = {
    ws: null,
    status: 'disconnected', // 'connected' | 'disconnected' | 'connecting' | 'reconnecting'
    url: '',
    reconnectTimer: null,
    onPredictionCallback: null,
    onStatusChangeCallback: null,
    reconnectAttempts: 0,

    connect(url) {
        this.url = url || this.url;
        this.setStatus('connecting');
        
        try {
            if (this.ws) {
                this.ws.onclose = null;
                this.ws.onerror = null;
                this.ws.close();
            }

            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log('[WS] WebSocket connected.');
                this.setStatus('connected');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                if (this.onPredictionCallback) {
                    try {
                        const data = JSON.parse(event.data);
                        this.onPredictionCallback(data);
                    } catch (e) {
                        console.error('[WS] Failed to parse prediction JSON:', e);
                    }
                }
            };
            
            this.ws.onclose = () => {
                console.log('[WS] WebSocket disconnected.');
                this.setStatus('disconnected');
                this.scheduleReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('[WS] WebSocket error:', error);
                this.setStatus('disconnected');
            };
        } catch (e) {
            console.error('[WS] Failed to create WebSocket:', e);
            this.setStatus('disconnected');
            this.scheduleReconnect();
        }
    },

    setStatus(newStatus) {
        if (this.status !== newStatus) {
            this.status = newStatus;
            if (this.onStatusChangeCallback) {
                this.onStatusChangeCallback(newStatus);
            }
        }
    },

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        
        this.setStatus('reconnecting');
        const backoff = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 5000);
        this.reconnectAttempts++;
        
        console.log(`[WS] Reconnecting in ${Math.round(backoff)}ms (Attempt ${this.reconnectAttempts})...`);
        this.reconnectTimer = setTimeout(() => {
            if (this.status !== 'connected') {
                this.connect(this.url);
            }
        }, backoff);
    },

    send(blob) {
        if (this.status === 'connected' && this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(blob);
            return true;
        }
        return false;
    },

    onPrediction(callback) {
        this.onPredictionCallback = callback;
    },

    onStatusChange(callback) {
        this.onStatusChangeCallback = callback;
    },

    getStatus() {
        return this.status;
    },

    isConnected() {
        return this.status === 'connected';
    },

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
            this.ws = null;
        }
        this.setStatus('disconnected');
        console.log('[WS] WebSocket explicitly closed.');
    }
};
