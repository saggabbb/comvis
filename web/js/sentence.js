window.SentenceModule = {
    displayElement: null,
    countElement: null,
    sentence: '',
    onChangeCallback: null,

    init(displayElementId, countElementId) {
        this.displayElement = document.getElementById(displayElementId);
        if (countElementId) {
            this.countElement = document.getElementById(countElementId);
        }
        this.updateDisplay();
    },

    addLetter(letter) {
        if (!letter || letter === '-' || letter === '?' || letter.length === 0) return;
        
        // Handle special character codes
        if (letter.toLowerCase() === 'space') {
            this.sentence += ' ';
        } else if (letter.toLowerCase() === 'del' || letter.toLowerCase() === 'delete') {
            this.deleteLast();
            return;
        } else if (letter.toLowerCase() === 'nothing') {
            return;
        } else {
            this.sentence += letter.toUpperCase();
        }

        this.updateDisplay();
        this.triggerCallback();
    },

    deleteLast() {
        if (this.sentence.length > 0) {
            this.sentence = this.sentence.slice(0, -1);
            this.updateDisplay();
            this.triggerCallback();
        }
    },

    clear() {
        this.sentence = '';
        this.updateDisplay();
        this.triggerCallback();
    },

    getSentence() {
        return this.sentence;
    },

    getCharCount() {
        return this.sentence.length;
    },

    onChange(callback) {
        this.onChangeCallback = callback;
    },

    triggerCallback() {
        if (this.onChangeCallback) {
            this.onChangeCallback(this.sentence, this.sentence.length);
        }
    },

    checkTrigger() {
        return this.sentence.toUpperCase().includes('HENGKY');
    },

    updateDisplay() {
        if (this.countElement) {
            const len = this.sentence.length;
            this.countElement.textContent = `${len} character${len === 1 ? '' : 's'}`;
        }

        if (!this.displayElement) return;
        
        this.displayElement.innerHTML = '';
        
        if (this.sentence.length === 0) {
            const placeholder = document.createElement('span');
            placeholder.className = 'sentence-placeholder';
            placeholder.textContent = 'Start signing and press [+ ADD LETTER] to build words...';
            this.displayElement.appendChild(placeholder);
            return;
        }

        for (let i = 0; i < this.sentence.length; i++) {
            const span = document.createElement('span');
            span.className = 'sentence-char';
            
            if (this.sentence[i] === ' ') {
                span.innerHTML = '&nbsp;';
                span.style.minWidth = '1.2rem';
            } else {
                span.textContent = this.sentence[i];
            }
            
            this.displayElement.appendChild(span);
        }
    }
};
