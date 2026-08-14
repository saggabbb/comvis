window.SentenceModule = {
    displayElement: null,
    sentence: '',

    init(displayElementId) {
        this.displayElement = document.getElementById(displayElementId);
        this.updateDisplay();
    },

    addLetter(letter) {
        if (!letter || letter === '-') return;
        
        this.sentence += letter;
        this.updateDisplay();
    },

    deleteLast() {
        if (this.sentence.length > 0) {
            this.sentence = this.sentence.slice(0, -1);
            this.updateDisplay();
        }
    },

    clear() {
        this.sentence = '';
        this.updateDisplay();
    },

    getSentence() {
        return this.sentence;
    },

    checkTrigger() {
        return this.sentence.toUpperCase() === 'HENGKY';
    },

    updateDisplay() {
        if (!this.displayElement) return;
        
        this.displayElement.innerHTML = '';
        
        for (let i = 0; i < this.sentence.length; i++) {
            const span = document.createElement('span');
            span.className = 'sentence-char';
            
            if (this.sentence[i] === ' ') {
                span.innerHTML = '&nbsp;';
            } else {
                span.textContent = this.sentence[i];
            }
            
            this.displayElement.appendChild(span);
        }
    }
};
