import numpy as np
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Ilokano and Tagalog function words that carry no predictive signal
STOPWORDS = {
    'ti', 'a', 'iti', 'ken', 'nga', 'ket', 'kadagiti', 'ni', 'dagiti', 'idi',
    'ang', 'mga', 'ng', 'sa', 'at', 'ay', 'na', 'pa', 'ba', 'is', 'are', 'the', 'a', 'of'
}

def tokenize(sentence):
    """Splits sentence into lowercase tokens, removing common stopwords."""
    tokens = nltk.word_tokenize(sentence.lower())
    return [w for w in tokens if w not in STOPWORDS]

def stem(word):
    """
    Skipping English stemming (PorterStemmer) because it aggressively chops 
    Ilokano words (e.g., nangisit -> nangisi), ruining pattern matching.
    """
    return word.lower()

def bag_of_words(tokenized_sentence, words):
    """Creates a binary vector for the neural network."""
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag