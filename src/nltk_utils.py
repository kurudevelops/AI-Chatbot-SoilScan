import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Automatically download the tokenizer model if missing
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()

def tokenize(sentence):
    """Splits sentence into an array of tokens/words."""
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    """Finds the root form of the token."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Returns array of 0 or 1 for each word in the vocabulary that allows
    the sentence to be represented as a vector for the neural network.
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag