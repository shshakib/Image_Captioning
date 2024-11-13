# src/vocab_builder.py
import spacy
from collections import Counter
from torchtext.vocab import vocab
import torch
import os

nlp = spacy.load('en_core_web_lg')

def spacy_tokenizer(text):
    """
    Tokenizer using SpaCy, returning lemmatized tokens, lowercase, removing punctuation and whitespace.

    Parameters: text (str): Text to tokenize.
    Returns: List[str]: List of lemmatized, lowercase tokens.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return tokens

def build_vocab(text_lines, min_freq=2, save_path='saved_model/vocab.pth'):
    """
    Builds vocabulary from list of text. Saving the vocabulary.

    Parameters:
    text_lines (List[str]): List of text to build the vocab from.
    min_freq (int): Minimum frequency threshold (to include in the vocabulary)
    save_path (str): Path to save the voca.

    Returns:
    torchtext.vocab.Vocab: The built vocabulary.
    """
    counter = Counter()
    for line in text_lines:
        counter.update(spacy_tokenizer(line))
    vocabulary = vocab(counter, min_freq=min_freq)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(vocabulary, save_path)
    print(f"Vocabulary saved to {save_path}")
    return vocabulary


def load_vocab(save_path='saved_model/vocab.pth'):
    """
    Loads vocab from the file path.

    Parameters: save_path (str): Path to the saved vocab

    Returns: torchtext.vocab.Vocab: Loaded vocab
    """
    return torch.load(save_path)
