#src/vocab_builder.py
import spacy
from collections import Counter
from torchtext.vocab import vocab
import torch
import os

class VocabularyBuilder:
    #Special tokens
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    def __init__(self, min_freq=2, save_path='saved_model/vocab.pth', special_token=True):
        """
        Init VocabularyBuilder with our settings.

        Parameters:
            min_freq (int): Minimum frequency threshold for tokens to be in our vocabulary.
            save_path (str): Path to save or load the vocabulary file.
        """
        self.min_freq = min_freq
        self.save_path = save_path
        self.nlp = spacy.load('en_core_web_lg')
        self.vocabulary = None
        self.special_token = special_token

    def spacy_tokenizer(self, text):
        """
        Tokenizer using SpaCy, returning lemmatized tokens, lowercase, removing punctuation and whitespace.

        Parameters:
            text (str): Text to tokenize.

        Returns:
            List[str]: List of lemmatized, lowercase tokens.
        """
        if not isinstance(text, str):
                print(f"Warning: Expected a string but got {type(text)}")
                return []
        
        doc = self.nlp(text)
        #tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        return tokens
    

    def build_vocab(self, text_lines):
        """
        Builds vocab from list of text, also saving the vocab.

        Parameters:
            text_lines (List[str]): List of text to build the vocab from.

        Returns:
            torchtext.vocab.Vocab: vocab with special tokens added.
        """
        counter = Counter()
        for line in text_lines:
            if line is None or (isinstance(line, float) and torch.isnan(torch.tensor(line))):
                continue

            counter.update(self.spacy_tokenizer(line))
        
        self.vocabulary = vocab(counter, min_freq=self.min_freq)

        if self.special_token:
            self.add_special_tokens()
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.vocabulary, self.save_path)
        print(f"Vocabulary saved to {self.save_path}")
        return self.vocabulary
    


    def load_vocab(self, device='cpu'):
        """
        Loads vocab from the file path.

        Parameters:
            device (str): Device to load vocab to

        Returns:
            torchtext.vocab.Vocab: Loaded vocab
        """
        self.vocabulary = torch.load(self.save_path, map_location=torch.device(device))
        return self.vocabulary


    def add_special_tokens(self):
        """
        Add special tokens and set the default index.

        Returns:
            torchtext.vocab.Vocab: The vocabulary with special tokens added and default index set.
        """
        self.vocabulary.insert_token(self.UNK_TOKEN, 0)
        self.vocabulary.insert_token(self.PAD_TOKEN, 1)
        self.vocabulary.insert_token(self.SOS_TOKEN, 2)
        self.vocabulary.insert_token(self.EOS_TOKEN, 3)
        self.vocabulary.set_default_index(self.vocabulary[self.UNK_TOKEN])
        return self.vocabulary
    
    def token_ids_to_caption(self, token_ids):
        """
        Converts list of token IDs to caption string.

        Parameters:
            token_ids (List[int]): List of token IDs

        Returns:
            str: Caption
        """
        itos = self.vocabulary.get_itos()
        words = [itos[token] for token in token_ids if token < len(itos)]
        return " ".join(words)
