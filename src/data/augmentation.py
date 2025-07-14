"""
Data augmentation utilities
"""

import random
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Any, Optional
import re


class DataAugmenter:
    """
    Data augmentation for text data
    """
    
    def __init__(self, augmentation_prob: float = 0.1):
        self.augmentation_prob = augmentation_prob
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words with synonyms
        
        Args:
            text (str): Input text
            n (int): Number of words to replace
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Insert n random synonyms into the text
        
        Args:
            text (str): Input text
            n (int): Number of words to insert
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if not words:
                break
            
            random_word = random.choice([word for word in words if word.isalpha()])
            synonyms = self._get_synonyms(random_word)
            
            if synonyms:
                synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(new_words))
                new_words.insert(random_idx, synonym)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Swap n pairs of words randomly
        
        Args:
            text (str): Input text
            n (int): Number of swaps
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) < 2:
                break
            
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Delete words randomly with probability p
        
        Args:
            text (str): Input text
            p (float): Deletion probability
            
        Returns:
            str: Augmented text
        """
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # If all words are deleted, return original text
        if not new_words:
            return text
        
        return ' '.join(new_words)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').replace('-', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def augment_text(self, text: str, techniques: List[str] = None) -> str:
        """
        Apply augmentation techniques to text
        
        Args:
            text (str): Input text
            techniques (List[str]): List of techniques to apply
            
        Returns:
            str: Augmented text
        """
        if techniques is None:
            techniques = ["synonym_replacement", "random_insertion", "random_swap", "random_deletion"]
        
        # Apply augmentation with probability
        if random.random() > self.augmentation_prob:
            return text
        
        # Randomly select a technique
        technique = random.choice(techniques)
        
        if technique == "synonym_replacement":
            return self.synonym_replacement(text, n=1)
        elif technique == "random_insertion":
            return self.random_insertion(text, n=1)
        elif technique == "random_swap":
            return self.random_swap(text, n=1)
        elif technique == "random_deletion":
            return self.random_deletion(text, p=0.1)
        else:
            return text
    
    def augment_dataset(
        self,
        data: List[Dict[str, Any]],
        text_column: str = "text",
        augmentation_factor: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Augment entire dataset
        
        Args:
            data (List[Dict]): Input dataset
            text_column (str): Name of text column
            augmentation_factor (int): Number of augmented samples per original sample
            
        Returns:
            List[Dict]: Augmented dataset
        """
        augmented_data = []
        
        for item in data:
            # Add original item
            augmented_data.append(item)
            
            # Add augmented versions
            for _ in range(augmentation_factor):
                augmented_item = item.copy()
                augmented_item[text_column] = self.augment_text(item[text_column])
                augmented_data.append(augmented_item)
        
        return augmented_data
