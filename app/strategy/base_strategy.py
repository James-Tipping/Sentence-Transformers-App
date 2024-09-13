from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class BaseStrategy(ABC):
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.setup_strategy()
            
            
    @abstractmethod
    def setup_strategy(self):
        pass
    
    @abstractmethod
    def process(self, text: str, limit: int):
        pass
    
    