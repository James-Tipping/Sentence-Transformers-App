
import nltk
import numpy as np
from sentence_transformers import util
from BaseStrategy import BaseStrategy
from LexRank import degree_centrality_scores


class SummarisationStrategy(BaseStrategy):
    
    def setup_strategy(self):
        pass
    
    def process(self, text: str, limit: int):
        if not text:
            raise ValueError("Text and n_sentences are required")
        
        sentences = nltk.sent_tokenize(text)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True) # type: ignore
        cos_sim = util.cos_sim(sentence_embeddings, sentence_embeddings).cpu().numpy() # type: ignore
        scores = degree_centrality_scores(cos_sim)
        scores_highest_first = np.argsort(-scores)
        
        summarised_sentences = []
        for id in scores_highest_first[:limit]:
            summarised_sentences.append(sentences[id])
            
        if len(summarised_sentences) >= 1:
            return summarised_sentences
        else:
            raise ValueError("No summarised text available")