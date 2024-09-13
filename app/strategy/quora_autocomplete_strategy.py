import h5py
import os
import pandas as pd
import numpy as np
from .abstract_base_strategy_with_dataset import BaseStrategyWithDataset
from sentence_transformers import SentenceTransformer, util
from app.constants import StrategyEmbeddingsData


class QuoraAutocompleteStrategy(BaseStrategyWithDataset):
    
    def setup_embeddings(self):
        if not self.has_gcloud_dataset():
            try: 
                self.download_gcloud_dataset()
            except Exception as e:
                print(f"Error getting embeddings from gcloud: {e}")
                raise e
        
        self.load_embeddings_and_texts()
        
    def load_embeddings_and_texts(self):
        with h5py.File(self.strategy_data["h5_filename"], "r") as f:
            embeddings = np.array(f["embeddings"][:]) # type: ignore
            texts = np.array(f["texts"][:]) # type: ignore
            
        self.data_df = pd.DataFrame({
            "embeddings": list(embeddings), 
            "questions" : [text.decode("utf-8") for text in texts]
        })
        
    def process(self, text: str, limit: int):
        embeddings = self.model.encode([text])

        if embeddings is None:
            raise ValueError("No embeddings generated")

        scores = util.semantic_search(embeddings, np.array(self.data_df['embeddings'].to_list()), top_k=limit)[0] # type: ignore

        questions_list = [
            {
                "question": self.data_df.iloc[score['corpus_id']]['questions'], # type: ignore
                "score": f"{score['score']:.3f}"
            }
            for score in sorted(scores, key=lambda x: x['score'], reverse=True)[:limit]
        ]

        return questions_list
        
    @staticmethod
    def create_embeddings(strategy_data: StrategyEmbeddingsData, model: SentenceTransformer):
        
        if not os.path.exists(strategy_data["h5_filename"]):
            QuoraAutocompleteStrategy.initialise_h5_datasets(strategy_data=strategy_data, model=model)
    
        QuoraAutocompleteStrategy.download_corpus(strategy_data=strategy_data)
        
        file_path = strategy_data['filename']
        df = pd.read_csv(file_path, sep='\t')
        corpus_sentences = list(set(df['question1'].tolist() + df['question2'].tolist()))
        corpus_sentences = [sentence for sentence in corpus_sentences if type(sentence) == str]
        
        number_of_texts = len(corpus_sentences)
        if strategy_data['max_no_steps'] < number_of_texts:
            number_of_texts = strategy_data['max_no_steps']
        embeddings_length = model.get_sentence_embedding_dimension() or 384
        
        for i in range(0, number_of_texts, strategy_data['texts_step']):
            embeddings = model.encode(corpus_sentences[i : i+strategy_data['texts_step']], show_progress_bar=True)

            with h5py.File(strategy_data["h5_filename"], "a") as f:
                current_length = f["embeddings"].shape[0] # type: ignore
                new_length = current_length + len(embeddings)
                embeddings_database = f["embeddings"]
                texts_database = f["texts"]
                
                embeddings_database.resize((new_length, embeddings_length)) # type: ignore
                embeddings_database[current_length:new_length] = embeddings # type: ignore
                texts_database.resize((new_length, )) # type: ignore
                texts_database[current_length:new_length] = corpus_sentences[i : i+texts_step] # type: ignore
                