import gzip
import json
import os
import h5py
import numpy as np
import pandas as pd
from BaseStrategyWithDataset import BaseStrategyWithDataset
from sentence_transformers import SentenceTransformer, util
from Constants import StrategyEmbeddingsData


class QuestionAnswerStrategy(BaseStrategyWithDataset):
    
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
            "titles": [text.decode("utf-8") for text in texts[:, 0]],
            "answers" : [text.decode("utf-8") for text in texts[:, 1]]
        })
        
    def process(self, text: str, limit: int):
        embeddings = self.model.encode([text])

        if embeddings is None:
            raise ValueError("No embeddings generated")

        scores = util.semantic_search(embeddings, np.array(self.data_df['embeddings'].to_list()), top_k=limit)[0] # type: ignore

        questions_list = [
            {
                "title": self.data_df.iloc[score['corpus_id']]['titles'], # type: ignore
                "answer": self.data_df.iloc[score['corpus_id']]['answers'], # type: ignore
                "score": f"{score['score']:.3f}"
            }
            for score in sorted(scores, key=lambda x: x['score'], reverse=True)[:limit]
        ]

        return questions_list
        
        
    @staticmethod
    def create_embeddings(model: SentenceTransformer, strategy_data: StrategyEmbeddingsData):
        
        if not os.path.exists(strategy_data["h5_filename"]):
            QuestionAnswerStrategy.initialise_h5_datasets(strategy_data=strategy_data, model=model)
            
        file_path = strategy_data["filename"]
            
        QuestionAnswerStrategy.download_corpus(strategy_data=strategy_data)
        
        passages = []
        
        with gzip.open(file_path, "rt", encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                for paragraph in data['paragraphs']:
                    passages.append([data['title'], paragraph])
                    if len(passages) >= strategy_data['max_no_steps']:
                        break
                passages.append([data['title'], ' '.join(data['paragraphs']) ])
                if len(passages) >= strategy_data['max_no_steps']:
                    break
                
        number_of_texts = len(passages)
        if strategy_data['max_no_steps'] < number_of_texts:
            number_of_texts = strategy_data['max_no_steps']
        embeddings_length = model.get_sentence_embedding_dimension() or 384

        for i in range(0, number_of_texts, strategy_data['texts_step']):
            texts = [passage for passage in passages[i:i+strategy_data['texts_step']]]
            embeddings = model.encode(texts, show_progress_bar=True, device='cpu')
        
            print(f'Embeddings generated {i + strategy_data["texts_step"]}')
            
            if embeddings is not None and len(embeddings) > 0:
                with h5py.File(strategy_data["h5_filename"], "a") as f:
                    current_length = f["embeddings"].shape[0] # type: ignore
                    new_length = current_length + len(embeddings)
                    embeddings_database = f["embeddings"]
                    texts_database = f["texts"]
                    
                    embeddings_database.resize((new_length, embeddings_length)) # type: ignore
                    embeddings_database[current_length:new_length] = embeddings # type: ignore
                    texts_database.resize((new_length, 2)) # type: ignore
                    texts_database[current_length:new_length] = texts # type: ignore
