from abc import abstractmethod
import os
from google.cloud import storage
from sentence_transformers import SentenceTransformer, util
from .base_strategy import BaseStrategy
import h5py
from app.constants import StrategyData, StrategyEmbeddingsData


class BaseStrategyWithDataset(BaseStrategy):
    
    def __init__(self, model: SentenceTransformer, strategy_data: StrategyData):
        self.strategy_data: StrategyData = strategy_data
        super().__init__(model)
        
    def setup_strategy(self):
        self.setup_embeddings()
        
    def setup_embeddings(self):
        pass
    
    def has_gcloud_dataset(self):
        return os.path.exists(self.strategy_data["h5_filename"])
            
    def download_gcloud_dataset(self):
        if not self.strategy_data:
            raise ValueError("Strategy data is required")
        if self.has_gcloud_dataset():
            return
        
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(self.strategy_data["h5_file_bucket"])
        blob = bucket.blob(self.strategy_data["h5_filename"])
        blob.download_to_filename(self.strategy_data["h5_filename"])
        
    @staticmethod
    def initialise_h5_datasets(strategy_data: StrategyEmbeddingsData, model: SentenceTransformer):
        if not strategy_data:
            raise ValueError("Strategy data is required")
        
        dataset_path = strategy_data["h5_filename"]
            
        if not os.path.exists(dataset_path):
            embeddings_length = model.get_sentence_embedding_dimension() or 384
            with h5py.File(dataset_path, "w") as f:
                f.create_dataset(
                    'embeddings', 
                    shape=(0, embeddings_length), 
                    maxshape=(None, embeddings_length), 
                    dtype='f'
                )
                f.create_dataset(
                    'texts', 
                    shape=strategy_data['database_texts_shape'], 
                    maxshape=strategy_data['database_texts_max_shape'], 
                    dtype=h5py.special_dtype(vlen=str)
                )      


    @staticmethod
    @abstractmethod
    def create_embeddings(strategy_data: StrategyEmbeddingsData, model: SentenceTransformer):
        pass
    
    @staticmethod
    def download_corpus(strategy_data: StrategyEmbeddingsData):
        if not strategy_data:
            raise ValueError("Strategy data is required")
        
        url = strategy_data["url"]
        filename = strategy_data["filename"]
        
        if not os.path.exists(filename):
            print('downloading dataset')
            util.http_get(url, filename)
        else:
            print('dataset already exists')