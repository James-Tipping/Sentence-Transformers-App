from typing import TypedDict
from pydantic import BaseModel


MODEL_NAME = "all-MiniLM-L6-v2"

class RequestStructure(BaseModel):
    text: str
    n_answers: int = 5

class StrategyData(TypedDict):
    h5_filename: str
    h5_file_bucket: str

class StrategyEmbeddingsData(TypedDict):
    url: str
    filename: str
    database_texts_shape: tuple
    database_texts_max_shape: tuple
    max_no_steps: int
    texts_step: int
    h5_filename: str
    
QuestionAnswerStrategyData: StrategyData = {
    "h5_filename": "question_answer_model_database.h5",
    "h5_file_bucket": "ml-models-data-jamestipping",
}

QuoraAutocompleteStrategyData: StrategyData = {
    "h5_filename": "quora_autocomplete_model_database.h5",
    "h5_file_bucket": "ml-models-data-jamestipping",
}
    
QuestionAnswerStrategyEmbeddingsData: StrategyEmbeddingsData = {
    "url": "http://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/simplewiki-2020-11-01.jsonl.gz",
    "filename": "simplewiki-2020-11-01.jsonl.gz",
    "database_texts_shape": (0, 2),
    "database_texts_max_shape": (None, 2),
    "max_no_steps": 1_500_000,
    "texts_step": 50_000,
    "h5_filename": QuestionAnswerStrategyData["h5_filename"],
}

QuoraAutocompleteStrategyEmbeddingsData: StrategyEmbeddingsData = {
    "url": "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv",
    "filename": "quora_duplicate_questions.tsv",
    "database_texts_shape": (0,),
    "database_texts_max_shape": (None,),
    "max_no_steps": 1_500_000,
    "texts_step": 50_000,
    "h5_filename": QuoraAutocompleteStrategyData["h5_filename"],
}
