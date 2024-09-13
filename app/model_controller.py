from typing import Optional
from sentence_transformers import SentenceTransformer
from app.strategy.base_strategy import BaseStrategy
from app.strategy.strategy_factory import StrategyFactory
import asyncio
from app.constants import MODEL_NAME
import logging


class ModelController:
    def __init__(self):
        self.ready = False
        self.model = None
        self.quora_autocomplete_strategy: Optional[BaseStrategy] = None
        self.question_answer_strategy: Optional[BaseStrategy] = None
        self.summarisation_strategy: Optional[BaseStrategy] = None

    async def initialise_model_and_strategies(self):
        self.model = await self.load_model(MODEL_NAME)
        logging.info('Model loaded')
        
        strategies = await asyncio.gather(
            StrategyFactory.load_strategy(self.model, "quora_autocomplete"),
            StrategyFactory.load_strategy(self.model, "question_answer"),
            StrategyFactory.load_strategy(self.model, "summarisation"),
        )
        logging.info('Strategies loaded')
        self.quora_autocomplete_strategy, self.question_answer_strategy, self.summarisation_strategy = strategies
        
        self.ready = True

    async def load_model(self, model_name: str):
        return await asyncio.to_thread(SentenceTransformer, model_name)

    async def process_request(self, strategy, text: str, n_items: int):
        if not self.ready:
            return
        elif strategy is not None:
            return await asyncio.to_thread(strategy.process, text, n_items)

    async def get_summarised_text(self, text: str, n_sentences: int):
        return await self.process_request(self.summarisation_strategy, text, n_sentences)

    async def get_answers_to_question(self, text: str, n_answers: int):
        return await self.process_request(self.question_answer_strategy, text, n_answers)

    async def get_autocomplete_suggestions(self, text: str, n_answers: int):
        return await self.process_request(self.quora_autocomplete_strategy, text, n_answers)
