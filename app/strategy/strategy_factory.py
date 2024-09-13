import asyncio
from sentence_transformers import SentenceTransformer
from app.constants import QuestionAnswerStrategyData, QuoraAutocompleteStrategyData
from .question_answer_strategy import QuestionAnswerStrategy
from .quora_autocomplete_strategy import QuoraAutocompleteStrategy
from .summarisation_strategy import SummarisationStrategy


class StrategyFactory:
    
    @staticmethod
    async def load_strategy(model: SentenceTransformer, strategy_name: str):
        if strategy_name == "quora_autocomplete":
            return await asyncio.to_thread(QuoraAutocompleteStrategy, model, QuoraAutocompleteStrategyData)
        elif strategy_name == "question_answer":
            return await asyncio.to_thread(QuestionAnswerStrategy, model, QuestionAnswerStrategyData)
        elif strategy_name == "summarisation":
            return await asyncio.to_thread(SummarisationStrategy, model)