import asyncio
from sentence_transformers import SentenceTransformer
from Constants import QuestionAnswerStrategyData, QuoraAutocompleteStrategyData
from QuestionAnswerStrategy import QuestionAnswerStrategy
from QuoraAutocompleteStrategy import QuoraAutocompleteStrategy
from SummarisationStrategy import SummarisationStrategy


class StrategyFactory:
    
    @staticmethod
    async def load_strategy(model: SentenceTransformer, strategy_name: str):
        if strategy_name == "quora_autocomplete":
            return await asyncio.to_thread(QuoraAutocompleteStrategy, model, QuoraAutocompleteStrategyData)
        elif strategy_name == "question_answer":
            return await asyncio.to_thread(QuestionAnswerStrategy, model, QuestionAnswerStrategyData)
        elif strategy_name == "summarisation":
            return await asyncio.to_thread(SummarisationStrategy, model)