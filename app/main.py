import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from .constants import RequestStructure
from .model_controller import ModelController

import uvicorn


model_controller = ModelController()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_controller.initialise_model_and_strategies())
    
async def handle_request(text: str, n_answers: int, handler):
    if not text:
        raise HTTPException(status_code=400, detail="Provided query is necessary for model to work")
    
    if not model_controller.ready:
        return JSONResponse(status_code=503, content={"detail": "Model is not ready yet"}, headers={"Retry-After": "10"})
    
    try:
        answers = await handler(text, n_answers)
    except ValueError as e:
        logging.error(f"Error retrieving data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    else:
        return answers

@app.post("/nlp/summarise-text")
async def summarise_text(request: RequestStructure):
    
    return await handle_request(request.text, request.n_answers, model_controller.get_summarised_text)
    
@app.post("/nlp/question-answer-retrieval")
async def get_answer_to_question(request: RequestStructure):
    
    return await handle_request(request.text, request.n_answers, model_controller.get_answers_to_question)
    
@app.post("/nlp/quora-autocomplete")
async def get_autocomplete_suggestions(request: RequestStructure):
    
    return await handle_request(request.text, request.n_answers, model_controller.get_autocomplete_suggestions)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)