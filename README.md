# Sentence Similarity API

This is a Python application that provides an API for calculating the the semantic similarity between sentences and paragraphs.
There are several strategies, each serving a different use case.

This API uses the SentenceTransformer library, with an underlying SBERT (Sentence-BERT) model to perform various functions. One
strategy uses the model to summarise paragraphs into a given number of sentences by choosing the most important sentences. 
The other two strategies find similar texts from
databases: a Quora database of questions, and a database of SimpleWiki entries. These are intended to show functionality similar to
the autocomplete functions on websites, and AI suggested answers to search prompts.

The SentenceTransformer model creates embeddings of each sentence. Central to each strategy is the use of cosine similarity to find
semantically similar sentences and texts to those provided to the API. Examples of usage are shown below, and you are welcome to 
clone the repository and attempt yourself.

## Installation

1. Clone the repository:

```
git clone https://github.com/James-Tipping/Sentence-Transformers-App.git
```

2. Navigate to the project directory:

```
cd "Sentence Similarity API"
```

3. Install the required dependencies and activate environment:

```
conda env create -f environment.yml
```

If you are not running on Apple Silicon architecture, a platform independent dependency list has been created. Run:

```
conda env create -f environment_portable.yml
```

Then activate the required environment

```
conda activate pytorch
```

It is strongly recommended to use conda to install dependencies, particularly as some libraries such as h5py have native dependencies that
are troublesome to install with pip.

There is also an included Dockerfile which may aid in setting up the API. This has been created primarily to allow the API to run on Google Cloud Run. As such, no container is provided at this time.

## Usage

### 1. Start the API server:

```
python -m app.main
```

### 2. Send a POST request to the `/nlp/summarise-text` endpoint with a sentence in the request body.

```
POST http://0.0.0.0:8000/nlp/summarise-text
Content-Type: application/json

{
    "text": "There will be no extra NHS funding without reform, Sir Keir Starmer says, as he promised to draw up a new 10-year plan for the health service. The pledge came after a damning report warned the NHS in England was in a \"critical condition\". The prime minister said the new plan, expected to be published in the spring, would be the \"the biggest reimagining of the NHS\" since it was formed.",
    "n_answers": 1
}
```

where `n_answers` refers to the number of sentences to be returned in the summarised text.
The API will respond with a list of sentences most central to the overall meaning of the text.

```
[
    "The prime minister said the new plan, expected to be published in the spring, would be the \"the biggest reimagining of the NHS\" since it was formed."
]
```

### 3. Send a POST request to the `/nlp/question-answer-retrieval` endpoint 

Requests to this endpoint search a SimpleWiki database for semantically similar texts. This is similar to Google's AI suggested
answers to questions in Google search.

```
POST http://0.0.0.0:8000/nlp/question-answer-retrieval
Content-Type: application/json

{
    "text": "What is the capital of China?",
    "n_answers": 6
}
```

where `question` is the question you wish to search for in the SimpleWiki database and `n_answers` is the number of semantically similar
texts to be returned as answers.

The API will respond with a list of texts that are semantically similar to the given question, in this format: 

```
[
    {
        "title": "Beijing",
        "answer": "Beijing is the capital of the People's Republic of China. The city used to be known as Peking. It is in the northern and eastern parts of the country. It is the world's most populous capital city.",
        "score": "0.699"
    },
    {
        "title": "China, Texas",
        "answer": "China is a city in the U.S. state of Texas.",
        "score": "0.640"
    },
    ...
]
```

Where `title` refers to the article from which the paragraph was taken, `answer` refers to the paragraph, and `score` refers to the cosine
similarity score between the queried text and the answer.

Feel free to experiment with different questions and adjust the `n_answers` parameter to get the desired number of results.


### 4. Send a POST request to the `/nlp/quora-autocomplete` endpoint

Requests sent to this enpoint search a Quora database of questions to find semantically similar questions. This is similar to the autocomplete
function found on many websites.

```
POST http://0.0.0.0:8000/nlp/quora-autocomplete
Content-Type: application/json

{
    "text": "What's the biggest city in the world?",
    "n_answers": 5
}
```

The response is similar to before: 

```
[
    {
        "question": "Which is biggest city in the world?",
        "score": "0.978"
    },
    {
        "question": "What's the world's largest city?",
        "score": "0.970"
    },
    {
        "question": "Which is the largest city in the world?",
        "score": "0.961"
    },
    ...
]
```

## Contributing

This is a small project intended for personal development. However, if you wish to help develop this further, you are welcome to! Likewise, 
if any issues are found with the setup or execution of the program, please add an issue.

## Licence

This project is licensed under the MIT Licence.
