# Smart Text Categorizer API

A FastAPI-based service that intelligently categorizes text messages into predefined categories using a hybrid approach of keyword matching and LLM (Large Language Model) processing.

## Features

- Hybrid categorization system combining rule-based and LLM approaches
- Smart batching for efficient processing
- LRU caching for improved performance
- Fallback mechanisms for reliability
- RESTful API endpoints
- Configurable categories and confidence thresholds

## Requirements

- Python 3.8+
- OpenAI API key
- FastAPI
- Poetry (recommended for dependency management)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart-categorizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install fastapi uvicorn python-dotenv openai pydantic
```

4. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, view the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

#### POST /categorize
Categorizes a list of messages into predefined categories.

Example request:
```json
{
  "messages": [
    "I need help with my billing",
    "Can you recommend a good product?"
  ],
  "categories": [
    "billing",
    "recommendation request",
    "account issue",
    "feedback",
    "unclear_or_ambiguous"
  ]
}
```
