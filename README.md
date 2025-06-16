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
## Optional Follow-up Discussion

### How would you scale this?
#### Current Foundation: 
Caching (lru_cache), Tiered Logic (fallback first), and Smart Batching for LLM calls.
#### Next Steps:
- Distributed Cache: Replace lru_cache with Redis for multi-server caching.
- Message Queue: Use RabbitMQ or SQS for asynchronous processing.
- Persistent Database: Store all results for analytics.

### How would you deal with new categories?
#### Current Foundation: 
The unclear_or_ambiguous category acts as a collection bucket.
#### Discovery Pipeline:
- Cluster: Group unclear messages using embeddings model (as sentence-transformers).
- Review: A human reviews the clusters to identify new themes.
- Promote: Add the new theme as a formal category.
### When would you consider fine-tuning?
Yes, absolutely. Fine-tuning is the logical next step for optimization. I would use the performance of my current tiered system to decide when to fine-tune.
#### Performance Triggers
- High LLM Usage: When the fallback categorizer isn't effective enough.
- Low Accuracy: When the model struggles with key business categories.
- Cost/Speed: To replace the expensive LLM API with a cheaper, faster custom model at scale.
### How would you integrate feedback loops?
A feedback loop is critical for turning our classification system from a static tool into a learning system that improves over time
#### Mechanism
- Store Results: Log every classification in a database.
- Feedback Endpoint: Create an API endpoint (/feedback) to receive corrections.
- Collect Signals: Gather explicit feedback (user clicks "wrong category") and implicit feedback (agent re-categorizes a ticket).
#### Purpose
To create a "ground truth" dataset for monitoring and future fine-tuning.
