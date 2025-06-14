import os
import json
from functools import lru_cache
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()
app = FastAPI(
    title="Smart Text Categorizer API",
    description="Classifies free-form user messages into predefined categories using an LLM.",
    version="1.0.0",
)

# Constants
BATCH_SIZE = 20  # Maximum number of messages to send in one API call
MAX_CACHE_SIZE = 1000  # Maximum number of cached results

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=api_key)

# FastAPI Models
class CategorizationRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1, max_length=50, description="A list of 1 to 50 text messages to categorize.")
    categories: List[str] = Field(
        default=["account issue", "recommendation request", "billing", "feedback", "unclear_or_ambiguous"],
        description="A list of categories to classify messages into."
    )

class CategorizedMessage(BaseModel):
    message: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str

class CategorizationResponse(BaseModel):
    results: List[CategorizedMessage]

# Enhanced fallback categorizer with more keywords and better confidence scoring
def enhanced_fallback_categorizer(message: str, categories: List[str]) -> dict:
    """An enhanced keyword-based categorizer with confidence scoring."""
    message_lower = message.lower()
    
    # Enhanced rules with more keywords and weights
    rules = {
        "billing": {
            "high_confidence": ["invoice", "refund", "payment", "subscription", "bill", "charge"],
            "medium_confidence": ["price", "cost", "paid", "money", "credit card"],
        },
        "account issue": {
            "high_confidence": ["login", "password", "cannot access", "locked out", "reset password"],
            "medium_confidence": ["account", "sign in", "username", "email", "profile"],
        },
        "feedback": {
            "high_confidence": ["excellent", "terrible", "amazing", "awful", "outstanding"],
            "medium_confidence": ["good", "bad", "great", "poor", "love", "hate", "like", "smooth!", "difficult"],
        },
        "recommendation request": {
            "high_confidence": ["recommend", "suggest", "looking for", "what's the best"],
            "medium_confidence": ["advice", "recommendation", "suggestion", "guide", "help"],
        }
    }

    best_match = {
        "category": "unclear_or_ambiguous",
        "confidence": 0.0,
        "matches": 0
    }

    for category, keywords in rules.items():
        if category not in categories:
            continue

        matches_high = sum(1 for kw in keywords["high_confidence"] if kw in message_lower)
        matches_medium = sum(1 for kw in keywords["medium_confidence"] if kw in message_lower)
        
        # Calculate confidence score
        confidence = (matches_high * 0.8 + matches_medium * 0.4)
        total_matches = matches_high + matches_medium

        if confidence > best_match["confidence"]:
            best_match = {
                "category": category,
                "confidence": min(0.8, confidence),  # Cap at 0.8 to reflect it's not LLM
                "matches": total_matches
            }

    return {
        "message": message,
        "category": best_match["category"],
        "confidence": best_match["confidence"],
        "explanation": f"Matched {best_match['matches']} keywords using enhanced fallback rules."
    }

@lru_cache(maxsize=MAX_CACHE_SIZE)
def categorize_message(message: str, categories_key: str) -> dict:
    """
    Attempt to categorize a single message using fallback categorizer.
    Results are automatically cached via lru_cache decorator.
    """
    categories = categories_key.split(',')
    return enhanced_fallback_categorizer(message, categories)

def categorize_with_llm(messages: List[str], categories: List[str]) -> List[dict]:
    """Batch categorize messages using the LLM."""
    messages_text = "\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(messages)])
    
    prompt = f"""
    You are an expert text classifier for a support system.
    Analyze each message and categorize it into one of these categories: {', '.join(categories)}

    Messages to analyze:
    {messages_text}

    For each message, provide a classification in this exact format:
    {{"results": [
        {{"category": "category_name", "confidence": 0.95, "explanation": "explanation text"}}
    ]}}

    Respond with ONLY the JSON object.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        parsed_response = json.loads(response.choices[0].message.content)
        results = parsed_response.get("results", [])
        
        # Add original messages to results and ensure correct structure
        final_results = []
        for msg, result in zip(messages, results):
            final_results.append({
                "message": msg,
                "category": result.get("category", "unclear_or_ambiguous"),
                "confidence": float(result.get("confidence", 0.0)),
                "explanation": result.get("explanation", "No explanation provided")
            })
            
        return final_results
    except Exception as e:
        print(f"ERROR: Batch LLM API call failed: {str(e)}")
        # Fallback to individual processing
        return [enhanced_fallback_categorizer(msg, categories) for msg in messages]

@app.post("/categorize", response_model=CategorizationResponse)
async def categorize_messages(request: CategorizationRequest):
    """
    Efficiently categorizes messages using smart batching:
    1. Try fallback categorizer first for all messages
    2. Batch process remaining low-confidence messages with LLM
    3. Cache all results for future use
    """
    categories_key = ','.join(sorted(request.categories))
    categories = request.categories
    results = []
    llm_needed = []
    
    # First pass: Try fallback categorizer for all messages
    for i, message in enumerate(request.messages):
        # Try to get from cache first (automatic via @lru_cache)
        result = categorize_message(message, categories_key)
        if result["confidence"] < 0.7:
            # Mark for LLM processing
            llm_needed.append((i, message))
            results.append(None)  # Placeholder
        else:
            results.append(result)
    
    # Process LLM-needed messages in batches
    for i in range(0, len(llm_needed), BATCH_SIZE):
        batch = llm_needed[i:i + BATCH_SIZE]
        batch_messages = [msg for _, msg in batch]
        
        try:
            batch_results = categorize_with_llm(batch_messages, categories)
            
            # Update results at original indexes
            for (orig_idx, _), batch_result in zip(batch, batch_results):
                results[orig_idx] = batch_result
        except Exception as e:
            print(f"ERROR: Batch LLM processing failed: {str(e)}")
            # Fallback to individual processing
            for orig_idx, msg in batch:
                results[orig_idx] = enhanced_fallback_categorizer(msg, categories)
    
    return {"results": results}

@app.get("/")
def read_root():
    return {"status": "Smart Text Categorizer API is running."}

