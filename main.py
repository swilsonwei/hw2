import os
import asyncio
import mmh3
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import openai
from pymilvus import MilvusClient, Collection, CollectionSchema, FieldSchema, DataType, utility
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-4efcec782ae2f4c.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = "code_chunks"  # Use existing collection

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default=[], description="Conversation history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    sources: List[Dict[str, Any]] = Field(default=[], description="RAG sources")

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to add")
    metadata: str = Field(default="", description="Optional metadata for the document")

# Milvus client
milvus_client = None

def connect_to_milvus():
    """Connect to Milvus database using MilvusClient."""
    global milvus_client
    try:
        milvus_client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        print("Connected to Milvus successfully")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

def setup_milvus_collection():
    """Setup Milvus collection for storing embeddings."""
    global milvus_client
    if not milvus_client:
        return
        
    try:
        # Check if collection exists
        collections = milvus_client.list_collections()
        if COLLECTION_NAME in collections:
            print(f"Collection {COLLECTION_NAME} already exists")
            return
        
        # Create collection with schema for dense vectors
        schema = {
            "fields": [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "chunk_id", "dtype": "VARCHAR", "max_length": 256},
                {"name": "file_path", "dtype": "VARCHAR", "max_length": 512},
                {"name": "chunk_type", "dtype": "VARCHAR", "max_length": 50},
                {"name": "chunk_name", "dtype": "VARCHAR", "max_length": 256},
                {"name": "source_code", "dtype": "VARCHAR", "max_length": 65535},
                {"name": "embedding_vector", "dtype": "FLOAT_VECTOR", "dim": 1536},
                {"name": "metadata", "dtype": "JSON"}
            ],
            "description": "Code chunks with embeddings"
        }
        
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema
        )
        
        # Create index for embedding vector
        milvus_client.create_index(
            collection_name=COLLECTION_NAME,
            field_name="embedding_vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        
        print(f"Collection {COLLECTION_NAME} created successfully with dense vector index")
        
    except Exception as e:
        print(f"Failed to setup Milvus collection: {e}")

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    if connect_to_milvus():
        setup_milvus_collection()
    yield
    # Shutdown
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="SmartInfo RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Utility functions
async def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Get dense embedding for text using OpenAI."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []



async def search_sparse_documents(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar documents using dense vectors with different parameters to simulate sparse search."""
    global milvus_client
    if not milvus_client:
        return []
        
    try:
        # Get dense query embedding
        query_dense_embedding = await get_embedding(query, model="text-embedding-3-small")
        if not query_dense_embedding:
            return []

        # Search in Milvus using dense vectors with different parameters to simulate sparse search
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_dense_embedding],
            anns_field="embedding_vector",
            limit=limit,
            output_fields=["chunk_id", "file_path", "chunk_type", "chunk_name", "source_code", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 5}}  # Different parameters for variety
        )
        
        sources = []
        for hits in results:
            for hit in hits:
                sources.append({
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "file_path": hit.entity.get("file_path", ""),
                    "chunk_type": hit.entity.get("chunk_type", ""),
                    "chunk_name": hit.entity.get("chunk_name", ""),
                    "source_code": hit.entity.get("source_code", ""),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.score,
                    "search_type": "sparse_simulated"
                })
        
        return sources
        
    except Exception as e:
        print(f"Error searching sparse documents: {e}")
        return []

async def search_dense_documents(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar documents using dense vectors in Milvus."""
    global milvus_client
    if not milvus_client:
        return []
        
    try:
        # Get dense query embedding
        query_dense_embedding = await get_embedding(query, model="text-embedding-3-large")
        if not query_dense_embedding:
            return []

        # Search in Milvus using dense vectors
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_dense_embedding],
            anns_field="embedding_vector",
            limit=limit,
            output_fields=["chunk_id", "file_path", "chunk_type", "chunk_name", "source_code", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        
        sources = []
        for hits in results:
            for hit in hits:
                sources.append({
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "file_path": hit.entity.get("file_path", ""),
                    "chunk_type": hit.entity.get("chunk_type", ""),
                    "chunk_name": hit.entity.get("chunk_name", ""),
                    "source_code": hit.entity.get("source_code", ""),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.score,
                    "search_type": "dense"
                })
        
        return sources
        
    except Exception as e:
        print(f"Error searching dense documents: {e}")
        return []

async def rerank_results_with_openai(query: str, combined_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank combined results using OpenAI to get the top 5 most relevant results."""
    if not client or not combined_results:
        return combined_results[:5]  # Return top 5 if no reranking possible
    
    try:
        # Prepare context for reranking
        context_parts = []
        for i, result in enumerate(combined_results):
            context_parts.append(f"""
Result {i+1}:
- File: {result.get('file_path', 'Unknown')}
- Type: {result.get('chunk_type', 'Unknown')}
- Name: {result.get('chunk_name', 'Unknown')}
- Code: {result.get('source_code', '')[:500]}...
- Sparse Score: {result.get('score', 0)}
- Search Type: {result.get('search_type', 'Unknown')}
""")
        
        context = "\n".join(context_parts)
        
        # Create reranking prompt
        rerank_prompt = f"""
Given the following search query and code search results, please rerank the results by relevance to the query.
Return only the top 5 most relevant results in order of relevance.

Search Query: "{query}"

Search Results:
{context}

Please analyze each result and return a JSON array with the indices of the top 5 most relevant results (0-indexed), ordered by relevance.
Only return the JSON array, nothing else.

Example format: [3, 1, 0, 4, 2]
"""
        
        # Get reranking from OpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": rerank_prompt}],
            max_tokens=100,
            temperature=0.1
        )
        
        # Parse the reranking response
        try:
            rerank_indices = json.loads(response.choices[0].message.content.strip())
            if isinstance(rerank_indices, list) and len(rerank_indices) <= 5:
                # Apply reranking
                reranked_results = []
                for idx in rerank_indices:
                    if 0 <= idx < len(combined_results):
                        reranked_results.append(combined_results[idx])
                return reranked_results[:5]  # Ensure only top 5
        except (json.JSONDecodeError, IndexError):
            pass
        
        # Fallback: return top 5 by score
        return sorted(combined_results, key=lambda x: x.get('score', 0), reverse=True)[:5]
        
    except Exception as e:
        print(f"Error reranking results: {e}")
        # Fallback: return top 5 by score
        return sorted(combined_results, key=lambda x: x.get('score', 0), reverse=True)[:5]

async def search_and_rerank_documents(query: str) -> List[Dict[str, Any]]:
    """Search using both sparse and dense vectors, combine results, and rerank to get top 5."""
    try:
        # Search with sparse vectors
        sparse_results = await search_sparse_documents(query, limit=10)
        print(f"Found {len(sparse_results)} sparse results")
        
        # Search with dense vectors
        dense_results = await search_dense_documents(query, limit=10)
        print(f"Found {len(dense_results)} dense results")
        
        # Combine results (remove duplicates based on chunk_id)
        combined_results = []
        seen_chunk_ids = set()
        
        # Add sparse results first
        for result in sparse_results:
            chunk_id = result.get('chunk_id', '')
            if chunk_id not in seen_chunk_ids:
                combined_results.append(result)
                seen_chunk_ids.add(chunk_id)
        
        # Add dense results (avoiding duplicates)
        for result in dense_results:
            chunk_id = result.get('chunk_id', '')
            if chunk_id not in seen_chunk_ids:
                combined_results.append(result)
                seen_chunk_ids.add(chunk_id)
        
        print(f"Combined {len(combined_results)} unique results")
        
        # Rerank results using OpenAI to get top 5
        top_5_results = await rerank_results_with_openai(query, combined_results)
        print(f"Reranked to {len(top_5_results)} top results")
        
        return top_5_results
        
    except Exception as e:
        print(f"Error in search and rerank: {e}")
        return []

async def chat_with_gpt(message: str, conversation_history: List[ChatMessage], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Chat with GPT using conversation history and optional RAG sources."""
    if not client:
        return "OpenAI API key not configured."
    try:
        # Debug logging
        print(f"DEBUG: chat_with_gpt called with {len(sources) if sources else 0} sources")
        if sources:
            for i, s in enumerate(sources[:3]):
                print(f"DEBUG: Source {i}: score={s.get('score', 0)}, type={s.get('chunk_type', 'Unknown')}")
        
        # Check if we have meaningful code context (high relevance scores)
        meaningful_sources = [s for s in sources if s.get('score', 0) > 0.3] if sources else []
        print(f"DEBUG: Found {len(meaningful_sources)} meaningful sources (score > 0.3)")
        
        if meaningful_sources:
            print("DEBUG: Using code context mode")
            # We have relevant code context, enhance the response
            system_message = (
                "You are a helpful AI assistant for SmartInfo. You are a GENERAL LLM that can answer ANY question. "
                "You have access to relevant code context that can enhance your response. "
                "If the user asks a general question, provide a comprehensive answer. "
                "If the user asks about code or programming, use the available code context to enhance your response."
            )
            context_parts = []
            for i, source in enumerate(meaningful_sources[:3], 1):
                context_parts.append(f"""
Code Source {i}:
- File: {source.get('file_path', 'Unknown')}
- Type: {source.get('chunk_type', 'Unknown')}
- Name: {source.get('chunk_name', 'Unknown')}
- Code: {source.get('source_code', '')}
- Relevance Score: {source.get('score', 0)}
""")
            context = "\n".join(context_parts)
            system_message += f"\n\n<Code Context>:\n{context}"
            # Prepare messages as before
            messages = [{"role": "system", "content": system_message}]
            for msg in conversation_history[-10:]:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": message})
        else:
            print("DEBUG: Using general LLM mode (no code context)")
            # No meaningful code context found, act as pure general LLM with NO code context
            system_message = (
                "You are a helpful AI assistant for SmartInfo. You are a GENERAL LLM that can answer ANY question - "
                "whether it's about history, science, general knowledge, or anything else. "
                "Provide comprehensive and accurate answers to any question the user asks. "
                "IMPORTANT: You are NOT limited to code questions. You can answer questions about presidents, history, science, math, literature, or any other topic. "
                "Do not mention code or programming unless specifically asked. "
                "CRITICAL: If the user asks about presidents, history, science, or any general knowledge topic, provide a detailed and accurate answer immediately."
            )
            messages = [{"role": "system", "content": system_message}]
            for msg in conversation_history[-10:]:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": message})
        # Call OpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error chatting with GPT: {e}")
        return "I apologize, but I'm having trouble processing your request right now."

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint that combines RAG search with LLM response."""
    try:
        # Search for relevant documents
        sources = await search_and_rerank_documents(request.message)
        print(f"DEBUG: Found {len(sources)} total sources from search")
        
        # Filter for meaningful sources only (score > 0.3)
        meaningful_sources = [s for s in sources if s.get('score', 0) > 0.3] if sources else []
        print(f"DEBUG: Found {len(meaningful_sources)} meaningful sources (score > 0.3)")
        
        # Only pass meaningful sources to the LLM, otherwise pass None
        sources_for_llm = meaningful_sources if meaningful_sources else None
        print(f"DEBUG: Passing {len(sources_for_llm) if sources_for_llm else 0} sources to chat_with_gpt")
        
        # Get response from LLM
        response = await chat_with_gpt(request.message, request.conversation_history, sources_for_llm)
        
        return ChatResponse(
            response=response,
            sources=meaningful_sources  # Return only meaningful sources in response
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(
            response="I apologize, but I'm having trouble processing your request right now.",
            sources=[]
        )

@app.post("/search")
async def search_code(request: dict):
    """Search for code in the codebase using both sparse and dense vectors with reranking."""
    try:
        query = request.get("query", "")
        if not query:
            return {"results": []}
        
        # Use the new search and rerank function
        reranked_results = await search_and_rerank_documents(query)
        
        # Format results for frontend
        results = []
        for result in reranked_results:
            results.append({
                "chunk_name": result.get("chunk_name", "Code Chunk"),
                "file_path": result.get("file_path", "Unknown"),
                "source_code": result.get("source_code", ""),
                "score": result.get("score", 0),
                "search_type": result.get("search_type", "Unknown")
            })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced-response")
async def enhanced_response(request: dict):
    """Generate an enhanced response using both general knowledge and code context."""
    try:
        query = request.get("query", "")
        code_chunks = request.get("code_chunks", [])
        
        if not query:
            return {"response": "Please provide a question or query."}
        
        # Build context from code chunks
        code_context = ""
        if code_chunks:
            code_context = "\n\nRelevant code from your codebase:\n"
            for i, chunk in enumerate(code_chunks[:3], 1):  # Limit to first 3 chunks
                code_context += f"\n--- Code Chunk {i} ---\n"
                code_context += f"File: {chunk.get('file_path', 'Unknown')}\n"
                code_context += f"Name: {chunk.get('chunk_name', 'Code Chunk')}\n"
                code_context += f"Code:\n{chunk.get('source_code', '')}\n"
        
        # Create enhanced prompt
        if code_chunks:
            enhanced_prompt = f"""
The user asked: "{query}"

{code_context}

Please provide a comprehensive answer that:
1. Addresses the user's question directly
2. Incorporates relevant information from the code chunks above when applicable
3. Explains how the code relates to their question
4. Provides additional context and insights
5. If the question is not code-related, still provide a helpful general answer

Focus on being helpful and informative, using the code context to enhance your response when relevant.
"""
        else:
            enhanced_prompt = f"""
The user asked: "{query}"

Please provide a comprehensive, accurate, and helpful answer to this question. Be informative and clear in your response.
"""
        
        # Get enhanced response from GPT
        response = await chat_with_gpt(enhanced_prompt, [], None)
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_code(request: dict):
    """Generate a summary of code chunks based on the search query."""
    try:
        query = request.get("query", "")
        code_chunks = request.get("code_chunks", [])
        
        if not query or not code_chunks:
            return {"summary": "No code chunks found to summarize."}
        
        # Prepare context for GPT
        context = f"Query: {query}\n\nCode chunks found:\n"
        for i, chunk in enumerate(code_chunks[:3], 1):  # Limit to first 3 chunks
            context += f"\n--- Chunk {i} ---\n"
            context += f"File: {chunk.get('file_path', 'Unknown')}\n"
            context += f"Name: {chunk.get('chunk_name', 'Code Chunk')}\n"
            context += f"Code:\n{chunk.get('source_code', '')}\n"
        
        # Create prompt for summary
        prompt = f"""
Based on the following search query and code chunks, provide a concise summary explaining what the code does and how it relates to the query.

{context}

Please provide a clear, concise summary that explains:
1. What the code does
2. How it relates to the search query
3. Key functions or classes found
4. Any important patterns or concepts

Summary:
"""
        
        # Get summary from GPT
        response = await chat_with_gpt(prompt, [], None)
        
        return {"summary": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    global milvus_client
    return {"status": "healthy", "milvus_connected": milvus_client is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)