#!/usr/bin/env python3
"""
Load Data Script for SmartOrg
Loads Python files from a GitHub repository into a Milvus sparse vector index.
"""

import os
import ast
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
import git
import openai
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import hashlib
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeChunker:
    """Handles extraction and chunking of Python code files."""
    
    def __init__(self):
        self.chunks = []
    
    def extract_functions_and_classes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract functions and classes from a Python file using AST.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of dictionaries containing function/class information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            extracted_items = []
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Get the source code for this node
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    # Extract the actual source code
                    lines = content.split('\n')
                    node_source = '\n'.join(lines[start_line-1:end_line])
                    
                    item = {
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'name': node.name,
                        'source_code': node_source,
                        'start_line': start_line,
                        'end_line': end_line,
                        'file_path': file_path
                    }
                    extracted_items.append(item)
            
            return extracted_items
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []
    
    def chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk a Python file by extracting the entire file content and individual functions/classes.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        try:
            # Read the entire file
            with open(file_path, 'r', encoding='utf-8') as file:
                full_content = file.read()
            
            # Create a chunk for the entire file
            file_chunk = {
                'type': 'file',
                'name': Path(file_path).name,
                'source_code': full_content,
                'file_path': file_path,
                'chunk_id': f"file_{Path(file_path).stem}"
            }
            chunks.append(file_chunk)
            
            # Extract functions and classes
            extracted_items = self.extract_functions_and_classes(file_path)
            
            for item in extracted_items:
                chunk = {
                    'type': item['type'],
                    'name': item['name'],
                    'source_code': item['source_code'],
                    'file_path': file_path,
                    'start_line': item['start_line'],
                    'end_line': item['end_line'],
                    'chunk_id': f"{item['type']}_{Path(file_path).stem}_{item['name']}"
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {e}")
            return []

class GitHubRepoLoader:
    """Handles cloning and loading of GitHub repositories."""
    
    def __init__(self, github_pat: str):
        self.github_pat = github_pat
    
    def clone_repo(self, repo_url: str, local_path: str) -> bool:
        """
        Clone a GitHub repository using the provided PAT.
        
        Args:
            repo_url: GitHub repository URL
            local_path: Local path to clone the repository
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Clone the repository
            logger.info(f"Cloning repository: {repo_url}")
            git.Repo.clone_from(
                repo_url.replace('https://', f'https://{self.github_pat}@'),
                local_path
            )
            logger.info(f"Successfully cloned repository to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    
    def get_python_files(self, repo_path: str) -> List[str]:
        """
        Get all Python files from the repository.
        
        Args:
            repo_path: Path to the cloned repository
            
        Returns:
            List of Python file paths
        """
        python_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories that shouldn't be processed
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(python_files)} Python files")
        return python_files

class EmbeddingGenerator:
    """Generates embeddings using OpenAI's API."""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generate embedding for the given text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Dictionary containing embedding data
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding data
            embedding_data = response.data[0].embedding
            
            return {
                'embedding': embedding_data,
                'text': text
            }
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

class MilvusManager:
    """Manages Milvus connection and collection operations."""
    
    def __init__(self, uri: str, token: str = None):
        self.uri = uri
        self.token = token
        self.collection_name = "code_chunks"
    
    def connect(self) -> bool:
        """
        Connect to Milvus instance.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token if self.token else ""
            )
            logger.info("Successfully connected to Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            return False
    
    def create_collection(self) -> bool:
        """
        Create a sparse vector collection for code chunks.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="source_code", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields, description="Code chunks with sparse embeddings")
            
            # Create collection
            collection = Collection(self.collection_name, schema)
            
            # Create index for embedding vector field
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding_vector", index_params)
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def upsert_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Upsert data into the collection.
        
        Args:
            data: List of dictionaries containing chunk data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = Collection(self.collection_name)
            collection.load()
            
            # Prepare data for insertion
            insert_data = []
            for chunk in data:
                insert_data.append([
                    chunk['chunk_id'],
                    chunk['file_path'],
                    chunk['type'],
                    chunk['name'],
                    chunk['source_code'],
                    chunk['embedding'],
                    json.dumps(chunk.get('metadata', {}))
                ])
            
            # Insert data
            collection.insert(insert_data)
            collection.flush()
            
            logger.info(f"Successfully upserted {len(data)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting data: {e}")
            return False

def main():
    """Main function to orchestrate the data loading process."""
    
    # Load environment variables
    github_pat = os.getenv('GITHUB_PAT')
    milvus_uri = os.getenv('MILVUS_URI', 'localhost:19530')
    milvus_token = os.getenv('MILVUS_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Validate required environment variables
    if not github_pat:
        logger.error("GITHUB_PAT not found in environment variables")
        return
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Configuration
    repo_url = "https://github.com/swilsonwei/hw2.git"  # GitHub repository URL
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Use current directory instead of cloning
        logger.info("Step 1: Using current directory for Python files")
        current_dir = os.getcwd()
        
        # Step 2: Extract and chunk Python files
        logger.info("Step 2: Extracting and chunking Python files")
        python_files = []
        for root, dirs, files in os.walk(current_dir):
            # Skip common directories that shouldn't be processed
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        chunker = CodeChunker()
        
        all_chunks = []
        for file_path in python_files:
            chunks = chunker.chunk_file(file_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(python_files)} files")
        
        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings")
        embedding_generator = EmbeddingGenerator(openai_api_key)
        
        chunks_with_embeddings = []
        for chunk in all_chunks:
            # Generate embedding for the source code
            embedding = embedding_generator.generate_embedding(chunk['source_code'])
            if embedding:
                chunk['embedding'] = embedding['embedding']
                chunk['metadata'] = {
                    'start_line': chunk.get('start_line'),
                    'end_line': chunk.get('end_line'),
                    'file_size': len(chunk['source_code'])
                }
                chunks_with_embeddings.append(chunk)
        
        logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
        
        # Step 4: Connect to Milvus
        logger.info("Step 4: Connecting to Milvus")
        milvus_manager = MilvusManager(milvus_uri, milvus_token)
        if not milvus_manager.connect():
            logger.error("Failed to connect to Milvus")
            return
        
        # Step 5: Create collection if it doesn't exist
        logger.info("Step 5: Creating/checking collection")
        if not utility.has_collection(milvus_manager.collection_name):
            if not milvus_manager.create_collection():
                logger.error("Failed to create collection")
                return
        
        # Step 6: Upsert data into Milvus
        logger.info("Step 6: Upserting data into Milvus")
        if not milvus_manager.upsert_data(chunks_with_embeddings):
            logger.error("Failed to upsert data")
            return
        
        logger.info("Data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary directory")

if __name__ == "__main__":
    main() 