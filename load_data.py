#!/usr/bin/env python3
"""
Load Data Script for SmartInfo
Loads Python files from a GitHub repository into a Milvus sparse vector index.
Experiments with different chunking strategies: by file, by function, by line ranges, by semantic blocks.
"""

import os
import ast
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv
import git
import openai
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Different chunking strategies for code analysis."""
    BY_FILE = "by_file"
    BY_FUNCTION = "by_function"
    BY_CLASS = "by_class"
    BY_LINE_RANGE = "by_line_range"
    BY_SEMANTIC_BLOCK = "by_semantic_block"
    BY_IMPORT_BLOCK = "by_import_block"
    BY_DOCSTRING = "by_docstring"

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    chunk_id: str
    chunk_type: str
    name: str
    source_code: str
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class AdvancedCodeChunker:
    """Advanced code chunking with multiple strategies."""
    
    def __init__(self):
        self.chunks = []
    
    def chunk_by_file(self, file_path: str) -> List[CodeChunk]:
        """Chunk by entire file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunk = CodeChunk(
                chunk_id=f"file_{Path(file_path).stem}",
                chunk_type="file",
                name=Path(file_path).name,
                source_code=content,
                file_path=file_path,
                metadata={
                    'file_size': len(content),
                    'line_count': len(content.split('\n')),
                    'strategy': ChunkingStrategy.BY_FILE.value
                }
            )
            return [chunk]
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            return []
    
    def chunk_by_function(self, file_path: str) -> List[CodeChunk]:
        """Chunk by individual functions and methods."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    lines = content.split('\n')
                    node_source = '\n'.join(lines[start_line-1:end_line])
                    
                    chunk = CodeChunk(
                        chunk_id=f"func_{Path(file_path).stem}_{node.name}",
                        chunk_type="function",
                        name=node.name,
                        source_code=node_source,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={
                            'is_async': isinstance(node, ast.AsyncFunctionDef),
                            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                            'strategy': ChunkingStrategy.BY_FUNCTION.value
                        }
                    )
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking functions in {file_path}: {e}")
            return []
    
    def chunk_by_class(self, file_path: str) -> List[CodeChunk]:
        """Chunk by individual classes."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    lines = content.split('\n')
                    node_source = '\n'.join(lines[start_line-1:end_line])
                    
                    chunk = CodeChunk(
                        chunk_id=f"class_{Path(file_path).stem}_{node.name}",
                        chunk_type="class",
                        name=node.name,
                        source_code=node_source,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={
                            'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                            'strategy': ChunkingStrategy.BY_CLASS.value
                        }
                    )
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking classes in {file_path}: {e}")
            return []
    
    def chunk_by_line_range(self, file_path: str, chunk_size: int = 50, overlap: int = 10) -> List[CodeChunk]:
        """Chunk by fixed line ranges with overlap."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            chunks = []
            total_lines = len(lines)
            
            for i in range(0, total_lines, chunk_size - overlap):
                end_line = min(i + chunk_size, total_lines)
                chunk_lines = lines[i:end_line]
                chunk_content = ''.join(chunk_lines)
                
                chunk = CodeChunk(
                    chunk_id=f"line_{Path(file_path).stem}_{i}_{end_line}",
                    chunk_type="line_range",
                    name=f"Lines {i+1}-{end_line}",
                    source_code=chunk_content,
                    file_path=file_path,
                    start_line=i+1,
                    end_line=end_line,
                    metadata={
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'strategy': ChunkingStrategy.BY_LINE_RANGE.value
                    }
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking by line range in {file_path}: {e}")
            return []
    
    def chunk_by_semantic_block(self, file_path: str) -> List[CodeChunk]:
        """Chunk by semantic blocks (imports, functions, classes, etc.)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunks = []
            lines = content.split('\n')
            current_block = []
            current_block_type = None
            current_start_line = 1
            
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                
                # Determine block type
                if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                    block_type = 'import'
                elif stripped_line.startswith('def ') or stripped_line.startswith('async def '):
                    block_type = 'function'
                elif stripped_line.startswith('class '):
                    block_type = 'class'
                elif stripped_line.startswith('#'):
                    block_type = 'comment'
                elif stripped_line == '':
                    block_type = 'empty'
                else:
                    block_type = 'code'
                
                # Start new block if type changes
                if current_block_type and current_block_type != block_type and current_block:
                    chunk_content = '\n'.join(current_block)
                    chunk = CodeChunk(
                        chunk_id=f"semantic_{Path(file_path).stem}_{current_start_line}_{i}",
                        chunk_type=f"semantic_{current_block_type}",
                        name=f"{current_block_type.title()} Block",
                        source_code=chunk_content,
                        file_path=file_path,
                        start_line=current_start_line,
                        end_line=i,
                        metadata={
                            'block_type': current_block_type,
                            'strategy': ChunkingStrategy.BY_SEMANTIC_BLOCK.value
                        }
                    )
                    chunks.append(chunk)
                    current_block = []
                    current_start_line = i + 1
                
                current_block.append(line)
                current_block_type = block_type
            
            # Add final block
            if current_block:
                chunk_content = '\n'.join(current_block)
                chunk = CodeChunk(
                    chunk_id=f"semantic_{Path(file_path).stem}_{current_start_line}_{len(lines)}",
                    chunk_type=f"semantic_{current_block_type}",
                    name=f"{current_block_type.title()} Block",
                    source_code=chunk_content,
                    file_path=file_path,
                    start_line=current_start_line,
                    end_line=len(lines),
                    metadata={
                        'block_type': current_block_type,
                        'strategy': ChunkingStrategy.BY_SEMANTIC_BLOCK.value
                    }
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking by semantic block in {file_path}: {e}")
            return []
    
    def chunk_by_import_block(self, file_path: str) -> List[CodeChunk]:
        """Extract import blocks separately."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunks = []
            lines = content.split('\n')
            import_lines = []
            
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                    import_lines.append((i+1, line))
            
            if import_lines:
                import_content = '\n'.join([line for _, line in import_lines])
                chunk = CodeChunk(
                    chunk_id=f"imports_{Path(file_path).stem}",
                    chunk_type="import_block",
                    name="Import Statements",
                    source_code=import_content,
                    file_path=file_path,
                    start_line=import_lines[0][0],
                    end_line=import_lines[-1][0],
                    metadata={
                        'import_count': len(import_lines),
                        'strategy': ChunkingStrategy.BY_IMPORT_BLOCK.value
                    }
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking imports in {file_path}: {e}")
            return []
    
    def chunk_by_docstring(self, file_path: str) -> List[CodeChunk]:
        """Extract docstrings as separate chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            chunks = []
            
            for node in ast.walk(tree):
                if hasattr(node, 'body') and node.body:
                    for item in node.body:
                        if hasattr(item, 'value') and isinstance(item.value, ast.Str):
                            docstring = item.value.s
                            if docstring.strip():
                                chunk = CodeChunk(
                                    chunk_id=f"docstring_{Path(file_path).stem}_{item.lineno}",
                                    chunk_type="docstring",
                                    name=f"Docstring at line {item.lineno}",
                                    source_code=docstring,
                                    file_path=file_path,
                                    start_line=item.lineno,
                                    end_line=item.lineno,
                                    metadata={
                                        'docstring_length': len(docstring),
                                        'strategy': ChunkingStrategy.BY_DOCSTRING.value
                                    }
                                )
                                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking docstrings in {file_path}: {e}")
            return []
    
    def chunk_file_with_strategies(self, file_path: str, strategies: List[ChunkingStrategy]) -> List[CodeChunk]:
        """Apply multiple chunking strategies to a file."""
        all_chunks = []
        
        for strategy in strategies:
            if strategy == ChunkingStrategy.BY_FILE:
                chunks = self.chunk_by_file(file_path)
            elif strategy == ChunkingStrategy.BY_FUNCTION:
                chunks = self.chunk_by_function(file_path)
            elif strategy == ChunkingStrategy.BY_CLASS:
                chunks = self.chunk_by_class(file_path)
            elif strategy == ChunkingStrategy.BY_LINE_RANGE:
                chunks = self.chunk_by_line_range(file_path)
            elif strategy == ChunkingStrategy.BY_SEMANTIC_BLOCK:
                chunks = self.chunk_by_semantic_block(file_path)
            elif strategy == ChunkingStrategy.BY_IMPORT_BLOCK:
                chunks = self.chunk_by_import_block(file_path)
            elif strategy == ChunkingStrategy.BY_DOCSTRING:
                chunks = self.chunk_by_docstring(file_path)
            else:
                continue
            
            all_chunks.extend(chunks)
            logger.info(f"Applied {strategy.value} strategy to {file_path}: {len(chunks)} chunks")
        
        return all_chunks

class GitHubRepoLoader:
    """Handles cloning and loading of GitHub repositories."""
    
    def __init__(self, github_pat: str):
        self.github_pat = github_pat
    
    def clone_repo(self, repo_url: str, local_path: str) -> bool:
        """Clone a GitHub repository using the provided PAT."""
        try:
            os.makedirs(local_path, exist_ok=True)
            
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
        """Get all Python files from the repository."""
        python_files = []
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(python_files)} Python files")
        return python_files

class SparseEmbeddingGenerator:
    """Generates sparse embeddings using OpenAI's API."""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def generate_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """Generate sparse embedding for the given text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="sparse"
            )
            
            # Extract sparse embedding data
            sparse_data = response.data[0].embedding
            
            return {
                'sparse_embedding': sparse_data,
                'text': text
            }
            
        except Exception as e:
            logger.error(f"Error generating sparse embedding: {e}")
            return None

class MilvusManager:
    """Manages Milvus connection and sparse vector collection operations."""
    
    def __init__(self, uri: str, token: str = None):
        self.uri = uri
        self.token = token
        self.collection_name = "code_chunks_sparse"
    
    def connect(self) -> bool:
        """Connect to Milvus instance."""
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
    
    def create_sparse_collection(self) -> bool:
        """Create a sparse vector collection for code chunks."""
        try:
            # Drop existing collection if it exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
                import time
                time.sleep(1)
            
            # Define collection schema with sparse vector
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="source_code", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields, description="Code chunks with sparse embeddings")
            
            # Create collection
            collection = Collection(self.collection_name, schema)
            
            # Create index for sparse vector field
            index_params = {
                "metric_type": "IP",  # Inner Product for sparse vectors
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2}
            }
            collection.create_index("sparse_vector", index_params)
            
            logger.info(f"Successfully created sparse collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sparse collection: {e}")
            return False
    
    def upsert_sparse_data(self, data: List[Dict[str, Any]]) -> bool:
        """Upsert data with sparse embeddings into the collection."""
        try:
            collection = Collection(self.collection_name)
            collection.load()
            
            # Prepare data for insertion
            chunk_ids = [chunk['chunk_id'] for chunk in data]
            file_paths = [chunk['file_path'] for chunk in data]
            chunk_types = [chunk['chunk_type'] for chunk in data]
            chunk_names = [chunk['name'] for chunk in data]
            source_codes = [chunk['source_code'] for chunk in data]
            sparse_vectors = [chunk['sparse_embedding'] for chunk in data]
            metadatas = [json.dumps(chunk.get('metadata', {})) for chunk in data]
            
            # Insert data
            collection.insert([
                chunk_ids,
                file_paths,
                chunk_types,
                chunk_names,
                source_codes,
                sparse_vectors,
                metadatas
            ])
            collection.flush()
            
            logger.info(f"Successfully upserted {len(data)} sparse chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting sparse data: {e}")
            return False

def main():
    """Main function to orchestrate the data loading process with multiple chunking strategies."""
    
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
    repo_url = "https://github.com/swilsonwei/hw2.git"
    temp_dir = tempfile.mkdtemp()
    
    # Define chunking strategies to experiment with
    chunking_strategies = [
        ChunkingStrategy.BY_FILE,
        ChunkingStrategy.BY_FUNCTION,
        ChunkingStrategy.BY_CLASS,
        ChunkingStrategy.BY_LINE_RANGE,
        ChunkingStrategy.BY_SEMANTIC_BLOCK,
        ChunkingStrategy.BY_IMPORT_BLOCK,
        ChunkingStrategy.BY_DOCSTRING
    ]
    
    try:
        # Step 1: Clone repository or use current directory
        logger.info("Step 1: Setting up repository")
        if os.path.exists('.git'):
            logger.info("Using current directory as repository")
            repo_path = os.getcwd()
        else:
            logger.info("Cloning repository from GitHub")
            repo_loader = GitHubRepoLoader(github_pat)
            if not repo_loader.clone_repo(repo_url, temp_dir):
                logger.error("Failed to clone repository")
                return
            repo_path = temp_dir
        
        # Step 2: Get Python files
        logger.info("Step 2: Finding Python files")
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(python_files)} Python files")
        
        # Step 3: Apply multiple chunking strategies
        logger.info("Step 3: Applying multiple chunking strategies")
        chunker = AdvancedCodeChunker()
        
        all_chunks = []
        strategy_stats = {}
        
        for strategy in chunking_strategies:
            strategy_chunks = []
            for file_path in python_files:
                chunks = chunker.chunk_file_with_strategies(file_path, [strategy])
                strategy_chunks.extend(chunks)
            
            strategy_stats[strategy.value] = len(strategy_chunks)
            all_chunks.extend(strategy_chunks)
            logger.info(f"Strategy {strategy.value}: {len(strategy_chunks)} chunks")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        logger.info(f"Strategy breakdown: {strategy_stats}")
        
        # Step 4: Generate sparse embeddings
        logger.info("Step 4: Generating sparse embeddings")
        embedding_generator = SparseEmbeddingGenerator(openai_api_key)
        
        chunks_with_embeddings = []
        for chunk in all_chunks:
            # Generate sparse embedding for the source code
            embedding = embedding_generator.generate_sparse_embedding(chunk.source_code)
            if embedding:
                chunk_dict = {
                    'chunk_id': chunk.chunk_id,
                    'chunk_type': chunk.chunk_type,
                    'name': chunk.name,
                    'source_code': chunk.source_code,
                    'file_path': chunk.file_path,
                    'sparse_embedding': embedding['sparse_embedding'],
                    'metadata': chunk.metadata or {}
                }
                chunks_with_embeddings.append(chunk_dict)
        
        logger.info(f"Generated sparse embeddings for {len(chunks_with_embeddings)} chunks")
        
        # Step 5: Connect to Milvus
        logger.info("Step 5: Connecting to Milvus")
        milvus_manager = MilvusManager(milvus_uri, milvus_token)
        if not milvus_manager.connect():
            logger.error("Failed to connect to Milvus")
            return
        
        # Step 6: Create sparse collection
        logger.info("Step 6: Creating sparse collection")
        if not milvus_manager.create_sparse_collection():
            logger.error("Failed to create sparse collection")
            return
        
        # Step 7: Upsert data into Milvus
        logger.info("Step 7: Upserting sparse data into Milvus")
        if not milvus_manager.upsert_sparse_data(chunks_with_embeddings):
            logger.error("Failed to upsert sparse data")
            return
        
        logger.info("Sparse vector data loading completed successfully!")
        logger.info(f"Final statistics: {strategy_stats}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary directory")

if __name__ == "__main__":
    main() 