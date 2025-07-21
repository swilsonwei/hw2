# SmartOrg Data Loading Script

This script loads Python files from a GitHub repository into a Milvus sparse vector index for code search and analysis.

## Features

- **GitHub Repository Cloning**: Uses Personal Access Token (PAT) to clone private/public repositories
- **Code Chunking**: Extracts entire files and individual functions/classes using AST parsing
- **Sparse Embeddings**: Generates sparse embeddings using OpenAI's text-embedding-3-small model
- **Milvus Integration**: Stores code chunks with sparse vectors in a Milvus collection
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Prerequisites

1. **Python 3.8+**
2. **Milvus Instance**: Running Milvus server (local or cloud)
3. **OpenAI API Key**: For generating sparse embeddings
4. **GitHub Personal Access Token**: For repository access

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
# GitHub Personal Access Token
GITHUB_PAT=your_github_pat_here

# Milvus Connection Parameters
MILVUS_URI=localhost:19530
MILVUS_TOKEN=your_milvus_token_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. **Update the repository URL** in `load_data.py`:
```python
repo_url = "https://github.com/your-username/your-repo"
```

2. **Run the script**:
```bash
python load_data.py
```

## How It Works

### Step 1: Repository Cloning
- Uses GitPython to clone the specified GitHub repository
- Authenticates using the provided PAT
- Creates a temporary directory for processing

### Step 2: Code Extraction and Chunking
- Walks through the repository to find all `.py` files
- Skips common directories (`.git`, `__pycache__`, `node_modules`, etc.)
- For each Python file:
  - Creates a chunk for the entire file
  - Extracts individual functions and classes using AST parsing
  - Generates unique chunk IDs for each piece

### Step 3: Sparse Embedding Generation
- Uses OpenAI's `text-embedding-3-small` model with sparse encoding
- Generates embeddings for each code chunk
- Handles API rate limiting and errors gracefully

### Step 4: Milvus Integration
- Connects to Milvus using the provided URI and token
- Creates a collection with the following schema:
  - `id`: Primary key (auto-increment)
  - `chunk_id`: Unique identifier for the chunk
  - `file_path`: Path to the source file
  - `chunk_type`: Type of chunk (file, function, class)
  - `chunk_name`: Name of the function/class or filename
  - `source_code`: The actual code content
  - `sparse_vector`: Sparse embedding vector
  - `metadata`: JSON metadata (line numbers, file size, etc.)

### Step 5: Data Upsertion
- Inserts all chunks with their embeddings into the collection
- Creates an index on the sparse vector field for efficient searching
- Handles batch insertion for better performance

## Collection Schema

The Milvus collection stores the following information for each code chunk:

| Field | Type | Description |
|-------|------|-------------|
| `id` | INT64 | Primary key (auto-increment) |
| `chunk_id` | VARCHAR(256) | Unique chunk identifier |
| `file_path` | VARCHAR(512) | Path to the source file |
| `chunk_type` | VARCHAR(50) | Type: file, function, or class |
| `chunk_name` | VARCHAR(256) | Name of the chunk |
| `source_code` | VARCHAR(65535) | The actual code content |
| `sparse_vector` | SPARSE_FLOAT_VECTOR(1536) | Sparse embedding vector |
| `metadata` | JSON | Additional metadata |

## Error Handling

The script includes comprehensive error handling for:
- Repository cloning failures
- File parsing errors
- OpenAI API errors
- Milvus connection issues
- Data insertion failures

## Logging

The script provides detailed logging at each step:
- Repository cloning progress
- File discovery and processing
- Embedding generation status
- Milvus operations
- Error messages and debugging information

## Customization

You can customize the script by:
- Modifying the chunking strategy in `CodeChunker`
- Changing the embedding model in `SparseEmbeddingGenerator`
- Adjusting the collection schema in `MilvusManager`
- Adding filters for specific file types or directories

## Troubleshooting

### Common Issues

1. **GitHub PAT Issues**: Ensure your PAT has the necessary permissions
2. **OpenAI API Limits**: The script handles rate limiting, but you may need to adjust for large repositories
3. **Milvus Connection**: Verify your Milvus instance is running and accessible
4. **Memory Usage**: Large repositories may require significant memory for processing

### Debug Mode

Enable debug logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Security Notes

- Store your `.env` file securely and never commit it to version control
- Use environment-specific PATs with minimal required permissions
- Consider using Milvus cloud for production deployments
- Monitor API usage and costs for OpenAI embeddings 



Strategy by_file: 5 chunks
Strategy by_function: 15 chunks  
Strategy by_class: 3 chunks
Strategy by_line_range: 25 chunks
Strategy by_semantic_block: 12 chunks
Strategy by_import_block: 5 chunks
Strategy by_docstring: 8 chunks
Total chunks created: 73