# SmartInfo RAG System - Integration Testing Suite

This directory contains comprehensive integration tests for the SmartInfo RAG (Retrieval-Augmented Generation) system.

## 📋 Overview

The testing suite includes **20 total tests**:
- **10 Functionality Tests** - Ensure the RAG system works correctly
- **10 Security Tests** - Prevent abuse and security vulnerabilities

## 🚀 Quick Start

### Option 1: Automatic Test Runner (Recommended)
```bash
python run_tests.py
```

This script will:
- Install required dependencies
- Start the FastAPI server if needed
- Run all tests
- Provide detailed reporting

### Option 2: Manual Testing
```bash
# Install dependencies
pip install requests python-dotenv

# Start your server (if not running)
source venv/bin/activate
python main.py

# In another terminal, run tests
python test_rag.py
```

## 🧪 Test Categories

### Functionality Tests (10 tests)

These tests ensure your RAG system returns results correctly:

1. **Search Endpoint Health** - Basic connectivity test
2. **General Knowledge Query** - Tests general LLM responses
3. **Code-Related Query** - Tests enhanced responses with code context
4. **Milvus Search Functionality** - Tests vector search capabilities
5. **Frontend Availability** - Tests web interface accessibility
6. **Summary Endpoint** - Tests code summarization
7. **Health Endpoint** - Tests system health monitoring
8. **Response Time Performance** - Tests performance benchmarks
9. **Large Query Handling** - Tests system with large inputs
10. **Empty Query Handling** - Tests edge case handling

### Security Tests (10 tests)

These tests prevent abuse and security vulnerabilities:

1. **SQL Injection Prevention** - Tests against SQL injection attacks
2. **XSS Prevention** - Tests against cross-site scripting
3. **Rate Limiting** - Tests request throttling (if implemented)
4. **Large Payload Prevention** - Tests against oversized requests
5. **Authentication Bypass** - Tests endpoint security
6. **Input Validation** - Tests malformed input handling
7. **Path Traversal Prevention** - Tests against directory traversal
8. **DoS Prevention** - Tests denial-of-service protection
9. **Credential Exposure** - Tests for sensitive data leaks
10. **CORS Policy** - Tests cross-origin resource sharing

## 📊 Test Results

The test suite provides detailed reporting including:

- ✅/❌ Pass/Fail status for each test
- ⏱️ Response times for performance monitoring
- 📈 Success rate percentage
- 🔍 Detailed error messages for failed tests
- 📋 Summary statistics

### Example Output
```
🚀 Starting RAG Integration Tests
==================================================
📋 Running Functionality Tests...
✅ Test 'Search Endpoint Health' passed in 0.15s
✅ Test 'General Knowledge Query' passed in 1.23s
✅ Test 'Code-Related Query' passed in 1.45s
...

🔒 Running Security Tests...
✅ Test 'SQL Injection Prevention' passed in 0.89s
✅ Test 'XSS Prevention' passed in 0.67s
⚠️ Test 'Rate Limiting' passed in 0.34s (No rate limiting detected)

📊 Test Results Summary
==================================================
Total Tests: 20
Passed: 19
Failed: 1
Success Rate: 95.0%

⚡ Average Response Time: 0.87s
```

## 🔧 Configuration

### Environment Variables
The tests use the same environment variables as your main application:
- `MILVUS_URI` - Milvus connection URI
- `MILVUS_TOKEN` - Milvus authentication token
- `OPENAI_API_KEY` - OpenAI API key

### Test Configuration
You can modify test parameters in `test_rag.py`:
- `timeout` - Request timeout (default: 30s)
- `max_response_time` - Performance threshold (default: 10s)
- `base_url` - Server URL (default: http://localhost:8000)

## 🐛 Troubleshooting

### Common Issues

1. **Server Not Running**
   ```
   ❌ Cannot run tests without a running server
   ```
   **Solution**: Start your FastAPI server first or use `run_tests.py`

2. **Dependencies Missing**
   ```
   ModuleNotFoundError: No module named 'requests'
   ```
   **Solution**: Run `pip install requests python-dotenv`

3. **Connection Timeout**
   ```
   ❌ Server failed to start within 30 seconds
   ```
   **Solution**: Check if port 8000 is available, or modify the port in your server

4. **Milvus Connection Issues**
   ```
   Failed to connect to Milvus
   ```
   **Solution**: Verify your Milvus credentials and connection

### Debug Mode
For detailed debugging, you can run individual tests:

```python
# In Python console
from test_rag import RAGIntegrationTester

tester = RAGIntegrationTester()
result = tester.run_test("Search Endpoint Health", tester.test_search_endpoint_health)
print(result)
```

## 📈 Performance Benchmarks

The test suite includes performance monitoring:

- **Response Time**: Average time for API calls
- **Throughput**: Number of successful requests per second
- **Error Rate**: Percentage of failed requests
- **Resource Usage**: Memory and CPU monitoring (if available)

## 🔒 Security Considerations

The security tests cover:

- **Input Validation**: Malformed JSON, null values, wrong data types
- **Injection Attacks**: SQL injection, XSS, path traversal
- **Resource Abuse**: Large payloads, rapid requests, DoS attempts
- **Data Exposure**: Credential leaks, sensitive information disclosure
- **Access Control**: Authentication bypass attempts

## 🎯 GitHub Repository Context

Tests are configured for your specific repository:
- **Repository**: https://github.com/swilsonwei/hw2.git
- **Expected Content**: Python files with code chunks
- **Milvus Collection**: `code_chunks_sparse` (sparse vector index)

## 📝 Adding New Tests

To add new tests:

1. **Functionality Tests**: Add to `RAGIntegrationTester` class
2. **Security Tests**: Add to `RAGSecurityTester` class
3. **Update Test Lists**: Add to `functionality_tests` or `security_tests` in `run_all_tests()`

Example:
```python
def test_new_feature(self) -> Dict:
    """Test new feature functionality."""
    response = self.session.post(
        f"{self.base_url}/new-endpoint",
        json={"data": "test"},
        timeout=self.timeout
    )
    
    if response.status_code != 200:
        raise Exception(f"New endpoint returned status {response.status_code}")
    
    return response.json()
```

## 🤝 Contributing

When adding new tests:
- Follow the existing naming conventions
- Include proper error handling
- Add descriptive docstrings
- Update this README if adding new test categories

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test logs for specific error messages
3. Verify your server configuration
4. Ensure all dependencies are installed

---

**Note**: These tests are designed for the SmartInfo RAG system with Milvus sparse vector indexing and OpenAI embeddings. Adjust configurations as needed for your specific setup. 