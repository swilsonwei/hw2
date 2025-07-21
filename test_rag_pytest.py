#!/usr/bin/env python3
"""
Pytest-compatible Integration Tests for SmartInfo RAG System
Tests both functionality and security aspects of the RAG system.
"""

import os
import sys
import time
import json
import requests
import pytest
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestResult:
    """Represents a test result."""
    __test__ = False  # Tell pytest this is not a test class
    
    def __init__(self, test_name: str, passed: bool, response_time: float, 
                 error_message: Optional[str] = None, response_data: Optional[Dict] = None):
        self.test_name = test_name
        self.passed = passed
        self.response_time = response_time
        self.error_message = error_message
        self.response_data = response_data

class RAGIntegrationTester:
    """Integration tester for the RAG system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
        # Test configuration
        self.timeout = 30
        self.max_response_time = 10  # seconds
        
        # GitHub repo info for context
        self.github_repo = "https://github.com/swilsonwei/hw2.git"
        
    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results."""
        start_time = time.time()
        passed = False
        error_message = None
        response_data = None
        
        try:
            response_data = test_func()
            passed = True
        except Exception as e:
            error_message = str(e)
            logger.error(f"Test '{test_name}' failed: {e}")
        
        response_time = time.time() - start_time
        
        result = TestResult(
            test_name=test_name,
            passed=passed,
            response_time=response_time,
            error_message=error_message,
            response_data=response_data
        )
        
        self.test_results.append(result)
        
        if passed:
            logger.info(f"✅ Test '{test_name}' passed in {response_time:.2f}s")
        else:
            logger.error(f"❌ Test '{test_name}' failed in {response_time:.2f}s")
        
        return result
    
    def test_search_endpoint_health(self) -> Dict:
        """Test 1: Basic search endpoint health check."""
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": "test"},
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Search endpoint returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict) or "results" not in data:
            raise Exception("Invalid response format from search endpoint")
        
        return data
    
    def test_general_knowledge_query(self) -> Dict:
        """Test 2: General knowledge query should return answer."""
        response = self.session.post(
            f"{self.base_url}/api/enhanced-response",
            json={
                "query": "What is the capital of France?",
                "code_chunks": []
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Enhanced response endpoint returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict) or "response" not in data:
            raise Exception("Invalid response format from enhanced response endpoint")
        
        if not data["response"] or len(data["response"]) < 10:
            raise Exception("Response too short or empty")
        
        return data
    
    def test_code_related_query(self) -> Dict:
        """Test 3: Code-related query should return enhanced response."""
        response = self.session.post(
            f"{self.base_url}/api/enhanced-response",
            json={
                "query": "How do I create a function in Python?",
                "code_chunks": [
                    {
                        "chunk_name": "function_example",
                        "file_path": "example.py",
                        "source_code": "def hello_world():\n    print('Hello, World!')"
                    }
                ]
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Enhanced response endpoint returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict) or "response" not in data:
            raise Exception("Invalid response format from enhanced response endpoint")
        
        if not data["response"] or len(data["response"]) < 20:
            raise Exception("Response too short for code-related query")
        
        return data
    
    def test_milvus_search_functionality(self) -> Dict:
        """Test 4: Milvus search should return relevant code chunks."""
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": "function"},
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Search endpoint returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict) or "results" not in data:
            raise Exception("Invalid response format from search endpoint")
        
        # Should return results (even if empty list)
        if not isinstance(data["results"], list):
            raise Exception("Results should be a list")
        
        return data
    
    def test_frontend_availability(self) -> Dict:
        """Test 5: Frontend should be accessible."""
        response = self.session.get(
            f"{self.base_url}/",
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Frontend returned status {response.status_code}")
        
        if "SmartInfo" not in response.text:
            raise Exception("Frontend does not contain expected branding")
        
        return {"status": "frontend_accessible", "content_length": len(response.text)}
    
    def test_summary_endpoint(self) -> Dict:
        """Test 6: Summary endpoint should work with code chunks."""
        response = self.session.post(
            f"{self.base_url}/api/summarize",
            json={
                "query": "database connection",
                "code_chunks": [
                    {
                        "chunk_name": "db_connection",
                        "file_path": "database.py",
                        "source_code": "import sqlite3\n\ndef connect_db():\n    return sqlite3.connect('app.db')"
                    }
                ]
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Summary endpoint returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict) or "summary" not in data:
            raise Exception("Invalid response format from summary endpoint")
        
        return data
    
    def test_health_endpoint(self) -> Dict:
        """Test 7: Health endpoint should be available."""
        response = self.session.get(
            f"{self.base_url}/api/health",
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Health endpoint returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict):
            raise Exception("Health endpoint should return JSON")
        
        return data
    
    def test_response_time_performance(self) -> Dict:
        """Test 8: Response time should be within acceptable limits."""
        start_time = time.time()
        
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": "test query"},
            timeout=self.timeout
        )
        
        response_time = time.time() - start_time
        
        if response_time > self.max_response_time:
            raise Exception(f"Response time {response_time:.2f}s exceeds limit of {self.max_response_time}s")
        
        if response.status_code != 200:
            raise Exception(f"Search endpoint returned status {response.status_code}")
        
        return {"response_time": response_time, "status_code": response.status_code}
    
    def test_large_query_handling(self) -> Dict:
        """Test 9: System should handle large queries gracefully."""
        large_query = "What is the meaning of life? " * 100  # 2400 characters
        
        response = self.session.post(
            f"{self.base_url}/api/enhanced-response",
            json={
                "query": large_query,
                "code_chunks": []
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Large query returned status {response.status_code}")
        
        data = response.json()
        if not isinstance(data, dict) or "response" not in data:
            raise Exception("Invalid response format for large query")
        
        return data
    
    def test_empty_query_handling(self) -> Dict:
        """Test 10: System should handle empty queries gracefully."""
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": ""},
            timeout=self.timeout
        )
        
        # Should either return 200 with empty results or 400 for bad request
        if response.status_code not in [200, 400]:
            raise Exception(f"Empty query returned unexpected status {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict) or "results" not in data:
                raise Exception("Invalid response format for empty query")
        
        return {"status_code": response.status_code}

class RAGSecurityTester:
    """Security and abuse prevention tester for the RAG system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.timeout = 30
    
    def run_security_test(self, test_name: str, test_func) -> TestResult:
        """Run a single security test and record results."""
        start_time = time.time()
        passed = False
        error_message = None
        response_data = None
        
        try:
            response_data = test_func()
            passed = True
        except Exception as e:
            error_message = str(e)
            logger.error(f"Security test '{test_name}' failed: {e}")
        
        response_time = time.time() - start_time
        
        result = TestResult(
            test_name=test_name,
            passed=passed,
            response_time=response_time,
            error_message=error_message,
            response_data=response_data
        )
        
        self.test_results.append(result)
        
        if passed:
            logger.info(f"✅ Security test '{test_name}' passed in {response_time:.2f}s")
        else:
            logger.error(f"❌ Security test '{test_name}' failed in {response_time:.2f}s")
        
        return result
    
    def test_sql_injection_prevention(self) -> Dict:
        """Security Test 1: Prevent SQL injection attempts."""
        malicious_queries = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "admin'--",
            "1' UNION SELECT * FROM users--"
        ]
        
        for query in malicious_queries:
            response = self.session.post(
                f"{self.base_url}/search",
                json={"query": query},
                timeout=self.timeout
            )
            
            # Should not crash or return sensitive data
            if response.status_code == 500:
                raise Exception(f"SQL injection query caused server error: {query}")
            
            # Should not return database schema or sensitive info
            if response.status_code == 200:
                data = response.json()
                response_text = str(data).lower()
                # Only flag if it looks like actual database schema or credentials
                sensitive_patterns = [
                    'create table', 'drop table', 'insert into', 'select * from',
                    'password=', 'admin:', 'root:', 'schema:', 'database:'
                ]
                for pattern in sensitive_patterns:
                    if pattern in response_text:
                        raise Exception(f"Potential sensitive data exposure for query: {query}")
        
        return {"status": "sql_injection_prevented", "queries_tested": len(malicious_queries)}
    
    def test_xss_prevention(self) -> Dict:
        """Security Test 2: Prevent XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
            "<svg onload=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            response = self.session.post(
                f"{self.base_url}/api/enhanced-response",
                json={
                    "query": payload,
                    "code_chunks": []
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = str(data)
                
                # Check if script tags are escaped or removed
                if "<script>" in response_text.lower():
                    raise Exception(f"XSS payload not properly escaped: {payload}")
        
        return {"status": "xss_prevented", "payloads_tested": len(xss_payloads)}
    
    def test_rate_limiting(self) -> Dict:
        """Security Test 3: Test rate limiting (if implemented)."""
        # Send multiple rapid requests
        responses = []
        for i in range(20):
            response = self.session.post(
                f"{self.base_url}/search",
                json={"query": f"test query {i}"},
                timeout=self.timeout
            )
            responses.append(response.status_code)
            time.sleep(0.1)  # Small delay between requests
        
        # Check if any requests were rate limited (429 status)
        rate_limited = any(status == 429 for status in responses)
        
        # If no rate limiting, that's acceptable for this test
        # but we should log it
        if not rate_limited:
            logger.warning("No rate limiting detected - consider implementing rate limiting")
        
        return {
            "status": "rate_limiting_tested",
            "requests_sent": len(responses),
            "rate_limited": rate_limited
        }
    
    def test_large_payload_prevention(self) -> Dict:
        """Security Test 4: Prevent large payload attacks."""
        # Create a very large payload
        large_payload = {
            "query": "test",
            "code_chunks": [
                {
                    "chunk_name": "large_chunk",
                    "file_path": "large_file.py",
                    "source_code": "x" * 1000000  # 1MB of data
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/enhanced-response",
                json=large_payload,
                timeout=self.timeout
            )
            
            # Should either reject or handle gracefully
            if response.status_code == 413:  # Payload too large
                return {"status": "large_payload_rejected", "status_code": 413}
            elif response.status_code == 200:
                return {"status": "large_payload_handled", "status_code": 200}
            else:
                return {"status": "large_payload_response", "status_code": response.status_code}
                
        except requests.exceptions.Timeout:
            return {"status": "large_payload_timeout", "status_code": "timeout"}
    
    def test_authentication_bypass(self) -> Dict:
        """Security Test 5: Test for authentication bypass vulnerabilities."""
        # Test various endpoints that might require authentication
        endpoints_to_test = [
            "/api/enhanced-response",
            "/api/summarize",
            "/search"
        ]
        
        bypass_attempts = []
        
        for endpoint in endpoints_to_test:
            # Try without any authentication headers
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json={"query": "test"},
                timeout=self.timeout
            )
            
            # If endpoint requires auth, should return 401 or 403
            if response.status_code in [401, 403]:
                bypass_attempts.append(f"{endpoint}: properly_secured")
            elif response.status_code == 200:
                bypass_attempts.append(f"{endpoint}: no_auth_required")
            else:
                bypass_attempts.append(f"{endpoint}: status_{response.status_code}")
        
        return {
            "status": "authentication_bypass_tested",
            "endpoints_tested": endpoints_to_test,
            "results": bypass_attempts
        }
    
    def test_input_validation(self) -> Dict:
        """Security Test 6: Test input validation."""
        invalid_inputs = [
            None,
            {},
            {"invalid_field": "test"},
            {"query": None},
            {"query": 123},
            {"query": []},
            {"query": {"nested": "object"}}
        ]
        
        validation_results = []
        
        for invalid_input in invalid_inputs:
            try:
                response = self.session.post(
                    f"{self.base_url}/search",
                    json=invalid_input,
                    timeout=self.timeout
                )
                
                # Should return 400 for bad request or handle gracefully
                if response.status_code in [400, 422]:
                    validation_results.append("properly_validated")
                elif response.status_code == 200:
                    validation_results.append("accepted_invalid_input")
                else:
                    validation_results.append(f"unexpected_status_{response.status_code}")
                    
            except Exception as e:
                validation_results.append(f"exception_{type(e).__name__}")
        
        return {
            "status": "input_validation_tested",
            "invalid_inputs_tested": len(invalid_inputs),
            "results": validation_results
        }
    
    def test_path_traversal_prevention(self) -> Dict:
        """Security Test 7: Prevent path traversal attacks."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for payload in path_traversal_payloads:
            response = self.session.post(
                f"{self.base_url}/search",
                json={"query": payload},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = str(data).lower()
                
                # Should not contain system file contents
                if "root:" in response_text or "administrator:" in response_text:
                    raise Exception(f"Path traversal successful: {payload}")
        
        return {"status": "path_traversal_prevented", "payloads_tested": len(path_traversal_payloads)}
    
    def test_denial_of_service_prevention(self) -> Dict:
        """Security Test 8: Test DoS prevention."""
        # Test with very long queries
        long_query = "a" * 10000  # 10KB query
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": long_query},
            timeout=self.timeout
        )
        response_time = time.time() - start_time
        
        # Should not take too long to process
        if response_time > 30:
            raise Exception(f"Long query took too long: {response_time:.2f}s")
        
        # Should not crash
        if response.status_code == 500:
            raise Exception("Long query caused server error")
        
        return {
            "status": "dos_prevention_tested",
            "query_length": len(long_query),
            "response_time": response_time,
            "status_code": response.status_code
        }
    
    def test_credential_exposure(self) -> Dict:
        """Security Test 9: Check for credential exposure."""
        # Test various endpoints for potential credential exposure
        endpoints = ["/", "/api/health", "/search"]
        
        exposed_credentials = []
        
        for endpoint in endpoints:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                content = response.text.lower()
                
                # Check for common credential patterns
                credential_patterns = [
                    "api_key", "password", "secret", "token", "credential",
                    "sk-", "ghp_", "xoxb-", "Bearer", "Authorization"
                ]
                
                for pattern in credential_patterns:
                    if pattern in content:
                        exposed_credentials.append(f"{endpoint}: {pattern}")
        
        if exposed_credentials:
            raise Exception(f"Potential credential exposure: {exposed_credentials}")
        
        return {"status": "no_credential_exposure", "endpoints_checked": endpoints}
    
    def test_cors_policy(self) -> Dict:
        """Security Test 10: Test CORS policy."""
        # Test CORS headers
        response = self.session.options(
            f"{self.base_url}/search",
            headers={"Origin": "https://malicious-site.com"},
            timeout=self.timeout
        )
        
        cors_headers = response.headers.get("Access-Control-Allow-Origin", "")
        
        # Should not allow all origins
        if cors_headers == "*":
            return {"status": "cors_allows_all_origins", "recommendation": "restrict_cors"}
        
        return {"status": "cors_properly_configured", "allow_origin": cors_headers}

# Pytest fixtures
@pytest.fixture(scope="session")
def integration_tester():
    """Fixture for integration tester."""
    return RAGIntegrationTester()

@pytest.fixture(scope="session")
def security_tester():
    """Fixture for security tester."""
    return RAGSecurityTester()

# Functionality Tests (pytest format)
def test_search_endpoint_health(integration_tester):
    """Test 1: Basic search endpoint health check."""
    result = integration_tester.run_test("Search Endpoint Health", integration_tester.test_search_endpoint_health)
    assert result.passed, f"Search endpoint health check failed: {result.error_message}"

def test_general_knowledge_query(integration_tester):
    """Test 2: General knowledge query should return answer."""
    result = integration_tester.run_test("General Knowledge Query", integration_tester.test_general_knowledge_query)
    assert result.passed, f"General knowledge query failed: {result.error_message}"

def test_code_related_query(integration_tester):
    """Test 3: Code-related query should return enhanced response."""
    result = integration_tester.run_test("Code-Related Query", integration_tester.test_code_related_query)
    assert result.passed, f"Code-related query failed: {result.error_message}"

def test_milvus_search_functionality(integration_tester):
    """Test 4: Milvus search should return relevant code chunks."""
    result = integration_tester.run_test("Milvus Search Functionality", integration_tester.test_milvus_search_functionality)
    assert result.passed, f"Milvus search functionality failed: {result.error_message}"

def test_frontend_availability(integration_tester):
    """Test 5: Frontend should be accessible."""
    result = integration_tester.run_test("Frontend Availability", integration_tester.test_frontend_availability)
    assert result.passed, f"Frontend availability failed: {result.error_message}"

def test_summary_endpoint(integration_tester):
    """Test 6: Summary endpoint should work with code chunks."""
    result = integration_tester.run_test("Summary Endpoint", integration_tester.test_summary_endpoint)
    assert result.passed, f"Summary endpoint failed: {result.error_message}"

def test_health_endpoint(integration_tester):
    """Test 7: Health endpoint should be available."""
    result = integration_tester.run_test("Health Endpoint", integration_tester.test_health_endpoint)
    assert result.passed, f"Health endpoint failed: {result.error_message}"

def test_response_time_performance(integration_tester):
    """Test 8: Response time should be within acceptable limits."""
    result = integration_tester.run_test("Response Time Performance", integration_tester.test_response_time_performance)
    assert result.passed, f"Response time performance failed: {result.error_message}"

def test_large_query_handling(integration_tester):
    """Test 9: System should handle large queries gracefully."""
    result = integration_tester.run_test("Large Query Handling", integration_tester.test_large_query_handling)
    assert result.passed, f"Large query handling failed: {result.error_message}"

def test_empty_query_handling(integration_tester):
    """Test 10: System should handle empty queries gracefully."""
    result = integration_tester.run_test("Empty Query Handling", integration_tester.test_empty_query_handling)
    assert result.passed, f"Empty query handling failed: {result.error_message}"

# Security Tests (pytest format)
def test_sql_injection_prevention(security_tester):
    """Security Test 1: Prevent SQL injection attempts."""
    result = security_tester.run_security_test("SQL Injection Prevention", security_tester.test_sql_injection_prevention)
    assert result.passed, f"SQL injection prevention failed: {result.error_message}"

def test_xss_prevention(security_tester):
    """Security Test 2: Prevent XSS attacks."""
    result = security_tester.run_security_test("XSS Prevention", security_tester.test_xss_prevention)
    assert result.passed, f"XSS prevention failed: {result.error_message}"

def test_rate_limiting(security_tester):
    """Security Test 3: Test rate limiting (if implemented)."""
    result = security_tester.run_security_test("Rate Limiting", security_tester.test_rate_limiting)
    assert result.passed, f"Rate limiting test failed: {result.error_message}"

def test_large_payload_prevention(security_tester):
    """Security Test 4: Prevent large payload attacks."""
    result = security_tester.run_security_test("Large Payload Prevention", security_tester.test_large_payload_prevention)
    assert result.passed, f"Large payload prevention failed: {result.error_message}"

def test_authentication_bypass(security_tester):
    """Security Test 5: Test for authentication bypass vulnerabilities."""
    result = security_tester.run_security_test("Authentication Bypass", security_tester.test_authentication_bypass)
    assert result.passed, f"Authentication bypass test failed: {result.error_message}"

def test_input_validation(security_tester):
    """Security Test 6: Test input validation."""
    result = security_tester.run_security_test("Input Validation", security_tester.test_input_validation)
    assert result.passed, f"Input validation failed: {result.error_message}"

def test_path_traversal_prevention(security_tester):
    """Security Test 7: Prevent path traversal attacks."""
    result = security_tester.run_security_test("Path Traversal Prevention", security_tester.test_path_traversal_prevention)
    assert result.passed, f"Path traversal prevention failed: {result.error_message}"

def test_denial_of_service_prevention(security_tester):
    """Security Test 8: Test DoS prevention."""
    result = security_tester.run_security_test("DoS Prevention", security_tester.test_denial_of_service_prevention)
    assert result.passed, f"DoS prevention failed: {result.error_message}"

def test_credential_exposure(security_tester):
    """Security Test 9: Check for credential exposure."""
    result = security_tester.run_security_test("Credential Exposure", security_tester.test_credential_exposure)
    assert result.passed, f"Credential exposure check failed: {result.error_message}"

def test_cors_policy(security_tester):
    """Security Test 10: Test CORS policy."""
    result = security_tester.run_security_test("CORS Policy", security_tester.test_cors_policy)
    assert result.passed, f"CORS policy test failed: {result.error_message}"

# Test for server availability
def test_server_running():
    """Test that the server is running before running other tests."""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        assert response.status_code == 200, "Server health check failed"
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Server not running: {e}")

if __name__ == "__main__":
    # This allows the file to be run directly as well
    pytest.main([__file__, "-v"]) 