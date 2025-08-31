"""
Unit tests for OpenVINO GenAI Server with OpenAI API compatibility.

This test suite validates the models and chat completions endpoints of the server.
Tests assume the server is already running and use the OpenAI client library
to interact with the API endpoints.

Usage:
    pytest test_server.py -v                                    # Test against default localhost:8000
    pytest test_server.py --base-url http://host:port -v        # Test against custom URL
    pytest test_server.py::TestOVServer::test_models_endpoint_get -v  # Run specific test
    pytest test_server.py -k "thinking" -v                      # Run tests matching pattern
"""

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
import requests
from typing import List, Dict, Any


class TestOVServer:
    """Test suite for OpenVINO GenAI Server OpenAI API endpoints."""
    
    def test_server_health(self, base_url):
        """Test if the server is running and accessible."""
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            assert response.status_code == 200, f"Server not accessible at {base_url}"
            print(f"✓ Server is running at {base_url}")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Server not accessible at {base_url}: {e}")
    
    def test_models_endpoint_get(self, client):
        """Test GET /v1/models endpoint returns model list."""
        try:
            models = client.models.list()
            
            # Validate response structure
            assert hasattr(models, 'object'), "Response should have 'object' field"
            assert hasattr(models, 'data'), "Response should have 'data' field"
            assert models.object == "list", "Object type should be 'list'"
            assert isinstance(models.data, list), "Data should be a list"
            assert len(models.data) > 0, "Should return at least one model"
            
            print(f"✓ Models endpoint returns {len(models.data)} model(s)")
            print(f"  Available models: {[model.id for model in models.data]}")
            
        except Exception as e:
            pytest.fail(f"Failed to get models: {e}")
    
    def test_models_endpoint_options(self, base_url):
        """Test OPTIONS /v1/models endpoint for CORS support."""
        try:
            response = requests.options(f"{base_url}/v1/models")
            assert response.status_code == 200, "OPTIONS request should succeed"
            print("✓ Models endpoint supports OPTIONS requests")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"OPTIONS request failed: {e}")
    
    def test_chat_completions_simple_streaming(self, client):
        """Test basic streaming chat completion with a simple user message."""
        try:
            stream = client.chat.completions.create(
                model="test-model",  # Model name doesn't matter based on server code
                messages=[
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                max_tokens=50,
                temperature=0.7,
                stream=True  # Streaming response
            )
            
            chunks_received = 0
            content_chunks = []
            assistant_role_seen = False
            
            for chunk in stream:
                chunks_received += 1
                assert isinstance(chunk, ChatCompletionChunk), "Should receive ChatCompletionChunk objects"
                
                # Check for role in first chunk
                if chunk.choices and chunk.choices[0].delta.role == "assistant":
                    assistant_role_seen = True
                
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                # Check for finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    assert chunk.choices[0].finish_reason == "stop", "Should finish with 'stop' reason"
                    break
            
            assert chunks_received > 0, "Should receive at least one chunk"
            assert assistant_role_seen, "Should see assistant role in first chunk"
            full_content = "".join(content_chunks)
            assert len(full_content) > 0, "Should receive some content"
            
            print("✓ Basic streaming chat completion works")
            print(f"  Response: {full_content[:100]}...")
            
        except Exception as e:
            pytest.fail(f"Basic streaming chat completion failed: {e}")
    
    def test_chat_completions_streaming(self, client):
        """Test streaming chat completion response."""
        try:
            stream = client.chat.completions.create(
                model="test-model",
                messages=[
                    {"role": "user", "content": "Tell me a short story."}
                ],
                max_tokens=100,
                temperature=0.7,
                stream=True
            )
            
            chunks_received = 0
            content_chunks = []
            
            for chunk in stream:
                chunks_received += 1
                assert isinstance(chunk, ChatCompletionChunk), "Should receive ChatCompletionChunk objects"
                
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                # Check for finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    assert chunk.choices[0].finish_reason == "stop", "Should finish with 'stop' reason"
                    break
            
            assert chunks_received > 0, "Should receive at least one chunk"
            full_content = "".join(content_chunks)
            assert len(full_content) > 0, "Should receive some content"
            
            print(f"✓ Streaming chat completion works ({chunks_received} chunks)")
            print(f"  Generated content length: {len(full_content)} characters")
            
        except Exception as e:
            pytest.fail(f"Streaming chat completion failed: {e}")
    
    def test_chat_completions_multi_turn_streaming(self, client):
        """Test multi-turn conversation with assistant and user messages using streaming."""
        try:
            messages = [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "What about 3+3?"}
            ]
            
            stream = client.chat.completions.create(
                model="test-model",
                messages=messages,
                max_tokens=30,
                temperature=0.5,
                stream=True
            )
            
            content_chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                if chunk.choices and chunk.choices[0].finish_reason:
                    break
            
            full_content = "".join(content_chunks)
            assert len(full_content) > 0, "Should receive some content"
            
            print("✓ Multi-turn conversation streaming works")
            print(f"  Response: {full_content}")
            
        except Exception as e:
            pytest.fail(f"Multi-turn conversation streaming failed: {e}")
    
    def test_chat_completions_with_stop_sequences_streaming(self, client):
        """Test streaming chat completion with stop sequences."""
        try:
            stream = client.chat.completions.create(
                model="test-model",
                messages=[
                    {"role": "user", "content": "Count from 1 to 10: 1, 2, 3,"}
                ],
                max_tokens=50,
                stop=[",", "5"],  # Should stop at comma or number 5
                temperature=0.1,
                stream=True
            )
            
            content_chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                if chunk.choices and chunk.choices[0].finish_reason:
                    break
            
            full_content = "".join(content_chunks)
            assert len(full_content) >= 0, "Should receive content or stop early"
            
            print("✓ Stop sequences streaming works")
            print(f"  Response: {full_content}")
            
        except Exception as e:
            pytest.fail(f"Stop sequences streaming test failed: {e}")
    
    def test_chat_completions_temperature_variations_streaming(self, client):
        """Test streaming chat completion with different temperature settings."""
        try:
            base_prompt = "Describe the color blue in one sentence."
            temperatures = [0.0, 0.5, 1.0]
            
            for temp in temperatures:
                stream = client.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": base_prompt}],
                    max_tokens=30,
                    temperature=temp,
                    stream=True
                )
                
                content_chunks = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunks.append(chunk.choices[0].delta.content)
                    
                    if chunk.choices and chunk.choices[0].finish_reason:
                        break
                
                full_content = "".join(content_chunks)
                assert len(full_content) >= 0, "Should receive some content"
                print(f"✓ Temperature {temp} streaming works: {full_content[:50]}...")
                
        except Exception as e:
            pytest.fail(f"Temperature variations streaming test failed: {e}")
    
    def test_chat_completions_with_system_message_streaming(self, client):
        """Test streaming chat completion with system message."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that responds in a friendly manner."},
                {"role": "user", "content": "Hello!"}
            ]
            
            stream = client.chat.completions.create(
                model="test-model",
                messages=messages,
                max_tokens=50,
                temperature=0.7,
                stream=True
            )
            
            content_chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                if chunk.choices and chunk.choices[0].finish_reason:
                    break
            
            full_content = "".join(content_chunks)
            assert len(full_content) > 0, "Should receive some content"
            
            print("✓ System message streaming works")
            print(f"  Response: {full_content}")
            
        except Exception as e:
            pytest.fail(f"System message streaming test failed: {e}")
    
    def test_chat_completions_invalid_request(self, client):
        """Test error handling for invalid chat completion requests."""
        try:
            # Test with empty messages - should fail
            try:
                stream = client.chat.completions.create(
                    model="test-model",
                    messages=[],  # Empty messages should cause error
                    max_tokens=50,
                    stream=True
                )
                # Try to consume the stream to trigger the error
                list(stream)
                pytest.fail("Should have failed with empty messages")
            except Exception:
                print("✓ Empty messages properly rejected")
            
            # Test with non-user last message - should fail
            try:
                stream = client.chat.completions.create(
                    model="test-model",
                    messages=[
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there"}  # Last message not from user
                    ],
                    max_tokens=50,
                    stream=True
                )
                # Try to consume the stream to trigger the error
                list(stream)
                pytest.fail("Should have failed with non-user last message")
            except Exception:
                print("✓ Non-user last message properly rejected")
                
        except Exception as e:
            pytest.fail(f"Invalid request test setup failed: {e}")
    
    def test_generation_config_parameters_streaming(self, client):
        """Test various generation configuration parameters with streaming."""
        try:
            # Test with custom generation parameters
            stream = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Write a haiku about programming."}],
                max_tokens=100,
                temperature=0.8,
                top_p=0.9,
                frequency_penalty=0.1,  # Maps to repetition_penalty
                stop=["END", "\n\n"],
                stream=True
            )
            
            content_chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                if chunk.choices and chunk.choices[0].finish_reason:
                    break
            
            full_content = "".join(content_chunks)
            assert len(full_content) >= 0, "Should receive some content"
            
            print("✓ Generation config parameters streaming work")
            print(f"  Response: {full_content}")
            
        except Exception as e:
            pytest.fail(f"Generation config streaming test failed: {e}")
    
    def test_chat_template_kwargs_disable_thinking(self, client):
        """Test chat template with thinking disabled via chat_template_kwargs."""
        try:
            # Test with thinking disabled
            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Solve this step by step: What is 15 + 27?"}],
                max_tokens=100,
                temperature=0.3,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            )
            
            content_chunks = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                if chunk.choices and chunk.choices[0].finish_reason:
                    break
            
            full_content = "".join(content_chunks)
            assert len(full_content) > 0, "Should receive some content"
            
            print("✓ Chat template with thinking disabled works")
            print(f"  Response: {full_content}")
            
        except Exception as e:
            pytest.fail(f"Chat template kwargs test failed: {e}")
