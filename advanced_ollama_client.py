#!/usr/bin/env python3
"""
Advanced Ollama API Client Example
This script demonstrates more advanced features of the Ollama API, including:
- Parameter tuning
- Interactive chat interface
- Batch processing
- Error handling and retries
"""

import requests
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import sys

class OllamaClient:
    """A client for interacting with the Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        """Initialize the Ollama client with the API base URL."""
        self.base_url = base_url
        self.session = requests.Session()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models in Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(self, 
                model: str, 
                prompt: str, 
                system: Optional[str] = None,
                template: Optional[str] = None,
                context: Optional[List[int]] = None,
                stream: bool = False,
                raw: bool = False,
                format: Optional[str] = None,
                options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generate a response from a prompt using the specified model.
        
        Args:
            model: The model name to use for generation
            prompt: The prompt to generate a response for
            system: System prompt to send to the model
            template: The prompt template to use
            context: The context to use for generation
            stream: Whether to stream the response
            raw: Whether to return the raw response
            format: The format to return the response in (json)
            options: Additional model parameters
        
        Returns:
            The generated response or a stream of responses
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if context:
            payload["context"] = context
        if stream:
            payload["stream"] = stream
        if raw:
            payload["raw"] = raw
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        
        try:
            # Always use streaming for better reliability
            # If stream=False, we'll collect the full response internally
            payload["stream"] = True
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            if stream:
                return self._stream_response(url, payload)
            else:
                # Collect the full response internally
                full_response = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    try:
                        # Decode bytes to string if needed
                        if isinstance(line, bytes):
                            line = line.decode('utf-8')
                            
                        json_response = json.loads(line)
                        chunk = json_response.get("response", "")
                        full_response += chunk
                        
                        # Check if we're done
                        if json_response.get("done", False):
                            break
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                    except Exception as e:
                        print(f"Error processing generate response: {str(e)}")
                        continue
                
                # Return a response object similar to the non-streaming API
                return {"response": full_response, "model": model}
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return {"error": str(e)}
    
    def _stream_response(self, url: str, payload: Dict[str, Any]):
        """Stream the response from the API."""
        try:
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    chunk = json_response.get("response", "")
                    full_response += chunk
                    yield json_response
            
            return {"response": full_response, "done": True}
        except requests.exceptions.RequestException as e:
            print(f"Error streaming response: {e}")
            return {"error": str(e), "done": True}
    
    def chat(self, 
            model: str, 
            messages: List[Dict[str, str]], 
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Chat with the model using a list of messages.
        
        Args:
            model: The model name to use for chat
            messages: A list of messages to send to the model
            stream: Whether to stream the response
            options: Additional model parameters
        
        Returns:
            The chat response or a stream of responses
        """
        url = f"{self.base_url}/chat"
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        if options:
            payload["options"] = options
        
        try:
            # Always use streaming for better reliability
            # If stream=False, we'll collect the full response internally
            payload["stream"] = True
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            if stream:
                return self._stream_chat_response(url, payload)
            else:
                # Collect the full response internally
                full_content = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        # Decode bytes to string if needed
                        if isinstance(line, bytes):
                            line = line.decode('utf-8')
                            
                        line_data = json.loads(line)
                        if "message" in line_data and "content" in line_data["message"]:
                            chunk = line_data["message"]["content"]
                            full_content += chunk
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                    except Exception as e:
                        print(f"Error processing chat response: {str(e)}")
                        continue
                
                # Return a response object similar to the non-streaming API
                return {
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "model": model
                }
        except requests.exceptions.RequestException as e:
            print(f"Error chatting: {e}")
            return {"error": str(e)}
    
    def _stream_chat_response(self, url: str, payload: Dict[str, Any]):
        """Stream the chat response from the API."""
        try:
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        # Decode bytes to string if needed
                        if isinstance(line, bytes):
                            line = line.decode('utf-8')
                            
                        line_data = json.loads(line)
                        if "message" in line_data:
                            print(line_data["message"].get("content", ""), end="", flush=True)
                        yield line_data
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                    except Exception as e:
                        print(f"Error processing chat stream: {str(e)}")
                        continue
            
            print()  # New line after completion
        except requests.exceptions.RequestException as e:
            print(f"Error streaming chat response: {e}")
            yield {"error": str(e), "done": True}
    
    def get_embeddings(self, model: str, prompt: str) -> List[float]:
        """
        Get embeddings for the provided text using the specified model.
        
        Args:
            model: The model name to use for embeddings
            prompt: The text to get embeddings for
        
        Returns:
            A list of embeddings
        """
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except requests.exceptions.RequestException as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def batch_generate(self, 
                      model: str, 
                      prompts: List[str], 
                      system: Optional[str] = None,
                      options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts in batch.
        
        Args:
            model: The model name to use for generation
            prompts: A list of prompts to generate responses for
            system: System prompt to send to the model
            options: Additional model parameters
        
        Returns:
            A list of response objects
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(model, prompt, system, options=options)
            results.append(result)
        return results
    
    def batch_chat(self, 
                  model: str, 
                  message_sets: List[List[Dict[str, str]]], 
                  options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple chat conversations in batch.
        
        Args:
            model: The model name to use for chat
            message_sets: A list of message lists for different conversations
            options: Additional model parameters
        
        Returns:
            A list of chat response objects
        """
        results = []
        for i, messages in enumerate(message_sets):
            print(f"Processing conversation {i+1}/{len(message_sets)}")
            result = self.chat(model, messages, options=options)
            results.append(result)
        return results

def interactive_chat(client: OllamaClient, model: str, system_prompt: Optional[str] = None):
    """Run an interactive chat session with the specified model."""
    print(f"\n=== Interactive Chat with {model} ===")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to reset the conversation history")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.lower() == "clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("Conversation history cleared")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="", flush=True)
        
        # Stream the response for better user experience
        response_generator = client.chat(model, messages, stream=True)
        full_response = ""
        
        for response in response_generator:
            if isinstance(response, dict):
                if "error" in response:
                    print(f"\nError: {response['error']}")
                    break
                elif "message" in response and "content" in response["message"]:
                    chunk = response["message"]["content"]
                    full_response += chunk
        
        # Add the assistant's response to the conversation history
        messages.append({"role": "assistant", "content": full_response})

def main():
    parser = argparse.ArgumentParser(description="Advanced Ollama API Client")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Model to use")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument("--system", type=str, help="System prompt")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--stream", action="store_true", help="Stream the output")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--repeat_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--batch", action="store_true", help="Process prompts in batch mode")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts for batch processing")
    parser.add_argument("--api_url", type=str, default="http://localhost:11434/api", help="Ollama API URL")
    
    args = parser.parse_args()
    
    # Initialize the client
    client = OllamaClient(base_url=args.api_url)
    
    # List available models
    print("Available models:")
    models = client.list_models()
    for model in models:
        print(f"- {model.get('name')}: {model.get('size')}")
    print()
    
    # Set up generation options
    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty
    }
    
    # Check if the specified model is available
    model_names = [model.get('name') for model in models]
    if args.model not in model_names:
        if not models:
            print("No models available. Please pull a model first.")
            return
        print(f"Model '{args.model}' not found. Using '{models[0].get('name')}' instead.")
        args.model = models[0].get('name')
    
    # Process based on arguments
    if args.chat:
        interactive_chat(client, args.model, args.system)
    elif args.batch and args.prompt_file:
        # Batch processing from file
        try:
            with open(args.prompt_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(prompts)} prompts in batch mode...")
            results = client.batch_generate(args.model, prompts, args.system, options)
            
            for i, result in enumerate(results):
                print(f"\nPrompt {i+1}: {prompts[i]}")
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Response: {result.get('response', '')}")
        except Exception as e:
            print(f"Error processing batch: {e}")
    elif args.prompt:
        # Single prompt generation
        print(f"\n=== Generating with {args.model} ===")
        print(f"Prompt: {args.prompt}")
        
        start_time = time.time()
        
        if args.stream:
            print("Response: ", end="", flush=True)
            response_generator = client.generate(args.model, args.prompt, args.system, stream=True, options=options)
            for response in response_generator:
                pass  # The streaming is handled by the client
        else:
            response = client.generate(args.model, args.prompt, args.system, options=options)
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                print(f"Response: {response.get('response', '')}")
        
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    else:
        # Default to interactive chat
        interactive_chat(client, args.model, args.system)

if __name__ == "__main__":
    main()