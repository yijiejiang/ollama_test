#!/usr/bin/env python3
"""
Simple Ollama Client
使用官方推荐的方式调用 Ollama API
"""

import json
import time
import requests
import sys

# Ollama API 基础 URL
OLLAMA_API_URL = "http://localhost:11434/api"

def list_models():
    """列出所有可用的模型"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return models
        else:
            print(f"获取模型列表失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        return []

def generate(model, prompt, system=None, stream=False, options=None):
    """使用指定模型生成文本"""
    url = f"{OLLAMA_API_URL}/generate"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    if system:
        payload["system"] = system
    if stream:
        payload["stream"] = stream
    if options:
        payload["options"] = options
    
    try:
        if stream:
            response = requests.post(url, json=payload, stream=True)
            if response.status_code != 200:
                print(f"生成文本失败: {response.status_code} - {response.text}")
                return None
            
            full_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    line_data = json.loads(line)
                    chunk = line_data.get("response", "")
                    full_text += chunk
                    print(chunk, end="", flush=True)
                    
                    if line_data.get("done", False):
                        break
                except json.JSONDecodeError:
                    print(f"\n解析响应时出错: {line}")
            
            print()  # 完成后换行
            return full_text
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"生成文本失败: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return None

def chat(model, messages, stream=False, options=None):
    """使用聊天模式与模型交互"""
    url = f"{OLLAMA_API_URL}/chat"
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    if stream:
        payload["stream"] = stream
    if options:
        payload["options"] = options
    
    try:
        if stream:
            response = requests.post(url, json=payload, stream=True)
            if response.status_code != 200:
                print(f"聊天失败: {response.status_code} - {response.text}")
                return None
            
            full_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    line_data = json.loads(line)
                    if "message" in line_data:
                        chunk = line_data["message"].get("content", "")
                        full_content += chunk
                        print(chunk, end="", flush=True)
                except json.JSONDecodeError:
                    print(f"\n解析响应时出错: {line}")
            
            print()  # 完成后换行
            return full_content
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                if "message" in result:
                    return result["message"].get("content", "")
                return str(result)
            else:
                print(f"聊天失败: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"聊天时出错: {e}")
        return None

def embeddings(model, prompt):
    """获取文本的嵌入向量"""
    url = f"{OLLAMA_API_URL}/embeddings"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            print(f"获取嵌入向量失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        return []

def interactive_chat(model, system_prompt=None):
    """交互式聊天界面"""
    print(f"\n=== 与 {model} 进行交互式聊天 ===")
    print("输入 'exit' 或 'quit' 结束聊天")
    print("输入 'clear' 清除对话历史")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.lower() == "clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("对话历史已清除")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("\n助手: ", end="", flush=True)
        response = chat(model, messages, stream=True)
        
        if response:
            messages.append({"role": "assistant", "content": response})
        else:
            print("获取响应失败")

def main():
    # 列出可用模型
    print("可用模型:")
    models = list_models()
    for model in models:
        print(f"- {model.get('name')}: {model.get('size')}")
    print()
    
    # 选择默认模型
    default_model = "qwen2.5:7b"
    if not any(model.get('name').startswith(default_model) for model in models):
        if models:
            default_model = models[0].get('name')
        else:
            print("没有可用的模型")
            return
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="Simple Ollama Client")
    parser.add_argument("--model", type=str, default=default_model, help="Model to use")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument("--system", type=str, help="System prompt for chat")
    parser.add_argument("--generate", type=str, help="Generate text from prompt")
    parser.add_argument("--stream", action="store_true", help="Stream the output")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for generation")
    
    args = parser.parse_args()
    
    # 设置生成选项
    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }
    
    # 执行请求的操作
    if args.chat:
        interactive_chat(args.model, args.system)
    elif args.generate:
        print(f"\n=== 使用 {args.model} 生成文本 ===")
        print(f"提示词: {args.generate}")
        print("生成中...\n")
        
        start_time = time.time()
        response = generate(args.model, args.generate, args.system, args.stream, options)
        end_time = time.time()
        
        if not args.stream and response:
            print(f"响应: {response}")
        
        print(f"\n耗时: {end_time - start_time:.2f} 秒")
    else:
        # 默认行为：启动交互式聊天
        interactive_chat(args.model, args.system)

if __name__ == "__main__":
    main()