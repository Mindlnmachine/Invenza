"""
Enhanced LLM client supporting both Ollama models and local fine-tuned qwen with LoRA.

This module provides:
- OllamaClient: For llama3.2 via Ollama API
- LocalQwenClient: For fine-tuned qwen3:4b with LoRA adapters
- MultiModelLLMClient: Unified interface for both models
"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings


class ModelType(str, Enum):
    """Supported model types."""
    QWEN_4B = "qwen3:4b"
    LLAMA_32 = "llama3.2"


class OllamaClient:
    """Client for Ollama-hosted models (llama3.2)."""
    
    def __init__(self, base_url: str | None = None, model_name: str | None = None):
        configured_url = (base_url or settings.ollama_base_url).rstrip("/")
        if "host.docker.internal" in configured_url and not Path("/.dockerenv").exists():
            configured_url = configured_url.replace("host.docker.internal", "127.0.0.1")
        self.base_url = configured_url
        self.model_name = model_name or settings.ollama_model
    
    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> tuple[str, float]:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, inference_time_ms)
        """
        payload = {
            "model": model or self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            return data.get("response", ""), inference_time
        except Exception as e:
            raise RuntimeError(f"Ollama inference failed: {str(e)}")


class LocalQwenClient:
    """Client for local fine-tuned qwen3:4b with LoRA adapters."""
    
    def __init__(
        self,
        model_path: str = "qwen/qwen2.5-3b",
        lora_adapter_path: Optional[Path] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        """
        Initialize local qwen client.
        
        Args:
            model_path: Path to qwen model (HuggingFace model ID or local path)
            lora_adapter_path: Path to LoRA adapter (if None, use base model)
            device: Device to load model on ('cuda', 'cpu', 'auto')
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_path = model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading qwen from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.load_in_4bit:
            from bitsandbytes.cuda_setup import get_device
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
        
        # Load LoRA adapter if provided
        if self.lora_adapter_path and self.lora_adapter_path.exists():
            print(f"Loading LoRA adapter from {self.lora_adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(self.lora_adapter_path))
        
        self.model.eval()
        print(f"✓ Model loaded: {self.model_path}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> tuple[str, float]:
        """
        Generate response using local qwen model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, inference_time_ms)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Decode
            response_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response_text, inference_time
        except Exception as e:
            raise RuntimeError(f"Qwen inference failed: {str(e)}")


class MultiModelLLMClient:
    """Unified client supporting both Ollama and local qwen models."""
    
    def __init__(
        self,
        ollama_base_url: str = "http://host.docker.internal:11434",
        qwen_model_path: str = "qwen/qwen2.5-3b",
        qwen_lora_adapter_path: Optional[Path] = None,
        enable_qwen_local: bool = False,  # Only load local qwen if explicitly enabled
    ):
        """
        Initialize multi-model client.
        
        Args:
            ollama_base_url: Ollama API base URL
            qwen_model_path: Path to qwen model
            qwen_lora_adapter_path: Path to LoRA adapter for qwen
            enable_qwen_local: Whether to load qwen locally (memory-intensive)
        """
        self.ollama_client = OllamaClient(ollama_base_url, settings.ollama_model)
        self.qwen_ollama_client = OllamaClient(ollama_base_url, settings.qwen_ollama_model)
        
        self.qwen_client = None
        if enable_qwen_local:
            self.qwen_client = LocalQwenClient(
                model_path=qwen_model_path,
                lora_adapter_path=qwen_lora_adapter_path,
            )
    
    def generate(
        self,
        prompt: str,
        model: ModelType = ModelType.LLAMA_32,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> tuple[str, float, str]:
        """
        Generate response using specified model.
        
        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, inference_time_ms, model_used)
        """
        if model == ModelType.LLAMA_32:
            response, inference_time = self.ollama_client.generate(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
            return response, inference_time, ModelType.LLAMA_32.value
        
        elif model == ModelType.QWEN_4B:
            if self.qwen_client is None:
                response, inference_time = self.qwen_ollama_client.generate(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
                return response, inference_time, ModelType.QWEN_4B.value
            response, inference_time = self.qwen_client.generate(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
            return response, inference_time, ModelType.QWEN_4B.value
        
        else:
            raise ValueError(f"Unsupported model: {model}")


def build_idea_prompt(
    topic: str,
    evidence: list[dict[str, Any]],
    conversation: list[dict[str, str]] | None = None,
) -> str:
    """
    Build prompt for idea generation.
    
    Args:
        topic: User query/topic
        evidence: List of retrieved patent evidence
        
    Returns:
        Formatted prompt
    """
    evidence_text = "\n\n".join(
        (
            f"Patent {item['patent_id']} | Title: {item['title']}\n"
            f"Assignee: {item['assignee'] or 'unknown'}\n"
            f"Abstract: {item['abstract'] or item['text'] or 'N/A'}\n"
            f"CPC/IPC: {', '.join(item.get('cpc_codes', []) + item.get('ipc_codes', [])) or 'N/A'}"
        )
        for item in evidence
    )
    conversation_text = "\n".join(
        f"{item['role'].title()}: {item['content']}"
        for item in (conversation or [])[-8:]
    ) or "No earlier conversation."

    return f"""
You are a patent research assistant. Answer the user's latest message using the conversation and patent evidence below.
If the latest message asks a follow-up question, answer that question directly instead of generating a new unrelated idea.
Ground factual claims in the supplied patent evidence and clearly distinguish practical recommendations from patent-backed facts.

Latest user message: {topic}

Conversation:
{conversation_text}

Use these patents as your source basis and do not invent unsupported claims.

Patent evidence:
{evidence_text}

Return a detailed output with:
1. A concept name
2. A detailed answer of 4–6 paragraphs covering the solution, target customer, business model, implementation, risks, and differentiation
3. An evidence statement explaining what each of the 2–3 strongest patents contributes
4. Explicitly name the patent IDs in the evidence statement
5. Only claim what is clearly supported by the patents

Format your response as JSON with keys:
{{"concept_name": "...", "summary": "...", "evidence": "..."}}
"""
