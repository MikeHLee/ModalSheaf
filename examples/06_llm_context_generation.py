#!/usr/bin/env python3
"""
LLM Context Generation Example for ModalSheaf.

This example demonstrates how to use ModalSheaf to transform various common data types
into valid context for popular LLMs:

**Open-Source Models:**
- Llama 3.2 90B Vision (Meta)
- DeepSeek-V3 / DeepSeek-Coder
- GLM-4 (Zhipu AI)

**Frontier Model Providers:**
- OpenAI (GPT-4, GPT-4V)
- Anthropic (Claude 3.5)
- Google (Gemini Pro)
- Amazon Bedrock (Claude, Titan, Llama)

**Data Types Covered:**
- Text documents (web-downloadable)
- JSON/structured data
- Images (from URLs)
- Code files
- CSV/tabular data
- Mixed multimodal content
"""

import json
import urllib.request
import tempfile
import base64
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum, auto

import numpy as np

from modalsheaf import (
    ModalityGraph,
    Modality,
    Transformation,
    TransformationType,
    ConsistencyChecker,
    LocalSection,
    DocumentGluing,
    glue_with_protocol,
)


# ==================== LLM Provider Specifications ====================

class LLMProvider(Enum):
    """Supported LLM providers."""
    LLAMA_90B = auto()
    DEEPSEEK_V3 = auto()
    DEEPSEEK_CODER = auto()
    GLM_4 = auto()
    OPENAI_GPT4 = auto()
    OPENAI_GPT4V = auto()
    ANTHROPIC_CLAUDE = auto()
    GOOGLE_GEMINI = auto()
    BEDROCK_CLAUDE = auto()
    BEDROCK_TITAN = auto()
    BEDROCK_LLAMA = auto()


@dataclass
class LLMSpec:
    """Specification for an LLM's context requirements."""
    provider: LLMProvider
    name: str
    max_context_tokens: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_code: bool = True
    supports_json_mode: bool = False
    supports_function_calling: bool = False
    token_encoding: str = "cl100k_base"
    special_tokens: Dict[str, str] = field(default_factory=dict)
    image_formats: List[str] = field(default_factory=list)
    max_images: int = 0
    notes: str = ""


# LLM Specifications based on current research
LLM_SPECS: Dict[LLMProvider, LLMSpec] = {
    # === Open-Source Models ===
    LLMProvider.LLAMA_90B: LLMSpec(
        provider=LLMProvider.LLAMA_90B,
        name="Llama 3.2 90B Vision",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_code=True,
        supports_json_mode=True,
        token_encoding="llama3",
        special_tokens={
            "bos": "<|begin_of_text|>",
            "eos": "<|end_of_text|>",
            "start_header": "<|start_header_id|>",
            "end_header": "<|end_header_id|>",
            "eot": "<|eot_id|>",
        },
        image_formats=["png", "jpg", "webp", "gif"],
        max_images=5,
        notes="Llama 3.2 Vision supports interleaved image-text"
    ),
    
    LLMProvider.DEEPSEEK_V3: LLMSpec(
        provider=LLMProvider.DEEPSEEK_V3,
        name="DeepSeek-V3",
        max_context_tokens=128_000,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_code=True,
        supports_json_mode=True,
        supports_function_calling=True,
        token_encoding="deepseek",
        special_tokens={
            "bos": "<|begin_of_sentence|>",
            "eos": "<|end_of_sentence|>",
        },
        notes="DeepSeek-V3 excels at reasoning and code"
    ),
    
    LLMProvider.DEEPSEEK_CODER: LLMSpec(
        provider=LLMProvider.DEEPSEEK_CODER,
        name="DeepSeek-Coder-V2",
        max_context_tokens=128_000,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_code=True,
        supports_json_mode=True,
        token_encoding="deepseek",
        special_tokens={
            "fim_prefix": "<|fim_begin|>",
            "fim_middle": "<|fim_hole|>",
            "fim_suffix": "<|fim_end|>",
        },
        notes="Optimized for code completion with FIM support"
    ),
    
    LLMProvider.GLM_4: LLMSpec(
        provider=LLMProvider.GLM_4,
        name="GLM-4",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_code=True,
        supports_json_mode=True,
        supports_function_calling=True,
        token_encoding="glm4",
        special_tokens={
            "system": "[gMASK]<sop>",
        },
        image_formats=["png", "jpg", "webp"],
        max_images=10,
        notes="GLM-4V supports vision; strong multilingual"
    ),
    
    # === Frontier Providers ===
    LLMProvider.OPENAI_GPT4: LLMSpec(
        provider=LLMProvider.OPENAI_GPT4,
        name="GPT-4 Turbo",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_code=True,
        supports_json_mode=True,
        supports_function_calling=True,
        token_encoding="cl100k_base",
        notes="OpenAI's flagship text model"
    ),
    
    LLMProvider.OPENAI_GPT4V: LLMSpec(
        provider=LLMProvider.OPENAI_GPT4V,
        name="GPT-4 Vision",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_code=True,
        supports_json_mode=True,
        supports_function_calling=True,
        token_encoding="cl100k_base",
        image_formats=["png", "jpg", "webp", "gif"],
        max_images=20,
        notes="Supports high/low detail image modes"
    ),
    
    LLMProvider.ANTHROPIC_CLAUDE: LLMSpec(
        provider=LLMProvider.ANTHROPIC_CLAUDE,
        name="Claude 3.5 Sonnet",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_code=True,
        supports_json_mode=True,
        supports_function_calling=True,
        token_encoding="claude",
        image_formats=["png", "jpg", "webp", "gif"],
        max_images=20,
        notes="200K context; excellent at analysis"
    ),
    
    LLMProvider.GOOGLE_GEMINI: LLMSpec(
        provider=LLMProvider.GOOGLE_GEMINI,
        name="Gemini 1.5 Pro",
        max_context_tokens=2_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_code=True,
        supports_json_mode=True,
        supports_function_calling=True,
        token_encoding="gemini",
        image_formats=["png", "jpg", "webp"],
        max_images=3600,  # Can process video frames
        notes="2M context; supports audio/video natively"
    ),
    
    LLMProvider.BEDROCK_CLAUDE: LLMSpec(
        provider=LLMProvider.BEDROCK_CLAUDE,
        name="Claude 3.5 (Bedrock)",
        max_context_tokens=200_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_code=True,
        supports_json_mode=True,
        token_encoding="claude",
        image_formats=["png", "jpg", "webp", "gif"],
        max_images=20,
        notes="Claude via AWS Bedrock"
    ),
    
    LLMProvider.BEDROCK_TITAN: LLMSpec(
        provider=LLMProvider.BEDROCK_TITAN,
        name="Amazon Titan Text",
        max_context_tokens=32_000,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_code=True,
        supports_json_mode=False,
        token_encoding="titan",
        notes="AWS native model"
    ),
    
    LLMProvider.BEDROCK_LLAMA: LLMSpec(
        provider=LLMProvider.BEDROCK_LLAMA,
        name="Llama 3.1 (Bedrock)",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_code=True,
        supports_json_mode=True,
        token_encoding="llama3",
        notes="Llama via AWS Bedrock"
    ),
}


# ==================== Sample Data URLs ====================

SAMPLE_DATA_URLS = {
    # Public domain text
    "text_gutenberg": "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    
    # JSON data
    "json_countries": "https://restcountries.com/v3.1/all",
    "json_github_api": "https://api.github.com/repos/python/cpython",
    
    # CSV data  
    "csv_iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    
    # Images (public domain)
    "image_wikipedia": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    
    # Code (raw GitHub)
    "code_python": "https://raw.githubusercontent.com/python/cpython/main/Lib/json/__init__.py",
}


# ==================== Token Estimation ====================

def estimate_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """
    Estimate token count for text.
    
    Uses a simple heuristic: ~4 characters per token for English.
    For production, use tiktoken or the model's actual tokenizer.
    """
    # Simple heuristic - in production use tiktoken
    char_per_token = {
        "cl100k_base": 4.0,
        "llama3": 3.8,
        "deepseek": 3.5,
        "glm4": 2.5,  # CJK languages use more tokens
        "claude": 4.0,
        "gemini": 4.0,
        "titan": 4.0,
    }
    
    ratio = char_per_token.get(encoding, 4.0)
    return int(len(text) / ratio)


def estimate_image_tokens(width: int, height: int, detail: str = "auto") -> int:
    """
    Estimate token cost for an image.
    
    Based on OpenAI's image token calculation:
    - Low detail: 85 tokens
    - High detail: 170 tokens per 512x512 tile + 85 base
    """
    if detail == "low":
        return 85
    
    # High detail calculation
    # Scale to fit in 2048x2048
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)
    
    # Scale shortest side to 768
    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)
    
    # Count 512x512 tiles
    tiles_x = (width + 511) // 512
    tiles_y = (height + 511) // 512
    
    return 170 * tiles_x * tiles_y + 85


# ==================== Data Loaders ====================

def download_text(url: str, max_chars: int = 100_000) -> str:
    """Download text content from URL."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode('utf-8', errors='ignore')
            return content[:max_chars]
    except Exception as e:
        return f"[Error downloading: {e}]"


def download_json(url: str) -> Any:
    """Download and parse JSON from URL."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}


def download_image_base64(url: str) -> Tuple[str, int, int]:
    """Download image and return as base64 with dimensions."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            b64 = base64.b64encode(data).decode('utf-8')
            
            # Try to get dimensions (simplified - assumes JPEG/PNG)
            # In production, use PIL
            width, height = 320, 240  # Default estimate
            
            return b64, width, height
    except Exception as e:
        return "", 0, 0


def download_csv(url: str) -> List[List[str]]:
    """Download and parse CSV from URL."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode('utf-8')
            lines = content.strip().split('\n')
            return [line.split(',') for line in lines]
    except Exception as e:
        return [[f"Error: {e}"]]


# ==================== Context Formatters ====================

class LLMContextFormatter:
    """
    Formats data into valid context for specific LLM providers.
    
    Uses ModalSheaf to track transformations and information loss.
    """
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.spec = LLM_SPECS[provider]
        self.graph = self._build_modality_graph()
    
    def _build_modality_graph(self) -> ModalityGraph:
        """Build a modality graph for this LLM's supported transformations."""
        graph = ModalityGraph(name=f"{self.spec.name}_context")
        
        # Add modalities
        graph.add_modality("raw_text", dtype="str", description="Raw text input")
        graph.add_modality("json_data", dtype="object", description="JSON/dict data")
        graph.add_modality("csv_data", dtype="object", description="Tabular CSV data")
        graph.add_modality("code", dtype="str", description="Source code")
        graph.add_modality("image_url", dtype="str", description="Image URL")
        graph.add_modality("image_base64", dtype="str", description="Base64 image")
        graph.add_modality("llm_context", dtype="str", description="Formatted LLM context")
        graph.add_modality("llm_message", dtype="object", description="API message format")
        
        # Add transformations
        graph.add_transformation(
            "raw_text", "llm_context",
            forward=self._format_text,
            info_loss="low",
            name="text_to_context"
        )
        
        graph.add_transformation(
            "json_data", "llm_context",
            forward=self._format_json,
            info_loss="low",
            name="json_to_context"
        )
        
        graph.add_transformation(
            "csv_data", "llm_context",
            forward=self._format_csv,
            info_loss="medium",
            name="csv_to_context"
        )
        
        graph.add_transformation(
            "code", "llm_context",
            forward=self._format_code,
            info_loss="low",
            name="code_to_context"
        )
        
        if self.spec.supports_vision:
            graph.add_transformation(
                "image_url", "llm_message",
                forward=self._format_image_url,
                info_loss="medium",
                name="image_url_to_message"
            )
            
            graph.add_transformation(
                "image_base64", "llm_message",
                forward=self._format_image_base64,
                info_loss="medium",
                name="image_base64_to_message"
            )
        
        return graph
    
    def _format_text(self, text: str) -> str:
        """Format raw text for LLM context."""
        # Truncate if needed
        max_tokens = self.spec.max_context_tokens - 1000  # Reserve for response
        estimated = estimate_tokens(text, self.spec.token_encoding)
        
        if estimated > max_tokens:
            # Truncate proportionally
            ratio = max_tokens / estimated
            text = text[:int(len(text) * ratio)]
            text += "\n\n[... truncated ...]"
        
        return text
    
    def _format_json(self, data: Any) -> str:
        """Format JSON data for LLM context."""
        # Pretty print with reasonable indent
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        # Truncate if too large
        max_chars = self.spec.max_context_tokens * 3  # Rough estimate
        if len(formatted) > max_chars:
            formatted = formatted[:max_chars] + "\n... [truncated]"
        
        return f"```json\n{formatted}\n```"
    
    def _format_csv(self, data: List[List[str]]) -> str:
        """Format CSV data as markdown table."""
        if not data:
            return "[Empty CSV]"
        
        # Limit rows for context
        max_rows = 100
        if len(data) > max_rows:
            data = data[:max_rows]
            truncated = True
        else:
            truncated = False
        
        # Build markdown table
        lines = []
        
        # Header
        if data:
            header = data[0]
            lines.append("| " + " | ".join(str(h) for h in header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
        
        # Rows
        for row in data[1:]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        
        if truncated:
            lines.append(f"\n*[Showing first {max_rows} rows]*")
        
        return "\n".join(lines)
    
    def _format_code(self, code: str, language: str = "python") -> str:
        """Format code with syntax highlighting hint."""
        return f"```{language}\n{code}\n```"
    
    def _format_image_url(self, url: str) -> Dict[str, Any]:
        """Format image URL for vision models."""
        if self.provider in [LLMProvider.OPENAI_GPT4V]:
            return {
                "type": "image_url",
                "image_url": {"url": url, "detail": "auto"}
            }
        elif self.provider == LLMProvider.ANTHROPIC_CLAUDE:
            return {
                "type": "image",
                "source": {"type": "url", "url": url}
            }
        elif self.provider == LLMProvider.GOOGLE_GEMINI:
            return {
                "inline_data": {"mime_type": "image/jpeg", "url": url}
            }
        elif self.provider in [LLMProvider.LLAMA_90B, LLMProvider.GLM_4]:
            return {
                "type": "image",
                "url": url
            }
        else:
            return {"url": url}
    
    def _format_image_base64(self, b64_data: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Format base64 image for vision models."""
        if self.provider in [LLMProvider.OPENAI_GPT4V]:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
            }
        elif self.provider == LLMProvider.ANTHROPIC_CLAUDE:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_data
                }
            }
        elif self.provider == LLMProvider.GOOGLE_GEMINI:
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_data
                }
            }
        else:
            return {"base64": b64_data, "mime_type": mime_type}
    
    def format_for_api(
        self,
        system_prompt: str,
        user_content: List[Dict[str, Any]],
        assistant_prefill: str = ""
    ) -> Dict[str, Any]:
        """
        Format a complete API request for this provider.
        
        Args:
            system_prompt: System/instruction prompt
            user_content: List of content items (text, images, etc.)
            assistant_prefill: Optional assistant response prefix
        
        Returns:
            Provider-specific API request format
        """
        if self.provider in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT4V]:
            return self._format_openai(system_prompt, user_content, assistant_prefill)
        elif self.provider == LLMProvider.ANTHROPIC_CLAUDE:
            return self._format_anthropic(system_prompt, user_content, assistant_prefill)
        elif self.provider == LLMProvider.GOOGLE_GEMINI:
            return self._format_gemini(system_prompt, user_content)
        elif self.provider in [LLMProvider.LLAMA_90B, LLMProvider.DEEPSEEK_V3, 
                               LLMProvider.DEEPSEEK_CODER, LLMProvider.GLM_4]:
            return self._format_open_source(system_prompt, user_content)
        elif self.provider in [LLMProvider.BEDROCK_CLAUDE, LLMProvider.BEDROCK_TITAN,
                               LLMProvider.BEDROCK_LLAMA]:
            return self._format_bedrock(system_prompt, user_content)
        else:
            return {"messages": user_content}
    
    def _format_openai(self, system: str, content: List, prefill: str) -> Dict:
        """OpenAI API format."""
        messages = [{"role": "system", "content": system}]
        
        # Build user message content
        user_parts = []
        for item in content:
            if isinstance(item, str):
                user_parts.append({"type": "text", "text": item})
            elif isinstance(item, dict) and "type" in item:
                user_parts.append(item)
        
        if len(user_parts) == 1 and user_parts[0]["type"] == "text":
            messages.append({"role": "user", "content": user_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_parts})
        
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        
        return {
            "model": "gpt-4-turbo-preview",
            "messages": messages,
            "max_tokens": self.spec.max_output_tokens,
        }
    
    def _format_anthropic(self, system: str, content: List, prefill: str) -> Dict:
        """Anthropic API format."""
        user_parts = []
        for item in content:
            if isinstance(item, str):
                user_parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                user_parts.append(item)
        
        messages = [{"role": "user", "content": user_parts}]
        
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        
        return {
            "model": "claude-3-5-sonnet-20241022",
            "system": system,
            "messages": messages,
            "max_tokens": self.spec.max_output_tokens,
        }
    
    def _format_gemini(self, system: str, content: List) -> Dict:
        """Google Gemini API format."""
        parts = []
        
        for item in content:
            if isinstance(item, str):
                parts.append({"text": item})
            elif isinstance(item, dict) and "inline_data" in item:
                parts.append(item)
        
        return {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "maxOutputTokens": self.spec.max_output_tokens,
            }
        }
    
    def _format_open_source(self, system: str, content: List) -> Dict:
        """Format for open-source models (Llama, DeepSeek, GLM)."""
        # Use OpenAI-compatible format (common for vLLM, TGI, etc.)
        messages = [{"role": "system", "content": system}]
        
        user_text = []
        for item in content:
            if isinstance(item, str):
                user_text.append(item)
            elif isinstance(item, dict):
                # Handle images for vision models
                if self.spec.supports_vision and "url" in item:
                    user_text.append(f"[Image: {item.get('url', 'embedded')}]")
        
        messages.append({"role": "user", "content": "\n".join(user_text)})
        
        return {
            "model": self.spec.name.lower().replace(" ", "-"),
            "messages": messages,
            "max_tokens": self.spec.max_output_tokens,
        }
    
    def _format_bedrock(self, system: str, content: List) -> Dict:
        """AWS Bedrock API format."""
        if self.provider == LLMProvider.BEDROCK_CLAUDE:
            return self._format_anthropic(system, content, "")
        elif self.provider == LLMProvider.BEDROCK_TITAN:
            text_content = "\n".join(
                item if isinstance(item, str) else str(item) 
                for item in content
            )
            return {
                "inputText": f"{system}\n\nUser: {text_content}\n\nAssistant:",
                "textGenerationConfig": {
                    "maxTokenCount": self.spec.max_output_tokens,
                }
            }
        else:  # BEDROCK_LLAMA
            return self._format_open_source(system, content)
    
    def estimate_context_usage(self, content: List[Any]) -> Dict[str, Any]:
        """Estimate token usage for given content."""
        total_tokens = 0
        breakdown = []
        
        for item in content:
            if isinstance(item, str):
                tokens = estimate_tokens(item, self.spec.token_encoding)
                breakdown.append({"type": "text", "tokens": tokens})
                total_tokens += tokens
            elif isinstance(item, dict):
                if "image_url" in item or "url" in item or "base64" in item:
                    # Estimate image tokens
                    tokens = estimate_image_tokens(512, 512)  # Default estimate
                    breakdown.append({"type": "image", "tokens": tokens})
                    total_tokens += tokens
        
        return {
            "total_tokens": total_tokens,
            "max_tokens": self.spec.max_context_tokens,
            "usage_percent": (total_tokens / self.spec.max_context_tokens) * 100,
            "remaining_tokens": self.spec.max_context_tokens - total_tokens,
            "breakdown": breakdown,
        }


# ==================== Context Builder with Gluing ====================

class MultiSourceContextBuilder:
    """
    Build LLM context from multiple data sources using ModalSheaf gluing.
    
    This demonstrates how sheaf-theoretic gluing ensures consistency
    when combining data from different sources.
    """
    
    def __init__(self, formatter: LLMContextFormatter):
        self.formatter = formatter
        self.sections: List[LocalSection] = []
    
    def add_text(self, text: str, source_id: str, metadata: Dict = None) -> "MultiSourceContextBuilder":
        """Add a text section."""
        formatted = self.formatter._format_text(text)
        self.sections.append(LocalSection(
            id=source_id,
            data=formatted,
            domain="text",
            metadata=metadata or {"page_number": len(self.sections)}
        ))
        return self
    
    def add_json(self, data: Any, source_id: str, metadata: Dict = None) -> "MultiSourceContextBuilder":
        """Add a JSON section."""
        formatted = self.formatter._format_json(data)
        self.sections.append(LocalSection(
            id=source_id,
            data=formatted,
            domain="json",
            metadata=metadata or {"page_number": len(self.sections)}
        ))
        return self
    
    def add_code(self, code: str, source_id: str, language: str = "python", 
                 metadata: Dict = None) -> "MultiSourceContextBuilder":
        """Add a code section."""
        formatted = self.formatter._format_code(code, language)
        self.sections.append(LocalSection(
            id=source_id,
            data=formatted,
            domain="code",
            metadata=metadata or {"page_number": len(self.sections)}
        ))
        return self
    
    def add_csv(self, data: List[List[str]], source_id: str, 
                metadata: Dict = None) -> "MultiSourceContextBuilder":
        """Add a CSV/table section."""
        formatted = self.formatter._format_csv(data)
        self.sections.append(LocalSection(
            id=source_id,
            data=formatted,
            domain="csv",
            metadata=metadata or {"page_number": len(self.sections)}
        ))
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Glue all sections into a unified context.
        
        Returns:
            Tuple of (glued_context, diagnostics)
        """
        if not self.sections:
            return "", {"error": "No sections added"}
        
        # Use DocumentGluing to assemble sections
        gluing = DocumentGluing(order_key="page_number")
        
        # Create overlaps between consecutive sections
        overlaps = []
        for i in range(len(self.sections) - 1):
            overlaps.append({
                "sections": (self.sections[i].id, self.sections[i + 1].id),
                "region": "sequential",
            })
        
        result = gluing.glue(self.sections, [])
        
        # Get the glued document
        if isinstance(result.global_section, str):
            context = result.global_section
        else:
            context = "\n\n---\n\n".join(s.data for s in self.sections)
        
        # Estimate final token usage
        usage = self.formatter.estimate_context_usage([context])
        
        diagnostics = {
            "gluing_success": result.success,
            "h1_obstruction": result.h1_obstruction,
            "num_sections": len(self.sections),
            "consistency_errors": result.consistency_errors,
            "token_usage": usage,
        }
        
        return context, diagnostics


# ==================== Demo Functions ====================

def demo_text_context():
    """Demonstrate text-to-context transformation."""
    print("\n" + "=" * 60)
    print("Demo: Text Document to LLM Context")
    print("=" * 60)
    
    # Download sample text (first 5000 chars of Pride and Prejudice)
    print("\nDownloading sample text from Project Gutenberg...")
    text = download_text(SAMPLE_DATA_URLS["text_gutenberg"], max_chars=5000)
    
    print(f"Downloaded {len(text)} characters")
    
    # Format for different providers
    providers = [
        LLMProvider.OPENAI_GPT4,
        LLMProvider.ANTHROPIC_CLAUDE,
        LLMProvider.LLAMA_90B,
        LLMProvider.DEEPSEEK_V3,
    ]
    
    for provider in providers:
        formatter = LLMContextFormatter(provider)
        spec = LLM_SPECS[provider]
        
        formatted = formatter._format_text(text)
        usage = formatter.estimate_context_usage([formatted])
        
        print(f"\n{spec.name}:")
        print(f"  Max context: {spec.max_context_tokens:,} tokens")
        print(f"  Estimated usage: {usage['total_tokens']:,} tokens ({usage['usage_percent']:.1f}%)")
        print(f"  Remaining: {usage['remaining_tokens']:,} tokens")


def demo_json_context():
    """Demonstrate JSON-to-context transformation."""
    print("\n" + "=" * 60)
    print("Demo: JSON Data to LLM Context")
    print("=" * 60)
    
    # Use a small sample JSON instead of downloading
    sample_json = {
        "repository": "cpython",
        "language": "Python",
        "stars": 63000,
        "forks": 30000,
        "topics": ["python", "programming-language", "cpython"],
        "license": "PSF-2.0",
        "created_at": "2017-02-10",
    }
    
    print(f"\nSample JSON data: {json.dumps(sample_json, indent=2)[:200]}...")
    
    formatter = LLMContextFormatter(LLMProvider.ANTHROPIC_CLAUDE)
    formatted = formatter._format_json(sample_json)
    
    print(f"\nFormatted for Claude:")
    print(formatted[:500])
    
    usage = formatter.estimate_context_usage([formatted])
    print(f"\nToken usage: {usage['total_tokens']} tokens")


def demo_code_context():
    """Demonstrate code-to-context transformation."""
    print("\n" + "=" * 60)
    print("Demo: Code to LLM Context")
    print("=" * 60)
    
    # Sample Python code
    sample_code = '''
def fibonacci(n: int) -> list[int]:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib


def main():
    result = fibonacci(10)
    print(f"First 10 Fibonacci numbers: {result}")


if __name__ == "__main__":
    main()
'''
    
    # Format for code-focused models
    providers = [
        LLMProvider.DEEPSEEK_CODER,
        LLMProvider.OPENAI_GPT4,
        LLMProvider.ANTHROPIC_CLAUDE,
    ]
    
    for provider in providers:
        formatter = LLMContextFormatter(provider)
        spec = LLM_SPECS[provider]
        
        formatted = formatter._format_code(sample_code, "python")
        
        print(f"\n{spec.name}:")
        print(f"  Code support: {spec.supports_code}")
        print(f"  JSON mode: {spec.supports_json_mode}")
        if spec.notes:
            print(f"  Notes: {spec.notes}")


def demo_csv_context():
    """Demonstrate CSV-to-context transformation."""
    print("\n" + "=" * 60)
    print("Demo: CSV/Tabular Data to LLM Context")
    print("=" * 60)
    
    # Sample CSV data (Iris dataset format)
    sample_csv = [
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        ["5.1", "3.5", "1.4", "0.2", "setosa"],
        ["4.9", "3.0", "1.4", "0.2", "setosa"],
        ["7.0", "3.2", "4.7", "1.4", "versicolor"],
        ["6.4", "3.2", "4.5", "1.5", "versicolor"],
        ["6.3", "3.3", "6.0", "2.5", "virginica"],
        ["5.8", "2.7", "5.1", "1.9", "virginica"],
    ]
    
    formatter = LLMContextFormatter(LLMProvider.GOOGLE_GEMINI)
    formatted = formatter._format_csv(sample_csv)
    
    print("\nFormatted as Markdown table:")
    print(formatted)
    
    usage = formatter.estimate_context_usage([formatted])
    print(f"\nToken usage: {usage['total_tokens']} tokens")
    print(f"Gemini context: {LLM_SPECS[LLMProvider.GOOGLE_GEMINI].max_context_tokens:,} tokens (2M!)")


def demo_multimodal_context():
    """Demonstrate multimodal context for vision models."""
    print("\n" + "=" * 60)
    print("Demo: Multimodal Context (Text + Image)")
    print("=" * 60)
    
    vision_providers = [
        LLMProvider.OPENAI_GPT4V,
        LLMProvider.ANTHROPIC_CLAUDE,
        LLMProvider.GOOGLE_GEMINI,
        LLMProvider.LLAMA_90B,
        LLMProvider.GLM_4,
    ]
    
    image_url = SAMPLE_DATA_URLS["image_wikipedia"]
    
    for provider in vision_providers:
        formatter = LLMContextFormatter(provider)
        spec = LLM_SPECS[provider]
        
        if spec.supports_vision:
            image_content = formatter._format_image_url(image_url)
            
            print(f"\n{spec.name}:")
            print(f"  Vision support: Yes")
            print(f"  Max images: {spec.max_images}")
            print(f"  Image formats: {spec.image_formats}")
            print(f"  Image content format: {list(image_content.keys())}")


def demo_multi_source_gluing():
    """Demonstrate gluing multiple data sources into unified context."""
    print("\n" + "=" * 60)
    print("Demo: Multi-Source Context Gluing")
    print("=" * 60)
    
    formatter = LLMContextFormatter(LLMProvider.ANTHROPIC_CLAUDE)
    builder = MultiSourceContextBuilder(formatter)
    
    # Add various data sources
    builder.add_text(
        "This analysis examines Python repository statistics and code patterns.",
        source_id="intro"
    )
    
    builder.add_json(
        {"repo": "cpython", "stars": 63000, "language": "Python"},
        source_id="repo_stats"
    )
    
    builder.add_code(
        "def hello():\n    print('Hello, World!')",
        source_id="sample_code",
        language="python"
    )
    
    builder.add_csv(
        [["metric", "value"], ["lines", "1.5M"], ["contributors", "2000"]],
        source_id="metrics_table"
    )
    
    # Build unified context
    context, diagnostics = builder.build()
    
    print(f"\nGluing result:")
    print(f"  Success: {diagnostics['gluing_success']}")
    print(f"  H¹ obstruction: {diagnostics['h1_obstruction']}")
    print(f"  Sections glued: {diagnostics['num_sections']}")
    print(f"  Token usage: {diagnostics['token_usage']['total_tokens']} tokens")
    print(f"  Usage percent: {diagnostics['token_usage']['usage_percent']:.2f}%")
    
    print(f"\nGlued context preview (first 500 chars):")
    print("-" * 40)
    print(context[:500])
    print("-" * 40)


def demo_api_request_formats():
    """Demonstrate complete API request formatting for each provider."""
    print("\n" + "=" * 60)
    print("Demo: Complete API Request Formats")
    print("=" * 60)
    
    system_prompt = "You are a helpful assistant that analyzes data."
    user_content = [
        "Please analyze this Python code:",
        "```python\ndef add(a, b): return a + b\n```"
    ]
    
    providers = [
        LLMProvider.OPENAI_GPT4,
        LLMProvider.ANTHROPIC_CLAUDE,
        LLMProvider.GOOGLE_GEMINI,
        LLMProvider.DEEPSEEK_V3,
        LLMProvider.BEDROCK_TITAN,
    ]
    
    for provider in providers:
        formatter = LLMContextFormatter(provider)
        spec = LLM_SPECS[provider]
        
        request = formatter.format_for_api(system_prompt, user_content)
        
        print(f"\n{spec.name} API format:")
        print(json.dumps(request, indent=2, default=str)[:600])
        if len(json.dumps(request)) > 600:
            print("  ...")


def demo_context_limits():
    """Demonstrate context limit handling across providers."""
    print("\n" + "=" * 60)
    print("Demo: Context Limits Comparison")
    print("=" * 60)
    
    print("\n{:<25} {:>15} {:>15} {:>10}".format(
        "Provider", "Max Context", "Max Output", "Vision"
    ))
    print("-" * 70)
    
    for provider, spec in sorted(LLM_SPECS.items(), 
                                  key=lambda x: x[1].max_context_tokens, 
                                  reverse=True):
        vision = "Yes" if spec.supports_vision else "No"
        print("{:<25} {:>15,} {:>15,} {:>10}".format(
            spec.name, spec.max_context_tokens, spec.max_output_tokens, vision
        ))


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("ModalSheaf LLM Context Generation Examples")
    print("=" * 60)
    print("""
This example demonstrates transforming various data types into
valid LLM context for multiple providers:

- Open-source: Llama 90B, DeepSeek, GLM-4
- Frontier: OpenAI, Anthropic, Google, AWS Bedrock

Using ModalSheaf to track transformations and ensure consistency.
""")
    
    # Run demos
    demo_context_limits()
    demo_text_context()
    demo_json_context()
    demo_code_context()
    demo_csv_context()
    demo_multimodal_context()
    demo_multi_source_gluing()
    demo_api_request_formats()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
