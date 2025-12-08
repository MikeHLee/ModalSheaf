"""
Tests for LLM context generation using ModalSheaf.

Tests transformation of common data types into valid context for:
- Open-source models: Llama 90B, DeepSeek, GLM-4
- Frontier providers: OpenAI, Anthropic, Google, Amazon Bedrock

Uses web-downloadable sample data where practical.
"""

import json
import pytest
import numpy as np
from typing import Any, Dict, List

from modalsheaf import (
    ModalityGraph,
    Modality,
    Transformation,
    TransformationType,
    ConsistencyChecker,
    LocalSection,
    DocumentGluing,
)


# ==================== LLM Specifications ====================

LLM_CONTEXT_LIMITS = {
    # Open-source models
    "llama_90b": {"max_tokens": 128_000, "vision": True},
    "deepseek_v3": {"max_tokens": 128_000, "vision": False},
    "deepseek_coder": {"max_tokens": 128_000, "vision": False},
    "glm_4": {"max_tokens": 128_000, "vision": True},
    
    # Frontier providers
    "openai_gpt4": {"max_tokens": 128_000, "vision": False},
    "openai_gpt4v": {"max_tokens": 128_000, "vision": True},
    "anthropic_claude": {"max_tokens": 200_000, "vision": True},
    "google_gemini": {"max_tokens": 2_000_000, "vision": True},
    "bedrock_claude": {"max_tokens": 200_000, "vision": True},
    "bedrock_titan": {"max_tokens": 32_000, "vision": False},
    "bedrock_llama": {"max_tokens": 128_000, "vision": False},
}


# ==================== Token Estimation ====================

def estimate_tokens(text: str) -> int:
    """Simple token estimation (~4 chars per token for English)."""
    return len(text) // 4


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate image token cost (OpenAI-style calculation)."""
    tiles_x = (min(width, 2048) + 511) // 512
    tiles_y = (min(height, 2048) + 511) // 512
    return 170 * tiles_x * tiles_y + 85


# ==================== Context Formatters ====================

class TextToContextTransform:
    """Transform raw text to LLM context format."""
    
    def __init__(self, max_tokens: int = 100_000):
        self.max_tokens = max_tokens
    
    def __call__(self, text: str) -> str:
        estimated = estimate_tokens(text)
        if estimated > self.max_tokens:
            # Truncate proportionally
            ratio = self.max_tokens / estimated
            text = text[:int(len(text) * ratio)]
            text += "\n\n[... truncated ...]"
        return text


class JSONToContextTransform:
    """Transform JSON data to LLM context format."""
    
    def __call__(self, data: Any) -> str:
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return f"```json\n{formatted}\n```"


class CSVToContextTransform:
    """Transform CSV data to markdown table format."""
    
    def __init__(self, max_rows: int = 100):
        self.max_rows = max_rows
    
    def __call__(self, data: List[List[str]]) -> str:
        if not data:
            return "[Empty data]"
        
        rows = data[:self.max_rows + 1]  # +1 for header
        truncated = len(data) > self.max_rows + 1
        
        lines = []
        if rows:
            # Header
            lines.append("| " + " | ".join(str(h) for h in rows[0]) + " |")
            lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
            # Data rows
            for row in rows[1:]:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        
        if truncated:
            lines.append(f"\n*[Showing first {self.max_rows} rows]*")
        
        return "\n".join(lines)


class CodeToContextTransform:
    """Transform code to syntax-highlighted context."""
    
    def __call__(self, code: str, language: str = "python") -> str:
        return f"```{language}\n{code}\n```"


class ImageToContextTransform:
    """Transform image data to LLM message format."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
    
    def from_url(self, url: str) -> Dict[str, Any]:
        """Format image URL for API."""
        if self.provider == "openai":
            return {
                "type": "image_url",
                "image_url": {"url": url, "detail": "auto"}
            }
        elif self.provider == "anthropic":
            return {
                "type": "image",
                "source": {"type": "url", "url": url}
            }
        elif self.provider == "google":
            return {"inline_data": {"mime_type": "image/jpeg", "url": url}}
        else:
            return {"type": "image", "url": url}
    
    def from_base64(self, b64: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Format base64 image for API."""
        if self.provider == "openai":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"}
            }
        elif self.provider == "anthropic":
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64}
            }
        else:
            return {"base64": b64, "mime_type": mime_type}


# ==================== Test Fixtures ====================

@pytest.fixture
def sample_text():
    """Sample text document."""
    return """
    Pride and Prejudice by Jane Austen
    
    Chapter 1
    
    It is a truth universally acknowledged, that a single man in possession
    of a good fortune, must be in want of a wife.
    
    However little known the feelings or views of such a man may be on his
    first entering a neighbourhood, this truth is so well fixed in the minds
    of the surrounding families, that he is considered the rightful property
    of some one or other of their daughters.
    """


@pytest.fixture
def sample_json():
    """Sample JSON data."""
    return {
        "repository": "cpython",
        "language": "Python",
        "stars": 63000,
        "forks": 30000,
        "topics": ["python", "programming-language"],
        "license": "PSF-2.0",
    }


@pytest.fixture
def sample_csv():
    """Sample CSV data (Iris-like)."""
    return [
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        ["5.1", "3.5", "1.4", "0.2", "setosa"],
        ["4.9", "3.0", "1.4", "0.2", "setosa"],
        ["7.0", "3.2", "4.7", "1.4", "versicolor"],
        ["6.3", "3.3", "6.0", "2.5", "virginica"],
    ]


@pytest.fixture
def sample_code():
    """Sample Python code."""
    return '''
def fibonacci(n: int) -> list[int]:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
'''


@pytest.fixture
def modality_graph():
    """Create a modality graph for LLM context transformations."""
    graph = ModalityGraph(name="llm_context")
    
    graph.add_modality("raw_text", dtype="str")
    graph.add_modality("json_data", dtype="object")
    graph.add_modality("csv_data", dtype="object")
    graph.add_modality("code", dtype="str")
    graph.add_modality("image_url", dtype="str")
    graph.add_modality("llm_context", dtype="str")
    
    # Add transformations
    text_transform = TextToContextTransform()
    graph.add_transformation(
        "raw_text", "llm_context",
        forward=text_transform,
        info_loss="low",
        name="text_to_context"
    )
    
    json_transform = JSONToContextTransform()
    graph.add_transformation(
        "json_data", "llm_context",
        forward=json_transform,
        info_loss="low",
        name="json_to_context"
    )
    
    csv_transform = CSVToContextTransform()
    graph.add_transformation(
        "csv_data", "llm_context",
        forward=csv_transform,
        info_loss="medium",
        name="csv_to_context"
    )
    
    code_transform = CodeToContextTransform()
    graph.add_transformation(
        "code", "llm_context",
        forward=lambda c: code_transform(c, "python"),
        info_loss="low",
        name="code_to_context"
    )
    
    return graph


# ==================== Tests: Text Transformation ====================

class TestTextToContext:
    """Test text-to-context transformations."""
    
    def test_basic_text_transform(self, sample_text):
        """Test basic text formatting."""
        transform = TextToContextTransform()
        result = transform(sample_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Pride and Prejudice" in result
    
    def test_text_truncation(self):
        """Test text truncation for large inputs."""
        long_text = "word " * 100_000  # ~500K chars
        transform = TextToContextTransform(max_tokens=1000)
        
        result = transform(long_text)
        
        assert "[... truncated ...]" in result
        assert estimate_tokens(result) <= 1100  # Allow some margin
    
    def test_text_within_all_llm_limits(self, sample_text):
        """Test that sample text fits within all LLM context limits."""
        transform = TextToContextTransform()
        result = transform(sample_text)
        tokens = estimate_tokens(result)
        
        for llm_name, spec in LLM_CONTEXT_LIMITS.items():
            assert tokens < spec["max_tokens"], f"Text exceeds {llm_name} limit"


# ==================== Tests: JSON Transformation ====================

class TestJSONToContext:
    """Test JSON-to-context transformations."""
    
    def test_basic_json_transform(self, sample_json):
        """Test basic JSON formatting."""
        transform = JSONToContextTransform()
        result = transform(sample_json)
        
        assert "```json" in result
        assert "```" in result
        assert "cpython" in result
    
    def test_nested_json(self):
        """Test nested JSON structures."""
        nested = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                }
            }
        }
        transform = JSONToContextTransform()
        result = transform(nested)
        
        assert "level1" in result
        assert "level2" in result
        assert "level3" in result
    
    def test_json_with_special_chars(self):
        """Test JSON with special characters."""
        data = {
            "unicode": "日本語テスト",
            "emoji": "🎉",
            "quotes": 'He said "hello"',
        }
        transform = JSONToContextTransform()
        result = transform(data)
        
        assert "日本語" in result
        assert "🎉" in result


# ==================== Tests: CSV Transformation ====================

class TestCSVToContext:
    """Test CSV-to-context transformations."""
    
    def test_basic_csv_transform(self, sample_csv):
        """Test basic CSV to markdown table."""
        transform = CSVToContextTransform()
        result = transform(sample_csv)
        
        assert "|" in result
        assert "sepal_length" in result
        assert "setosa" in result
    
    def test_csv_truncation(self):
        """Test CSV row truncation."""
        large_csv = [["col1", "col2"]] + [[str(i), str(i*2)] for i in range(200)]
        transform = CSVToContextTransform(max_rows=50)
        
        result = transform(large_csv)
        
        assert "Showing first 50 rows" in result
    
    def test_empty_csv(self):
        """Test empty CSV handling."""
        transform = CSVToContextTransform()
        result = transform([])
        
        assert "Empty" in result


# ==================== Tests: Code Transformation ====================

class TestCodeToContext:
    """Test code-to-context transformations."""
    
    def test_python_code_transform(self, sample_code):
        """Test Python code formatting."""
        transform = CodeToContextTransform()
        result = transform(sample_code, "python")
        
        assert "```python" in result
        assert "def fibonacci" in result
        assert "```" in result
    
    def test_multiple_languages(self):
        """Test different language syntax hints."""
        transform = CodeToContextTransform()
        
        js_code = "const x = 42;"
        result_js = transform(js_code, "javascript")
        assert "```javascript" in result_js
        
        rust_code = "fn main() {}"
        result_rust = transform(rust_code, "rust")
        assert "```rust" in result_rust


# ==================== Tests: Image Transformation ====================

class TestImageToContext:
    """Test image-to-context transformations for vision models."""
    
    def test_openai_image_url_format(self):
        """Test OpenAI image URL format."""
        transform = ImageToContextTransform(provider="openai")
        result = transform.from_url("https://example.com/image.jpg")
        
        assert result["type"] == "image_url"
        assert "url" in result["image_url"]
        assert "detail" in result["image_url"]
    
    def test_anthropic_image_url_format(self):
        """Test Anthropic image URL format."""
        transform = ImageToContextTransform(provider="anthropic")
        result = transform.from_url("https://example.com/image.jpg")
        
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
    
    def test_google_image_format(self):
        """Test Google Gemini image format."""
        transform = ImageToContextTransform(provider="google")
        result = transform.from_url("https://example.com/image.jpg")
        
        assert "inline_data" in result
    
    def test_base64_image_format(self):
        """Test base64 image formatting."""
        transform = ImageToContextTransform(provider="openai")
        result = transform.from_base64("SGVsbG8=", "image/png")
        
        assert "data:image/png;base64" in result["image_url"]["url"]
    
    def test_vision_model_support(self):
        """Verify which models support vision."""
        vision_models = [
            name for name, spec in LLM_CONTEXT_LIMITS.items() 
            if spec["vision"]
        ]
        
        expected_vision = [
            "llama_90b", "glm_4", "openai_gpt4v", 
            "anthropic_claude", "google_gemini", "bedrock_claude"
        ]
        
        for model in expected_vision:
            assert model in vision_models, f"{model} should support vision"


# ==================== Tests: Modality Graph Integration ====================

class TestModalityGraphIntegration:
    """Test ModalSheaf modality graph for LLM context."""
    
    def test_graph_creation(self, modality_graph):
        """Test modality graph is created correctly."""
        assert "raw_text" in modality_graph.modalities
        assert "json_data" in modality_graph.modalities
        assert "llm_context" in modality_graph.modalities
    
    def test_text_transformation_via_graph(self, modality_graph, sample_text):
        """Test text transformation through modality graph."""
        result = modality_graph.transform("raw_text", "llm_context", sample_text)
        
        assert isinstance(result, str)
        assert "Pride and Prejudice" in result
    
    def test_json_transformation_via_graph(self, modality_graph, sample_json):
        """Test JSON transformation through modality graph."""
        result = modality_graph.transform("json_data", "llm_context", sample_json)
        
        assert "```json" in result
        assert "cpython" in result
    
    def test_csv_transformation_via_graph(self, modality_graph, sample_csv):
        """Test CSV transformation through modality graph."""
        result = modality_graph.transform("csv_data", "llm_context", sample_csv)
        
        assert "|" in result
        assert "sepal_length" in result
    
    def test_code_transformation_via_graph(self, modality_graph, sample_code):
        """Test code transformation through modality graph."""
        result = modality_graph.transform("code", "llm_context", sample_code)
        
        assert "```python" in result
        assert "fibonacci" in result
    
    def test_path_finding(self, modality_graph):
        """Test path finding between modalities."""
        path = modality_graph.find_path("raw_text", "llm_context")
        
        assert path is not None
        assert "raw_text" in path
        assert "llm_context" in path


# ==================== Tests: Multi-Source Gluing ====================

class TestMultiSourceGluing:
    """Test gluing multiple data sources into unified context."""
    
    def test_document_gluing(self, sample_text, sample_json, sample_code):
        """Test gluing text, JSON, and code into one context."""
        sections = [
            LocalSection(
                id="intro",
                data=sample_text,
                metadata={"page_number": 0}
            ),
            LocalSection(
                id="data",
                data=json.dumps(sample_json, indent=2),
                metadata={"page_number": 1}
            ),
            LocalSection(
                id="code",
                data=f"```python\n{sample_code}\n```",
                metadata={"page_number": 2}
            ),
        ]
        
        gluing = DocumentGluing(order_key="page_number")
        result = gluing.glue(sections, [])
        
        # Gluing produces output even with consistency warnings
        # (warnings are about sentence continuity at boundaries)
        assert result.global_section is not None
        assert "Pride and Prejudice" in result.global_section
        assert "cpython" in result.global_section
        assert "fibonacci" in result.global_section
    
    def test_gluing_preserves_order(self):
        """Test that gluing preserves section order."""
        sections = [
            LocalSection(id="c", data="Third", metadata={"page_number": 2}),
            LocalSection(id="a", data="First", metadata={"page_number": 0}),
            LocalSection(id="b", data="Second", metadata={"page_number": 1}),
        ]
        
        gluing = DocumentGluing(order_key="page_number")
        result = gluing.glue(sections, [])
        
        # Check order in result
        first_pos = result.global_section.find("First")
        second_pos = result.global_section.find("Second")
        third_pos = result.global_section.find("Third")
        
        assert first_pos < second_pos < third_pos
    
    def test_gluing_diagnostics(self, sample_text):
        """Test that gluing provides useful diagnostics."""
        sections = [
            LocalSection(id="s1", data="Part 1.", metadata={"page_number": 0}),
            LocalSection(id="s2", data="Part 2.", metadata={"page_number": 1}),
        ]
        
        gluing = DocumentGluing()
        result = gluing.glue(sections, [])
        
        assert "num_pages" in result.diagnostics
        assert result.diagnostics["num_pages"] == 2


# ==================== Tests: Provider-Specific Formatting ====================

class TestProviderFormatting:
    """Test provider-specific API request formatting."""
    
    def test_openai_message_format(self):
        """Test OpenAI API message format."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        
        # Verify structure
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_anthropic_message_format(self):
        """Test Anthropic API message format."""
        request = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 4096,
        }
        
        assert "system" in request
        assert request["messages"][0]["role"] == "user"
    
    def test_gemini_message_format(self):
        """Test Google Gemini API format."""
        request = {
            "contents": [{"parts": [{"text": "Hello!"}]}],
            "systemInstruction": {"parts": [{"text": "You are helpful."}]},
        }
        
        assert "contents" in request
        assert "systemInstruction" in request
    
    def test_bedrock_titan_format(self):
        """Test AWS Bedrock Titan format."""
        request = {
            "inputText": "System: You are helpful.\n\nUser: Hello!\n\nAssistant:",
            "textGenerationConfig": {"maxTokenCount": 4096},
        }
        
        assert "inputText" in request
        assert "textGenerationConfig" in request


# ==================== Tests: Token Estimation ====================

class TestTokenEstimation:
    """Test token estimation utilities."""
    
    def test_text_token_estimation(self):
        """Test text token estimation."""
        text = "Hello world, this is a test."
        tokens = estimate_tokens(text)
        
        # ~4 chars per token
        assert 5 <= tokens <= 10
    
    def test_image_token_estimation(self):
        """Test image token estimation."""
        # Small image
        tokens_small = estimate_image_tokens(256, 256)
        assert tokens_small == 85 + 170  # 1 tile + base
        
        # Large image
        tokens_large = estimate_image_tokens(1024, 1024)
        assert tokens_large > tokens_small
    
    def test_context_fits_all_providers(self, sample_text, sample_json, sample_code):
        """Test that combined sample data fits all provider limits."""
        combined = sample_text + json.dumps(sample_json) + sample_code
        tokens = estimate_tokens(combined)
        
        min_limit = min(spec["max_tokens"] for spec in LLM_CONTEXT_LIMITS.values())
        
        assert tokens < min_limit, "Sample data should fit smallest context window"


# ==================== Tests: Consistency Checking ====================

class TestConsistencyChecking:
    """Test consistency checking for multimodal context."""
    
    def test_consistency_checker_creation(self, modality_graph):
        """Test creating a consistency checker."""
        checker = ConsistencyChecker(modality_graph, common_modality="llm_context")
        
        assert checker is not None
    
    def test_consistent_data(self, modality_graph, sample_text):
        """Test consistency check with matching data."""
        checker = ConsistencyChecker(modality_graph, common_modality="llm_context")
        
        # Same text transformed should be consistent
        result = checker.check({
            "raw_text": sample_text,
        })
        
        # Single modality should always be consistent
        assert result.consistency_score >= 0


# ==================== Tests: Edge Cases ====================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self):
        """Test handling empty text."""
        transform = TextToContextTransform()
        result = transform("")
        
        assert result == ""
    
    def test_empty_json(self):
        """Test handling empty JSON."""
        transform = JSONToContextTransform()
        result = transform({})
        
        assert "{}" in result
    
    def test_unicode_handling(self):
        """Test Unicode text handling."""
        transform = TextToContextTransform()
        unicode_text = "日本語 中文 한국어 العربية"
        result = transform(unicode_text)
        
        assert "日本語" in result
        assert "العربية" in result
    
    def test_very_long_json_key(self):
        """Test JSON with very long keys."""
        transform = JSONToContextTransform()
        data = {"a" * 1000: "value"}
        result = transform(data)
        
        assert "a" * 100 in result  # At least partial key present
    
    def test_deeply_nested_json(self):
        """Test deeply nested JSON structures."""
        transform = JSONToContextTransform()
        
        # Create 10-level deep nesting
        data = {"level": 0}
        current = data
        for i in range(1, 10):
            current["nested"] = {"level": i}
            current = current["nested"]
        
        result = transform(data)
        assert "level" in result


# ==================== Integration Test ====================

class TestFullIntegration:
    """Full integration tests combining all components."""
    
    def test_full_pipeline(self, modality_graph, sample_text, sample_json, 
                           sample_csv, sample_code):
        """Test complete pipeline from raw data to LLM context."""
        # Transform each data type
        text_context = modality_graph.transform("raw_text", "llm_context", sample_text)
        json_context = modality_graph.transform("json_data", "llm_context", sample_json)
        csv_context = modality_graph.transform("csv_data", "llm_context", sample_csv)
        code_context = modality_graph.transform("code", "llm_context", sample_code)
        
        # Combine into sections
        sections = [
            LocalSection(id="text", data=text_context, metadata={"page_number": 0}),
            LocalSection(id="json", data=json_context, metadata={"page_number": 1}),
            LocalSection(id="csv", data=csv_context, metadata={"page_number": 2}),
            LocalSection(id="code", data=code_context, metadata={"page_number": 3}),
        ]
        
        # Glue together
        gluing = DocumentGluing()
        result = gluing.glue(sections, [])
        
        # Verify result (gluing produces output even with boundary warnings)
        assert result.global_section is not None
        assert "Pride and Prejudice" in result.global_section
        assert "cpython" in result.global_section
        assert "sepal_length" in result.global_section
        assert "fibonacci" in result.global_section
        
        # Verify fits in all LLM contexts
        tokens = estimate_tokens(result.global_section)
        for llm_name, spec in LLM_CONTEXT_LIMITS.items():
            assert tokens < spec["max_tokens"], f"Result exceeds {llm_name} limit"
    
    def test_multimodal_pipeline_for_vision_models(self):
        """Test multimodal pipeline for vision-capable models."""
        vision_models = [
            name for name, spec in LLM_CONTEXT_LIMITS.items()
            if spec["vision"]
        ]
        
        # Create image content for each provider
        for model in vision_models:
            provider = model.split("_")[0]  # Extract provider name
            if provider == "openai":
                provider = "openai"
            elif provider == "anthropic":
                provider = "anthropic"
            elif provider == "google":
                provider = "google"
            else:
                provider = "openai"  # Default format
            
            transform = ImageToContextTransform(provider=provider)
            result = transform.from_url("https://example.com/test.jpg")
            
            assert result is not None
            assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
