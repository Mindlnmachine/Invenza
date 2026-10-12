"""
Tests for model router and multi-model inference.

Usage:
    pytest tests/test_router.py -v
"""

import pytest

from app.model_router import ModelRouter, ModelType, QueryComplexity, get_router


class TestModelRouter:
    """Test model routing logic."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = ModelRouter(
            simple_token_threshold=50,
            fallback_model=ModelType.LLAMA_32,
            enable_logging=False,
        )
        assert router.simple_token_threshold == 50
        assert router.fallback_model == ModelType.LLAMA_32
    
    def test_complexity_classification_simple(self):
        """Test that short queries are classified as simple."""
        router = ModelRouter(simple_token_threshold=50, enable_logging=False)
        
        simple_queries = [
            "agriculture innovation",  # ~3 words
            "sensors in farming",  # ~3 words
            "IoT for precision",  # ~3 words
        ]
        
        for query in simple_queries:
            complexity = router.classify_complexity(query)
            assert complexity == QueryComplexity.SIMPLE, f"Query '{query}' should be SIMPLE"
    
    def test_complexity_classification_complex(self):
        """Test that long queries are classified as complex."""
        router = ModelRouter(simple_token_threshold=50, enable_logging=False)
        
        complex_query = " ".join(["word"] * 50)  # 50 words = ~65 tokens
        complexity = router.classify_complexity(complex_query)
        assert complexity == QueryComplexity.COMPLEX, f"Long query should be COMPLEX"
    
    def test_routing_simple_query(self):
        """Test routing of simple query."""
        router = ModelRouter(simple_token_threshold=50, enable_logging=False)
        
        model, complexity = router.route("agriculture")
        assert model == ModelType.QWEN_4B
        assert complexity == QueryComplexity.SIMPLE
    
    def test_routing_complex_query(self):
        """Test routing of complex query."""
        router = ModelRouter(simple_token_threshold=50, enable_logging=False)
        
        complex_query = " ".join(["word"] * 60)  # 60 words = ~78 tokens
        model, complexity = router.route(complex_query)
        assert model == ModelType.LLAMA_32
        assert complexity == QueryComplexity.COMPLEX
    
    def test_fallback_model(self):
        """Test fallback model selection."""
        router = ModelRouter(enable_logging=False)
        
        # Fallback from qwen should be llama
        fallback = router.get_fallback_model(ModelType.QWEN_4B)
        assert fallback == ModelType.LLAMA_32
        
        # Fallback from llama should be qwen
        fallback = router.get_fallback_model(ModelType.LLAMA_32)
        assert fallback == ModelType.QWEN_4B
    
    def test_routing_stats(self):
        """Test routing statistics tracking."""
        router = ModelRouter(enable_logging=False)
        
        # Route some queries
        router.route("simple")
        router.route(" ".join(["word"] * 60))
        router.route("another")
        
        stats = router.get_stats()
        assert stats["total_queries"] == 3
        assert stats["simple_routed"] == 2
        assert stats["complex_routed"] == 1
        assert stats["simple_percentage"] > 0
        assert stats["complex_percentage"] > 0
    
    def test_router_singleton(self):
        """Test that get_router returns singleton."""
        router1 = get_router(enable_logging=False)
        router2 = get_router(enable_logging=False)
        assert router1 is router2
    
    def test_stats_reset(self):
        """Test resetting routing statistics."""
        router = ModelRouter(enable_logging=False)
        
        router.route("query1")
        router.route("query2")
        
        stats_before = router.get_stats()
        assert stats_before["total_queries"] == 2
        
        router.reset_stats()
        stats_after = router.get_stats()
        assert stats_after["total_queries"] == 0


class TestRouterEdgeCases:
    """Test edge cases in router logic."""
    
    def test_empty_query(self):
        """Test routing with empty query."""
        router = ModelRouter(enable_logging=False)
        complexity = router.classify_complexity("")
        assert complexity == QueryComplexity.SIMPLE
    
    def test_single_word_query(self):
        """Test routing with single word."""
        router = ModelRouter(enable_logging=False)
        complexity = router.classify_complexity("innovation")
        assert complexity == QueryComplexity.SIMPLE
    
    def test_threshold_boundary(self):
        """Test classification at threshold boundary."""
        router = ModelRouter(simple_token_threshold=40, enable_logging=False)
        
        # 40 words ≈ 52 tokens (just over threshold)
        query_40_words = " ".join(["word"] * 40)
        complexity = router.classify_complexity(query_40_words)
        # At boundary, should still be considered within simple range
        # (since we compute tokens as word_count * 1.3)
        assert complexity == QueryComplexity.SIMPLE
        
        # 45 words ≈ 58.5 tokens (over threshold)
        query_45_words = " ".join(["word"] * 45)
        complexity = router.classify_complexity(query_45_words)
        assert complexity == QueryComplexity.COMPLEX


class TestMultiModelIntegration:
    """Test multi-model inference integration."""
    
    def test_ollama_client_initialization(self):
        """Test Ollama client can be initialized."""
        from app.llm import OllamaClient
        
        client = OllamaClient(base_url="http://localhost:11434")
        assert client.base_url == "http://localhost:11434"
        assert client.model_name == "llama3.2"
    
    def test_multi_model_llm_initialization(self):
        """Test MultiModelLLMClient can be initialized."""
        from app.llm import MultiModelLLMClient
        
        client = MultiModelLLMClient(
            ollama_base_url="http://localhost:11434",
            enable_qwen_local=False,  # Don't load qwen in test
        )
        assert client.ollama_client is not None
        assert client.qwen_client is None  # Should be None since enable_qwen_local=False
    
    def test_build_idea_prompt(self):
        """Test prompt building."""
        from app.llm import build_idea_prompt
        
        evidence = [
            {
                "patent_id": "US123",
                "title": "Agricultural sensor system",
                "assignee": "AgriTech Inc",
                "abstract": "A system for monitoring soil",
                "text": "A system for monitoring soil",
                "cpc_codes": ["G01N"],
                "ipc_codes": ["G01N33"],
            }
        ]
        
        prompt = build_idea_prompt("precision agriculture", evidence)
        assert "Agricultural sensor system" in prompt
        assert "AgriTech Inc" in prompt
        assert "precision agriculture" in prompt
        assert "G01N" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
