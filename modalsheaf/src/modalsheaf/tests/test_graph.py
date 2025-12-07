"""Tests for graph module."""

import pytest
import numpy as np

from modalsheaf.graph import ModalityGraph, create_standard_graph
from modalsheaf.core import TransformationType


class TestModalityGraph:
    """Tests for ModalityGraph class."""
    
    def test_create_empty_graph(self):
        graph = ModalityGraph()
        assert len(graph.modalities) == 0
    
    def test_add_modality(self):
        graph = ModalityGraph()
        mod = graph.add_modality("image", shape=(224, 224, 3))
        
        assert "image" in graph.modalities
        assert graph.get_modality("image") == mod
    
    def test_add_duplicate_modality_fails(self):
        graph = ModalityGraph()
        graph.add_modality("image")
        
        with pytest.raises(ValueError):
            graph.add_modality("image")
    
    def test_add_transformation(self):
        graph = ModalityGraph()
        graph.add_modality("image")
        graph.add_modality("embedding")
        
        t = graph.add_transformation(
            "image", "embedding",
            forward=lambda x: x.mean(),
            info_loss="high"
        )
        
        assert graph.get_transformation("image", "embedding") == t
        assert graph.has_path("image", "embedding")
    
    def test_add_invertible_transformation(self):
        graph = ModalityGraph()
        graph.add_modality("a")
        graph.add_modality("b")
        
        graph.add_transformation(
            "a", "b",
            forward=lambda x: x * 2,
            inverse=lambda x: x / 2
        )
        
        # Should create both directions
        assert graph.has_path("a", "b")
        assert graph.has_path("b", "a")
    
    def test_transform_data(self):
        graph = ModalityGraph()
        graph.add_modality("a")
        graph.add_modality("b")
        graph.add_transformation("a", "b", forward=lambda x: x * 2)
        
        result = graph.transform("a", "b", 5)
        assert result == 10
    
    def test_transform_along_path(self):
        graph = ModalityGraph()
        graph.add_modality("a")
        graph.add_modality("b")
        graph.add_modality("c")
        graph.add_transformation("a", "b", forward=lambda x: x * 2)
        graph.add_transformation("b", "c", forward=lambda x: x + 1)
        
        result = graph.transform("a", "c", 5)
        assert result == 11  # (5 * 2) + 1
    
    def test_find_path(self):
        graph = ModalityGraph()
        graph.add_modality("a")
        graph.add_modality("b")
        graph.add_modality("c")
        graph.add_transformation("a", "b", forward=lambda x: x)
        graph.add_transformation("b", "c", forward=lambda x: x)
        
        path = graph.find_path("a", "c")
        assert path == ["a", "b", "c"]
    
    def test_no_path_returns_none(self):
        graph = ModalityGraph()
        graph.add_modality("a")
        graph.add_modality("b")
        # No transformation added
        
        assert graph.find_path("a", "b") is None
    
    def test_estimate_path_info_loss(self):
        graph = ModalityGraph()
        graph.add_modality("a")
        graph.add_modality("b")
        graph.add_modality("c")
        graph.add_transformation("a", "b", forward=lambda x: x, info_loss=0.5)
        graph.add_transformation("b", "c", forward=lambda x: x, info_loss=0.5)
        
        loss = graph.estimate_path_info_loss("a", "c")
        assert loss == 0.75  # 1 - (1-0.5)*(1-0.5)


class TestStandardGraph:
    """Tests for create_standard_graph."""
    
    def test_creates_standard_modalities(self):
        graph = create_standard_graph()
        
        assert "image" in graph.modalities
        assert "text" in graph.modalities
        assert "embedding" in graph.modalities
        assert "tokens" in graph.modalities
