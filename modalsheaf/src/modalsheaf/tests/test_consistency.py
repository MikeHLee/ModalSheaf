"""Tests for consistency module."""

import pytest
import numpy as np

from modalsheaf.graph import ModalityGraph
from modalsheaf.consistency import (
    ConsistencyChecker,
    CohomologyResult,
    compute_sheaf_laplacian,
    diffuse_to_consensus,
)


class TestCohomologyResult:
    """Tests for CohomologyResult class."""
    
    def test_consistent_result(self):
        result = CohomologyResult(h0_dim=1, h1_dim=0, consistency_score=1.0)
        assert result.is_consistent
        assert "consistent" in result.diagnosis.lower()
    
    def test_inconsistent_result(self):
        result = CohomologyResult(h0_dim=1, h1_dim=2, consistency_score=0.3)
        assert not result.is_consistent
        assert "inconsisten" in result.diagnosis.lower()
    
    def test_repr(self):
        result = CohomologyResult(h0_dim=1, h1_dim=0, consistency_score=0.95)
        repr_str = repr(result)
        assert "H⁰=1" in repr_str
        assert "H¹=0" in repr_str


class TestConsistencyChecker:
    """Tests for ConsistencyChecker class."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        graph = ModalityGraph()
        graph.add_modality("a", shape=(4,))
        graph.add_modality("b", shape=(4,))
        graph.add_modality("embedding", shape=(4,))
        
        # Identity transforms to embedding
        graph.add_transformation("a", "embedding", forward=lambda x: x)
        graph.add_transformation("b", "embedding", forward=lambda x: x)
        
        return graph
    
    def test_consistent_data(self, simple_graph):
        """Test that identical embeddings are consistent."""
        checker = ConsistencyChecker(simple_graph, common_modality="embedding")
        
        # Same data for both modalities
        data = {
            "a": np.array([1.0, 0.0, 0.0, 0.0]),
            "b": np.array([1.0, 0.0, 0.0, 0.0])
        }
        
        result = checker.check(data)
        
        assert result.consistency_score > 0.99
        assert result.h1_dim == 0
    
    def test_inconsistent_data(self, simple_graph):
        """Test that different embeddings are inconsistent."""
        checker = ConsistencyChecker(simple_graph, common_modality="embedding")
        
        # Orthogonal vectors
        data = {
            "a": np.array([1.0, 0.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0, 0.0])
        }
        
        result = checker.check(data)
        
        assert result.consistency_score < 0.5
        assert result.h1_dim > 0
    
    def test_single_modality_trivially_consistent(self, simple_graph):
        """Test that single modality is trivially consistent."""
        checker = ConsistencyChecker(simple_graph, common_modality="embedding")
        
        data = {"a": np.array([1.0, 0.0, 0.0, 0.0])}
        
        result = checker.check(data)
        
        assert result.is_consistent
    
    def test_diagnose_inconsistency(self, simple_graph):
        """Test diagnosis of inconsistencies."""
        checker = ConsistencyChecker(simple_graph, common_modality="embedding")
        
        data = {
            "a": np.array([1.0, 0.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0, 0.0])
        }
        
        result = checker.check(data)
        diagnosis = checker.diagnose_inconsistency(result)
        
        assert diagnosis["status"] == "inconsistent"
        assert len(diagnosis["problematic_modalities"]) > 0


class TestSheafLaplacian:
    """Tests for sheaf Laplacian computation."""
    
    def test_laplacian_shape(self):
        graph = ModalityGraph()
        graph.add_modality("a", shape=(3,))
        graph.add_modality("b", shape=(3,))
        graph.add_transformation("a", "b", forward=lambda x: x)
        
        embeddings = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0])
        }
        
        L = compute_sheaf_laplacian(graph, embeddings)
        
        # 2 modalities * 3 dimensions = 6x6 matrix
        assert L.shape == (6, 6)
    
    def test_laplacian_symmetric(self):
        """Sheaf Laplacian should be symmetric for undirected graphs."""
        graph = ModalityGraph()
        graph.add_modality("a", shape=(2,))
        graph.add_modality("b", shape=(2,))
        graph.add_transformation(
            "a", "b",
            forward=lambda x: x,
            inverse=lambda x: x  # Makes it undirected
        )
        
        embeddings = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0])
        }
        
        L = compute_sheaf_laplacian(graph, embeddings)
        
        # Should be approximately symmetric
        assert np.allclose(L, L.T, atol=1e-10)


class TestDiffusion:
    """Tests for diffusion to consensus."""
    
    def test_diffusion_reduces_distance(self):
        graph = ModalityGraph()
        graph.add_modality("a", shape=(3,))
        graph.add_modality("b", shape=(3,))
        graph.add_transformation(
            "a", "b",
            forward=lambda x: x,
            inverse=lambda x: x
        )
        
        # Start with different embeddings
        initial = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0])
        }
        
        # Diffuse
        diffused = diffuse_to_consensus(graph, initial, num_steps=100)
        
        # Distance should decrease
        initial_dist = np.linalg.norm(initial["a"] - initial["b"])
        diffused_dist = np.linalg.norm(diffused["a"] - diffused["b"])
        
        assert diffused_dist < initial_dist
