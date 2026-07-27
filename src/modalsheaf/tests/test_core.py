"""Tests for core module."""

import pytest
import numpy as np

from modalsheaf.core import (
    Modality,
    Transformation,
    TransformationType,
    compose_transformations,
)


class TestModality:
    """Tests for Modality class."""
    
    def test_create_modality(self):
        mod = Modality("test", shape=(224, 224, 3), dtype="float32")
        assert mod.name == "test"
        assert mod.shape == (224, 224, 3)
        assert mod.dtype == "float32"
    
    def test_modality_dimensionality(self):
        mod = Modality("test", shape=(224, 224, 3))
        assert mod.dimensionality == 224 * 224 * 3
    
    def test_modality_variable_shape(self):
        mod = Modality("text", shape=None)
        assert mod.dimensionality is None
    
    def test_modality_equality(self):
        mod1 = Modality("test")
        mod2 = Modality("test")
        mod3 = Modality("other")
        
        assert mod1 == mod2
        assert mod1 != mod3
        assert mod1 == "test"  # Compare with string
    
    def test_modality_hash(self):
        mod1 = Modality("test")
        mod2 = Modality("test")
        
        # Should be usable in sets/dicts
        s = {mod1, mod2}
        assert len(s) == 1


class TestTransformation:
    """Tests for Transformation class."""
    
    def test_create_transformation(self):
        t = Transformation(
            source="a",
            target="b",
            forward=lambda x: x * 2
        )
        assert t.source == "a"
        assert t.target == "b"
        assert t(5) == 10
    
    def test_transformation_with_inverse(self):
        t = Transformation(
            source="a",
            target="b",
            forward=lambda x: x * 2,
            inverse=lambda x: x / 2
        )
        assert t.is_invertible
        assert t.transform_type == TransformationType.ISOMORPHISM
        assert t(5) == 10
        assert t.apply_inverse(10) == 5
    
    def test_transformation_without_inverse(self):
        t = Transformation(
            source="a",
            target="b",
            forward=lambda x: x * 2
        )
        assert not t.is_invertible
        assert t.transform_type == TransformationType.LOSSY
        
        with pytest.raises(ValueError):
            t.apply_inverse(10)
    
    def test_transformation_info_loss(self):
        t_lossless = Transformation(
            source="a", target="b",
            forward=lambda x: x,
            inverse=lambda x: x,
            transform_type=TransformationType.ISOMORPHISM,
            info_loss_estimate=0.0
        )
        assert t_lossless.is_lossless
        
        t_lossy = Transformation(
            source="a", target="b",
            forward=lambda x: x,
            transform_type=TransformationType.LOSSY,
            info_loss_estimate=0.8
        )
        assert not t_lossy.is_lossless
    
    def test_transformation_auto_name(self):
        t = Transformation(source="image", target="embedding", forward=lambda x: x)
        assert t.name == "image_to_embedding"


class TestComposition:
    """Tests for transformation composition."""
    
    def test_compose_two_transforms(self):
        t1 = Transformation(
            source="a", target="b",
            forward=lambda x: x * 2,
            info_loss_estimate=0.1
        )
        t2 = Transformation(
            source="b", target="c",
            forward=lambda x: x + 1,
            info_loss_estimate=0.2
        )
        
        composed = compose_transformations(t1, t2)
        
        assert composed.source == "a"
        assert composed.target == "c"
        assert composed(5) == 11  # (5 * 2) + 1
    
    def test_compose_incompatible_fails(self):
        t1 = Transformation(source="a", target="b", forward=lambda x: x)
        t2 = Transformation(source="c", target="d", forward=lambda x: x)
        
        with pytest.raises(ValueError):
            compose_transformations(t1, t2)
    
    def test_compose_preserves_invertibility(self):
        t1 = Transformation(
            source="a", target="b",
            forward=lambda x: x * 2,
            inverse=lambda x: x / 2
        )
        t2 = Transformation(
            source="b", target="c",
            forward=lambda x: x + 1,
            inverse=lambda x: x - 1
        )
        
        composed = compose_transformations(t1, t2)
        
        assert composed.is_invertible
        assert composed(5) == 11
        assert composed.apply_inverse(11) == 5
    
    def test_compose_info_loss_accumulates(self):
        t1 = Transformation(
            source="a", target="b",
            forward=lambda x: x,
            info_loss_estimate=0.5
        )
        t2 = Transformation(
            source="b", target="c",
            forward=lambda x: x,
            info_loss_estimate=0.5
        )
        
        composed = compose_transformations(t1, t2)
        
        # Combined loss: 1 - (1-0.5)*(1-0.5) = 0.75
        assert composed.info_loss_estimate == 0.75


class TestTransformationType:
    """Tests for TransformationType enum."""
    
    def test_types_exist(self):
        assert TransformationType.ISOMORPHISM
        assert TransformationType.EMBEDDING
        assert TransformationType.PROJECTION
        assert TransformationType.LOSSY
        assert TransformationType.UNKNOWN
