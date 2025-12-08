"""
ModalityGraph - The central data structure for managing modalities and transformations.

This represents the "base space" of our sheaf, with modalities as vertices
and transformations as edges.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np

from .core import Modality, Transformation, TransformationType, compose_transformations


class ModalityGraph:
    """
    A graph of modalities connected by transformations.
    
    This is the main interface for ModalSheaf. It allows you to:
    - Define modalities (data types)
    - Register transformations between them
    - Find paths between modalities
    - Transform data along paths
    - Measure consistency of multimodal data
    
    In sheaf-theoretic terms:
    - Vertices = Modalities (stalks)
    - Edges = Transformations (restriction maps)
    - The graph structure = Base topology
    
    Example:
        >>> graph = ModalityGraph()
        >>> graph.add_modality("image", shape=(224, 224, 3))
        >>> graph.add_modality("embedding", shape=(768,))
        >>> graph.add_transformation("image", "embedding", encoder_fn)
        >>> 
        >>> emb = graph.transform("image", "embedding", my_image)
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize an empty modality graph.
        
        Args:
            name: Name for this graph (useful when managing multiple)
        """
        self.name = name
        self._modalities: Dict[str, Modality] = {}
        self._graph = nx.DiGraph()  # Directed graph of transformations
        self._transforms: Dict[Tuple[str, str], Transformation] = {}
    
    # ==================== Modality Management ====================
    
    def add_modality(
        self,
        name: str,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        description: str = "",
        **metadata
    ) -> Modality:
        """
        Add a modality to the graph.
        
        Args:
            name: Unique identifier
            shape: Expected data shape (None for variable)
            dtype: Data type
            description: Human-readable description
            **metadata: Additional properties
        
        Returns:
            The created Modality object
        
        Example:
            >>> graph.add_modality("image", shape=(224, 224, 3), dtype="float32")
            >>> graph.add_modality("text", dtype="str", description="Raw text input")
        """
        if name in self._modalities:
            raise ValueError(f"Modality '{name}' already exists")
        
        modality = Modality(
            name=name,
            shape=shape,
            dtype=dtype,
            description=description,
            metadata=metadata
        )
        
        self._modalities[name] = modality
        self._graph.add_node(name, modality=modality)
        
        return modality
    
    def get_modality(self, name: str) -> Modality:
        """Get a modality by name."""
        if name not in self._modalities:
            raise KeyError(f"Modality '{name}' not found")
        return self._modalities[name]
    
    @property
    def modalities(self) -> List[str]:
        """List all modality names."""
        return list(self._modalities.keys())
    
    # ==================== Transformation Management ====================
    
    def add_transformation(
        self,
        source: str,
        target: str,
        forward: Callable[[Any], Any],
        inverse: Optional[Callable[[Any], Any]] = None,
        transform_type: TransformationType = TransformationType.UNKNOWN,
        info_loss: Union[str, float] = "auto",
        name: Optional[str] = None,
        **metadata
    ) -> Transformation:
        """
        Add a transformation between two modalities.
        
        Args:
            source: Source modality name
            target: Target modality name
            forward: Forward transformation function
            inverse: Inverse function (None if not invertible)
            transform_type: Type of transformation
            info_loss: Information loss estimate:
                - "auto": Estimate from transform_type
                - "none"/"low"/"medium"/"high": Qualitative
                - float: Exact value 0.0-1.0
            name: Optional name for this transformation
            **metadata: Additional properties
        
        Returns:
            The created Transformation object
        
        Example:
            >>> graph.add_transformation(
            ...     "image", "embedding",
            ...     forward=clip_encoder,
            ...     info_loss="high"
            ... )
        """
        # Validate modalities exist
        if source not in self._modalities:
            raise KeyError(f"Source modality '{source}' not found")
        if target not in self._modalities:
            raise KeyError(f"Target modality '{target}' not found")
        
        # Parse info_loss
        if isinstance(info_loss, str):
            info_loss_map = {
                "auto": None,
                "none": 0.0,
                "low": 0.2,
                "medium": 0.5,
                "high": 0.8,
            }
            info_loss_value = info_loss_map.get(info_loss.lower())
            if info_loss_value is None:
                # Auto-estimate from transform type
                if inverse is not None:
                    info_loss_value = 0.0
                else:
                    info_loss_value = 0.5
        else:
            info_loss_value = float(info_loss)
        
        transform = Transformation(
            source=source,
            target=target,
            forward=forward,
            inverse=inverse,
            transform_type=transform_type,
            info_loss_estimate=info_loss_value,
            name=name,
            metadata=metadata
        )
        
        # Add to graph
        self._graph.add_edge(source, target, transform=transform)
        self._transforms[(source, target)] = transform
        
        # If invertible, also add reverse edge
        if inverse is not None:
            reverse_transform = Transformation(
                source=target,
                target=source,
                forward=inverse,
                inverse=forward,
                transform_type=transform_type,
                info_loss_estimate=info_loss_value,
                name=f"{name}_inverse" if name else None,
                metadata={"inverse_of": transform.name}
            )
            self._graph.add_edge(target, source, transform=reverse_transform)
            self._transforms[(target, source)] = reverse_transform
        
        return transform
    
    def get_transformation(self, source: str, target: str) -> Optional[Transformation]:
        """Get direct transformation between two modalities."""
        return self._transforms.get((source, target))
    
    def has_path(self, source: str, target: str) -> bool:
        """Check if there's any path between two modalities."""
        return nx.has_path(self._graph, source, target)
    
    def find_path(
        self, 
        source: str, 
        target: str,
        prefer_lossless: bool = True
    ) -> Optional[List[str]]:
        """
        Find a path from source to target modality.
        
        Args:
            source: Starting modality
            target: Ending modality
            prefer_lossless: If True, prefer paths with less information loss
        
        Returns:
            List of modality names forming the path, or None if no path exists
        """
        if not self.has_path(source, target):
            return None
        
        if prefer_lossless:
            # Use info_loss as edge weight
            def weight_fn(u, v, d):
                t = d.get('transform')
                return t.info_loss_estimate if t else 1.0
            
            try:
                return nx.dijkstra_path(self._graph, source, target, weight=weight_fn)
            except nx.NetworkXNoPath:
                return None
        else:
            return nx.shortest_path(self._graph, source, target)
    
    # ==================== Data Transformation ====================
    
    def transform(
        self,
        source: str,
        target: str,
        data: Any,
        path: Optional[List[str]] = None
    ) -> Any:
        """
        Transform data from source modality to target modality.
        
        Args:
            source: Source modality name
            target: Target modality name
            data: Input data
            path: Optional explicit path to use (if None, finds shortest)
        
        Returns:
            Transformed data
        
        Example:
            >>> embedding = graph.transform("image", "embedding", my_image)
        """
        if source == target:
            return data
        
        # Find path if not provided
        if path is None:
            path = self.find_path(source, target)
            if path is None:
                raise ValueError(f"No path from '{source}' to '{target}'")
        
        # Apply transformations along path
        result = data
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            transform = self._transforms.get((src, tgt))
            if transform is None:
                raise ValueError(f"No transformation from '{src}' to '{tgt}'")
            result = transform(result)
        
        return result
    
    def get_composed_transformation(
        self,
        source: str,
        target: str,
        path: Optional[List[str]] = None
    ) -> Transformation:
        """
        Get a single composed transformation for a path.
        
        Useful for analyzing the total information loss along a path.
        """
        if path is None:
            path = self.find_path(source, target)
            if path is None:
                raise ValueError(f"No path from '{source}' to '{target}'")
        
        if len(path) < 2:
            raise ValueError("Path must have at least 2 nodes")
        
        # Compose all transformations
        composed = self._transforms[(path[0], path[1])]
        for i in range(1, len(path) - 1):
            next_t = self._transforms[(path[i], path[i + 1])]
            composed = compose_transformations(composed, next_t)
        
        return composed
    
    # ==================== Analysis ====================
    
    def estimate_path_info_loss(
        self,
        source: str,
        target: str,
        path: Optional[List[str]] = None
    ) -> float:
        """
        Estimate total information loss along a path.
        
        Returns:
            Float between 0.0 (no loss) and 1.0 (total loss)
        """
        composed = self.get_composed_transformation(source, target, path)
        return composed.info_loss_estimate
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the modality graph."""
        return {
            "num_modalities": len(self._modalities),
            "num_transformations": len(self._transforms),
            "is_connected": nx.is_weakly_connected(self._graph),
            "modalities": self.modalities,
            "edges": list(self._transforms.keys()),
        }
    
    def __repr__(self) -> str:
        stats = self.get_graph_stats()
        return (
            f"ModalityGraph(name='{self.name}', "
            f"modalities={stats['num_modalities']}, "
            f"transformations={stats['num_transformations']})"
        )


# ==================== Convenience Functions ====================

def create_standard_graph() -> ModalityGraph:
    """
    Create a graph with common ML modalities pre-defined.
    
    Includes: image, text, audio, embedding, tokens
    (Transformations must still be added by user)
    """
    graph = ModalityGraph(name="standard")
    
    # Visual modalities
    graph.add_modality("image", shape=None, dtype="float32", 
                       description="Image as numpy array (H, W, C)")
    graph.add_modality("image_224", shape=(224, 224, 3), dtype="float32",
                       description="224x224 RGB image")
    graph.add_modality("patches", shape=None, dtype="float32",
                       description="Image patches (N, P, P, C)")
    
    # Text modalities
    graph.add_modality("text", dtype="str",
                       description="Raw text string")
    graph.add_modality("tokens", dtype="int64",
                       description="Token IDs")
    graph.add_modality("token_embeddings", dtype="float32",
                       description="Per-token embeddings (seq_len, dim)")
    
    # Audio modalities
    graph.add_modality("audio_waveform", dtype="float32",
                       description="Raw audio waveform")
    graph.add_modality("spectrogram", dtype="float32",
                       description="Audio spectrogram")
    
    # Embedding modalities
    graph.add_modality("embedding", dtype="float32",
                       description="Generic embedding vector")
    graph.add_modality("clip_embedding", shape=(768,), dtype="float32",
                       description="CLIP embedding")
    
    return graph
