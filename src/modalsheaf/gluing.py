"""
Gluing operations for assembling local data into global structures.

This module implements the core sheaf operation: given local sections that
agree on overlaps, construct the unique global section.

Mathematical Background:
    
    A sheaf F on a space X satisfies:
    1. Locality: If s, t ∈ F(U) agree on a cover, then s = t
    2. Gluing: If local sections agree on overlaps, they glue to a global section
    
    The obstruction to gluing is measured by H¹ (first cohomology).
    
    Key concepts:
    - Cover: A collection of "local" regions that together cover the "global"
    - Overlap: Where two local regions intersect
    - Cocycle condition: Consistency requirement on triple overlaps
    - Descent data: Local data + compatibility on overlaps

Examples:
    - Panorama stitching: images + overlap alignment → panorama
    - Coordinate fusion: sensor frames + transforms → world frame
    - Geographic assembly: states + border alignment → country
    - Document assembly: pages + page order → document
    - Codebase: files + import graph → unified module
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union
)
import numpy as np

from .core import Modality, Transformation, TransformationType


# Type variables for generic gluing
L = TypeVar('L')  # Local type
G = TypeVar('G')  # Global type
O = TypeVar('O')  # Overlap type


# ==================== Core Data Structures ====================

@dataclass
class LocalSection:
    """
    A piece of local data with its domain information.
    
    Attributes:
        id: Unique identifier for this local section
        data: The actual data
        domain: Description of what region this covers
        metadata: Additional information (coordinates, timestamps, etc.)
    """
    id: str
    data: Any
    domain: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Overlap:
    """
    Describes where two local sections overlap.
    
    Attributes:
        section_ids: Tuple of the two overlapping section IDs
        region: Description of the overlap region
        transform: How to align data from section 1 to section 2
        weight: Importance/confidence of this overlap (for weighted gluing)
    """
    section_ids: Tuple[str, str]
    region: Any = None
    transform: Optional[Callable] = None
    weight: float = 1.0
    
    @property
    def id(self) -> str:
        return f"{self.section_ids[0]}∩{self.section_ids[1]}"


@dataclass 
class GluingResult:
    """
    Result of a gluing operation.
    
    Attributes:
        success: Whether gluing succeeded
        global_section: The assembled global data (if successful)
        consistency_errors: List of overlap regions where data disagreed
        h1_obstruction: Measure of the obstruction to gluing (0 = perfect)
        diagnostics: Detailed information about the gluing process
    """
    success: bool
    global_section: Optional[Any] = None
    consistency_errors: List[Dict[str, Any]] = field(default_factory=list)
    h1_obstruction: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_consistent(self) -> bool:
        return len(self.consistency_errors) == 0
    
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"GluingResult({status}, H¹={self.h1_obstruction:.3f}, errors={len(self.consistency_errors)})"


# ==================== Abstract Gluing Protocol ====================

class GluingProtocol(ABC, Generic[L, G]):
    """
    Abstract base class for gluing operations.
    
    Subclasses implement specific gluing algorithms for different
    data types (images, coordinates, documents, etc.).
    """
    
    @abstractmethod
    def extract_overlap(
        self, 
        section1: LocalSection, 
        section2: LocalSection,
        overlap: Overlap,
    ) -> Tuple[Any, Any]:
        """
        Extract the overlapping portions from two sections.
        
        Returns:
            Tuple of (data1_in_overlap, data2_in_overlap)
        """
        pass
    
    @abstractmethod
    def measure_consistency(
        self,
        data1: Any,
        data2: Any,
        overlap: Overlap,
    ) -> float:
        """
        Measure how well two sections agree on their overlap.
        
        Returns:
            0.0 = perfect agreement, higher = more disagreement
        """
        pass
    
    @abstractmethod
    def glue(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> GluingResult:
        """
        Attempt to glue local sections into a global section.
        
        Returns:
            GluingResult with the assembled data or error information
        """
        pass
    
    def check_cocycle_condition(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> List[Dict[str, Any]]:
        """
        Check the cocycle condition on triple overlaps.
        
        For sections i, j, k with pairwise overlaps, the transforms must satisfy:
        T_ij ∘ T_jk = T_ik on the triple overlap.
        
        Returns:
            List of violations (empty if cocycle condition holds)
        """
        # Build overlap graph
        overlap_map = {o.section_ids: o for o in overlaps}
        section_ids = [s.id for s in sections]
        
        violations = []
        
        # Check all triples
        for i, id_i in enumerate(section_ids):
            for j, id_j in enumerate(section_ids[i+1:], i+1):
                for k, id_k in enumerate(section_ids[j+1:], j+1):
                    # Check if all three pairs overlap
                    pair_ij = (id_i, id_j) if (id_i, id_j) in overlap_map else (id_j, id_i)
                    pair_jk = (id_j, id_k) if (id_j, id_k) in overlap_map else (id_k, id_j)
                    pair_ik = (id_i, id_k) if (id_i, id_k) in overlap_map else (id_k, id_i)
                    
                    if all(p in overlap_map for p in [pair_ij, pair_jk, pair_ik]):
                        # All three overlap - check cocycle
                        # This is a simplified check; real implementation would
                        # compose transforms and compare
                        pass
        
        return violations


# ==================== Image Panorama Gluing ====================

class PanoramaGluing(GluingProtocol[np.ndarray, np.ndarray]):
    """
    Glue multiple overlapping images into a panorama.
    
    This implements a simplified version of panorama stitching:
    1. Images are assumed to be roughly aligned (same scale/rotation)
    2. Overlaps are specified by pixel offsets
    3. Blending uses linear interpolation in overlap regions
    """
    
    def __init__(
        self,
        blend_mode: str = "linear",  # "linear", "max", "min", "average"
        consistency_threshold: float = 50.0,  # Max allowed pixel difference
    ):
        self.blend_mode = blend_mode
        self.consistency_threshold = consistency_threshold
    
    def extract_overlap(
        self,
        section1: LocalSection,
        section2: LocalSection,
        overlap: Overlap,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract overlapping regions based on offset metadata."""
        img1 = section1.data
        img2 = section2.data
        
        # Overlap region specified as (x_offset, y_offset) of img2 relative to img1
        offset = overlap.region or (0, 0)
        x_off, y_off = offset
        
        # Calculate overlap bounds
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Overlap in img1 coordinates
        x1_start = max(0, x_off)
        y1_start = max(0, y_off)
        x1_end = min(w1, x_off + w2)
        y1_end = min(h1, y_off + h2)
        
        # Corresponding region in img2
        x2_start = max(0, -x_off)
        y2_start = max(0, -y_off)
        x2_end = x2_start + (x1_end - x1_start)
        y2_end = y2_start + (y1_end - y1_start)
        
        overlap1 = img1[y1_start:y1_end, x1_start:x1_end]
        overlap2 = img2[y2_start:y2_end, x2_start:x2_end]
        
        return overlap1, overlap2
    
    def measure_consistency(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        overlap: Overlap,
    ) -> float:
        """Measure pixel-wise difference in overlap region."""
        if data1.shape != data2.shape:
            return float('inf')
        
        diff = np.abs(data1.astype(float) - data2.astype(float))
        return float(np.mean(diff))
    
    def glue(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> GluingResult:
        """Stitch images into a panorama."""
        if not sections:
            return GluingResult(success=False, diagnostics={"error": "No sections"})
        
        # Check consistency on all overlaps
        consistency_errors = []
        total_error = 0.0
        
        for overlap in overlaps:
            id1, id2 = overlap.section_ids
            sec1 = next((s for s in sections if s.id == id1), None)
            sec2 = next((s for s in sections if s.id == id2), None)
            
            if sec1 and sec2:
                data1, data2 = self.extract_overlap(sec1, sec2, overlap)
                error = self.measure_consistency(data1, data2, overlap)
                total_error += error
                
                if error > self.consistency_threshold:
                    consistency_errors.append({
                        "overlap": overlap.id,
                        "error": error,
                        "threshold": self.consistency_threshold,
                    })
        
        # Calculate canvas size
        positions = {}
        positions[sections[0].id] = (0, 0)
        
        # Propagate positions through overlaps
        for overlap in overlaps:
            id1, id2 = overlap.section_ids
            if id1 in positions and id2 not in positions:
                offset = overlap.region or (0, 0)
                positions[id2] = (
                    positions[id1][0] + offset[0],
                    positions[id1][1] + offset[1]
                )
        
        # Calculate canvas bounds
        min_x = min_y = 0
        max_x = max_y = 0
        
        for sec in sections:
            if sec.id in positions:
                x, y = positions[sec.id]
                h, w = sec.data.shape[:2]
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
        
        # Create canvas
        canvas_h = max_y - min_y
        canvas_w = max_x - min_x
        channels = sections[0].data.shape[2] if sections[0].data.ndim == 3 else 1
        
        if channels > 1:
            canvas = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
        else:
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # Place images on canvas
        for sec in sections:
            if sec.id not in positions:
                continue
                
            x, y = positions[sec.id]
            x -= min_x
            y -= min_y
            
            img = sec.data.astype(np.float32)
            h, w = img.shape[:2]
            
            if self.blend_mode == "linear":
                # Create distance-based weight
                weight = np.ones((h, w), dtype=np.float32)
                
                if channels > 1:
                    canvas[y:y+h, x:x+w] += img * weight[:, :, np.newaxis]
                else:
                    canvas[y:y+h, x:x+w] += img * weight
                    
                weight_map[y:y+h, x:x+w] += weight
            else:
                # Simple overwrite
                if channels > 1:
                    canvas[y:y+h, x:x+w] = img
                else:
                    canvas[y:y+h, x:x+w] = img
                weight_map[y:y+h, x:x+w] = 1.0
        
        # Normalize by weights
        if self.blend_mode == "linear":
            weight_map = np.maximum(weight_map, 1e-6)
            if channels > 1:
                canvas = canvas / weight_map[:, :, np.newaxis]
            else:
                canvas = canvas / weight_map
        
        h1_obstruction = total_error / max(len(overlaps), 1)
        
        return GluingResult(
            success=len(consistency_errors) == 0,
            global_section=canvas.astype(np.uint8) if canvas.max() > 1 else canvas,
            consistency_errors=consistency_errors,
            h1_obstruction=h1_obstruction,
            diagnostics={
                "canvas_size": (canvas_h, canvas_w),
                "num_sections": len(sections),
                "num_overlaps": len(overlaps),
                "positions": positions,
            }
        )


# ==================== Coordinate Frame Gluing ====================

@dataclass
class CoordinateFrame:
    """A coordinate frame with position and orientation."""
    name: str
    origin: np.ndarray  # 3D position
    rotation: np.ndarray  # 3x3 rotation matrix
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from this frame to world frame."""
        return self.rotation @ point + self.origin
    
    def inverse_transform(self, world_point: np.ndarray) -> np.ndarray:
        """Transform a point from world frame to this frame."""
        return self.rotation.T @ (world_point - self.origin)


class CoordinateGluing(GluingProtocol):
    """
    Glue sensor coordinate frames into a unified world frame.
    
    Given multiple sensors with known relative transforms, compute
    the global coordinate system and transform all data into it.
    """
    
    def __init__(self, reference_frame: str = "world"):
        self.reference_frame = reference_frame
    
    def extract_overlap(
        self,
        section1: LocalSection,
        section2: LocalSection,
        overlap: Overlap,
    ) -> Tuple[Any, Any]:
        """Extract points visible in both frames."""
        # Points in section1's frame that are also in section2's FOV
        points1 = section1.data.get("points", [])
        points2 = section2.data.get("points", [])
        
        # For simplicity, assume overlap.region contains indices of shared points
        shared_indices = overlap.region or []
        
        return (
            [points1[i] for i in shared_indices if i < len(points1)],
            [points2[i] for i in shared_indices if i < len(points2)]
        )
    
    def measure_consistency(
        self,
        data1: Any,
        data2: Any,
        overlap: Overlap,
    ) -> float:
        """Measure alignment error between transformed points."""
        if not data1 or not data2:
            return 0.0
        
        points1 = np.array(data1)
        points2 = np.array(data2)
        
        if overlap.transform:
            # Transform points1 to points2's frame
            transformed = np.array([overlap.transform(p) for p in points1])
            error = np.mean(np.linalg.norm(transformed - points2, axis=1))
        else:
            error = np.mean(np.linalg.norm(points1 - points2, axis=1))
        
        return float(error)
    
    def glue(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> GluingResult:
        """Compute unified world frame and transform all data."""
        if not sections:
            return GluingResult(success=False)
        
        # Build transform graph
        transforms = {}  # (from_id, to_id) -> transform
        
        for overlap in overlaps:
            if overlap.transform:
                transforms[overlap.section_ids] = overlap.transform
        
        # Compute transforms to reference frame
        # (simplified: assumes first section is reference)
        reference_id = sections[0].id
        to_world = {reference_id: lambda x: x}
        
        # BFS to find transforms to all frames
        queue = [reference_id]
        visited = {reference_id}
        
        while queue:
            current = queue.pop(0)
            
            for (id1, id2), transform in transforms.items():
                if id1 == current and id2 not in visited:
                    # Compose transforms
                    parent_transform = to_world[id1]
                    to_world[id2] = lambda x, t=transform, p=parent_transform: p(t(x))
                    visited.add(id2)
                    queue.append(id2)
        
        # Transform all points to world frame
        world_points = {}
        
        for section in sections:
            if section.id in to_world:
                transform = to_world[section.id]
                points = section.data.get("points", [])
                world_points[section.id] = [transform(p) for p in points]
        
        # Check consistency
        consistency_errors = []
        total_error = 0.0
        
        for overlap in overlaps:
            id1, id2 = overlap.section_ids
            if id1 in world_points and id2 in world_points:
                data1, data2 = self.extract_overlap(
                    next(s for s in sections if s.id == id1),
                    next(s for s in sections if s.id == id2),
                    overlap
                )
                
                if data1 and data2:
                    # Transform both to world and compare
                    world1 = [to_world[id1](p) for p in data1]
                    world2 = [to_world[id2](p) for p in data2]
                    
                    error = np.mean([
                        np.linalg.norm(np.array(w1) - np.array(w2))
                        for w1, w2 in zip(world1, world2)
                    ])
                    
                    total_error += error
                    
                    if error > 0.1:  # Threshold
                        consistency_errors.append({
                            "overlap": overlap.id,
                            "error": error,
                        })
        
        return GluingResult(
            success=len(consistency_errors) == 0,
            global_section={
                "reference_frame": self.reference_frame,
                "points": world_points,
                "transforms": {k: "computed" for k in to_world},
            },
            consistency_errors=consistency_errors,
            h1_obstruction=total_error / max(len(overlaps), 1),
            diagnostics={
                "num_frames": len(sections),
                "frames_resolved": len(to_world),
            }
        )


# ==================== Hierarchical Gluing ====================

class HierarchicalGluing(GluingProtocol):
    """
    Glue hierarchical structures (states→country, pages→document, etc.).
    
    This handles cases where:
    - Local sections have a natural ordering or adjacency
    - Overlaps are at boundaries
    - Global structure emerges from local adjacencies
    """
    
    def __init__(
        self,
        boundary_key: str = "boundary",
        merge_strategy: str = "union",  # "union", "intersection", "first"
    ):
        self.boundary_key = boundary_key
        self.merge_strategy = merge_strategy
    
    def extract_overlap(
        self,
        section1: LocalSection,
        section2: LocalSection,
        overlap: Overlap,
    ) -> Tuple[Any, Any]:
        """Extract boundary data from adjacent sections."""
        boundary1 = section1.metadata.get(self.boundary_key, {})
        boundary2 = section2.metadata.get(self.boundary_key, {})
        
        # Get the specific boundary between these two
        shared_boundary_id = overlap.region
        
        return (
            boundary1.get(shared_boundary_id),
            boundary2.get(shared_boundary_id)
        )
    
    def measure_consistency(
        self,
        data1: Any,
        data2: Any,
        overlap: Overlap,
    ) -> float:
        """Check if boundaries match."""
        if data1 is None or data2 is None:
            return 0.0  # No boundary to check
        
        if data1 == data2:
            return 0.0
        
        # For numeric boundaries, compute difference
        if isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
            return abs(data1 - data2)
        
        # For other types, binary match
        return 1.0 if data1 != data2 else 0.0
    
    def glue(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> GluingResult:
        """Assemble sections into a unified structure."""
        if not sections:
            return GluingResult(success=False)
        
        # Check boundary consistency
        consistency_errors = []
        
        for overlap in overlaps:
            id1, id2 = overlap.section_ids
            sec1 = next((s for s in sections if s.id == id1), None)
            sec2 = next((s for s in sections if s.id == id2), None)
            
            if sec1 and sec2:
                b1, b2 = self.extract_overlap(sec1, sec2, overlap)
                error = self.measure_consistency(b1, b2, overlap)
                
                if error > 0:
                    consistency_errors.append({
                        "overlap": overlap.id,
                        "boundary1": b1,
                        "boundary2": b2,
                        "error": error,
                    })
        
        # Merge sections
        if self.merge_strategy == "union":
            merged_data = {}
            for section in sections:
                if isinstance(section.data, dict):
                    merged_data.update(section.data)
                else:
                    merged_data[section.id] = section.data
        else:
            merged_data = {s.id: s.data for s in sections}
        
        return GluingResult(
            success=len(consistency_errors) == 0,
            global_section=merged_data,
            consistency_errors=consistency_errors,
            h1_obstruction=len(consistency_errors) / max(len(overlaps), 1),
            diagnostics={
                "num_sections": len(sections),
                "merge_strategy": self.merge_strategy,
            }
        )


# ==================== Document Assembly Gluing ====================

class DocumentGluing(GluingProtocol):
    """
    Glue document fragments (pages, sections, chunks) into a complete document.
    
    Handles:
    - Page ordering
    - Section continuity
    - Cross-references
    """
    
    def __init__(self, order_key: str = "page_number"):
        self.order_key = order_key
    
    def extract_overlap(
        self,
        section1: LocalSection,
        section2: LocalSection,
        overlap: Overlap,
    ) -> Tuple[Any, Any]:
        """Extract continuation markers between sections."""
        # End of section1 should match start of section2
        text1 = section1.data if isinstance(section1.data, str) else ""
        text2 = section2.data if isinstance(section2.data, str) else ""
        
        # Get last/first N characters for continuity check
        n = 50
        return text1[-n:] if len(text1) >= n else text1, text2[:n] if len(text2) >= n else text2
    
    def measure_consistency(
        self,
        data1: Any,
        data2: Any,
        overlap: Overlap,
    ) -> float:
        """Check for broken sentences/words at boundaries."""
        if not data1 or not data2:
            return 0.0
        
        # Check if section1 ends mid-sentence
        ends_complete = data1.rstrip().endswith(('.', '!', '?', '"', ')'))
        starts_complete = data2.lstrip()[0].isupper() if data2.lstrip() else True
        
        if ends_complete and starts_complete:
            return 0.0
        
        return 0.5  # Possible mid-sentence break
    
    def glue(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> GluingResult:
        """Assemble pages/sections into complete document."""
        if not sections:
            return GluingResult(success=False)
        
        # Sort by order key
        sorted_sections = sorted(
            sections,
            key=lambda s: s.metadata.get(self.order_key, 0)
        )
        
        # Check continuity
        consistency_errors = []
        
        for i in range(len(sorted_sections) - 1):
            sec1 = sorted_sections[i]
            sec2 = sorted_sections[i + 1]
            
            # Find or create overlap
            overlap = next(
                (o for o in overlaps if set(o.section_ids) == {sec1.id, sec2.id}),
                Overlap(section_ids=(sec1.id, sec2.id))
            )
            
            data1, data2 = self.extract_overlap(sec1, sec2, overlap)
            error = self.measure_consistency(data1, data2, overlap)
            
            if error > 0:
                consistency_errors.append({
                    "between": (sec1.id, sec2.id),
                    "error": error,
                    "end_of_prev": data1,
                    "start_of_next": data2,
                })
        
        # Concatenate
        if all(isinstance(s.data, str) for s in sorted_sections):
            document = "\n\n".join(s.data for s in sorted_sections)
        else:
            document = [s.data for s in sorted_sections]
        
        return GluingResult(
            success=len(consistency_errors) == 0,
            global_section=document,
            consistency_errors=consistency_errors,
            h1_obstruction=len(consistency_errors) / max(len(sorted_sections) - 1, 1),
            diagnostics={
                "num_pages": len(sorted_sections),
                "order": [s.id for s in sorted_sections],
            }
        )


# ==================== Codebase Gluing ====================

class CodebaseGluing(GluingProtocol):
    """
    Glue code files into a unified module structure.
    
    Handles:
    - Import resolution
    - Dependency ordering
    - Symbol conflicts
    """
    
    def extract_overlap(
        self,
        section1: LocalSection,
        section2: LocalSection,
        overlap: Overlap,
    ) -> Tuple[Any, Any]:
        """Extract import/export relationships."""
        # What section1 exports that section2 imports
        exports1 = section1.metadata.get("exports", [])
        imports2 = section2.metadata.get("imports", [])
        
        shared = set(exports1) & set(imports2)
        
        return list(shared), list(shared)
    
    def measure_consistency(
        self,
        data1: Any,
        data2: Any,
        overlap: Overlap,
    ) -> float:
        """Check if imports are satisfied."""
        # data1 = exports, data2 = imports
        # Error if imports not in exports
        if not data2:
            return 0.0
        
        missing = set(data2) - set(data1)
        return len(missing) / len(data2) if data2 else 0.0
    
    def glue(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> GluingResult:
        """Build unified module from files."""
        if not sections:
            return GluingResult(success=False)
        
        # Build dependency graph
        dependencies = {}
        all_exports = {}
        all_imports = {}
        
        for section in sections:
            exports = section.metadata.get("exports", [])
            imports = section.metadata.get("imports", [])
            
            all_exports[section.id] = exports
            all_imports[section.id] = imports
            
            for exp in exports:
                if exp in all_exports:
                    pass  # Could track conflicts
        
        # Check for unresolved imports
        all_available = set()
        for exports in all_exports.values():
            all_available.update(exports)
        
        consistency_errors = []
        
        for section in sections:
            imports = section.metadata.get("imports", [])
            missing = set(imports) - all_available
            
            if missing:
                consistency_errors.append({
                    "file": section.id,
                    "missing_imports": list(missing),
                })
        
        # Topological sort for dependency order
        # (simplified - just return as-is)
        
        return GluingResult(
            success=len(consistency_errors) == 0,
            global_section={
                "files": {s.id: s.data for s in sections},
                "exports": all_exports,
                "imports": all_imports,
            },
            consistency_errors=consistency_errors,
            h1_obstruction=len(consistency_errors) / max(len(sections), 1),
            diagnostics={
                "num_files": len(sections),
                "total_exports": len(all_available),
            }
        )


# ==================== Utility Functions ====================

def create_gluing_problem(
    sections: List[Dict[str, Any]],
    overlaps: List[Dict[str, Any]],
) -> Tuple[List[LocalSection], List[Overlap]]:
    """
    Create a gluing problem from dictionaries.
    
    Args:
        sections: List of {"id": str, "data": Any, "domain": Any, "metadata": dict}
        overlaps: List of {"sections": (id1, id2), "region": Any, "weight": float}
    
    Returns:
        Tuple of (LocalSection list, Overlap list)
    """
    local_sections = [
        LocalSection(
            id=s["id"],
            data=s.get("data"),
            domain=s.get("domain"),
            metadata=s.get("metadata", {}),
        )
        for s in sections
    ]
    
    overlap_list = [
        Overlap(
            section_ids=tuple(o["sections"]),
            region=o.get("region"),
            transform=o.get("transform"),
            weight=o.get("weight", 1.0),
        )
        for o in overlaps
    ]
    
    return local_sections, overlap_list


def glue_with_protocol(
    protocol: GluingProtocol,
    sections: List[Dict[str, Any]],
    overlaps: List[Dict[str, Any]],
) -> GluingResult:
    """
    Convenience function to glue using a protocol.
    
    Example:
        result = glue_with_protocol(
            PanoramaGluing(),
            sections=[
                {"id": "left", "data": img1},
                {"id": "right", "data": img2},
            ],
            overlaps=[
                {"sections": ("left", "right"), "region": (100, 0)},
            ]
        )
    """
    local_sections, overlap_list = create_gluing_problem(sections, overlaps)
    return protocol.glue(local_sections, overlap_list)
