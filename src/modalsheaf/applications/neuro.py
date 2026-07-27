"""
Neuro-Topological Pipeline - Sheaf Theoretic Analysis of Brain Networks.

This module provides tools for analyzing brain pathologies (cognitive dissonance,
anxiety, etc.) as topological obstructions in neural networks using sheaf theory.

Mathematical Framework:
    
    We model the brain as a Sheaf (X, F) where:
    - Base Space (X): Graph of brain regions derived from correlation matrix
    - Stalks (F_U): Time-series signal intensity at each region
    - Obstruction: H¹(X, F) ≠ 0 indicates failure to integrate information
    
    The key insight is that non-trivial first cohomology (H¹) represents
    "holes" in information flow - regions that cannot be consistently
    integrated, potentially corresponding to cognitive pathologies.

Example Usage:

    >>> from modalsheaf.applications.neuro import BrainSheaf, load_fmri_data
    >>> 
    >>> # Load fMRI data
    >>> data = load_fmri_data("path/to/nifti.nii.gz", atlas="harvard_oxford")
    >>> 
    >>> # Create BrainSheaf
    >>> brain = BrainSheaf(
    ...     correlation_matrix=data.correlation_matrix,
    ...     time_series=data.time_series,
    ...     region_labels=data.labels
    ... )
    >>> 
    >>> # Build simplicial complex at threshold
    >>> brain.build_complex(threshold=0.3)
    >>> 
    >>> # Detect dissonance at timepoint t
    >>> result = brain.detect_dissonance(t=50)
    >>> print(f"Dissonance metric: {result.metric}")
    >>> 
    >>> # Find persistent cycles (topological obstructions)
    >>> cycles = brain.compute_persistent_homology()
    >>> brain.plot_active_holes(cycles)

References:
    - Robinson (2014). Topological Signal Processing.
    - Giusti et al. (2016). Two's company, three (or more) is a simplex.
    - Petri et al. (2014). Homological scaffolds of brain functional networks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False


# ==================== Data Structures ====================

@dataclass
class BrainRegion:
    """
    Represents a brain region (node in the sheaf base space).
    
    Attributes:
        index: Integer index in the correlation matrix
        label: Anatomical name (e.g., "Amygdala_L")
        coordinates: Optional MNI coordinates (x, y, z)
        time_series: Signal intensity over time at this region
    """
    index: int
    label: str
    coordinates: Optional[Tuple[float, float, float]] = None
    time_series: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.index)
    
    def __eq__(self, other):
        if isinstance(other, BrainRegion):
            return self.index == other.index
        return False


@dataclass
class DissonanceResult:
    """
    Result of dissonance detection at a specific timepoint.
    
    Attributes:
        timepoint: The timepoint analyzed
        metric: The dissonance metric (norm of coboundary)
        coboundary_values: Dictionary mapping edges to coboundary values
        top_dissonant_edges: Most dissonant region pairs
        is_consistent: True if dissonance is below threshold
    """
    timepoint: int
    metric: float
    coboundary_values: Dict[Tuple[str, str], float] = field(default_factory=dict)
    top_dissonant_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    is_consistent: bool = True
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Dissonance Analysis at t={self.timepoint}",
            "=" * 40,
            f"Global Dissonance Metric: {self.metric:.4f}",
            f"Consistent: {'Yes' if self.is_consistent else 'No'}",
        ]
        
        if self.top_dissonant_edges:
            lines.append("\nTop Dissonant Connections:")
            for r1, r2, val in self.top_dissonant_edges[:5]:
                lines.append(f"  {r1} ↔ {r2}: {val:.4f}")
        
        return "\n".join(lines)


@dataclass
class PersistentCycleResult:
    """
    Result of persistent homology computation.
    
    Attributes:
        persistence_pairs: List of (birth, death) pairs for each feature
        betti_numbers: Betti numbers at each filtration level
        significant_cycles: Cycles with persistence above threshold
        h1_generators: Representative cycles for H¹ (loops)
    """
    persistence_pairs: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    betti_numbers: Dict[float, Dict[int, int]] = field(default_factory=dict)
    significant_cycles: List[Dict[str, Any]] = field(default_factory=list)
    h1_generators: List[List[Tuple[str, str]]] = field(default_factory=list)
    
    def get_h1_persistence(self) -> List[Tuple[float, float]]:
        """Get persistence pairs for H¹ (1-dimensional holes / loops)."""
        return self.persistence_pairs.get(1, [])
    
    def total_persistence(self, dimension: int = 1) -> float:
        """Sum of all persistence values for a given dimension."""
        pairs = self.persistence_pairs.get(dimension, [])
        return sum(
            death - birth 
            for birth, death in pairs 
            if death != float('inf')
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Persistent Homology Analysis",
            "=" * 40,
        ]
        
        for dim in sorted(self.persistence_pairs.keys()):
            pairs = self.persistence_pairs[dim]
            finite_pairs = [(b, d) for b, d in pairs if d != float('inf')]
            
            lines.append(f"\nH{dim}: {len(pairs)} features")
            if finite_pairs:
                max_pers = max(d - b for b, d in finite_pairs)
                lines.append(f"  Max persistence: {max_pers:.4f}")
                lines.append(f"  Total persistence: {sum(d - b for b, d in finite_pairs):.4f}")
        
        if self.significant_cycles:
            lines.append(f"\nSignificant Cycles: {len(self.significant_cycles)}")
            for i, cycle in enumerate(self.significant_cycles[:3]):
                regions = cycle.get('regions', [])
                pers = cycle.get('persistence', 0)
                lines.append(f"  Cycle {i+1}: {' → '.join(regions[:4])}... (pers={pers:.3f})")
        
        return "\n".join(lines)


# ==================== fMRI Data Loading ====================

@dataclass
class FMRIData:
    """
    Container for loaded fMRI data.
    
    Attributes:
        time_series: (timepoints, regions) array of BOLD signals
        correlation_matrix: (regions, regions) functional connectivity
        labels: List of region labels
        atlas_name: Name of the atlas used
        metadata: Additional information
    """
    time_series: np.ndarray
    correlation_matrix: np.ndarray
    labels: List[str]
    atlas_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_fmri_data(
    func_path: Union[str, Path],
    atlas: str = "harvard_oxford",
    confounds_path: Optional[Union[str, Path]] = None,
    standardize: bool = True,
    **kwargs
) -> FMRIData:
    """
    Load fMRI data and extract time series using a brain atlas.
    
    This is a high-level wrapper around nilearn that handles:
    - Loading NIfTI files
    - Applying atlas parcellation
    - Extracting regional time series
    - Computing functional connectivity
    
    Args:
        func_path: Path to functional NIfTI file (.nii or .nii.gz)
        atlas: Atlas to use. Options:
            - "harvard_oxford": Harvard-Oxford cortical atlas
            - "aal": AAL atlas
            - "schaefer_100": Schaefer 100-region parcellation
            - "schaefer_400": Schaefer 400-region parcellation
            - Or path to custom atlas NIfTI
        confounds_path: Optional path to confounds file for denoising
        standardize: Whether to z-score the time series
        **kwargs: Additional arguments passed to NiftiLabelsMasker
    
    Returns:
        FMRIData containing time series, correlation matrix, and labels
    
    Example:
        >>> data = load_fmri_data("sub-01_bold.nii.gz", atlas="harvard_oxford")
        >>> print(f"Shape: {data.time_series.shape}")
        >>> print(f"Regions: {len(data.labels)}")
    
    Raises:
        ImportError: If nilearn is not installed
        FileNotFoundError: If func_path doesn't exist
    """
    try:
        from nilearn import datasets, input_data
        from nilearn.connectome import ConnectivityMeasure
    except ImportError:
        raise ImportError(
            "nilearn is required for fMRI loading. "
            "Install with: pip install nilearn"
        )
    
    func_path = Path(func_path)
    if not func_path.exists():
        raise FileNotFoundError(f"fMRI file not found: {func_path}")
    
    # Load atlas
    atlas_img, labels = _load_atlas(atlas)
    
    # Create masker
    masker = input_data.NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=standardize,
        memory='nilearn_cache',
        verbose=0,
        **kwargs
    )
    
    # Extract time series
    if confounds_path:
        import pandas as pd
        confounds = pd.read_csv(confounds_path, sep='\t')
        time_series = masker.fit_transform(str(func_path), confounds=confounds)
    else:
        time_series = masker.fit_transform(str(func_path))
    
    # Compute correlation matrix
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    
    return FMRIData(
        time_series=time_series,
        correlation_matrix=correlation_matrix,
        labels=labels,
        atlas_name=atlas,
        metadata={
            "source_file": str(func_path),
            "n_timepoints": time_series.shape[0],
            "n_regions": time_series.shape[1],
        }
    )


def _load_atlas(atlas: str) -> Tuple[Any, List[str]]:
    """Load a brain atlas and return the image and labels."""
    from nilearn import datasets
    
    if atlas == "harvard_oxford":
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        return atlas_data.maps, atlas_data.labels[1:]  # Skip background
    
    elif atlas == "aal":
        atlas_data = datasets.fetch_atlas_aal()
        return atlas_data.maps, atlas_data.labels
    
    elif atlas.startswith("schaefer"):
        n_rois = int(atlas.split("_")[1]) if "_" in atlas else 100
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
        return atlas_data.maps, atlas_data.labels
    
    elif Path(atlas).exists():
        # Custom atlas path
        import nibabel as nib
        img = nib.load(atlas)
        n_labels = int(np.max(img.get_fdata()))
        labels = [f"Region_{i}" for i in range(1, n_labels + 1)]
        return atlas, labels
    
    else:
        raise ValueError(f"Unknown atlas: {atlas}")


def load_connectivity_matrix(
    path: Union[str, Path],
    labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a precomputed connectivity matrix from CSV or numpy file.
    
    Args:
        path: Path to the matrix file (.csv or .npy)
        labels: Optional region labels (auto-generated if None)
    
    Returns:
        Tuple of (correlation_matrix, labels)
    
    Example:
        >>> matrix, labels = load_connectivity_matrix("connectivity.csv")
    """
    path = Path(path)
    
    if path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(path, index_col=0)
        matrix = df.values
        if labels is None:
            labels = list(df.columns)
    elif path.suffix == '.npy':
        matrix = np.load(path)
        if labels is None:
            labels = [f"Region_{i}" for i in range(matrix.shape[0])]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return matrix, labels


# ==================== BrainSheaf Class ====================

class BrainSheaf:
    """
    Sheaf-theoretic representation of brain functional connectivity.
    
    Models the brain as a sheaf (X, F) where:
    - X: Graph of brain regions (base space / nerve of cover)
    - F: Time-series signals (stalks over each region)
    - Restriction maps: Correlation-weighted connections
    
    The key mathematical objects:
    - Simplicial complex: Built from thresholded correlation matrix
    - Local sections: Time-series signals at each region
    - Coboundary map d⁰: Measures disagreement between connected regions
    - H¹(X, F): Cohomology detecting information integration failures
    
    Example:
        >>> brain = BrainSheaf(correlation_matrix, time_series, labels)
        >>> brain.build_complex(threshold=0.3)
        >>> 
        >>> # Detect dissonance at timepoint 50
        >>> result = brain.detect_dissonance(t=50)
        >>> 
        >>> # Compute persistent homology
        >>> cycles = brain.compute_persistent_homology()
    """
    
    def __init__(
        self,
        correlation_matrix: np.ndarray,
        time_series: np.ndarray,
        region_labels: Optional[List[str]] = None,
        region_coordinates: Optional[np.ndarray] = None,
    ):
        """
        Initialize a BrainSheaf from connectivity data.
        
        Args:
            correlation_matrix: (n_regions, n_regions) correlation matrix
            time_series: (n_timepoints, n_regions) BOLD signal matrix
            region_labels: Optional list of region names
            region_coordinates: Optional (n_regions, 3) MNI coordinates
        """
        self.correlation_matrix = np.asarray(correlation_matrix)
        self.time_series = np.asarray(time_series)
        
        n_regions = self.correlation_matrix.shape[0]
        
        # Validate shapes
        if self.correlation_matrix.shape != (n_regions, n_regions):
            raise ValueError("Correlation matrix must be square")
        if self.time_series.shape[1] != n_regions:
            raise ValueError(
                f"Time series has {self.time_series.shape[1]} regions, "
                f"but correlation matrix has {n_regions}"
            )
        
        # Create region labels
        if region_labels is None:
            region_labels = [f"Region_{i}" for i in range(n_regions)]
        
        # Create BrainRegion objects
        self.regions: List[BrainRegion] = []
        for i in range(n_regions):
            coords = tuple(region_coordinates[i]) if region_coordinates is not None else None
            self.regions.append(BrainRegion(
                index=i,
                label=region_labels[i],
                coordinates=coords,
                time_series=self.time_series[:, i]
            ))
        
        self._label_to_idx = {r.label: r.index for r in self.regions}
        
        # Complex state
        self._graph: Optional[nx.Graph] = None
        self._simplex_tree = None
        self._distance_matrix: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
    
    @property
    def n_regions(self) -> int:
        """Number of brain regions."""
        return len(self.regions)
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points in the fMRI data."""
        return self.time_series.shape[0]
    
    @property
    def labels(self) -> List[str]:
        """List of region labels."""
        return [r.label for r in self.regions]
    
    # ==================== Complex Construction ====================
    
    def build_complex(
        self,
        threshold: float = 0.3,
        use_absolute: bool = True,
        max_dimension: int = 2
    ) -> None:
        """
        Build a simplicial complex from the correlation matrix.
        
        Regions are connected if their correlation exceeds the threshold.
        This creates the "nerve" of the covering - the base space X.
        
        Args:
            threshold: Minimum correlation to form an edge (default: 0.3)
            use_absolute: If True, use |correlation| (default: True)
            max_dimension: Maximum simplex dimension for persistence (default: 2)
        
        Note:
            After calling this, use compute_local_sections() to populate stalks.
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required: pip install networkx")
        
        self._threshold = threshold
        
        # Build adjacency from correlation
        if use_absolute:
            adjacency = np.abs(self.correlation_matrix) > threshold
        else:
            adjacency = self.correlation_matrix > threshold
        
        # Zero diagonal (no self-loops)
        np.fill_diagonal(adjacency, False)
        
        # Create networkx graph
        self._graph = nx.Graph()
        
        # Add nodes
        for region in self.regions:
            self._graph.add_node(
                region.label,
                index=region.index,
                coordinates=region.coordinates
            )
        
        # Add edges
        for i in range(self.n_regions):
            for j in range(i + 1, self.n_regions):
                if adjacency[i, j]:
                    weight = self.correlation_matrix[i, j]
                    self._graph.add_edge(
                        self.regions[i].label,
                        self.regions[j].label,
                        weight=weight,
                        correlation=weight
                    )
        
        # Build distance matrix for persistent homology
        # Distance = 1 - |correlation|
        self._distance_matrix = 1.0 - np.abs(self.correlation_matrix)
        np.fill_diagonal(self._distance_matrix, 0)
        
        # Build GUDHI simplex tree if available
        if HAS_GUDHI:
            rips = gudhi.RipsComplex(
                distance_matrix=self._distance_matrix,
                max_edge_length=1.0 - threshold
            )
            self._simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
    
    def get_edges(self) -> List[Tuple[str, str, float]]:
        """Get all edges in the complex with their correlation weights."""
        if self._graph is None:
            raise RuntimeError("Call build_complex() first")
        
        return [
            (u, v, d.get('correlation', 0))
            for u, v, d in self._graph.edges(data=True)
        ]
    
    def get_neighbors(self, region: str) -> List[str]:
        """Get neighboring regions for a given region."""
        if self._graph is None:
            raise RuntimeError("Call build_complex() first")
        return list(self._graph.neighbors(region))
    
    # ==================== Local Sections (Stalks) ====================
    
    def compute_local_sections(self, t: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get the local section data (signal values) at each region.
        
        In sheaf terms, this maps each open set U_i to its stalk F(U_i).
        
        Args:
            t: Optional specific timepoint. If None, returns all timepoints.
        
        Returns:
            Dictionary mapping region labels to signal values
        """
        sections = {}
        for region in self.regions:
            if t is not None:
                sections[region.label] = np.array([self.time_series[t, region.index]])
            else:
                sections[region.label] = self.time_series[:, region.index]
        return sections
    
    def get_section_at_time(self, t: int) -> np.ndarray:
        """Get the signal vector across all regions at timepoint t."""
        return self.time_series[t, :]
    
    # ==================== Coboundary / Dissonance Detection ====================
    
    def detect_dissonance(
        self,
        t: int,
        normalize: bool = True,
        top_k: int = 10
    ) -> DissonanceResult:
        """
        Detect cognitive dissonance at a specific timepoint.
        
        Computes the coboundary map d⁰ which measures the difference
        in signal between connected regions. Large differences indicate
        "dissonance" - a failure of information integration.
        
        Mathematical formulation:
            (d⁰s)_{ij} = s_j - s_i  for each edge (i,j) in the complex
            
            The dissonance metric is ||d⁰s||, the norm of this vector.
        
        Args:
            t: Timepoint to analyze
            normalize: Whether to normalize by edge count (default: True)
            top_k: Number of top dissonant edges to report (default: 10)
        
        Returns:
            DissonanceResult with metric and edge-wise analysis
        """
        if self._graph is None:
            raise RuntimeError("Call build_complex() first")
        
        if t < 0 or t >= self.n_timepoints:
            raise ValueError(f"Timepoint {t} out of range [0, {self.n_timepoints})")
        
        # Get signal at timepoint t
        signal = self.get_section_at_time(t)
        
        # Compute coboundary values for each edge
        coboundary_values = {}
        for u, v, _ in self.get_edges():
            i = self._label_to_idx[u]
            j = self._label_to_idx[v]
            # Coboundary: difference of restrictions to overlap
            diff = signal[j] - signal[i]
            coboundary_values[(u, v)] = diff
        
        # Compute dissonance metric (L2 norm of coboundary)
        if coboundary_values:
            values = np.array(list(coboundary_values.values()))
            metric = np.linalg.norm(values)
            if normalize:
                metric /= np.sqrt(len(coboundary_values))
        else:
            metric = 0.0
        
        # Find top dissonant edges
        sorted_edges = sorted(
            [(u, v, abs(d)) for (u, v), d in coboundary_values.items()],
            key=lambda x: -x[2]
        )
        top_dissonant = sorted_edges[:top_k]
        
        # Threshold for consistency (heuristic: mean + 2*std of signal)
        signal_std = np.std(signal)
        consistency_threshold = 2 * signal_std
        is_consistent = metric < consistency_threshold
        
        return DissonanceResult(
            timepoint=t,
            metric=metric,
            coboundary_values=coboundary_values,
            top_dissonant_edges=top_dissonant,
            is_consistent=is_consistent
        )
    
    def compute_dissonance_timeseries(
        self,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute dissonance metric across all timepoints.
        
        Useful for identifying periods of high/low integration.
        
        Returns:
            Array of dissonance values (one per timepoint)
        """
        return np.array([
            self.detect_dissonance(t, normalize=normalize).metric
            for t in range(self.n_timepoints)
        ])
    
    # ==================== Persistent Homology ====================
    
    def compute_persistent_homology(
        self,
        max_dimension: int = 2,
        significance_threshold: float = 0.1
    ) -> PersistentCycleResult:
        """
        Compute persistent homology of the brain network.
        
        Uses Vietoris-Rips filtration on the distance matrix (1 - |correlation|).
        Long-lived H¹ features represent stable "loops" in the brain network
        that may correspond to information processing cycles.
        
        Args:
            max_dimension: Maximum homology dimension (default: 2)
            significance_threshold: Minimum persistence to report (default: 0.1)
        
        Returns:
            PersistentCycleResult with persistence pairs and cycle info
        """
        if not HAS_GUDHI:
            raise ImportError("gudhi is required: pip install gudhi")
        
        if self._simplex_tree is None:
            # Build complex if not done
            self.build_complex(threshold=0.3, max_dimension=max_dimension)
        
        # Compute persistence
        self._simplex_tree.compute_persistence()
        persistence = self._simplex_tree.persistence()
        
        # Organize by dimension
        persistence_pairs: Dict[int, List[Tuple[float, float]]] = {}
        for dim, (birth, death) in persistence:
            if dim not in persistence_pairs:
                persistence_pairs[dim] = []
            persistence_pairs[dim].append((birth, death))
        
        # Find significant cycles
        significant_cycles = []
        h1_pairs = persistence_pairs.get(1, [])
        
        for birth, death in h1_pairs:
            pers = death - birth if death != float('inf') else float('inf')
            if pers > significance_threshold:
                # Try to identify the cycle regions
                cycle_info = self._identify_cycle_regions(birth, death)
                significant_cycles.append({
                    'birth': birth,
                    'death': death,
                    'persistence': pers,
                    'regions': cycle_info
                })
        
        return PersistentCycleResult(
            persistence_pairs=persistence_pairs,
            significant_cycles=significant_cycles
        )
    
    def _identify_cycle_regions(
        self,
        birth: float,
        death: float
    ) -> List[str]:
        """
        Attempt to identify which regions participate in a cycle.
        
        This is a heuristic - proper cycle identification requires
        more sophisticated methods.
        """
        # Find edges that exist at the midpoint of the interval
        midpoint = (birth + death) / 2 if death != float('inf') else birth + 0.1
        threshold_corr = 1.0 - midpoint
        
        # Get edges present at this filtration value
        active_edges = []
        for i in range(self.n_regions):
            for j in range(i + 1, self.n_regions):
                if np.abs(self.correlation_matrix[i, j]) >= threshold_corr:
                    active_edges.append((
                        self.regions[i].label,
                        self.regions[j].label
                    ))
        
        # Try to find a cycle using networkx
        if HAS_NETWORKX and active_edges:
            G = nx.Graph()
            G.add_edges_from(active_edges)
            try:
                cycle = nx.find_cycle(G)
                return [edge[0] for edge in cycle]
            except nx.NetworkXNoCycle:
                pass
        
        # Fallback: return nodes with highest degree
        if self._graph is not None:
            sorted_nodes = sorted(
                self._graph.degree(),
                key=lambda x: -x[1]
            )[:4]
            return [n for n, _ in sorted_nodes]
        
        return []
    
    # ==================== Visualization ====================
    
    def plot_connectivity_matrix(
        self,
        ax=None,
        cmap: str = 'coolwarm',
        title: str = "Functional Connectivity Matrix"
    ):
        """
        Plot the correlation matrix as a heatmap.
        
        Args:
            ax: Optional matplotlib axis
            cmap: Colormap name
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask diagonal for clarity
        matrix = self.correlation_matrix.copy()
        np.fill_diagonal(matrix, 0)
        
        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Brain Regions')
        ax.set_ylabel('Brain Regions')
        plt.colorbar(im, ax=ax)
        
        return ax
    
    def plot_persistence_diagram(
        self,
        result: Optional[PersistentCycleResult] = None,
        ax=None
    ):
        """
        Plot persistence diagram showing topological features.
        
        Points far from the diagonal represent significant features.
        H¹ points (red) are loops in information flow.
        
        Args:
            result: PersistentCycleResult (computed if None)
            ax: Optional matplotlib axis
        """
        if not HAS_GUDHI:
            raise ImportError("gudhi is required for persistence plots")
        
        import matplotlib.pyplot as plt
        
        if result is None:
            result = self.compute_persistent_homology()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        gudhi.plot_persistence_diagram(
            self._simplex_tree.persistence(),
            axes=ax
        )
        ax.set_title("Persistence Diagram\n(Points far from diagonal = Significant features)")
        
        return ax
    
    def plot_persistence_barcode(
        self,
        result: Optional[PersistentCycleResult] = None,
        ax=None
    ):
        """
        Plot persistence barcode showing feature lifespans.
        
        Long bars represent significant topological features.
        
        Args:
            result: PersistentCycleResult (computed if None)
            ax: Optional matplotlib axis
        """
        if not HAS_GUDHI:
            raise ImportError("gudhi is required for persistence plots")
        
        import matplotlib.pyplot as plt
        
        if result is None:
            result = self.compute_persistent_homology()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        gudhi.plot_persistence_barcode(
            self._simplex_tree.persistence(),
            axes=ax
        )
        ax.set_title("Persistence Barcode\n(Long bars = Significant structure)")
        ax.set_xlabel("Filtration Value (1 - |Correlation|)")
        
        return ax
    
    def plot_active_holes(
        self,
        result: Optional[PersistentCycleResult] = None,
        highlight_cycles: bool = True,
        node_size: int = 500,
        ax=None
    ):
        """
        Plot the brain network graph highlighting active topological holes.
        
        If a specific loop (e.g., Amygdala-PFC-Thalamus) has high persistent
        homology, this highlights that cycle on the graph.
        
        Args:
            result: PersistentCycleResult (computed if None)
            highlight_cycles: Whether to highlight significant cycles
            node_size: Size of nodes in the plot
            ax: Optional matplotlib axis
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for network plots")
        
        import matplotlib.pyplot as plt
        
        if self._graph is None:
            raise RuntimeError("Call build_complex() first")
        
        if result is None:
            result = self.compute_persistent_homology()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use spring layout
        pos = nx.spring_layout(self._graph, seed=42)
        
        # Base node colors
        node_colors = ['lightblue'] * self._graph.number_of_nodes()
        
        # Highlight nodes in significant cycles
        cycle_nodes = set()
        if highlight_cycles and result.significant_cycles:
            for cycle in result.significant_cycles:
                cycle_nodes.update(cycle.get('regions', []))
        
        node_list = list(self._graph.nodes())
        for i, node in enumerate(node_list):
            if node in cycle_nodes:
                node_colors[i] = 'red'
        
        # Draw network
        nx.draw_networkx_nodes(
            self._graph, pos, 
            node_color=node_colors,
            node_size=node_size,
            ax=ax
        )
        
        # Draw edges
        edge_weights = [
            abs(self._graph[u][v].get('correlation', 0.5))
            for u, v in self._graph.edges()
        ]
        nx.draw_networkx_edges(
            self._graph, pos,
            width=[w * 2 for w in edge_weights],
            alpha=0.5,
            ax=ax
        )
        
        # Draw labels (abbreviated)
        labels = {n: n[:10] for n in self._graph.nodes()}
        nx.draw_networkx_labels(
            self._graph, pos,
            labels=labels,
            font_size=8,
            ax=ax
        )
        
        ax.set_title(
            f"Brain Network - Active Holes Highlighted\n"
            f"({len(result.significant_cycles)} significant cycles detected)"
        )
        ax.axis('off')
        
        return ax
    
    def plot_3d_brain(
        self,
        result: Optional[PersistentCycleResult] = None,
        highlight_cycles: bool = True
    ):
        """
        Plot brain regions on a 3D glass brain (requires nilearn).
        
        Highlights regions participating in significant topological cycles.
        
        Args:
            result: PersistentCycleResult (computed if None)
            highlight_cycles: Whether to highlight cycle nodes
        
        Note:
            Requires region_coordinates to be set during initialization.
        """
        try:
            from nilearn import plotting
        except ImportError:
            raise ImportError("nilearn is required for 3D brain plots")
        
        if result is None:
            result = self.compute_persistent_homology()
        
        # Get coordinates
        coords = []
        colors = []
        
        cycle_nodes = set()
        if highlight_cycles and result.significant_cycles:
            for cycle in result.significant_cycles:
                cycle_nodes.update(cycle.get('regions', []))
        
        for region in self.regions:
            if region.coordinates is not None:
                coords.append(region.coordinates)
                colors.append('red' if region.label in cycle_nodes else 'blue')
        
        if not coords:
            raise ValueError("No coordinates available for 3D plotting")
        
        coords = np.array(coords)
        
        # Plot on glass brain
        display = plotting.plot_markers(
            node_coords=coords,
            node_values=np.ones(len(coords)),
            node_cmap='coolwarm',
            node_size=50,
            display_mode='ortho',
            title="Brain Regions with Topological Obstructions"
        )
        
        return display
    
    # ==================== Analysis Utilities ====================
    
    def compute_cohomology_obstruction(
        self,
        t: int
    ) -> Dict[str, Any]:
        """
        Compute the sheaf cohomology obstruction at a timepoint.
        
        This is a higher-level analysis combining dissonance detection
        with topological features to identify integration failures.
        
        Args:
            t: Timepoint to analyze
        
        Returns:
            Dictionary with cohomology analysis results
        """
        from ..cech import compute_cech_cohomology
        
        # Get local sections at timepoint t
        sections = self.compute_local_sections(t)
        
        # Prepare data for Čech computation
        data = {k: v.flatten() for k, v in sections.items()}
        
        # Compute Čech cohomology
        result = compute_cech_cohomology(data)
        
        dissonance = self.detect_dissonance(t)
        
        return {
            'timepoint': t,
            'h0': result.h0,
            'h1': result.h1,
            'is_consistent': result.is_consistent,
            'dissonance_metric': dissonance.metric,
            'obstruction_dimension': result.obstruction_dimension,
            'summary': result.summary()
        }
    
    def summary(self) -> str:
        """Human-readable summary of the BrainSheaf."""
        lines = [
            "BrainSheaf Summary",
            "=" * 40,
            f"Regions: {self.n_regions}",
            f"Timepoints: {self.n_timepoints}",
        ]
        
        if self._graph is not None:
            lines.append(f"Complex built: Yes (threshold={self._threshold})")
            lines.append(f"  Nodes: {self._graph.number_of_nodes()}")
            lines.append(f"  Edges: {self._graph.number_of_edges()}")
        else:
            lines.append("Complex built: No (call build_complex())")
        
        # Connectivity stats
        corr_flat = self.correlation_matrix[
            np.triu_indices(self.n_regions, k=1)
        ]
        lines.append(f"\nConnectivity Statistics:")
        lines.append(f"  Mean correlation: {np.mean(corr_flat):.3f}")
        lines.append(f"  Max correlation: {np.max(corr_flat):.3f}")
        lines.append(f"  Min correlation: {np.min(corr_flat):.3f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"BrainSheaf(regions={self.n_regions}, timepoints={self.n_timepoints})"
