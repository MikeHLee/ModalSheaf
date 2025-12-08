"""
Diagnostic tools for identifying problematic viewpoints/contributors.

This module provides tools to:
1. Identify which local sections (sensors, sources, contributors) disagree
2. Compute trust/reliability scores for each contributor
3. Detect outliers, compromised sources, or systematic biases
4. Suggest which sources to exclude for consistent gluing
5. Perform consensus analysis across multiple observations

Mathematical basis:
- Each local section contributes to overlaps with neighbors
- Inconsistency on an overlap implicates both contributors
- By analyzing the pattern of inconsistencies, we can identify:
  - Isolated bad actors (one source disagrees with everyone)
  - Systematic drift (gradual disagreement)
  - Clusters of agreement (factions)
  - Compromised sources (sudden change in behavior)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import numpy as np

from .gluing import LocalSection, Overlap, GluingResult, GluingProtocol


# ==================== Diagnostic Results ====================

@dataclass
class ContributorScore:
    """Trust/reliability score for a single contributor."""
    id: str
    trust_score: float  # 0.0 = untrusted, 1.0 = fully trusted
    consistency_score: float  # How consistent with neighbors
    num_overlaps: int  # How many overlaps this contributor participates in
    num_agreements: int  # How many overlaps are consistent
    num_disagreements: int  # How many overlaps are inconsistent
    disagreement_partners: List[str] = field(default_factory=list)
    agreement_partners: List[str] = field(default_factory=list)
    anomaly_flags: List[str] = field(default_factory=list)
    
    @property
    def is_outlier(self) -> bool:
        """True if this contributor disagrees with most neighbors."""
        if self.num_overlaps == 0:
            return False
        return self.num_disagreements / self.num_overlaps > 0.5
    
    @property
    def is_isolated(self) -> bool:
        """True if this contributor has no overlaps."""
        return self.num_overlaps == 0
    
    def __repr__(self) -> str:
        status = "⚠️" if self.is_outlier else "✓"
        return f"ContributorScore({self.id}: trust={self.trust_score:.2f}, {status})"


@dataclass
class ClusterAnalysis:
    """Analysis of agreement clusters (factions)."""
    clusters: List[Set[str]]  # Groups that agree internally
    inter_cluster_conflicts: List[Tuple[int, int, float]]  # (cluster_i, cluster_j, disagreement)
    bridge_contributors: List[str]  # Contributors that connect clusters
    
    @property
    def num_factions(self) -> int:
        return len(self.clusters)
    
    @property
    def is_polarized(self) -> bool:
        """True if there are distinct factions that disagree."""
        return self.num_factions > 1 and len(self.inter_cluster_conflicts) > 0


@dataclass 
class DiagnosticReport:
    """Complete diagnostic report for a gluing problem."""
    # Per-contributor analysis
    contributor_scores: Dict[str, ContributorScore]
    
    # Overall metrics
    global_consistency: float  # 0.0 = total disagreement, 1.0 = perfect agreement
    num_contributors: int
    num_overlaps: int
    num_consistent_overlaps: int
    num_inconsistent_overlaps: int
    
    # Identified problems
    outliers: List[str]  # Contributors that disagree with most
    suspects: List[str]  # Contributors with anomalous patterns
    trusted: List[str]  # Contributors with high trust scores
    
    # Cluster analysis
    clusters: Optional[ClusterAnalysis] = None
    
    # Recommendations
    exclude_for_consensus: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"DiagnosticReport(\n"
            f"  consistency={self.global_consistency:.1%},\n"
            f"  contributors={self.num_contributors},\n"
            f"  outliers={self.outliers},\n"
            f"  trusted={self.trusted[:3]}...\n"
            f")"
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "DIAGNOSTIC REPORT",
            "=" * 60,
            f"Global consistency: {self.global_consistency:.1%}",
            f"Contributors: {self.num_contributors}",
            f"Overlaps: {self.num_consistent_overlaps}/{self.num_overlaps} consistent",
            "",
        ]
        
        if self.outliers:
            lines.append("⚠️  OUTLIERS (disagree with most):")
            for o in self.outliers:
                score = self.contributor_scores[o]
                lines.append(f"    - {o}: trust={score.trust_score:.2f}, "
                           f"disagrees with {score.disagreement_partners}")
        
        if self.suspects:
            lines.append("\n🔍 SUSPECTS (anomalous patterns):")
            for s in self.suspects:
                score = self.contributor_scores[s]
                lines.append(f"    - {s}: {score.anomaly_flags}")
        
        if self.clusters and self.clusters.is_polarized:
            lines.append(f"\n📊 FACTIONS DETECTED: {self.clusters.num_factions} clusters")
            for i, cluster in enumerate(self.clusters.clusters):
                lines.append(f"    Cluster {i}: {cluster}")
        
        if self.exclude_for_consensus:
            lines.append(f"\n💡 RECOMMENDATION: Exclude {self.exclude_for_consensus} for consensus")
            lines.append(f"   Confidence: {self.confidence:.1%}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ==================== Diagnostic Analyzer ====================

class DiagnosticAnalyzer:
    """
    Analyzes gluing problems to identify problematic contributors.
    
    Usage:
        analyzer = DiagnosticAnalyzer(protocol)
        report = analyzer.analyze(sections, overlaps)
        print(report.summary())
    """
    
    def __init__(
        self,
        protocol: GluingProtocol,
        consistency_threshold: float = 0.1,
        outlier_threshold: float = 0.5,
    ):
        """
        Args:
            protocol: The gluing protocol to use for consistency checks
            consistency_threshold: Max error for "consistent" overlap
            outlier_threshold: Fraction of disagreements to be an outlier
        """
        self.protocol = protocol
        self.consistency_threshold = consistency_threshold
        self.outlier_threshold = outlier_threshold
    
    def analyze(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> DiagnosticReport:
        """
        Analyze a gluing problem and identify problematic contributors.
        """
        # Build section lookup
        section_map = {s.id: s for s in sections}
        
        # Compute consistency for each overlap
        overlap_results = []
        for overlap in overlaps:
            id1, id2 = overlap.section_ids
            sec1 = section_map.get(id1)
            sec2 = section_map.get(id2)
            
            if sec1 and sec2:
                data1, data2 = self.protocol.extract_overlap(sec1, sec2, overlap)
                error = self.protocol.measure_consistency(data1, data2, overlap)
                is_consistent = error <= self.consistency_threshold
                
                overlap_results.append({
                    "overlap": overlap,
                    "sections": (id1, id2),
                    "error": error,
                    "is_consistent": is_consistent,
                })
        
        # Compute per-contributor scores
        contributor_scores = self._compute_contributor_scores(
            sections, overlap_results
        )
        
        # Identify outliers
        outliers = [
            cid for cid, score in contributor_scores.items()
            if score.is_outlier
        ]
        
        # Identify trusted contributors
        trusted = sorted(
            contributor_scores.keys(),
            key=lambda cid: contributor_scores[cid].trust_score,
            reverse=True
        )
        
        # Detect anomalies
        suspects = self._detect_anomalies(contributor_scores, overlap_results)
        
        # Cluster analysis
        clusters = self._analyze_clusters(sections, overlap_results)
        
        # Compute recommendations
        exclude, confidence = self._recommend_exclusions(
            contributor_scores, overlap_results, clusters
        )
        
        # Global metrics
        num_consistent = sum(1 for r in overlap_results if r["is_consistent"])
        global_consistency = num_consistent / len(overlap_results) if overlap_results else 1.0
        
        return DiagnosticReport(
            contributor_scores=contributor_scores,
            global_consistency=global_consistency,
            num_contributors=len(sections),
            num_overlaps=len(overlaps),
            num_consistent_overlaps=num_consistent,
            num_inconsistent_overlaps=len(overlap_results) - num_consistent,
            outliers=outliers,
            suspects=suspects,
            trusted=trusted,
            clusters=clusters,
            exclude_for_consensus=exclude,
            confidence=confidence,
        )
    
    def _compute_contributor_scores(
        self,
        sections: List[LocalSection],
        overlap_results: List[Dict],
    ) -> Dict[str, ContributorScore]:
        """Compute trust/consistency scores for each contributor."""
        scores = {}
        
        # Initialize
        for section in sections:
            scores[section.id] = ContributorScore(
                id=section.id,
                trust_score=1.0,
                consistency_score=1.0,
                num_overlaps=0,
                num_agreements=0,
                num_disagreements=0,
                disagreement_partners=[],
                agreement_partners=[],
            )
        
        # Accumulate overlap results
        for result in overlap_results:
            id1, id2 = result["sections"]
            is_consistent = result["is_consistent"]
            
            for cid, partner in [(id1, id2), (id2, id1)]:
                if cid in scores:
                    scores[cid].num_overlaps += 1
                    if is_consistent:
                        scores[cid].num_agreements += 1
                        scores[cid].agreement_partners.append(partner)
                    else:
                        scores[cid].num_disagreements += 1
                        scores[cid].disagreement_partners.append(partner)
        
        # Compute final scores
        for cid, score in scores.items():
            if score.num_overlaps > 0:
                score.consistency_score = score.num_agreements / score.num_overlaps
                
                # Trust score: consistency weighted by number of overlaps
                # More overlaps = more confidence in the score
                weight = min(1.0, score.num_overlaps / 5)  # Cap at 5 overlaps
                score.trust_score = score.consistency_score * weight + (1 - weight) * 0.5
            else:
                score.consistency_score = 0.5  # Unknown
                score.trust_score = 0.5  # Unknown
        
        return scores
    
    def _detect_anomalies(
        self,
        contributor_scores: Dict[str, ContributorScore],
        overlap_results: List[Dict],
    ) -> List[str]:
        """Detect contributors with anomalous patterns."""
        suspects = []
        
        for cid, score in contributor_scores.items():
            flags = []
            
            # Flag 1: Disagrees with everyone
            if score.num_overlaps >= 3 and score.num_agreements == 0:
                flags.append("disagrees_with_all")
            
            # Flag 2: Only agrees with one other (possible collusion)
            if score.num_overlaps >= 3 and score.num_agreements == 1:
                flags.append(f"only_agrees_with_{score.agreement_partners[0]}")
            
            # Flag 3: High variance in disagreement (inconsistent behavior)
            # Would need error magnitudes to compute this properly
            
            if flags:
                score.anomaly_flags = flags
                suspects.append(cid)
        
        return suspects
    
    def _analyze_clusters(
        self,
        sections: List[LocalSection],
        overlap_results: List[Dict],
    ) -> ClusterAnalysis:
        """Identify clusters of agreement (factions)."""
        # Build agreement graph
        agreement_graph = defaultdict(set)
        disagreement_graph = defaultdict(set)
        
        for result in overlap_results:
            id1, id2 = result["sections"]
            if result["is_consistent"]:
                agreement_graph[id1].add(id2)
                agreement_graph[id2].add(id1)
            else:
                disagreement_graph[id1].add(id2)
                disagreement_graph[id2].add(id1)
        
        # Find connected components in agreement graph
        all_ids = {s.id for s in sections}
        visited = set()
        clusters = []
        
        for start_id in all_ids:
            if start_id in visited:
                continue
            
            # BFS to find cluster
            cluster = set()
            queue = [start_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.add(current)
                
                for neighbor in agreement_graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if cluster:
                clusters.append(cluster)
        
        # Find inter-cluster conflicts
        inter_cluster_conflicts = []
        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters[i+1:], i+1):
                # Count disagreements between clusters
                conflicts = 0
                for id1 in cluster_i:
                    for id2 in cluster_j:
                        if id2 in disagreement_graph[id1]:
                            conflicts += 1
                
                if conflicts > 0:
                    inter_cluster_conflicts.append((i, j, conflicts))
        
        # Find bridge contributors (in multiple clusters or connecting them)
        bridge_contributors = []
        for cid in all_ids:
            agrees_with = agreement_graph[cid]
            disagrees_with = disagreement_graph[cid]
            
            # Check if this contributor connects different clusters
            clusters_connected = set()
            for i, cluster in enumerate(clusters):
                if cid in cluster or agrees_with & cluster:
                    clusters_connected.add(i)
            
            if len(clusters_connected) > 1:
                bridge_contributors.append(cid)
        
        return ClusterAnalysis(
            clusters=clusters,
            inter_cluster_conflicts=inter_cluster_conflicts,
            bridge_contributors=bridge_contributors,
        )
    
    def _recommend_exclusions(
        self,
        contributor_scores: Dict[str, ContributorScore],
        overlap_results: List[Dict],
        clusters: ClusterAnalysis,
    ) -> Tuple[List[str], float]:
        """Recommend which contributors to exclude for consensus."""
        exclude = []
        
        # Strategy 1: Exclude clear outliers
        for cid, score in contributor_scores.items():
            if score.is_outlier and score.num_overlaps >= 2:
                exclude.append(cid)
        
        # Strategy 2: If polarized, recommend excluding smaller faction
        if clusters.is_polarized and len(clusters.clusters) == 2:
            cluster_sizes = [len(c) for c in clusters.clusters]
            if cluster_sizes[0] != cluster_sizes[1]:
                smaller_idx = 0 if cluster_sizes[0] < cluster_sizes[1] else 1
                # Only exclude if much smaller
                if cluster_sizes[smaller_idx] < cluster_sizes[1 - smaller_idx] / 2:
                    exclude.extend(clusters.clusters[smaller_idx])
        
        # Compute confidence
        if not exclude:
            confidence = 1.0  # No exclusions needed
        else:
            # Confidence based on how clear the outliers are
            avg_outlier_score = np.mean([
                contributor_scores[cid].consistency_score 
                for cid in exclude
                if cid in contributor_scores
            ]) if exclude else 0.5
            
            confidence = 1.0 - avg_outlier_score
        
        return list(set(exclude)), confidence


# ==================== Temporal Analysis ====================

class TemporalAnalyzer:
    """
    Analyze contributor behavior over time to detect drift or compromise.
    
    Tracks:
    - Consistency history per contributor
    - Sudden changes in behavior
    - Gradual drift
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: Dict[str, List[Any]] = defaultdict(list)
    
    def record_observation(
        self,
        contributor_id: str,
        consistency_score: float,
        timestamp: Any = None,
    ):
        """Record a consistency observation for a contributor."""
        self.history[contributor_id].append(consistency_score)
        self.timestamps[contributor_id].append(timestamp)
        
        # Keep only recent history
        if len(self.history[contributor_id]) > self.window_size * 2:
            self.history[contributor_id] = self.history[contributor_id][-self.window_size*2:]
            self.timestamps[contributor_id] = self.timestamps[contributor_id][-self.window_size*2:]
    
    def detect_drift(self, contributor_id: str) -> Optional[Dict[str, Any]]:
        """Detect if a contributor's behavior is drifting."""
        history = self.history.get(contributor_id, [])
        
        if len(history) < self.window_size:
            return None
        
        # Compare recent window to older window
        old_window = history[:self.window_size]
        new_window = history[-self.window_size:]
        
        old_mean = np.mean(old_window)
        new_mean = np.mean(new_window)
        
        drift = new_mean - old_mean
        
        if abs(drift) > 0.2:  # Significant drift
            return {
                "contributor": contributor_id,
                "drift": drift,
                "direction": "improving" if drift > 0 else "degrading",
                "old_consistency": old_mean,
                "new_consistency": new_mean,
            }
        
        return None
    
    def detect_sudden_change(self, contributor_id: str) -> Optional[Dict[str, Any]]:
        """Detect sudden changes in behavior (possible compromise)."""
        history = self.history.get(contributor_id, [])
        
        if len(history) < 3:
            return None
        
        # Look for sudden drops
        for i in range(1, len(history)):
            change = history[i] - history[i-1]
            
            if change < -0.5:  # Sudden large drop
                return {
                    "contributor": contributor_id,
                    "type": "sudden_drop",
                    "change": change,
                    "at_index": i,
                    "timestamp": self.timestamps[contributor_id][i],
                    "before": history[i-1],
                    "after": history[i],
                }
        
        return None
    
    def get_contributor_trend(self, contributor_id: str) -> Dict[str, Any]:
        """Get trend analysis for a contributor."""
        history = self.history.get(contributor_id, [])
        
        if not history:
            return {"status": "no_data"}
        
        return {
            "contributor": contributor_id,
            "num_observations": len(history),
            "current_consistency": history[-1],
            "mean_consistency": np.mean(history),
            "std_consistency": np.std(history),
            "trend": "stable" if np.std(history) < 0.1 else "variable",
            "drift": self.detect_drift(contributor_id),
            "sudden_change": self.detect_sudden_change(contributor_id),
        }


# ==================== Consensus Builder ====================

class ConsensusBuilder:
    """
    Build consensus by iteratively excluding problematic contributors.
    
    Strategies:
    1. Majority voting: Keep contributors that agree with majority
    2. Weighted consensus: Weight by trust scores
    3. Iterative exclusion: Remove worst outlier until consistent
    """
    
    def __init__(
        self,
        protocol: GluingProtocol,
        strategy: str = "iterative",  # "majority", "weighted", "iterative"
        max_exclusions: int = None,
    ):
        self.protocol = protocol
        self.strategy = strategy
        self.max_exclusions = max_exclusions
        self.analyzer = DiagnosticAnalyzer(protocol)
    
    def build_consensus(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> Tuple[GluingResult, List[str]]:
        """
        Build consensus by excluding problematic contributors.
        
        Returns:
            Tuple of (gluing_result, excluded_contributors)
        """
        if self.strategy == "iterative":
            return self._iterative_consensus(sections, overlaps)
        elif self.strategy == "majority":
            return self._majority_consensus(sections, overlaps)
        else:
            return self._weighted_consensus(sections, overlaps)
    
    def _iterative_consensus(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> Tuple[GluingResult, List[str]]:
        """Iteratively exclude worst outlier until consistent."""
        current_sections = list(sections)
        current_overlaps = list(overlaps)
        excluded = []
        
        max_iter = self.max_exclusions or len(sections) // 2
        
        for _ in range(max_iter):
            # Try gluing
            result = self.protocol.glue(current_sections, current_overlaps)
            
            if result.success:
                return result, excluded
            
            # Analyze to find worst contributor
            report = self.analyzer.analyze(current_sections, current_overlaps)
            
            if not report.outliers:
                # No clear outlier, stop
                break
            
            # Exclude worst outlier
            worst = min(
                report.outliers,
                key=lambda cid: report.contributor_scores[cid].trust_score
            )
            
            excluded.append(worst)
            current_sections = [s for s in current_sections if s.id != worst]
            current_overlaps = [
                o for o in current_overlaps 
                if worst not in o.section_ids
            ]
        
        # Final attempt
        result = self.protocol.glue(current_sections, current_overlaps)
        return result, excluded
    
    def _majority_consensus(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> Tuple[GluingResult, List[str]]:
        """Keep only contributors that agree with majority."""
        report = self.analyzer.analyze(sections, overlaps)
        
        # Find majority cluster
        if report.clusters and report.clusters.clusters:
            largest_cluster = max(report.clusters.clusters, key=len)
            
            # Exclude everyone not in largest cluster
            excluded = [
                s.id for s in sections 
                if s.id not in largest_cluster
            ]
            
            remaining_sections = [s for s in sections if s.id in largest_cluster]
            remaining_overlaps = [
                o for o in overlaps
                if all(sid in largest_cluster for sid in o.section_ids)
            ]
            
            result = self.protocol.glue(remaining_sections, remaining_overlaps)
            return result, excluded
        
        return self.protocol.glue(sections, overlaps), []
    
    def _weighted_consensus(
        self,
        sections: List[LocalSection],
        overlaps: List[Overlap],
    ) -> Tuple[GluingResult, List[str]]:
        """Weight contributions by trust scores."""
        report = self.analyzer.analyze(sections, overlaps)
        
        # Exclude low-trust contributors
        threshold = 0.3
        excluded = [
            cid for cid, score in report.contributor_scores.items()
            if score.trust_score < threshold
        ]
        
        remaining_sections = [s for s in sections if s.id not in excluded]
        remaining_overlaps = [
            o for o in overlaps
            if all(sid not in excluded for sid in o.section_ids)
        ]
        
        result = self.protocol.glue(remaining_sections, remaining_overlaps)
        return result, excluded


# ==================== Utility Functions ====================

def diagnose_gluing_problem(
    protocol: GluingProtocol,
    sections: List[LocalSection],
    overlaps: List[Overlap],
) -> DiagnosticReport:
    """
    Convenience function to diagnose a gluing problem.
    
    Example:
        report = diagnose_gluing_problem(
            PanoramaGluing(),
            sections,
            overlaps
        )
        print(report.summary())
    """
    analyzer = DiagnosticAnalyzer(protocol)
    return analyzer.analyze(sections, overlaps)


def find_consensus(
    protocol: GluingProtocol,
    sections: List[LocalSection],
    overlaps: List[Overlap],
    strategy: str = "iterative",
) -> Tuple[GluingResult, List[str], DiagnosticReport]:
    """
    Find consensus by excluding problematic contributors.
    
    Returns:
        Tuple of (result, excluded_ids, diagnostic_report)
    """
    # Initial diagnosis
    analyzer = DiagnosticAnalyzer(protocol)
    initial_report = analyzer.analyze(sections, overlaps)
    
    # Build consensus
    builder = ConsensusBuilder(protocol, strategy=strategy)
    result, excluded = builder.build_consensus(sections, overlaps)
    
    return result, excluded, initial_report
