#!/usr/bin/env python3
"""
Diagnostic Analysis Example for ModalSheaf.

This example demonstrates how to:
1. Identify which contributors (sensors, sources) are causing inconsistencies
2. Compute trust scores for each contributor
3. Detect outliers, compromised sources, and factions
4. Build consensus by excluding problematic contributors
5. Track contributor behavior over time

Real-world applications:
- Sensor fusion: Which sensor is miscalibrated?
- Multi-source data: Which source is unreliable?
- Voting systems: Which voters are anomalous?
- Distributed systems: Which node is Byzantine?
"""

import numpy as np
from modalsheaf.gluing import (
    LocalSection,
    Overlap,
    HierarchicalGluing,
    glue_with_protocol,
)
from modalsheaf.diagnostics import (
    DiagnosticAnalyzer,
    TemporalAnalyzer,
    ConsensusBuilder,
    diagnose_gluing_problem,
    find_consensus,
)


def demo_sensor_calibration():
    """Identify a miscalibrated sensor in a sensor network."""
    print("\n" + "=" * 60)
    print("SENSOR CALIBRATION DIAGNOSIS")
    print("=" * 60)
    print("""
    Scenario: 5 temperature sensors in a room
    - Sensors should agree on overlapping measurements
    - One sensor is miscalibrated (reads 5°C too high)
    - Goal: Identify the bad sensor
    """)
    
    # Create sensor readings
    # True temperature is ~20°C
    np.random.seed(42)
    
    sections = [
        LocalSection(
            id="sensor_A",
            data={"temp": 20.1 + np.random.randn() * 0.5},
            metadata={"location": "north"}
        ),
        LocalSection(
            id="sensor_B", 
            data={"temp": 19.8 + np.random.randn() * 0.5},
            metadata={"location": "south"}
        ),
        LocalSection(
            id="sensor_C",
            data={"temp": 20.3 + np.random.randn() * 0.5},
            metadata={"location": "east"}
        ),
        LocalSection(
            id="sensor_D",
            data={"temp": 25.2 + np.random.randn() * 0.5},  # MISCALIBRATED! +5°C
            metadata={"location": "west"}
        ),
        LocalSection(
            id="sensor_E",
            data={"temp": 20.0 + np.random.randn() * 0.5},
            metadata={"location": "center"}
        ),
    ]
    
    # All sensors have overlapping coverage
    overlaps = [
        Overlap(section_ids=("sensor_A", "sensor_B")),
        Overlap(section_ids=("sensor_A", "sensor_C")),
        Overlap(section_ids=("sensor_A", "sensor_E")),
        Overlap(section_ids=("sensor_B", "sensor_C")),
        Overlap(section_ids=("sensor_B", "sensor_D")),
        Overlap(section_ids=("sensor_C", "sensor_D")),
        Overlap(section_ids=("sensor_C", "sensor_E")),
        Overlap(section_ids=("sensor_D", "sensor_E")),
    ]
    
    # Custom protocol for temperature comparison
    class TemperatureGluing(HierarchicalGluing):
        def extract_overlap(self, sec1, sec2, overlap):
            return sec1.data.get("temp", 0), sec2.data.get("temp", 0)
        
        def measure_consistency(self, data1, data2, overlap):
            return abs(data1 - data2)
    
    protocol = TemperatureGluing()
    
    # Diagnose
    analyzer = DiagnosticAnalyzer(protocol, consistency_threshold=1.0)
    report = analyzer.analyze(sections, overlaps)
    
    print(report.summary())
    
    # Show individual scores
    print("\nDetailed contributor scores:")
    for cid, score in sorted(
        report.contributor_scores.items(),
        key=lambda x: x[1].trust_score
    ):
        section = next(s for s in sections if s.id == cid)
        temp = section.data["temp"]
        print(f"  {cid}: temp={temp:.1f}°C, trust={score.trust_score:.2f}, "
              f"agrees={score.agreement_partners}, disagrees={score.disagreement_partners}")
    
    # Build consensus by excluding bad sensor
    print("\n--- Building consensus ---")
    builder = ConsensusBuilder(protocol, strategy="iterative")
    result, excluded = builder.build_consensus(sections, overlaps)
    
    print(f"Excluded: {excluded}")
    print(f"Consensus achieved: {result.success}")
    
    if excluded:
        print(f"\n✓ Identified miscalibrated sensor: {excluded[0]}")


def demo_news_sources():
    """Identify unreliable news sources based on fact agreement."""
    print("\n" + "=" * 60)
    print("NEWS SOURCE RELIABILITY")
    print("=" * 60)
    print("""
    Scenario: Multiple news sources reporting on events
    - Reliable sources should agree on facts
    - One source is spreading misinformation
    - Goal: Identify unreliable sources
    """)
    
    # Simulate news sources reporting facts
    # Fact: "Event happened at 3pm"
    sections = [
        LocalSection(
            id="reuters",
            data={"time": "3:00 PM", "location": "downtown", "casualties": 0},
            metadata={"type": "wire_service"}
        ),
        LocalSection(
            id="ap_news",
            data={"time": "3:05 PM", "location": "downtown", "casualties": 0},
            metadata={"type": "wire_service"}
        ),
        LocalSection(
            id="local_paper",
            data={"time": "3:00 PM", "location": "downtown", "casualties": 0},
            metadata={"type": "newspaper"}
        ),
        LocalSection(
            id="tabloid",
            data={"time": "2:00 PM", "location": "uptown", "casualties": 50},  # WRONG!
            metadata={"type": "tabloid"}
        ),
        LocalSection(
            id="tv_news",
            data={"time": "3:00 PM", "location": "downtown", "casualties": 0},
            metadata={"type": "broadcast"}
        ),
    ]
    
    # All sources cover the same story
    overlaps = [
        Overlap(section_ids=("reuters", "ap_news")),
        Overlap(section_ids=("reuters", "local_paper")),
        Overlap(section_ids=("reuters", "tabloid")),
        Overlap(section_ids=("reuters", "tv_news")),
        Overlap(section_ids=("ap_news", "local_paper")),
        Overlap(section_ids=("ap_news", "tabloid")),
        Overlap(section_ids=("local_paper", "tabloid")),
        Overlap(section_ids=("local_paper", "tv_news")),
        Overlap(section_ids=("tabloid", "tv_news")),
    ]
    
    class NewsGluing(HierarchicalGluing):
        def extract_overlap(self, sec1, sec2, overlap):
            return sec1.data, sec2.data
        
        def measure_consistency(self, data1, data2, overlap):
            # Count disagreements
            disagreements = 0
            for key in data1:
                if key in data2 and data1[key] != data2[key]:
                    disagreements += 1
            return disagreements
    
    protocol = NewsGluing()
    report = diagnose_gluing_problem(protocol, sections, overlaps)
    
    print(report.summary())
    
    print("\nSource reliability ranking:")
    for cid in report.trusted:
        score = report.contributor_scores[cid]
        section = next(s for s in sections if s.id == cid)
        print(f"  {cid} ({section.metadata['type']}): trust={score.trust_score:.2f}")


def demo_faction_detection():
    """Detect factions/polarization in a group."""
    print("\n" + "=" * 60)
    print("FACTION DETECTION")
    print("=" * 60)
    print("""
    Scenario: Committee voting on proposals
    - Two factions have formed with opposing views
    - Goal: Identify the factions and bridge members
    """)
    
    # Simulate committee members with votes
    # Faction A: members 1, 2, 3 vote YES
    # Faction B: members 4, 5, 6 vote NO
    # Member 7 is a bridge (votes mixed)
    
    sections = [
        LocalSection(id="member_1", data={"vote": "YES", "faction": "A"}),
        LocalSection(id="member_2", data={"vote": "YES", "faction": "A"}),
        LocalSection(id="member_3", data={"vote": "YES", "faction": "A"}),
        LocalSection(id="member_4", data={"vote": "NO", "faction": "B"}),
        LocalSection(id="member_5", data={"vote": "NO", "faction": "B"}),
        LocalSection(id="member_6", data={"vote": "NO", "faction": "B"}),
        LocalSection(id="member_7", data={"vote": "ABSTAIN", "faction": "bridge"}),
    ]
    
    # Create overlaps based on who interacts with whom
    overlaps = [
        # Within faction A
        Overlap(section_ids=("member_1", "member_2")),
        Overlap(section_ids=("member_2", "member_3")),
        Overlap(section_ids=("member_1", "member_3")),
        # Within faction B
        Overlap(section_ids=("member_4", "member_5")),
        Overlap(section_ids=("member_5", "member_6")),
        Overlap(section_ids=("member_4", "member_6")),
        # Cross-faction (these will disagree)
        Overlap(section_ids=("member_3", "member_4")),
        Overlap(section_ids=("member_2", "member_5")),
        # Bridge member
        Overlap(section_ids=("member_7", "member_1")),
        Overlap(section_ids=("member_7", "member_4")),
    ]
    
    class VotingGluing(HierarchicalGluing):
        def extract_overlap(self, sec1, sec2, overlap):
            return sec1.data["vote"], sec2.data["vote"]
        
        def measure_consistency(self, data1, data2, overlap):
            if data1 == data2:
                return 0.0
            if "ABSTAIN" in (data1, data2):
                return 0.3  # Partial disagreement
            return 1.0  # Full disagreement
    
    protocol = VotingGluing()
    report = diagnose_gluing_problem(protocol, sections, overlaps)
    
    print(report.summary())
    
    if report.clusters:
        print("\nDetected clusters (factions):")
        for i, cluster in enumerate(report.clusters.clusters):
            print(f"  Faction {i+1}: {cluster}")
        
        if report.clusters.bridge_contributors:
            print(f"\nBridge members: {report.clusters.bridge_contributors}")
        
        if report.clusters.inter_cluster_conflicts:
            print("\nInter-faction conflicts:")
            for c1, c2, count in report.clusters.inter_cluster_conflicts:
                print(f"  Faction {c1+1} vs Faction {c2+1}: {count} disagreements")


def demo_temporal_drift():
    """Detect sensor drift over time."""
    print("\n" + "=" * 60)
    print("TEMPORAL DRIFT DETECTION")
    print("=" * 60)
    print("""
    Scenario: Monitoring sensor reliability over time
    - Sensors may drift or suddenly fail
    - Goal: Detect degradation before it causes problems
    """)
    
    # Simulate sensor readings over time
    temporal = TemporalAnalyzer(window_size=5)
    
    # Sensor A: Stable
    for i in range(15):
        temporal.record_observation("sensor_A", 0.95 + np.random.randn() * 0.02, timestamp=i)
    
    # Sensor B: Gradual drift
    for i in range(15):
        drift = i * 0.03  # Gets worse over time
        temporal.record_observation("sensor_B", 0.95 - drift + np.random.randn() * 0.02, timestamp=i)
    
    # Sensor C: Sudden failure at t=10
    for i in range(15):
        if i < 10:
            temporal.record_observation("sensor_C", 0.95 + np.random.randn() * 0.02, timestamp=i)
        else:
            temporal.record_observation("sensor_C", 0.2 + np.random.randn() * 0.1, timestamp=i)
    
    print("\nSensor trend analysis:")
    for sensor_id in ["sensor_A", "sensor_B", "sensor_C"]:
        trend = temporal.get_contributor_trend(sensor_id)
        print(f"\n  {sensor_id}:")
        print(f"    Current consistency: {trend['current_consistency']:.2f}")
        print(f"    Mean consistency: {trend['mean_consistency']:.2f}")
        print(f"    Trend: {trend['trend']}")
        
        if trend['drift']:
            print(f"    ⚠️  DRIFT DETECTED: {trend['drift']['direction']}")
            print(f"       Old: {trend['drift']['old_consistency']:.2f} → "
                  f"New: {trend['drift']['new_consistency']:.2f}")
        
        if trend['sudden_change']:
            print(f"    🚨 SUDDEN CHANGE at t={trend['sudden_change']['at_index']}")
            print(f"       Before: {trend['sudden_change']['before']:.2f} → "
                  f"After: {trend['sudden_change']['after']:.2f}")


def demo_consensus_building():
    """Build consensus by excluding problematic contributors."""
    print("\n" + "=" * 60)
    print("CONSENSUS BUILDING")
    print("=" * 60)
    print("""
    Scenario: Distributed system with Byzantine nodes
    - Most nodes are honest and agree
    - Some nodes are Byzantine (malicious/faulty)
    - Goal: Achieve consensus despite bad actors
    """)
    
    # Simulate distributed nodes
    # Honest nodes report value ~100
    # Byzantine nodes report random values
    np.random.seed(123)
    
    sections = []
    
    # 7 honest nodes
    for i in range(7):
        sections.append(LocalSection(
            id=f"node_{i}",
            data={"value": 100 + np.random.randn() * 2},
            metadata={"type": "honest"}
        ))
    
    # 3 Byzantine nodes
    for i in range(7, 10):
        sections.append(LocalSection(
            id=f"node_{i}",
            data={"value": np.random.randint(0, 200)},  # Random!
            metadata={"type": "byzantine"}
        ))
    
    # All nodes communicate with neighbors
    overlaps = []
    for i in range(len(sections)):
        for j in range(i+1, min(i+4, len(sections))):  # Each node talks to 3 neighbors
            overlaps.append(Overlap(section_ids=(sections[i].id, sections[j].id)))
    
    class DistributedGluing(HierarchicalGluing):
        def extract_overlap(self, sec1, sec2, overlap):
            return sec1.data["value"], sec2.data["value"]
        
        def measure_consistency(self, data1, data2, overlap):
            return abs(data1 - data2)
    
    protocol = DistributedGluing()
    
    # Initial diagnosis
    print("\nInitial state:")
    report = diagnose_gluing_problem(protocol, sections, overlaps)
    print(f"  Global consistency: {report.global_consistency:.1%}")
    print(f"  Outliers detected: {report.outliers}")
    
    # Try different consensus strategies
    for strategy in ["iterative", "majority"]:
        print(f"\n--- Strategy: {strategy} ---")
        
        builder = ConsensusBuilder(protocol, strategy=strategy)
        result, excluded = builder.build_consensus(sections, overlaps)
        
        print(f"  Excluded: {excluded}")
        print(f"  Consensus achieved: {result.success}")
        
        # Check if we correctly identified Byzantine nodes
        actual_byzantine = {s.id for s in sections if s.metadata["type"] == "byzantine"}
        excluded_set = set(excluded)
        
        true_positives = excluded_set & actual_byzantine
        false_positives = excluded_set - actual_byzantine
        false_negatives = actual_byzantine - excluded_set
        
        print(f"  True positives (correctly excluded): {true_positives}")
        print(f"  False positives (wrongly excluded): {false_positives}")
        print(f"  False negatives (missed): {false_negatives}")


def main():
    print("=" * 60)
    print("ModalSheaf Diagnostic Analysis")
    print("=" * 60)
    print("""
    The diagnostic system answers key questions:
    
    1. WHO is causing inconsistency?
       → ContributorScore with trust/consistency metrics
    
    2. WHAT pattern of disagreement exists?
       → Cluster analysis reveals factions
    
    3. WHEN did behavior change?
       → Temporal analysis detects drift/failure
    
    4. HOW to achieve consensus?
       → ConsensusBuilder excludes bad actors
    
    Mathematical basis:
    - Each overlap is a constraint
    - Inconsistent overlaps implicate both parties
    - Pattern analysis reveals isolated bad actors vs factions
    - H¹ cohomology measures total obstruction
    """)
    
    demo_sensor_calibration()
    demo_news_sources()
    demo_faction_detection()
    demo_temporal_drift()
    demo_consensus_building()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
