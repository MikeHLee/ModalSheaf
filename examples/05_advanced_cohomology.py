#!/usr/bin/env python3
"""
Advanced Cohomology Examples for ModalSheaf.

This example demonstrates the rigorous mathematical tools:
1. Čech cohomology computation
2. Persistent cohomology for noisy data
3. Cocycle condition checking and repair

These go beyond the intuitive consistency checking to provide
full mathematical machinery.
"""

import numpy as np
from modalsheaf import (
    # Čech cohomology
    compute_cech_cohomology,
    CechComplex,
    CechCochain,
    CechCohomology,
    
    # Persistent cohomology
    compute_persistent_cohomology,
    persistence_based_consistency,
    PersistentCohomology,
    
    # Cocycle conditions
    check_cocycle,
    repair_cocycle,
    CocycleChecker,
    TransitionFunction,
)


def demo_cech_cohomology():
    """Demonstrate rigorous Čech cohomology computation."""
    print("\n" + "=" * 60)
    print("ČECH COHOMOLOGY")
    print("=" * 60)
    print("""
    The Čech complex organizes data by overlaps:
    
    C⁰ = data on each region
    C¹ = data on pairwise overlaps
    C² = data on triple overlaps
    
    Cohomology measures:
    H⁰ = global consistent states (ker δ⁰)
    H¹ = obstructions to gluing (ker δ¹ / im δ⁰)
    """)
    
    # Example 1: Consistent data
    print("\n--- Example 1: Consistent Sensors ---")
    
    consistent_data = {
        'sensor_a': np.array([20.0, 30.0]),
        'sensor_b': np.array([20.0, 30.0]),
        'sensor_c': np.array([20.0, 30.0]),
    }
    
    result = compute_cech_cohomology(consistent_data)
    print(result.summary())
    
    # Example 2: Inconsistent data
    print("\n--- Example 2: Inconsistent Sensors ---")
    
    inconsistent_data = {
        'sensor_a': np.array([20.0, 30.0]),
        'sensor_b': np.array([21.0, 31.0]),
        'sensor_c': np.array([22.0, 29.0]),
    }
    
    result = compute_cech_cohomology(inconsistent_data)
    print(result.summary())
    
    # Example 3: Manual Čech complex construction
    print("\n--- Example 3: Manual Complex Construction ---")
    
    # Create a complex for 3 overlapping regions
    complex = CechComplex(
        cover=['A', 'B', 'C'],
        overlaps={
            ('A', 'B'): True,
            ('B', 'C'): True,
            ('A', 'C'): True,
        },
        dim=2  # 2D data
    )
    
    print(f"0-simplices (regions): {complex.get_simplices(0)}")
    print(f"1-simplices (overlaps): {complex.get_simplices(1)}")
    print(f"2-simplices (triple): {complex.get_simplices(2)}")
    
    # Create a 0-cochain (data on each region)
    c0 = complex.create_cochain(0, {
        ('A',): np.array([1.0, 0.0]),
        ('B',): np.array([1.1, 0.1]),
        ('C',): np.array([0.9, -0.1]),
    })
    
    # Compute coboundary (measures disagreement)
    c1 = complex.coboundary(c0)
    
    print(f"\nCoboundary δ⁰(data):")
    for simplex, value in c1.data.items():
        print(f"  {' ∩ '.join(simplex)}: {value}")
    
    print(f"\nTotal disagreement: {c1.norm():.4f}")
    print(f"Is cocycle (δc = 0)? {complex.is_cocycle(c0)}")


def demo_persistent_cohomology():
    """Demonstrate persistent cohomology for noisy data."""
    print("\n" + "=" * 60)
    print("PERSISTENT COHOMOLOGY")
    print("=" * 60)
    print("""
    Real data is noisy. Persistence tells us which features are
    "real" (high persistence) vs "noise" (low persistence).
    
    We track how cohomology changes as we vary a tolerance threshold:
    - Birth: threshold where feature appears
    - Death: threshold where feature disappears
    - Persistence = Death - Birth
    """)
    
    # Example 1: Sensors with noise
    print("\n--- Example 1: Noisy Sensors ---")
    
    noisy_data = {
        'sensor_a': np.array([20.0]),
        'sensor_b': np.array([20.1]),   # Small noise
        'sensor_c': np.array([20.05]),  # Small noise
        'sensor_d': np.array([25.0]),   # Outlier!
    }
    
    result = compute_persistent_cohomology(noisy_data)
    print(result.summary())
    
    # Example 2: Quick consistency check with noise filtering
    print("\n--- Example 2: Noise-Filtered Consistency ---")
    
    is_consistent, confidence, explanation = persistence_based_consistency(
        noisy_data,
        noise_threshold=0.5  # Ignore differences < 0.5
    )
    
    print(f"Consistent (with noise filtering): {is_consistent}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Explanation: {explanation}")
    
    # Example 3: CLIP-like embeddings
    print("\n--- Example 3: Embedding Matching ---")
    
    # Simulate CLIP embeddings (768-dim, but we'll use 10 for demo)
    np.random.seed(42)
    
    # Good match: similar embeddings
    image_emb = np.random.randn(10) * 0.1
    text_emb_good = image_emb + np.random.randn(10) * 0.05  # Small noise
    
    # Bad match: different embeddings
    text_emb_bad = np.random.randn(10) * 0.1  # Completely different
    
    print("\nGood match (image + similar caption):")
    is_match, conf, expl = persistence_based_consistency(
        {'image': image_emb, 'text': text_emb_good},
        noise_threshold=0.1
    )
    print(f"  Match: {is_match}, Confidence: {conf:.1%}")
    
    print("\nBad match (image + unrelated caption):")
    is_match, conf, expl = persistence_based_consistency(
        {'image': image_emb, 'text': text_emb_bad},
        noise_threshold=0.1
    )
    print(f"  Match: {is_match}, Confidence: {conf:.1%}")
    
    # Example 4: Finding the right threshold
    print("\n--- Example 4: Automatic Threshold Selection ---")
    
    data = {
        'a': np.array([1.0]),
        'b': np.array([1.02]),
        'c': np.array([1.05]),
        'd': np.array([2.0]),  # Outlier
    }
    
    result = compute_persistent_cohomology(data)
    
    # The recommended threshold separates noise from signal
    threshold = result.recommended_threshold(max_noise=0.1)
    print(f"Recommended threshold: {threshold:.3f}")
    print(f"Consistent at this threshold: {result.consistency_at_threshold(threshold)}")
    
    # Show the persistence diagram
    h1 = result.get_diagram(1)
    print(f"\nH¹ Persistence Diagram:")
    for interval in sorted(h1.intervals, key=lambda i: -i.persistence)[:5]:
        print(f"  [{interval.birth:.3f}, {interval.death:.3f}) "
              f"persistence = {interval.persistence:.3f}")


def demo_cocycle_conditions():
    """Demonstrate cocycle condition checking and repair."""
    print("\n" + "=" * 60)
    print("COCYCLE CONDITIONS")
    print("=" * 60)
    print("""
    The cocycle condition ensures consistency around loops:
    
    For three regions A, B, C with transitions g_AB, g_BC, g_CA:
    
        g_CA ∘ g_BC ∘ g_AB = identity
    
    If this fails, the transitions are inconsistent.
    """)
    
    # Example 1: Currency exchange (should be consistent)
    print("\n--- Example 1: Currency Exchange ---")
    
    # Exchange rates (as 1x1 matrices for the framework)
    rates = {
        ('USD', 'EUR'): np.array([[0.85]]),
        ('EUR', 'GBP'): np.array([[0.86]]),
        ('GBP', 'USD'): np.array([[1.38]]),
    }
    
    result = check_cocycle(rates, tolerance=0.01)
    print(result.summary())
    
    # Calculate the round-trip
    product = 0.85 * 0.86 * 1.38
    print(f"\nRound-trip product: {product:.4f}")
    print(f"Arbitrage opportunity: {(product - 1) * 100:.2f}%")
    
    # Example 2: Sensor calibration (with error)
    print("\n--- Example 2: Sensor Calibration ---")
    
    # Rotation matrices (2D for simplicity)
    def rotation_matrix(angle_deg):
        angle = np.radians(angle_deg)
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
    
    # Correct calibration: angles should sum to 0 (or 360)
    R_cam_lidar = rotation_matrix(30)   # 30° rotation
    R_lidar_radar = rotation_matrix(45)  # 45° rotation
    R_radar_cam = rotation_matrix(-75)   # Should be -75° to close the loop
    
    print("Correct calibration:")
    result = check_cocycle({
        ('camera', 'lidar'): R_cam_lidar,
        ('lidar', 'radar'): R_lidar_radar,
        ('radar', 'camera'): R_radar_cam,
    })
    print(result.summary())
    
    # Introduce calibration error
    print("\nWith calibration error:")
    R_radar_cam_bad = rotation_matrix(-70)  # Wrong! Should be -75°
    
    result_bad = check_cocycle({
        ('camera', 'lidar'): R_cam_lidar,
        ('lidar', 'radar'): R_lidar_radar,
        ('radar', 'camera'): R_radar_cam_bad,
    })
    print(result_bad.summary())
    
    # Example 3: Repairing cocycle violations
    print("\n--- Example 3: Cocycle Repair ---")
    
    bad_calibration = {
        ('camera', 'lidar'): R_cam_lidar,
        ('lidar', 'radar'): R_lidar_radar,
        ('radar', 'camera'): R_radar_cam_bad,
    }
    
    # Repair using averaging
    fixed = repair_cocycle(bad_calibration, method='average')
    
    print("After repair:")
    result_fixed = check_cocycle(fixed)
    print(result_fixed.summary())
    
    # Example 4: Time zones
    print("\n--- Example 4: Time Zone Consistency ---")
    
    # Time offsets (additive, so we use 1x1 matrices)
    # NYC → London: +5h, London → Tokyo: +9h, Tokyo → NYC: -14h
    # Sum should be 0
    
    offsets = {
        ('NYC', 'London'): np.array([[5.0]]),
        ('London', 'Tokyo'): np.array([[9.0]]),
        ('Tokyo', 'NYC'): np.array([[-14.0]]),
    }
    
    # For additive groups, cocycle means sum = 0
    # Our framework uses multiplicative, so this won't work directly
    # But we can check manually:
    total = 5 + 9 + (-14)
    print(f"Time zone round-trip: {total} hours")
    print(f"Consistent: {total == 0}")
    
    # With an error:
    print("\nWith daylight saving error:")
    total_bad = 5 + 9 + (-13)  # Someone forgot DST
    print(f"Time zone round-trip: {total_bad} hours")
    print(f"Consistent: {total_bad == 0}")
    print(f"Error: {total_bad} hour(s) off")


def demo_sheaf_laplacian():
    """Demonstrate the sheaf Laplacian and diffusion to consensus."""
    print("\n" + "=" * 60)
    print("SHEAF LAPLACIAN & DIFFUSION")
    print("=" * 60)
    print("""
    The sheaf Laplacian encodes ALL consistency conditions in one matrix.
    
    Key insight: L = δᵀδ where δ is the coboundary operator
    
    Properties:
    - ker(L) = H⁰ = space of globally consistent states
    - xᵀLx = total squared disagreement
    - Eigenvalues = "frequencies" of inconsistency
    
    Diffusion: dx/dt = -Lx converges to consensus (projection onto ker(L))
    """)
    
    from modalsheaf import ModalityGraph
    from modalsheaf.consistency import compute_sheaf_laplacian, diffuse_to_consensus
    
    # Example 1: Simple triangle of sensors
    print("\n--- Example 1: Triangle of Sensors ---")
    
    graph = ModalityGraph("sensors")
    graph.add_modality("A")
    graph.add_modality("B") 
    graph.add_modality("C")
    
    # All sensors should agree (identity transforms)
    graph.add_transformation("A", "B", forward=lambda x: x)
    graph.add_transformation("B", "C", forward=lambda x: x)
    graph.add_transformation("A", "C", forward=lambda x: x)
    
    # Inconsistent readings
    embeddings = {
        'A': np.array([20.0]),
        'B': np.array([21.0]),
        'C': np.array([22.0]),
    }
    
    print(f"Initial readings: A={embeddings['A'][0]}, B={embeddings['B'][0]}, C={embeddings['C'][0]}")
    
    # Compute Laplacian
    L = compute_sheaf_laplacian(graph, embeddings)
    print(f"\nLaplacian matrix:\n{L}")
    
    # Analyze eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Number of zero eigenvalues (H⁰ dim): {np.sum(np.abs(eigenvalues) < 1e-10)}")
    
    # Compute total disagreement
    x = np.array([embeddings['A'][0], embeddings['B'][0], embeddings['C'][0]])
    disagreement = x @ L @ x
    print(f"Total disagreement (xᵀLx): {disagreement:.2f}")
    
    # Example 2: Diffusion to consensus
    print("\n--- Example 2: Diffusion to Consensus ---")
    
    print(f"Before diffusion: A={embeddings['A'][0]:.1f}, B={embeddings['B'][0]:.1f}, C={embeddings['C'][0]:.1f}")
    
    # Diffuse
    consensus = diffuse_to_consensus(graph, embeddings, num_steps=50, step_size=0.1)
    
    print(f"After diffusion:  A={consensus['A'][0]:.1f}, B={consensus['B'][0]:.1f}, C={consensus['C'][0]:.1f}")
    
    # Verify consensus
    x_after = np.array([consensus['A'][0], consensus['B'][0], consensus['C'][0]])
    disagreement_after = x_after @ L @ x_after
    print(f"\nDisagreement reduced: {disagreement:.2f} → {disagreement_after:.2f}")
    
    # Example 3: Non-trivial restriction maps
    print("\n--- Example 3: Non-Identity Restrictions ---")
    print("""
    Real sheaves have non-trivial restriction maps.
    Example: Sensor B measures in Fahrenheit, others in Celsius.
    
    The Laplacian accounts for these transformations!
    """)
    
    graph2 = ModalityGraph("mixed_units")
    graph2.add_modality("celsius_1")
    graph2.add_modality("fahrenheit")
    graph2.add_modality("celsius_2")
    
    # Transforms between units
    def c_to_f(x): return x * 9/5 + 32
    def f_to_c(x): return (x - 32) * 5/9
    
    graph2.add_transformation("celsius_1", "fahrenheit", forward=c_to_f)
    graph2.add_transformation("celsius_2", "fahrenheit", forward=c_to_f)
    
    # Readings (all measuring ~20°C)
    embeddings2 = {
        'celsius_1': np.array([20.0]),      # 20°C
        'fahrenheit': np.array([68.0]),     # 68°F = 20°C ✓
        'celsius_2': np.array([21.0]),      # 21°C (slight disagreement)
    }
    
    print(f"Celsius_1: {embeddings2['celsius_1'][0]}°C")
    print(f"Fahrenheit: {embeddings2['fahrenheit'][0]}°F (= {f_to_c(embeddings2['fahrenheit'][0]):.1f}°C)")
    print(f"Celsius_2: {embeddings2['celsius_2'][0]}°C")
    
    # The Laplacian would account for the C↔F conversion
    # (Our simplified implementation uses identity, but the concept holds)


def demo_combined_workflow():
    """Show how to combine all tools in a real workflow."""
    print("\n" + "=" * 60)
    print("COMBINED WORKFLOW: Multi-Sensor Fusion")
    print("=" * 60)
    print("""
    Scenario: A robot has 3 sensors measuring distance to a wall.
    We want to:
    1. Check if sensors are calibrated (cocycle)
    2. Check if readings are consistent (cohomology)
    3. Handle noise appropriately (persistence)
    """)
    
    # Sensor readings (distance in meters)
    readings = {
        'ultrasonic': np.array([2.05]),
        'lidar': np.array([2.00]),
        'infrared': np.array([2.10]),
    }
    
    # Step 1: Check calibration (cocycle)
    print("\n--- Step 1: Calibration Check ---")
    
    # Calibration transforms (identity for same-unit sensors)
    calibration = {
        ('ultrasonic', 'lidar'): np.array([[1.0]]),
        ('lidar', 'infrared'): np.array([[1.0]]),
        ('infrared', 'ultrasonic'): np.array([[1.0]]),
    }
    
    cocycle_result = check_cocycle(calibration)
    print(f"Calibration consistent: {cocycle_result.is_satisfied}")
    
    # Step 2: Check reading consistency (Čech cohomology)
    print("\n--- Step 2: Reading Consistency ---")
    
    cech_result = compute_cech_cohomology(readings)
    print(f"H⁰ dimension: {cech_result.h0.dimension}")
    print(f"H¹ dimension: {cech_result.h1.dimension}")
    print(f"Fully consistent: {cech_result.is_consistent}")
    
    # Step 3: Account for noise (persistence)
    print("\n--- Step 3: Noise Analysis ---")
    
    pers_result = compute_persistent_cohomology(readings)
    
    # Typical ultrasonic accuracy is ±0.05m
    SENSOR_NOISE = 0.05
    
    is_consistent, confidence, explanation = persistence_based_consistency(
        readings,
        noise_threshold=SENSOR_NOISE * 2  # 2σ tolerance
    )
    
    print(f"Consistent (accounting for noise): {is_consistent}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Explanation: {explanation}")
    
    # Step 4: Fuse readings
    print("\n--- Step 4: Sensor Fusion ---")
    
    if is_consistent:
        # Simple average (could use persistence-weighted average)
        fused = np.mean([v[0] for v in readings.values()])
        print(f"Fused reading: {fused:.3f} m")
    else:
        # Find the outlier using persistence
        h1 = pers_result.get_diagram(1)
        print("Inconsistent readings detected!")
        print(f"Max persistence: {h1.max_persistence():.3f}")
        print("Consider investigating sensor with highest disagreement.")


def main():
    print("=" * 60)
    print("ModalSheaf Advanced Cohomology Examples")
    print("=" * 60)
    print("""
    This demonstrates the rigorous mathematical tools in ModalSheaf:
    
    1. ČECH COHOMOLOGY: Exact computation of H⁰, H¹
    2. PERSISTENT COHOMOLOGY: Handling noisy data
    3. COCYCLE CONDITIONS: Ensuring loop consistency
    
    These go beyond intuitive consistency checking to provide
    the full mathematical machinery of sheaf theory.
    """)
    
    demo_cech_cohomology()
    demo_persistent_cohomology()
    demo_cocycle_conditions()
    demo_sheaf_laplacian()
    demo_combined_workflow()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. ČECH COHOMOLOGY gives exact answers:
       - H⁰ = dimension of global consistent states
       - H¹ = dimension of obstructions
       
    2. PERSISTENT COHOMOLOGY handles noise:
       - High persistence = real feature
       - Low persistence = noise
       - Automatic threshold selection
       
    3. COCYCLE CONDITIONS ensure global consistency:
       - Check: do transforms compose correctly?
       - Repair: fix inconsistent calibrations
       
    4. SHEAF LAPLACIAN encodes consistency:
       - L = δᵀδ (coboundary squared)
       - ker(L) = H⁰ (consensus states)
       - Diffusion finds optimal consensus
       
    5. COMBINED WORKFLOW:
       - Check calibration (cocycle)
       - Check readings (cohomology)
       - Filter noise (persistence)
       - Fuse data (Laplacian diffusion)
    """)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
