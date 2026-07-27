#!/usr/bin/env python3
"""
Gluing Operations Example for ModalSheaf.

This example demonstrates the core sheaf operation: gluing local data
into global structures when the local pieces agree on overlaps.

Examples covered:
1. Panorama stitching (images)
2. Coordinate frame fusion (sensors)
3. Geographic assembly (states → country)
4. Document assembly (pages → document)

Key insight: H¹ measures the OBSTRUCTION to gluing. When H¹ = 0,
local data glues perfectly into a global section.
"""

import numpy as np
from modalsheaf.gluing import (
    LocalSection,
    Overlap,
    GluingResult,
    PanoramaGluing,
    CoordinateGluing,
    HierarchicalGluing,
    DocumentGluing,
    CodebaseGluing,
    glue_with_protocol,
)


def demo_panorama_stitching():
    """Demonstrate image panorama gluing."""
    print("\n" + "=" * 60)
    print("PANORAMA STITCHING")
    print("=" * 60)
    print("""
    Mathematical setup:
    - Cover: {left_image, center_image, right_image}
    - Overlaps: left∩center, center∩right
    - Gluing: Stitch into panorama when overlaps match
    - H¹: Measures stitching errors (parallax, exposure)
    """)
    
    # Create synthetic images with overlapping regions
    # Left image: gradient from dark to light
    left = np.zeros((100, 150, 3), dtype=np.uint8)
    for i in range(150):
        left[:, i, :] = int(i * 255 / 150)
    
    # Center image: continues the gradient
    center = np.zeros((100, 150, 3), dtype=np.uint8)
    for i in range(150):
        left_val = 100 * 255 / 150  # Where left ends (at overlap start)
        center[:, i, :] = int(left_val + i * (255 - left_val) / 150)
    
    # Right image: continues further
    right = np.zeros((100, 150, 3), dtype=np.uint8)
    for i in range(150):
        center_val = 200  # Approximate where center ends
        right[:, i, :] = min(255, int(center_val + i * 0.3))
    
    print(f"Images: left={left.shape}, center={center.shape}, right={right.shape}")
    
    # Define overlaps (x_offset of second image relative to first)
    sections = [
        {"id": "left", "data": left},
        {"id": "center", "data": center},
        {"id": "right", "data": right},
    ]
    
    overlaps = [
        {"sections": ("left", "center"), "region": (100, 0)},  # 50px overlap
        {"sections": ("center", "right"), "region": (100, 0)},
    ]
    
    # Glue with consistent data
    protocol = PanoramaGluing(blend_mode="linear", consistency_threshold=30.0)
    result = glue_with_protocol(protocol, sections, overlaps)
    
    print(f"\nResult: {result}")
    print(f"  Canvas size: {result.diagnostics.get('canvas_size')}")
    print(f"  H¹ obstruction: {result.h1_obstruction:.2f}")
    
    if result.success:
        print("  ✓ Panorama stitched successfully!")
    else:
        print("  ✗ Stitching failed - overlaps don't match")
        for err in result.consistency_errors:
            print(f"    - {err['overlap']}: error={err['error']:.1f}")
    
    # Now introduce an inconsistency
    print("\n--- Introducing exposure mismatch ---")
    
    # Make center image brighter (simulating exposure difference)
    center_bright = (center.astype(float) * 1.5).clip(0, 255).astype(np.uint8)
    
    sections_bad = [
        {"id": "left", "data": left},
        {"id": "center", "data": center_bright},  # Exposure mismatch!
        {"id": "right", "data": right},
    ]
    
    result_bad = glue_with_protocol(protocol, sections_bad, overlaps)
    
    print(f"Result with mismatch: {result_bad}")
    print(f"  H¹ obstruction: {result_bad.h1_obstruction:.2f}")
    
    if not result_bad.success:
        print("  ✗ Detected inconsistency!")
        for err in result_bad.consistency_errors:
            print(f"    - {err['overlap']}: error={err['error']:.1f} > threshold={err['threshold']}")


def demo_coordinate_fusion():
    """Demonstrate sensor coordinate frame gluing."""
    print("\n" + "=" * 60)
    print("COORDINATE FRAME FUSION")
    print("=" * 60)
    print("""
    Mathematical setup:
    - Cover: {camera_frame, lidar_frame, radar_frame}
    - Overlaps: Sensors with overlapping field of view
    - Gluing: Unified world coordinate system
    - H¹: Calibration errors, sensor drift
    """)
    
    # Simulate three sensors observing the same scene
    # Each has points in its local coordinate frame
    
    # Camera: looking forward, origin at (0, 0, 0)
    camera_points = [
        np.array([1.0, 0.0, 5.0]),   # Object 1
        np.array([0.5, 0.2, 4.0]),   # Object 2
        np.array([-0.3, -0.1, 6.0]), # Object 3
    ]
    
    # Lidar: offset by (1, 0, 0), same orientation
    # Same objects should appear shifted
    lidar_points = [
        np.array([0.0, 0.0, 5.0]),   # Object 1 (shifted by -1 in x)
        np.array([-0.5, 0.2, 4.0]),  # Object 2
        np.array([-1.3, -0.1, 6.0]), # Object 3
    ]
    
    # Radar: offset by (0, 1, 0)
    radar_points = [
        np.array([1.0, -1.0, 5.0]),  # Object 1 (shifted by -1 in y)
        np.array([0.5, -0.8, 4.0]),  # Object 2
        np.array([-0.3, -1.1, 6.0]), # Object 3
    ]
    
    sections = [
        {
            "id": "camera",
            "data": {"points": camera_points},
            "metadata": {"origin": [0, 0, 0]},
        },
        {
            "id": "lidar",
            "data": {"points": lidar_points},
            "metadata": {"origin": [1, 0, 0]},
        },
        {
            "id": "radar",
            "data": {"points": radar_points},
            "metadata": {"origin": [0, 1, 0]},
        },
    ]
    
    # Define transforms between frames
    def lidar_to_camera(p):
        return p + np.array([1, 0, 0])  # Shift back
    
    def radar_to_camera(p):
        return p + np.array([0, 1, 0])  # Shift back
    
    overlaps = [
        {
            "sections": ("camera", "lidar"),
            "region": [0, 1, 2],  # Indices of shared points
            "transform": lidar_to_camera,
        },
        {
            "sections": ("camera", "radar"),
            "region": [0, 1, 2],
            "transform": radar_to_camera,
        },
    ]
    
    protocol = CoordinateGluing(reference_frame="world")
    result = glue_with_protocol(protocol, sections, overlaps)
    
    print(f"\nResult: {result}")
    print(f"  Frames resolved: {result.diagnostics.get('frames_resolved')}")
    
    if result.success:
        print("  ✓ All sensors aligned to world frame!")
    else:
        print("  ✗ Calibration errors detected")
        for err in result.consistency_errors:
            print(f"    - {err['overlap']}: error={err['error']:.3f}m")
    
    # Introduce calibration error
    print("\n--- Introducing calibration error ---")
    
    def bad_lidar_to_camera(p):
        return p + np.array([1.1, 0.05, 0])  # Slightly wrong!
    
    overlaps_bad = [
        {
            "sections": ("camera", "lidar"),
            "region": [0, 1, 2],
            "transform": bad_lidar_to_camera,  # Wrong calibration
        },
        {
            "sections": ("camera", "radar"),
            "region": [0, 1, 2],
            "transform": radar_to_camera,
        },
    ]
    
    result_bad = glue_with_protocol(protocol, sections, overlaps_bad)
    print(f"Result with bad calibration: {result_bad}")


def demo_geographic_assembly():
    """Demonstrate hierarchical geographic gluing."""
    print("\n" + "=" * 60)
    print("GEOGRAPHIC ASSEMBLY (States → Country)")
    print("=" * 60)
    print("""
    Mathematical setup:
    - Cover: {state1, state2, state3, ...}
    - Overlaps: Shared borders
    - Gluing: Unified country boundary
    - H¹: Border disputes, misaligned boundaries
    """)
    
    # Simplified example: Three states forming a country
    sections = [
        {
            "id": "california",
            "data": {"population": 39_500_000, "capital": "Sacramento"},
            "metadata": {
                "boundary": {
                    "nevada": "shared_border_CA_NV",
                    "oregon": "shared_border_CA_OR",
                }
            },
        },
        {
            "id": "nevada",
            "data": {"population": 3_100_000, "capital": "Carson City"},
            "metadata": {
                "boundary": {
                    "california": "shared_border_CA_NV",  # Must match!
                    "oregon": "shared_border_NV_OR",
                }
            },
        },
        {
            "id": "oregon",
            "data": {"population": 4_200_000, "capital": "Salem"},
            "metadata": {
                "boundary": {
                    "california": "shared_border_CA_OR",  # Must match!
                    "nevada": "shared_border_NV_OR",      # Must match!
                }
            },
        },
    ]
    
    overlaps = [
        {"sections": ("california", "nevada"), "region": "california"},
        {"sections": ("california", "oregon"), "region": "california"},
        {"sections": ("nevada", "oregon"), "region": "nevada"},
    ]
    
    protocol = HierarchicalGluing(boundary_key="boundary")
    result = glue_with_protocol(protocol, sections, overlaps)
    
    print(f"\nResult: {result}")
    
    if result.success:
        print("  ✓ All borders align!")
        total_pop = sum(
            s["data"]["population"] 
            for s in sections
        )
        print(f"  Total population: {total_pop:,}")
    else:
        print("  ✗ Border disputes detected!")
    
    # Introduce a border dispute
    print("\n--- Introducing border dispute ---")
    
    sections_dispute = [
        {
            "id": "california",
            "data": {"population": 39_500_000},
            "metadata": {
                "boundary": {
                    "nevada": "CA_claims_this_border",  # Different!
                }
            },
        },
        {
            "id": "nevada",
            "data": {"population": 3_100_000},
            "metadata": {
                "boundary": {
                    "california": "NV_claims_this_border",  # Different!
                }
            },
        },
    ]
    
    overlaps_dispute = [
        {"sections": ("california", "nevada"), "region": "california"},
    ]
    
    result_dispute = glue_with_protocol(protocol, sections_dispute, overlaps_dispute)
    print(f"Result with dispute: {result_dispute}")
    
    if result_dispute.consistency_errors:
        for err in result_dispute.consistency_errors:
            print(f"  Border conflict: {err['boundary1']} vs {err['boundary2']}")


def demo_document_assembly():
    """Demonstrate document page gluing."""
    print("\n" + "=" * 60)
    print("DOCUMENT ASSEMBLY (Pages → Document)")
    print("=" * 60)
    print("""
    Mathematical setup:
    - Cover: {page1, page2, page3, ...}
    - Overlaps: Page transitions (end of page n → start of page n+1)
    - Gluing: Complete document
    - H¹: Broken sentences, missing pages, wrong order
    """)
    
    # Document pages with proper sentence boundaries
    sections = [
        {
            "id": "page1",
            "data": "This is the first page of our document. It introduces the main concepts.",
            "metadata": {"page_number": 1},
        },
        {
            "id": "page2",
            "data": "The second page continues the discussion. More details are provided here.",
            "metadata": {"page_number": 2},
        },
        {
            "id": "page3",
            "data": "Finally, the third page concludes with a summary.",
            "metadata": {"page_number": 3},
        },
    ]
    
    overlaps = [
        {"sections": ("page1", "page2")},
        {"sections": ("page2", "page3")},
    ]
    
    protocol = DocumentGluing(order_key="page_number")
    result = glue_with_protocol(protocol, sections, overlaps)
    
    print(f"\nResult: {result}")
    print(f"  Page order: {result.diagnostics.get('order')}")
    
    if result.success:
        print("  ✓ Document assembled successfully!")
        print(f"\n  Full document:\n  {result.global_section[:100]}...")
    
    # Introduce a broken sentence
    print("\n--- Introducing broken sentence ---")
    
    sections_broken = [
        {
            "id": "page1",
            "data": "This is the first page. The sentence continues on the",  # Broken!
            "metadata": {"page_number": 1},
        },
        {
            "id": "page2",
            "data": "next page without proper break. This is problematic.",
            "metadata": {"page_number": 2},
        },
    ]
    
    overlaps_broken = [{"sections": ("page1", "page2")}]
    
    result_broken = glue_with_protocol(protocol, sections_broken, overlaps_broken)
    print(f"Result with broken sentence: {result_broken}")
    
    if result_broken.consistency_errors:
        for err in result_broken.consistency_errors:
            print(f"  Break between {err['between']}")
            print(f"    End: '...{err['end_of_prev'][-30:]}'")
            print(f"    Start: '{err['start_of_next'][:30]}...'")


def demo_codebase_assembly():
    """Demonstrate codebase module gluing."""
    print("\n" + "=" * 60)
    print("CODEBASE ASSEMBLY (Files → Module)")
    print("=" * 60)
    print("""
    Mathematical setup:
    - Cover: {file1.py, file2.py, file3.py, ...}
    - Overlaps: Import/export relationships
    - Gluing: Unified module with resolved dependencies
    - H¹: Unresolved imports, circular dependencies
    """)
    
    sections = [
        {
            "id": "utils.py",
            "data": "def helper(): pass\ndef format_data(): pass",
            "metadata": {
                "exports": ["helper", "format_data"],
                "imports": [],
            },
        },
        {
            "id": "models.py",
            "data": "from utils import helper\nclass Model: pass",
            "metadata": {
                "exports": ["Model"],
                "imports": ["helper"],
            },
        },
        {
            "id": "main.py",
            "data": "from models import Model\nfrom utils import format_data",
            "metadata": {
                "exports": ["main"],
                "imports": ["Model", "format_data"],
            },
        },
    ]
    
    overlaps = [
        {"sections": ("utils.py", "models.py")},
        {"sections": ("utils.py", "main.py")},
        {"sections": ("models.py", "main.py")},
    ]
    
    protocol = CodebaseGluing()
    result = glue_with_protocol(protocol, sections, overlaps)
    
    print(f"\nResult: {result}")
    print(f"  Total exports: {result.diagnostics.get('total_exports')}")
    
    if result.success:
        print("  ✓ All imports resolved!")
    
    # Introduce missing import
    print("\n--- Introducing missing import ---")
    
    sections_missing = [
        {
            "id": "main.py",
            "data": "from nonexistent import something",
            "metadata": {
                "exports": [],
                "imports": ["something"],  # Not exported by anyone!
            },
        },
    ]
    
    result_missing = glue_with_protocol(protocol, sections_missing, [])
    print(f"Result with missing import: {result_missing}")
    
    if result_missing.consistency_errors:
        for err in result_missing.consistency_errors:
            print(f"  File {err['file']} missing: {err['missing_imports']}")


def main():
    print("=" * 60)
    print("ModalSheaf Gluing Operations")
    print("=" * 60)
    print("""
    THE GLUING AXIOM (Core of Sheaf Theory)
    
    Given:
    - A cover {Uᵢ} of a space U
    - Local sections sᵢ ∈ F(Uᵢ) for each piece
    - Agreement on overlaps: sᵢ|_{Uᵢ∩Uⱼ} = sⱼ|_{Uᵢ∩Uⱼ}
    
    Then:
    - There exists a UNIQUE global section s ∈ F(U)
    - Such that s|_{Uᵢ} = sᵢ for all i
    
    H¹ COHOMOLOGY measures the OBSTRUCTION to gluing:
    - H¹ = 0: Local data glues perfectly
    - H¹ ≠ 0: There's an inconsistency preventing gluing
    
    This is DIFFERENT from restriction/extension maps:
    - Restriction: F(U) → F(V), extract local from global
    - Extension: F(V) → F(U), aggregate (not always possible)
    - Gluing: {F(Uᵢ)} → F(U), assemble locals into global
    """)
    
    demo_panorama_stitching()
    demo_coordinate_fusion()
    demo_geographic_assembly()
    demo_document_assembly()
    demo_codebase_assembly()
    
    print("\n" + "=" * 60)
    print("SUMMARY: When Does Gluing Fail?")
    print("=" * 60)
    print("""
    Gluing fails (H¹ ≠ 0) when local data doesn't agree on overlaps:
    
    1. PANORAMA: Exposure/parallax differences between images
    2. SENSORS: Calibration errors, timing drift
    3. GEOGRAPHY: Border disputes, inconsistent boundaries
    4. DOCUMENTS: Broken sentences, missing pages
    5. CODEBASE: Unresolved imports, symbol conflicts
    
    In ML/AI terms, H¹ detects:
    - Hallucinations (text doesn't match image)
    - Sensor fusion errors
    - Inconsistent multimodal embeddings
    - Knowledge graph contradictions
    """)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
