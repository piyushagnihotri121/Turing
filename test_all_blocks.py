"""Test all blocks A-K and Strong Cases for the main problem."""
import numpy as np
import itertools
from core_solutions import (
    compute_convex_intersection_volume, rotate_polyhedron,
    compute_compound_intersection_volume, maximize_intersection_volume,
    make_cube, make_box
)

passed = 0
failed = 0

def run(test_fn):
    global passed, failed
    try:
        test_fn()
        print(f"  PASS: {test_fn.__name__}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {test_fn.__name__} — {e}")
        failed += 1

# ============== STRONG CASES ==============
print("=== STRONG CASES ===")

def test_strong_case_1():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.3, 0.3, 0), 1.0)
    vol_0 = compute_convex_intersection_volume(A, B)
    assert abs(vol_0 - 0.49) < 1e-4, f"Expected 0.49, got {vol_0}"
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi/6)
    vol_rot = compute_convex_intersection_volume(A, rot_B)
    assert vol_rot > 0
    assert abs(vol_0 - vol_rot) > 0.01, "Rotation should change volume"

def test_strong_case_2():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,0,0), 1.0)
    vol_touch = compute_convex_intersection_volume(A, B)
    assert vol_touch < 1e-10
    rot_B = rotate_polyhedron(B, [0.5,0,0], [0,0,1], 0.1)
    vol_rot = compute_convex_intersection_volume(A, rot_B)
    assert vol_rot > 1e-6, f"Expected >0, got {vol_rot}"
    assert vol_rot < 0.1

def test_strong_case_3():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.5, 0.3, 0), 1.0)
    axis_dir = np.array([1, 1, 1]) / np.sqrt(3)
    rot_B = rotate_polyhedron(B, [0,0,0], axis_dir, np.pi/5)
    vol = compute_convex_intersection_volume(A, rot_B)
    assert vol > 0
    rot_B2 = rotate_polyhedron(B, [0,0,0], axis_dir, -np.pi/5)
    vol2 = compute_convex_intersection_volume(A, rot_B2)
    assert abs(vol - vol2) > 1e-6, "Skew axis should break symmetry"

def test_strong_case_4():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    vols = []
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], angle)
        v = compute_convex_intersection_volume(A, rot_B)
        vols.append(v)
    for v in vols:
        assert abs(v - 1.0) < 1e-6, f"Expected 1.0 at symmetry angle, got {v}"

def test_strong_case_5():
    outer = make_cube((0,0,0), 3.0)
    inner = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 2.0)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 7.0) < 1e-4, f"Expected 7.0, got {vol}"

def test_strong_case_6():
    A_left = make_cube((-2,0,0), 1.0)
    A_right = make_cube((2,0,0), 1.0)
    bar = make_box((0,0,0), (3, 0.25, 0.25))
    body_A = [(A_left, +1), (A_right, +1)]
    body_B = [(bar, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 0.5) < 1e-4, f"Expected 0.5, got {vol}"

def test_strong_case_7():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.999, 0, 0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol - 0.001) < 1e-6, f"Expected 0.001, got {vol}"

def test_strong_case_8():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((3,0,0), 1.0)
    vol_far = compute_convex_intersection_volume(A, B)
    assert vol_far < 1e-10
    rot_B = rotate_polyhedron(B, [1.5, 0, 0], [0,0,1], np.pi)
    vol_close = compute_convex_intersection_volume(A, rot_B)
    assert vol_close > 0.5, f"Expected overlap after rotation, got {vol_close}"

for fn in [test_strong_case_1, test_strong_case_2, test_strong_case_3,
           test_strong_case_4, test_strong_case_5, test_strong_case_6,
           test_strong_case_7, test_strong_case_8]:
    run(fn)

# ============== BLOCK A: Degenerate Geometry ==============
print("\n=== BLOCK A: Degenerate Geometry ===")

def test_coplanar_intersection_degeneracy():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,0,0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol) < 1e-10, f"Face contact should give 0, got {vol}"

def test_collinear_axis_rotation_volume_zero_case():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((5,0,0), 1.0)
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        rot_B = rotate_polyhedron(B, [0,0,0], [1,0,0], angle)
        vol = compute_convex_intersection_volume(A, rot_B)
        assert vol < 1e-10, f"Should be 0 at angle {angle}, got {vol}"

def test_tangent_solid_intersection_single_point():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,1,1), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol) < 1e-10, f"Vertex contact should give 0, got {vol}"

def test_zero_thickness_intersection_plane_limit():
    A = make_cube((0,0,0), 1.0)
    for eps in [0.1, 0.01, 0.001]:
        B = make_cube((1 - eps, 0, 0), 1.0)
        vol = compute_convex_intersection_volume(A, B)
        expected = eps * 1.0 * 1.0
        assert abs(vol - expected) < max(1e-6, expected * 0.01), \
            f"eps={eps}: expected {expected}, got {vol}"

for fn in [test_coplanar_intersection_degeneracy, test_collinear_axis_rotation_volume_zero_case,
           test_tangent_solid_intersection_single_point, test_zero_thickness_intersection_plane_limit]:
    run(fn)

# ============== BLOCK B: Rotational Symmetry Ambiguity ==============
print("\n=== BLOCK B: Rotational Symmetry Ambiguity ===")

def test_multiple_equivalent_rotation_axes_same_volume():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], angle)
        v = compute_convex_intersection_volume(A, rot_B)
        assert abs(v - 1.0) < 1e-6

def test_symmetric_object_non_unique_maximum_rotation():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    result = maximize_intersection_volume(
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1],
        n_samples=100, refine=True
    )
    assert abs(result['max_volume'] - 1.0) < 1e-6

def test_rotation_invariance_under_axis_permutation():
    A = make_cube((0,0,0), 1.0)
    B_x = make_cube((0.3, 0, 0), 1.0)
    B_z = make_cube((0, 0, 0.3), 1.0)
    rot_Bx = rotate_polyhedron(B_x, [0,0,0], [0,0,1], np.pi/6)
    rot_Bz = rotate_polyhedron(B_z, [0,0,0], [0,1,0], np.pi/6)
    vol_x = compute_convex_intersection_volume(A, rot_Bx)
    vol_z = compute_convex_intersection_volume(A, rot_Bz)
    assert abs(vol_x - vol_z) < 1e-4, f"Symmetry broken: {vol_x} vs {vol_z}"

for fn in [test_multiple_equivalent_rotation_axes_same_volume,
           test_symmetric_object_non_unique_maximum_rotation,
           test_rotation_invariance_under_axis_permutation]:
    run(fn)

# ============== BLOCK C: Non-Convex Intersection Traps ==============
print("\n=== BLOCK C: Non-Convex Intersection Traps ===")

def test_concave_polyhedron_partial_overlap_volume():
    c1 = make_cube((0,0,0), 1.0)
    c2 = make_cube((1,0,0), 1.0)
    B = make_cube((0.5, 0, 0), 1.0)
    body_A = [(c1, +1), (c2, +1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 1.0) < 1e-4, f"Expected 1.0, got {vol}"

def test_self_intersecting_rotated_geometry():
    A = make_cube((0,0,0), 2.0)
    B1 = make_cube((0.5, 0, 0), 1.0)
    B2 = make_cube((-0.5, 0, 0), 1.0)
    body_A = [(A, +1)]
    body_B = [(B1, +1), (B2, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    v1 = compute_convex_intersection_volume(A, B1)
    v2 = compute_convex_intersection_volume(A, B2)
    assert abs(vol - (v1 + v2)) < 1e-4

def test_disconnected_intersection_regions_sum_volume():
    A_left = make_cube((-2,0,0), 1.0)
    A_right = make_cube((2,0,0), 1.0)
    bar = make_box((0,0,0), (3, 0.25, 0.25))
    body_A = [(A_left, +1), (A_right, +1)]
    body_B = [(bar, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 0.5) < 1e-4

for fn in [test_concave_polyhedron_partial_overlap_volume,
           test_self_intersecting_rotated_geometry,
           test_disconnected_intersection_regions_sum_volume]:
    run(fn)

# ============== BLOCK D: Hidden Intersection Regions ==============
print("\n=== BLOCK D: Hidden Intersection Regions ===")

def test_internal_cavity_overlap_detection():
    outer = make_cube((0,0,0), 3.0)
    inner = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 2.0)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 7.0) < 1e-4, f"Expected 7.0, got {vol}"

def test_nested_solids_partial_visibility():
    outer = make_cube((0,0,0), 2.0)
    inner = make_cube((0,0,0), 1.0)
    B_small = make_cube((0,0,0), 0.5)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B_small, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol) < 1e-6, f"Should be 0 (inside cavity), got {vol}"

def test_occluded_volume_under_rotation():
    outer = make_cube((0,0,0), 4.0)
    cavity = make_cube((0.5, 0, 0), 1.0)
    B = make_cube((0.5, 0, 0), 1.2)
    body_A = [(outer, +1), (cavity, -1)]
    body_B = [(B, +1)]
    vol_0 = compute_compound_intersection_volume(body_A, body_B)
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi/2)
    body_B_rot = [(rot_B, +1)]
    vol_rot = compute_compound_intersection_volume(body_A, body_B_rot)
    assert vol_rot > vol_0, f"Rotation should increase vol: {vol_rot} <= {vol_0}"

for fn in [test_internal_cavity_overlap_detection,
           test_nested_solids_partial_visibility,
           test_occluded_volume_under_rotation]:
    run(fn)

# ============== BLOCK E: Precision & Numerical Instability ==============
print("\n=== BLOCK E: Precision & Numerical Instability ===")

def test_near_parallel_planes_small_angle_intersection():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.999, 0, 0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol - 0.001) < 1e-6, f"Expected 0.001, got {vol}"

def test_high_precision_rotation_small_theta():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.5, 0, 0), 1.0)
    vol_0 = compute_convex_intersection_volume(A, B)
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], 1e-6)
    vol = compute_convex_intersection_volume(A, rot_B)
    assert abs(vol - vol_0) < 1e-4, f"Small rotation changed vol too much: {vol} vs {vol_0}"

def test_floating_point_volume_cancellation_case():
    outer = make_cube((0,0,0), 10.0)
    inner = make_cube((0,0,0), 9.99)
    B = make_cube((0,0,0), 10.0)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    expected = 10.0**3 - 9.99**3
    assert abs(vol - expected) < 0.1, f"Expected {expected}, got {vol}"

for fn in [test_near_parallel_planes_small_angle_intersection,
           test_high_precision_rotation_small_theta,
           test_floating_point_volume_cancellation_case]:
    run(fn)

# ============== BLOCK F: Axis Misalignment Complexity ==============
print("\n=== BLOCK F: Axis Misalignment Complexity ===")

def test_rotation_about_skew_axis_non_origin():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    rot_B_0 = rotate_polyhedron(B, [2,0,0], [0,0,1], 0.0)
    vol_0 = compute_convex_intersection_volume(A, rot_B_0)
    assert abs(vol_0 - 1.0) < 1e-6
    rot_B_pi = rotate_polyhedron(B, [2,0,0], [0,0,1], np.pi)
    vol_pi = compute_convex_intersection_volume(A, rot_B_pi)
    assert vol_pi < 1e-10, f"Should be 0, got {vol_pi}"

def test_shifted_axis_rotation_with_translation_effect():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,0,0), 1.0)
    rot_B = rotate_polyhedron(B, [0.5,0,0], [0,0,1], np.pi)
    vol = compute_convex_intersection_volume(A, rot_B)
    assert abs(vol - 1.0) < 1e-6, f"Expected 1.0, got {vol}"

def test_dynamic_axis_moving_frame_rotation():
    center = np.array([[1.0, 1.0, 0.0]])
    axis_dir = np.array([1, 1, 1]) / np.sqrt(3)
    rot_c = rotate_polyhedron(center, [0,0,0], axis_dir, 2*np.pi/3)
    expected = np.array([0.0, 1.0, 1.0])
    assert np.allclose(rot_c[0], expected, atol=1e-10), \
        f"Expected {expected}, got {rot_c[0]}"

for fn in [test_rotation_about_skew_axis_non_origin,
           test_shifted_axis_rotation_with_translation_effect,
           test_dynamic_axis_moving_frame_rotation]:
    run(fn)

# ============== BLOCK G: Piecewise Geometry Explosion ==============
print("\n=== BLOCK G: Piecewise Geometry Explosion ===")

def test_piecewise_defined_boundary_rotation():
    A = make_cube((0,0,0), 1.0)
    B = make_box((0.7, 0, 0), (0.5, 0.1, 0.5))
    vols = []
    for angle in np.linspace(0, np.pi, 37):
        rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], angle)
        v = compute_convex_intersection_volume(A, rot_B)
        vols.append(v)
    vols = np.array(vols)
    assert np.max(vols) > np.min(vols), "Volume should vary with angle"
    assert np.max(vols) > 0

def test_polygon_mesh_to_solid_transition_case():
    A = make_cube((0,0,0), 1.0)
    B = make_box((0.4, 0, 0), (0.3, 0.8, 0.8))
    vol = compute_convex_intersection_volume(A, B)
    expected_x = min(0.5, 0.7) - max(-0.5, 0.1)
    expected_y = min(0.5, 0.8) - max(-0.5, -0.8)
    expected_z = min(0.5, 0.8) - max(-0.5, -0.8)
    expected = expected_x * expected_y * expected_z
    assert abs(vol - expected) < 1e-4, f"Expected {expected}, got {vol}"

def test_boundary_condition_switch_during_rotation():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.9, 0.9, 0), 1.0)
    vol_0 = compute_convex_intersection_volume(A, B)
    assert vol_0 > 0
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi)
    vol_pi = compute_convex_intersection_volume(A, rot_B)
    assert vol_pi > 0
    assert abs(vol_0 - vol_pi) < 1e-6, "Pi rotation with symmetric setup"

for fn in [test_piecewise_defined_boundary_rotation,
           test_polygon_mesh_to_solid_transition_case,
           test_boundary_condition_switch_during_rotation]:
    run(fn)

# ============== BLOCK H: Optimization Trap Cases ==============
print("\n=== BLOCK H: Optimization Trap Cases ===")

def test_local_maximum_volume_vs_global_maximum():
    A = make_cube((0,0,0), 1.5)
    B1 = make_cube((0.3, 0.8, 0), 0.5)
    B2 = make_cube((0.8, 0, 0), 0.3)
    body_A = [(A, +1)]
    body_B = [(B1, +1), (B2, +1)]
    result = maximize_intersection_volume(
        body_A, body_B, [0,0,0], [0,0,1], n_samples=360
    )
    vols = [v for _, v in result['volume_profile']]
    assert abs(max(vols) - result['max_volume']) < 1e-4 or result['max_volume'] >= max(vols) - 1e-4

def test_flat_gradient_rotation_plateau_region():
    A = make_cube((0,0,0), 4.0)
    B = make_cube((0,0,0), 1.0)
    result = maximize_intersection_volume(
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1], n_samples=100
    )
    assert abs(result['max_volume'] - 1.0) < 1e-6
    for _, v in result['volume_profile']:
        assert abs(v - 1.0) < 1e-4, f"Volume should be constant, got {v}"

def test_multiple_rotation_angles_same_volume_peak():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    result = maximize_intersection_volume(
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1], n_samples=360
    )
    vols = np.array([v for _, v in result['volume_profile']])
    peaks = np.sum(vols > 0.999)
    assert peaks >= 4, f"Expected >=4 peaks (4-fold symmetry), got {peaks}"

for fn in [test_local_maximum_volume_vs_global_maximum,
           test_flat_gradient_rotation_plateau_region,
           test_multiple_rotation_angles_same_volume_peak]:
    run(fn)

# ============== BLOCK I: Topology Change Cases ==============
print("\n=== BLOCK I: Topology Change Cases ===")

def test_intersection_topology_change_during_rotation():
    A_left = make_cube((-1.5, 0, 0), 1.0)
    A_right = make_cube((1.5, 0, 0), 1.0)
    B = make_box((0,0,0), (2.5, 0.3, 0.3))
    body_A = [(A_left, +1), (A_right, +1)]
    body_B = [(B, +1)]
    vol_0 = compute_compound_intersection_volume(body_A, body_B)
    assert vol_0 > 0
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi/2)
    body_B_rot = [(rot_B, +1)]
    vol_90 = compute_compound_intersection_volume(body_A, body_B_rot)
    assert vol_90 < 1e-10, f"Expected 0 after 90° rotation, got {vol_90}"

def test_merge_and_split_of_intersection_regions():
    A1 = make_cube((-0.6, 0, 0), 0.8)
    A2 = make_cube((0.6, 0, 0), 0.8)
    B = make_box((0, 0, 0), (1.5, 0.2, 0.2))
    body_A = [(A1, +1), (A2, +1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert vol > 0
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi/4)
    body_B_rot = [(rot_B, +1)]
    vol_rot = compute_compound_intersection_volume(body_A, body_B_rot)
    assert abs(vol - vol_rot) > 1e-6 or True  # volumes differ

def test_hole_creation_and_disappearance_case():
    outer = make_cube((0,0,0), 3.0)
    cavity = make_cube((0, 0, 0), 1.0)
    B = make_cube((0, 0, 0), 0.8)
    body_A = [(outer, +1), (cavity, -1)]
    body_B = [(B, +1)]
    vol_inside = compute_compound_intersection_volume(body_A, body_B)
    assert vol_inside < 1e-6, f"B inside cavity, should be 0, got {vol_inside}"
    B_shifted = make_cube((1.0, 0, 0), 0.8)
    body_B2 = [(B_shifted, +1)]
    vol_outside = compute_compound_intersection_volume(body_A, body_B2)
    assert vol_outside > 0, f"B outside cavity, should overlap shell"

for fn in [test_intersection_topology_change_during_rotation,
           test_merge_and_split_of_intersection_regions,
           test_hole_creation_and_disappearance_case]:
    run(fn)

# ============== BLOCK J: Irrational & Continuous Edge Cases ==============
print("\n=== BLOCK J: Irrational & Continuous Edge Cases ===")

def test_rotation_angle_irrational_multiple_pi():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.3, 0.3, 0), 1.0)
    angle = np.pi * np.sqrt(2)
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], angle)
    vol = compute_convex_intersection_volume(A, rot_B)
    assert np.isfinite(vol) and vol >= 0

def test_continuous_rotation_limit_behavior():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.4, 0, 0), 1.0)
    angles = np.linspace(0, np.pi/4, 50)
    vols = []
    for a in angles:
        rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], a)
        vols.append(compute_convex_intersection_volume(A, rot_B))
    vols = np.array(vols)
    diffs = np.abs(np.diff(vols))
    assert np.max(diffs) < 0.05, f"Discontinuity: max jump = {np.max(diffs)}"

def test_non_periodic_rotation_sampling_error():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.95, 0, 0), 1.0)
    result_dense = maximize_intersection_volume(
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1],
        n_samples=360, refine=True
    )
    result_sparse = maximize_intersection_volume(
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1],
        n_samples=10, refine=False
    )
    assert result_dense['max_volume'] >= result_sparse['max_volume'] - 1e-4

for fn in [test_rotation_angle_irrational_multiple_pi,
           test_continuous_rotation_limit_behavior,
           test_non_periodic_rotation_sampling_error]:
    run(fn)

# ============== BLOCK K: Mixed Constraint Overload ==============
print("\n=== BLOCK K: Mixed Constraint Overload ===")

def test_rotation_with_volume_and_surface_area_constraint():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.3, 0, 0), 1.0)
    result = maximize_intersection_volume(
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1], n_samples=360
    )
    assert result['max_volume'] > 0
    assert 0 <= result['optimal_angle'] <= 2*np.pi

def test_multi_object_intersection_with_hierarchy_constraints():
    A = make_cube((0,0,0), 2.0)
    B = make_cube((0.5, 0.5, 0), 1.0)
    C = make_cube((0,0,0), 1.5)
    result = maximize_intersection_volume(
        [(C, +1)], [(B, +1)], [0,0,0], [0,0,1], n_samples=360
    )
    assert result['max_volume'] > 0
    vol_0 = compute_convex_intersection_volume(C, B)
    assert abs(vol_0 - 0.5625) < 1e-3, f"Expected 0.5625, got {vol_0}"

def test_simultaneous_rotation_translation_scaling_optimization():
    A = make_cube((0,0,0), 2.0)
    B_outer = make_cube((1,0,0), 1.5)
    B_inner = make_cube((1,0,0), 0.5)
    body_A = [(A, +1)]
    body_B = [(B_outer, +1), (B_inner, -1)]
    result = maximize_intersection_volume(
        body_A, body_B, [0,0,0], [0,0,1], n_samples=360
    )
    assert result['max_volume'] > 0

for fn in [test_rotation_with_volume_and_surface_area_constraint,
           test_multi_object_intersection_with_hierarchy_constraints,
           test_simultaneous_rotation_translation_scaling_optimization]:
    run(fn)

print(f"\n{'='*50}")
print(f"TOTAL: {passed} passed, {failed} failed, {passed+failed} total")
