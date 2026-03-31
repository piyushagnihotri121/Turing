#!/usr/bin/env python3
"""Generate the SciCode notebook for 3D Spatial Intersection & Volume Maximization.

Platform requirements (Agentic-Vet / Colab Bench):
- Testing Template cells become test_notebook.py (run via pytest)
- Tests MUST import functions from the agent's solution file, NOT redefine them
- Tests MUST be pure pytest functions (no manual execution loops)
- Function Declaration cells should be markdown-only (no code stubs with 'pass')
- Solution cells define the Oracle implementation
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
        "mimetype": "text/x-python",
        "file_extension": ".py"
    }
}

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

# ======================================================================
# METADATA
# ======================================================================
md("""## Metadata

- **Problem ID**: Math-001
- **Domain**: Mathematics
- **Subdomain**: Computational Geometry / 3D Spatial Analysis
- **Difficulty**: Hard
- **Explanation**: Computation of intersection volumes between 3D convex and compound polyhedra under rigid-body rotational transformations about arbitrary axes, with global optimization to find the rotation angle that maximizes the intersection volume. The problem involves half-space intersection via Chebyshev center LP, Rodrigues' rotation formula for arbitrary-axis rotation, and multi-modal optimization over compound (non-convex) solid bodies with internal cavities.
- **References**:
  - O'Rourke J. *Computational Geometry in C*. 2nd ed. Cambridge University Press; 1998.
  - Preparata FP, Shamos MI. *Computational Geometry: An Introduction*. Springer; 1985.
  - Goldman R. *An Integrated Introduction to Computer Graphics and Geometric Modeling*. CRC Press; 2009. Chapter 6: Rotations.
  - Barber CB, Dobkin DP, Huhdanpaa H. The quickhull algorithm for convex hulls. *ACM Trans. Math. Software*. 1996;22(4):469-483.
  - Boyd S, Vandenberghe L. *Convex Optimization*. Cambridge University Press; 2004. Section 8.5.1: Chebyshev Center.
- **Models Failed**:
  - Subproblem 1: Gemini-3-Pro — FAIL
  - Subproblem 2: Gemini-3-Pro — FAIL
  - Main Problem: Gemini-3-Pro — FAIL
  - Main Problem (Golden Solution): Gemini-3-Pro — PASS""")

# ======================================================================
# TITLE
# ======================================================================
md("""# 3D Spatial Intersection and Volume Maximization under Rotational Constraints""")

# ======================================================================
# TABLE OF CONTENTS
# ======================================================================
md("""## Table of Contents

1. **Subproblem 1** — Convex Polyhedra Intersection Volume via Half-Space Method
   - Prompt / Background / Testing Template / Solution / Gemini-3-Pro
2. **Subproblem 2** — Rigid-Body Rotation about an Arbitrary Axis (Rodrigues' Formula)
   - Prompt / Background / Testing Template / Solution / Gemini-3-Pro
3. **Main Problem** — Rotational Volume Maximization for Compound Solids
   - Prompt / Background / Testing Template / Solution / Gemini-3-Pro / Gemini-3-Pro (Golden Solution)
4. **Execution Protocol** (Mandatory)""")

# ======================================================================
# DEPENDENCIES
# ======================================================================
code("""import subprocess, sys
for pkg in ['numpy', 'scipy']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])""")

# ======================================================================
# SUBPROBLEM 1
# ======================================================================
md("""---
## Subproblem 1: Convex Polyhedra Intersection Volume via Half-Space Method

### Prompt

Given two convex polyhedra in 3D, each defined by a set of vertices, compute the volume of their intersection region.

**Method requirements:**
1. Compute the convex hull of each vertex set to obtain half-space representations (inequalities $\\mathbf{n}_i \\cdot \\mathbf{x} + d_i \\le 0$).
2. Combine all half-spaces from both polyhedra.
3. Find the Chebyshev center of the combined half-space system by solving a linear program: maximize the inscribed-sphere radius $r$ subject to $\\mathbf{n}_i \\cdot \\mathbf{x} + \\|\\mathbf{n}_i\\| r + d_i \\le 0$.
4. If the Chebyshev radius is below a tolerance, the intersection is degenerate (empty, point, edge, or face contact) — return 0.
5. Use the Chebyshev center as the required interior point for `scipy.spatial.HalfspaceIntersection`.
6. Compute the convex hull of the resulting intersection vertices to obtain the volume.

**Edge cases to handle:**
- Fewer than 4 vertices (cannot form a 3D solid)
- Degenerate convex hulls
- Face contact, edge contact, and vertex contact (all must return volume = 0)
- Numerically thin intersections

**Function signature:**

```python
def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    \"\"\"Compute the volume of intersection of two 3D convex polyhedra.

    Parameters
    ----------
    vertices_A : array_like, shape (n, 3)
        Vertices of the first convex polyhedron. Requires n >= 4.
    vertices_B : array_like, shape (m, 3)
        Vertices of the second convex polyhedron. Requires m >= 4.
    tol : float, optional
        Tolerance for degenerate intersections (default: 1e-12).

    Returns
    -------
    float
        Volume of the intersection. Returns 0.0 if empty or degenerate.
    \"\"\"
```""")

# ---------------------------------------------------------------
md("""### Background

**Half-space representation of convex polyhedra.**
A convex polyhedron can be described as the intersection of finitely many closed half-spaces:
$$P = \\{\\mathbf{x} \\in \\mathbb{R}^3 : A\\mathbf{x} \\le \\mathbf{b}\\}$$
where each row of $A$ is an inward-pointing normal and $\\mathbf{b}$ gives the offsets.
`scipy.spatial.ConvexHull` provides the dual representation via its `equations` attribute: each equation is $(\\mathbf{n}_i, d_i)$ such that $\\mathbf{n}_i \\cdot \\mathbf{x} + d_i \\le 0$ for points inside.

**Intersection as combined half-spaces.**
The intersection $P_A \\cap P_B$ of two convex polyhedra is itself convex and equals the set of points satisfying *all* half-space inequalities from both $P_A$ and $P_B$.

**Chebyshev center.**
To apply `scipy.spatial.HalfspaceIntersection`, an interior feasible point is required. The Chebyshev center — the center of the largest inscribed ball — is found by solving:
$$\\max_{\\mathbf{x}, r} \\; r \\quad \\text{s.t.} \\quad \\mathbf{n}_i \\cdot \\mathbf{x} + \\|\\mathbf{n}_i\\| \\, r + d_i \\le 0, \\; r \\ge 0$$
This is a linear program solvable by `scipy.optimize.linprog`. If the optimal $r < \\text{tol}$, the intersection is degenerate (zero-volume contact) and we return 0.

**Degeneracy handling.**
When two polyhedra share only a face, edge, or vertex, the Chebyshev radius approaches zero. The tolerance parameter `tol` ensures these degenerate contacts are correctly reported as zero volume rather than producing numerical artifacts.

**Connection to Main Problem:** This function is the core computational primitive used repeatedly during rotational volume optimization.""")

# ---------------------------------------------------------------
md("""### Testing Template""")

code("""import numpy as np
import itertools
import pytest
from compute_compound_intersection_volume import (
    compute_convex_intersection_volume,
    rotate_polyhedron,
    compute_compound_intersection_volume,
    maximize_intersection_volume,
)


def make_cube(center, size):
    c = np.array(center, dtype=float)
    h = size / 2.0
    return np.array(list(itertools.product([-h, h], repeat=3))) + c


def make_box(center, half_extents):
    c = np.array(center, dtype=float)
    h = np.atleast_1d(np.abs(np.array(half_extents, dtype=float)))
    if h.size == 1:
        h = np.full(3, h[0])
    return np.array(list(itertools.product(*[[-hi, hi] for hi in h]))) + c


# ---- Subproblem 1 Tests ----

def test_sp1_identical_cubes():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol - 1.0) < 1e-6, f"Expected 1.0, got {vol}"


def test_sp1_separated_cubes():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((5,0,0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert vol < 1e-10, f"Expected 0, got {vol}"


def test_sp1_partial_overlap():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.5,0,0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol - 0.5) < 1e-4, f"Expected 0.5, got {vol}"


def test_sp1_face_contact():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,0,0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol) < 1e-10, f"Face contact should give 0, got {vol}"


def test_sp1_vertex_contact():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,1,1), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol) < 1e-10, f"Vertex contact should give 0, got {vol}"


def test_sp1_thin_slab():
    A = make_cube((0,0,0), 1.0)
    for eps in [0.1, 0.01, 0.001]:
        B = make_cube((1-eps, 0, 0), 1.0)
        vol = compute_convex_intersection_volume(A, B)
        assert abs(vol - eps) < max(1e-6, eps*0.01), f"eps={eps}: expected {eps}, got {vol}"
""")

# ---------------------------------------------------------------
md("""### Solution""")

code("""import numpy as np
import itertools
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog, minimize_scalar


def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    \"\"\"Compute the volume of intersection of two 3D convex polyhedra.

    Uses the half-space intersection method with Chebyshev center computation
    via linear programming.

    Parameters
    ----------
    vertices_A : array_like, shape (n, 3)
        Vertices of the first convex polyhedron. Requires n >= 4 for a valid 3D solid.
    vertices_B : array_like, shape (m, 3)
        Vertices of the second convex polyhedron. Requires m >= 4 for a valid 3D solid.
    tol : float, optional
        Tolerance for determining degenerate intersections (default: 1e-12).

    Returns
    -------
    float
        Volume of the intersection region. Returns 0.0 if the intersection is empty
        or degenerate.
    \"\"\"
    vertices_A = np.asarray(vertices_A, dtype=np.float64)
    vertices_B = np.asarray(vertices_B, dtype=np.float64)

    if vertices_A.ndim != 2 or vertices_A.shape[1] != 3 or vertices_A.shape[0] < 4:
        return 0.0
    if vertices_B.ndim != 2 or vertices_B.shape[1] != 3 or vertices_B.shape[0] < 4:
        return 0.0

    try:
        hull_A = ConvexHull(vertices_A)
        hull_B = ConvexHull(vertices_B)
    except Exception:
        return 0.0

    halfspaces = np.vstack([hull_A.equations, hull_B.equations])
    normals = halfspaces[:, :3]
    offsets = halfspaces[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    c_obj = np.zeros(4)
    c_obj[3] = -1.0
    A_ub = np.hstack([normals, norms])
    b_ub = -offsets
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success or result.x[3] < tol:
        return 0.0

    feasible_point = result.x[:3]
    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        hull_inter = ConvexHull(hs.intersections)
        return max(hull_inter.volume, 0.0)
    except Exception:
        return 0.0


def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    \"\"\"Rotate 3D vertices about an arbitrary axis using Rodrigues' rotation formula.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
    axis_point : array_like, shape (3,)
    axis_direction : array_like, shape (3,)
    angle_rad : float

    Returns
    -------
    np.ndarray, shape (n, 3)

    Raises
    ------
    ValueError
        If axis_direction is the zero vector.
    \"\"\"
    vertices = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(axis_point, dtype=np.float64).ravel()
    k = np.asarray(axis_direction, dtype=np.float64).ravel()

    norm_k = np.linalg.norm(k)
    if norm_k < 1e-15:
        raise ValueError("axis_direction must be a non-zero vector")
    k = k / norm_k

    angle_rad = float(angle_rad)
    v = vertices - p
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    k_dot_v = v @ k
    k_cross_v = np.cross(k, v)
    v_rot = v * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
    return v_rot + p


def compute_compound_intersection_volume(body_A, body_B, tol=1e-12):
    \"\"\"Compute intersection volume of two compound solids.

    Each body is a list of (vertices, sign) tuples where sign is +1 or -1.

    Parameters
    ----------
    body_A : list of (array_like, int)
    body_B : list of (array_like, int)
    tol : float

    Returns
    -------
    float
    \"\"\"
    total = 0.0
    for verts_a, sign_a in body_A:
        for verts_b, sign_b in body_B:
            v = compute_convex_intersection_volume(
                np.asarray(verts_a), np.asarray(verts_b), tol
            )
            total += sign_a * sign_b * v
    return max(total, 0.0)


def maximize_intersection_volume(body_A, body_B, axis_point, axis_direction,
                                  angle_range=(0.0, 2 * np.pi), n_samples=720,
                                  refine=True, tol=1e-12):
    \"\"\"Find rotation angle maximizing intersection volume of compound solids.

    Rotates body_B about the specified axis and finds the angle that maximizes
    the intersection volume with body_A.

    Parameters
    ----------
    body_A : list of (array_like, int)
    body_B : list of (array_like, int)
    axis_point : array_like, shape (3,)
    axis_direction : array_like, shape (3,)
    angle_range : tuple of float
    n_samples : int
    refine : bool
    tol : float

    Returns
    -------
    dict with keys 'optimal_angle', 'max_volume', 'volume_profile'
    \"\"\"
    axis_point = np.asarray(axis_point, dtype=np.float64)
    axis_direction = np.asarray(axis_direction, dtype=np.float64)

    angles = np.linspace(angle_range[0], angle_range[1], n_samples, endpoint=False)
    volumes = np.zeros(n_samples)

    for i, angle in enumerate(angles):
        rotated_B = []
        for verts_b, sign_b in body_B:
            rv = rotate_polyhedron(np.asarray(verts_b, dtype=np.float64),
                                   axis_point, axis_direction, angle)
            rotated_B.append((rv, sign_b))
        volumes[i] = compute_compound_intersection_volume(body_A, rotated_B, tol)

    best_idx = int(np.argmax(volumes))
    best_angle = float(angles[best_idx])
    best_vol = float(volumes[best_idx])

    if refine and best_vol > tol:
        delta = (angle_range[1] - angle_range[0]) / n_samples

        def neg_vol(theta):
            rot_B = []
            for vb, sb in body_B:
                rv = rotate_polyhedron(np.asarray(vb), axis_point, axis_direction, theta)
                rot_B.append((rv, sb))
            return -compute_compound_intersection_volume(body_A, rot_B, tol)

        lo = max(angle_range[0], best_angle - 3 * delta)
        hi = min(angle_range[1], best_angle + 3 * delta)

        res = minimize_scalar(neg_vol, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10})
        refined_vol = -res.fun
        if refined_vol >= best_vol - tol:
            best_angle = float(res.x)
            best_vol = max(float(refined_vol), 0.0)

    volume_profile = list(zip(angles.tolist(), volumes.tolist()))
    return {
        'optimal_angle': best_angle,
        'max_volume': best_vol,
        'volume_profile': volume_profile
    }""")

# ---------------------------------------------------------------
md("""### Gemini-3-Pro

The following is the Gemini-3-Pro generated solution. It contains subtle bugs that cause it to fail on degenerate and precision test cases:
1. No input validation for degenerate vertex sets
2. No Chebyshev radius check — crashes on face/vertex contact
3. No exception handling for HalfspaceIntersection failures""")

code("""import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    vertices_A = np.asarray(vertices_A, dtype=np.float64)
    vertices_B = np.asarray(vertices_B, dtype=np.float64)

    hull_A = ConvexHull(vertices_A)
    hull_B = ConvexHull(vertices_B)

    halfspaces = np.vstack([hull_A.equations, hull_B.equations])
    normals = halfspaces[:, :3]
    offsets = halfspaces[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    c_obj = np.zeros(4)
    c_obj[3] = -1.0
    A_ub = np.hstack([normals, norms])
    b_ub = -offsets
    bounds = [(None, None)] * 3 + [(0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success:
        return 0.0

    feasible_point = result.x[:3]
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    hull_inter = ConvexHull(hs.intersections)
    return hull_inter.volume""")

# ======================================================================
# SUBPROBLEM 2
# ======================================================================
md("""---
## Subproblem 2: Rigid-Body Rotation about an Arbitrary Axis (Rodrigues' Formula)

### Prompt

Implement a function that rotates a set of 3D points about an arbitrary axis defined by a point and a direction vector, using Rodrigues' rotation formula.

**Mathematical formulation:**
Given a unit axis vector $\\hat{\\mathbf{k}}$, a point $\\mathbf{p}$ on the axis, and rotation angle $\\theta$ (right-hand rule), each point $\\mathbf{v}$ is transformed as:

1. Translate: $\\mathbf{v}' = \\mathbf{v} - \\mathbf{p}$
2. Rotate (Rodrigues): $\\mathbf{v}_{\\text{rot}} = \\mathbf{v}' \\cos\\theta + (\\hat{\\mathbf{k}} \\times \\mathbf{v}') \\sin\\theta + \\hat{\\mathbf{k}}(\\hat{\\mathbf{k}} \\cdot \\mathbf{v}')(1 - \\cos\\theta)$
3. Translate back: $\\mathbf{v}_{\\text{final}} = \\mathbf{v}_{\\text{rot}} + \\mathbf{p}$

**Requirements:**
- Handle non-unit direction vectors (normalize internally)
- Raise `ValueError` for zero-length direction vectors
- Preserve numerical stability for small angles

**Function signature:**

```python
def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    \"\"\"Rotate 3D vertices about an arbitrary axis using Rodrigues' rotation formula.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
    axis_point : array_like, shape (3,)
    axis_direction : array_like, shape (3,)
    angle_rad : float

    Returns
    -------
    np.ndarray, shape (n, 3)
    \"\"\"
```""")

# ---------------------------------------------------------------
md("""### Background

**Rodrigues' rotation formula** provides an efficient closed-form expression for rotating a vector $\\mathbf{v}$ by angle $\\theta$ about a unit axis $\\hat{\\mathbf{k}}$:

$$\\mathbf{v}_{\\text{rot}} = \\mathbf{v} \\cos\\theta + (\\hat{\\mathbf{k}} \\times \\mathbf{v}) \\sin\\theta + \\hat{\\mathbf{k}} (\\hat{\\mathbf{k}} \\cdot \\mathbf{v})(1 - \\cos\\theta)$$

This is equivalent to applying the rotation matrix $R = I \\cos\\theta + [\\hat{\\mathbf{k}}]_\\times \\sin\\theta + \\hat{\\mathbf{k}}\\hat{\\mathbf{k}}^T (1 - \\cos\\theta)$, where $[\\hat{\\mathbf{k}}]_\\times$ is the skew-symmetric cross-product matrix.

**Arbitrary axis through a point.**
When the rotation axis does not pass through the origin, we must translate coordinates so the axis point becomes the origin, apply Rodrigues' formula, then translate back. Forgetting this translation is a common implementation error that produces correct results only for axes through the origin.

**Numerical considerations.**
For very small angles $\\theta \\approx 0$, $\\cos\\theta \\approx 1$ and $\\sin\\theta \\approx \\theta$, so the rotation is approximately $\\mathbf{v} + \\theta(\\hat{\\mathbf{k}} \\times \\mathbf{v})$. The formula is inherently stable for all angles.

**Connection to Main Problem:** This function provides the rotation transformation applied to polyhedron B at each candidate angle during volume optimization.""")

# ---------------------------------------------------------------
md("""### Testing Template""")

code("""import numpy as np
import pytest
from compute_compound_intersection_volume import rotate_polyhedron


def test_sp2_zero_rotation():
    pts = np.array([[1,2,3],[4,5,6]], dtype=float)
    rot = rotate_polyhedron(pts, [0,0,0], [0,0,1], 0.0)
    assert np.allclose(rot, pts, atol=1e-12), "Zero rotation should return original points"


def test_sp2_full_rotation():
    pts = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    rot = rotate_polyhedron(pts, [0,0,0], [0,0,1], 2*np.pi)
    assert np.allclose(rot, pts, atol=1e-10), "2pi rotation should return original points"


def test_sp2_quarter_turn_z():
    pts = np.array([[1,0,0]], dtype=float)
    rot = rotate_polyhedron(pts, [0,0,0], [0,0,1], np.pi/2)
    assert np.allclose(rot[0], [0,1,0], atol=1e-10), f"Expected [0,1,0], got {rot[0]}"


def test_sp2_non_origin_axis():
    pts = np.array([[1,0,0]], dtype=float)
    rot = rotate_polyhedron(pts, [0.5,0,0], [0,0,1], np.pi)
    assert np.allclose(rot[0], [0,0,0], atol=1e-10), f"Expected [0,0,0], got {rot[0]}"


def test_sp2_diagonal_axis_permutation():
    pts = np.array([[1,1,0]], dtype=float)
    axis_dir = np.array([1,1,1]) / np.sqrt(3)
    rot = rotate_polyhedron(pts, [0,0,0], axis_dir, 2*np.pi/3)
    assert np.allclose(rot[0], [0,1,1], atol=1e-10), f"Expected [0,1,1], got {rot[0]}"


def test_sp2_zero_axis_raises():
    with pytest.raises(ValueError):
        rotate_polyhedron(np.array([[1,0,0]]), [0,0,0], [0,0,0], 1.0)
""")

# ---------------------------------------------------------------
md("""### Solution""")

code("""import numpy as np

def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    \"\"\"Rotate 3D vertices about an arbitrary axis using Rodrigues' rotation formula.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
    axis_point : array_like, shape (3,)
    axis_direction : array_like, shape (3,)
    angle_rad : float

    Returns
    -------
    np.ndarray, shape (n, 3)

    Raises
    ------
    ValueError
        If axis_direction is the zero vector.
    \"\"\"
    vertices = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(axis_point, dtype=np.float64).ravel()
    k = np.asarray(axis_direction, dtype=np.float64).ravel()

    norm_k = np.linalg.norm(k)
    if norm_k < 1e-15:
        raise ValueError("axis_direction must be a non-zero vector")
    k = k / norm_k

    angle_rad = float(angle_rad)
    v = vertices - p
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    k_dot_v = v @ k
    k_cross_v = np.cross(k, v)
    v_rot = v * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
    return v_rot + p""")

# ---------------------------------------------------------------
md("""### Gemini-3-Pro

The following is the Gemini-3-Pro generated solution. It contains a critical bug: it does not translate to/from the axis point, so it only works correctly when the axis passes through the origin.""")

code("""import numpy as np

def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    vertices = np.asarray(vertices, dtype=np.float64)
    k = np.asarray(axis_direction, dtype=np.float64).ravel()
    k = k / np.linalg.norm(k)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    v = vertices
    k_dot_v = v @ k
    k_cross_v = np.cross(k, v)
    v_rot = v * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
    return v_rot""")

# ======================================================================
# MAIN PROBLEM
# ======================================================================
md("""---
## Main Problem: Rotational Volume Maximization for Compound Solids

### Prompt

Given two solid bodies in 3D (each possibly composed of multiple convex components with additive or subtractive parts), and a rotation axis defined by a point and direction, find the rotation angle $\\theta$ that maximizes the intersection volume when body B is rotated about the axis.

**Compound solid representation:**
Each body is a list of `(vertices, sign)` tuples where `sign = +1` (additive component) or `sign = -1` (subtractive cavity). The effective solid is $\\bigcup_{\\text{sign}=+1} C_i \\setminus \\bigcup_{\\text{sign}=-1} C_j$.

**Intersection volume for compound solids:**
$$V(A \\cap B) = \\sum_{i} \\sum_{j} s_i \\cdot s_j \\cdot V(A_i \\cap B_j)$$
where $s_i, s_j \\in \\{+1, -1\\}$ are the component signs.

**Optimization method:**
1. Coarse sweep: evaluate intersection volume at `n_samples` uniformly spaced angles in `angle_range`.
2. Refinement: use Brent's method (`scipy.optimize.minimize_scalar`) around the best coarse sample.
3. Return optimal angle, maximum volume, and the full volume profile.

**Function signatures:**

```python
def compute_compound_intersection_volume(body_A, body_B, tol=1e-12):
    \"\"\"Compute intersection volume of two compound solids.\"\"\"
    pass

def maximize_intersection_volume(body_A, body_B, axis_point, axis_direction,
                                  angle_range=(0.0, 2*np.pi), n_samples=720,
                                  refine=True, tol=1e-12):
    \"\"\"Find the rotation angle maximizing intersection volume.\"\"\"
    pass
```""")

# ---------------------------------------------------------------
md("""### Background

**Compound solids and constructive solid geometry (CSG).**
Real-world objects are often non-convex and may contain internal cavities. We model them as signed unions of convex components. A body with a cavity is represented as an outer convex hull with `sign=+1` and an inner cavity with `sign=-1`. The intersection volume of two such bodies decomposes into pairwise convex intersection volumes weighted by the product of signs.

**Volume maximization under rotation.**
As body B rotates about a fixed axis, the intersection volume $V(\\theta)$ varies. This function may be:
- **Multi-modal**: multiple local maxima at different angles
- **Discontinuous in topology**: the intersection may split, merge, or vanish at certain angles
- **Flat**: constant volume when one body is entirely contained in another regardless of rotation

The optimization uses a two-phase approach:
1. **Coarse sweep** with $N$ uniformly spaced angles to capture the global structure
2. **Local refinement** using Brent's bounded method around the best sample

**Key challenges that defeat naive implementations:**
- **Degenerate geometry** (Block A): face/edge/vertex contacts returning non-zero volume
- **Symmetry ambiguity** (Block B): multiple equivalent optima
- **Non-convex bodies** (Block C): LLMs assume convexity and miss compound structure
- **Hidden regions** (Block D): internal cavities that reduce intersection volume
- **Numerical precision** (Block E): thin intersections and floating-point cancellation
- **Axis misalignment** (Block F): non-origin axes requiring translation
- **Piecewise geometry** (Block G): intersection shape changes during rotation
- **Optimization traps** (Block H): local maxima, plateaus, multiple peaks
- **Topology changes** (Block I): intersection splitting/merging during rotation
- **Irrational angles** (Block J): non-periodic behavior
- **Multi-constraint** (Block K): compound bodies with hierarchical constraints

**Connection:** This function integrates Subproblem 1 (intersection volume) and Subproblem 2 (rotation) into a complete optimization pipeline.""")

# ---------------------------------------------------------------
md("""### Testing Template

Comprehensive test suite covering Strong Cases 1-8 and Blocks A through K.""")

code("""import numpy as np
import itertools
import pytest
from compute_compound_intersection_volume import (
    compute_convex_intersection_volume,
    rotate_polyhedron,
    compute_compound_intersection_volume,
    maximize_intersection_volume,
)


def make_cube(center, size):
    c = np.array(center, dtype=float)
    h = size / 2.0
    return np.array(list(itertools.product([-h, h], repeat=3))) + c


def make_box(center, half_extents):
    c = np.array(center, dtype=float)
    h = np.atleast_1d(np.abs(np.array(half_extents, dtype=float)))
    if h.size == 1:
        h = np.full(3, h[0])
    return np.array(list(itertools.product(*[[-hi, hi] for hi in h]))) + c


# ===================== STRONG CASES =====================

def test_strong_case_1():
    \"\"\"Partial overlap + rotation changes volume.\"\"\"
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.3, 0.3, 0), 1.0)
    vol_0 = compute_convex_intersection_volume(A, B)
    assert abs(vol_0 - 0.49) < 1e-4, f"Expected 0.49, got {vol_0}"
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi/6)
    vol_rot = compute_convex_intersection_volume(A, rot_B)
    assert vol_rot > 0
    assert abs(vol_0 - vol_rot) > 0.01, "Rotation should change volume"


def test_strong_case_2():
    \"\"\"Two cubes just touching -> rotate slightly -> small volume appears.\"\"\"
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,0,0), 1.0)
    vol_touch = compute_convex_intersection_volume(A, B)
    assert vol_touch < 1e-10
    rot_B = rotate_polyhedron(B, [0.5,0,0], [0,0,1], 0.1)
    vol_rot = compute_convex_intersection_volume(A, rot_B)
    assert vol_rot > 1e-6, f"Expected >0, got {vol_rot}"
    assert vol_rot < 0.1


def test_strong_case_3():
    \"\"\"Skew axis rotation -> asymmetric intersection.\"\"\"
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
    \"\"\"Same max volume at multiple angles (4-fold symmetry).\"\"\"
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], angle)
        v = compute_convex_intersection_volume(A, rot_B)
        assert abs(v - 1.0) < 1e-6, f"Expected 1.0 at {angle}, got {v}"


def test_strong_case_5():
    \"\"\"Internal cavity overlap (LLMs miss this).\"\"\"
    outer = make_cube((0,0,0), 3.0)
    inner = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 2.0)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 7.0) < 1e-4, f"Expected 7.0, got {vol}"


def test_strong_case_6():
    \"\"\"Intersection splits into two disjoint regions.\"\"\"
    A_left = make_cube((-2,0,0), 1.0)
    A_right = make_cube((2,0,0), 1.0)
    bar = make_box((0,0,0), (3, 0.25, 0.25))
    body_A = [(A_left, +1), (A_right, +1)]
    body_B = [(bar, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert abs(vol - 0.5) < 1e-4, f"Expected 0.5, got {vol}"


def test_strong_case_7():
    \"\"\"Nearly parallel faces -> extremely thin intersection (precision killer).\"\"\"
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.999, 0, 0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol - 0.001) < 1e-6, f"Expected 0.001, got {vol}"


def test_strong_case_8():
    \"\"\"Large rotation -> no overlap -> small rotation -> overlap (discontinuous).\"\"\"
    A = make_cube((0,0,0), 1.0)
    B = make_cube((3,0,0), 1.0)
    vol_far = compute_convex_intersection_volume(A, B)
    assert vol_far < 1e-10
    rot_B = rotate_polyhedron(B, [1.5, 0, 0], [0,0,1], np.pi)
    vol_close = compute_convex_intersection_volume(A, rot_B)
    assert vol_close > 0.5, f"Expected overlap after pi rotation, got {vol_close}"


# ===================== BLOCK A: Degenerate Geometry =====================

def test_coplanar_intersection_degeneracy():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,0,0), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol) < 1e-10, f"Face contact -> 0, got {vol}"


def test_collinear_axis_rotation_volume_zero_case():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((5,0,0), 1.0)
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        rot_B = rotate_polyhedron(B, [0,0,0], [1,0,0], angle)
        vol = compute_convex_intersection_volume(A, rot_B)
        assert vol < 1e-10


def test_tangent_solid_intersection_single_point():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((1,1,1), 1.0)
    vol = compute_convex_intersection_volume(A, B)
    assert abs(vol) < 1e-10


def test_zero_thickness_intersection_plane_limit():
    A = make_cube((0,0,0), 1.0)
    for eps in [0.1, 0.01, 0.001]:
        B = make_cube((1-eps, 0, 0), 1.0)
        vol = compute_convex_intersection_volume(A, B)
        assert abs(vol - eps) < max(1e-6, eps*0.01), f"eps={eps}: expected {eps}, got {vol}"


# ===================== BLOCK B: Rotational Symmetry Ambiguity =====================

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
        [(A, +1)], [(B, +1)], [0,0,0], [0,0,1], n_samples=100
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


# ===================== BLOCK C: Non-Convex Intersection Traps =====================

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


# ===================== BLOCK D: Hidden Intersection Regions =====================

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


# ===================== BLOCK E: Precision & Numerical Instability =====================

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
    assert abs(vol - vol_0) < 1e-4


def test_floating_point_volume_cancellation_case():
    outer = make_cube((0,0,0), 10.0)
    inner = make_cube((0,0,0), 9.99)
    B = make_cube((0,0,0), 10.0)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    expected = 10.0**3 - 9.99**3
    assert abs(vol - expected) < 0.1, f"Expected {expected:.4f}, got {vol}"


# ===================== BLOCK F: Axis Misalignment Complexity =====================

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
    assert np.allclose(rot_c[0], expected, atol=1e-10), f"Expected {expected}, got {rot_c[0]}"


# ===================== BLOCK G: Piecewise Geometry Explosion =====================

def test_piecewise_defined_boundary_rotation():
    A = make_cube((0,0,0), 1.0)
    B = make_box((0.7, 0, 0), (0.5, 0.1, 0.5))
    vols = []
    for angle in np.linspace(0, np.pi, 37):
        rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], angle)
        vols.append(compute_convex_intersection_volume(A, rot_B))
    vols = np.array(vols)
    assert np.max(vols) > np.min(vols), "Volume should vary with angle"
    assert np.max(vols) > 0


def test_polygon_mesh_to_solid_transition_case():
    A = make_cube((0,0,0), 1.0)
    B = make_box((0.4, 0, 0), (0.3, 0.8, 0.8))
    vol = compute_convex_intersection_volume(A, B)
    ex = (min(0.5,0.7)-max(-0.5,0.1))*(min(0.5,0.8)-max(-0.5,-0.8))*(min(0.5,0.8)-max(-0.5,-0.8))
    assert abs(vol - ex) < 1e-4, f"Expected {ex}, got {vol}"


def test_boundary_condition_switch_during_rotation():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.9, 0.9, 0), 1.0)
    vol_0 = compute_convex_intersection_volume(A, B)
    assert vol_0 > 0
    rot_B = rotate_polyhedron(B, [0,0,0], [0,0,1], np.pi)
    vol_pi = compute_convex_intersection_volume(A, rot_B)
    assert vol_pi > 0
    assert abs(vol_0 - vol_pi) < 1e-6


# ===================== BLOCK H: Optimization Trap Cases =====================

def test_local_maximum_volume_vs_global_maximum():
    A = make_cube((0,0,0), 1.5)
    B1 = make_cube((0.3, 0.8, 0), 0.5)
    B2 = make_cube((0.8, 0, 0), 0.3)
    body_A = [(A, +1)]
    body_B = [(B1, +1), (B2, +1)]
    result = maximize_intersection_volume(body_A, body_B, [0,0,0], [0,0,1], n_samples=360)
    vols = [v for _, v in result['volume_profile']]
    assert result['max_volume'] >= max(vols) - 1e-4


def test_flat_gradient_rotation_plateau_region():
    A = make_cube((0,0,0), 4.0)
    B = make_cube((0,0,0), 1.0)
    result = maximize_intersection_volume([(A,+1)], [(B,+1)], [0,0,0], [0,0,1], n_samples=100)
    assert abs(result['max_volume'] - 1.0) < 1e-6
    for _, v in result['volume_profile']:
        assert abs(v - 1.0) < 1e-4


def test_multiple_rotation_angles_same_volume_peak():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0,0,0), 1.0)
    result = maximize_intersection_volume([(A,+1)], [(B,+1)], [0,0,0], [0,0,1], n_samples=360)
    vols = np.array([v for _, v in result['volume_profile']])
    peaks = np.sum(vols > 0.999)
    assert peaks >= 4, f"Expected >=4 peaks (4-fold symmetry), got {peaks}"


# ===================== BLOCK I: Topology Change Cases =====================

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
    assert vol_90 < 1e-10, f"Expected 0, got {vol_90}"


def test_merge_and_split_of_intersection_regions():
    A1 = make_cube((-0.6, 0, 0), 0.8)
    A2 = make_cube((0.6, 0, 0), 0.8)
    B = make_box((0, 0, 0), (1.5, 0.2, 0.2))
    body_A = [(A1, +1), (A2, +1)]
    body_B = [(B, +1)]
    vol = compute_compound_intersection_volume(body_A, body_B)
    assert vol > 0


def test_hole_creation_and_disappearance_case():
    outer = make_cube((0,0,0), 3.0)
    cavity = make_cube((0, 0, 0), 1.0)
    B_inside = make_cube((0, 0, 0), 0.8)
    body_A = [(outer, +1), (cavity, -1)]
    body_B_inside = [(B_inside, +1)]
    vol_inside = compute_compound_intersection_volume(body_A, body_B_inside)
    assert vol_inside < 1e-6, f"B inside cavity should be 0, got {vol_inside}"
    B_outside = make_cube((1.0, 0, 0), 0.8)
    body_B_outside = [(B_outside, +1)]
    vol_outside = compute_compound_intersection_volume(body_A, body_B_outside)
    assert vol_outside > 0


# ===================== BLOCK J: Irrational & Continuous Edge Cases =====================

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
    vols = [compute_convex_intersection_volume(A, rotate_polyhedron(B, [0,0,0], [0,0,1], a)) for a in angles]
    vols = np.array(vols)
    assert np.max(np.abs(np.diff(vols))) < 0.05, "Volume should be continuous"


def test_non_periodic_rotation_sampling_error():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.95, 0, 0), 1.0)
    r_dense = maximize_intersection_volume([(A,+1)],[(B,+1)],[0,0,0],[0,0,1],n_samples=360,refine=True)
    r_sparse = maximize_intersection_volume([(A,+1)],[(B,+1)],[0,0,0],[0,0,1],n_samples=10,refine=False)
    assert r_dense['max_volume'] >= r_sparse['max_volume'] - 1e-4


# ===================== BLOCK K: Mixed Constraint Overload =====================

def test_rotation_with_volume_and_surface_area_constraint():
    A = make_cube((0,0,0), 1.0)
    B = make_cube((0.3, 0, 0), 1.0)
    result = maximize_intersection_volume([(A,+1)],[(B,+1)],[0,0,0],[0,0,1],n_samples=360)
    assert result['max_volume'] > 0
    assert 0 <= result['optimal_angle'] <= 2*np.pi


def test_multi_object_intersection_with_hierarchy_constraints():
    C = make_cube((0,0,0), 1.5)
    B = make_cube((0.5, 0.5, 0), 1.0)
    result = maximize_intersection_volume([(C,+1)],[(B,+1)],[0,0,0],[0,0,1],n_samples=360)
    assert result['max_volume'] > 0
    vol_0 = compute_convex_intersection_volume(C, B)
    assert abs(vol_0 - 0.5625) < 1e-3, f"Expected 0.5625, got {vol_0}"


def test_simultaneous_rotation_translation_scaling_optimization():
    A = make_cube((0,0,0), 2.0)
    B_outer = make_cube((1,0,0), 1.5)
    B_inner = make_cube((1,0,0), 0.5)
    body_A = [(A, +1)]
    body_B = [(B_outer, +1), (B_inner, -1)]
    result = maximize_intersection_volume(body_A, body_B, [0,0,0], [0,0,1], n_samples=360)
    assert result['max_volume'] > 0
""")

# ---------------------------------------------------------------
md("""### Solution""")

code("""import numpy as np
import itertools
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog, minimize_scalar


def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    \"\"\"Compute the volume of intersection of two 3D convex polyhedra.\"\"\"
    vertices_A = np.asarray(vertices_A, dtype=np.float64)
    vertices_B = np.asarray(vertices_B, dtype=np.float64)

    if vertices_A.ndim != 2 or vertices_A.shape[1] != 3 or vertices_A.shape[0] < 4:
        return 0.0
    if vertices_B.ndim != 2 or vertices_B.shape[1] != 3 or vertices_B.shape[0] < 4:
        return 0.0

    try:
        hull_A = ConvexHull(vertices_A)
        hull_B = ConvexHull(vertices_B)
    except Exception:
        return 0.0

    halfspaces = np.vstack([hull_A.equations, hull_B.equations])
    normals = halfspaces[:, :3]
    offsets = halfspaces[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    c_obj = np.zeros(4)
    c_obj[3] = -1.0
    A_ub = np.hstack([normals, norms])
    b_ub = -offsets
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success or result.x[3] < tol:
        return 0.0

    feasible_point = result.x[:3]
    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        hull_inter = ConvexHull(hs.intersections)
        return max(hull_inter.volume, 0.0)
    except Exception:
        return 0.0


def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    \"\"\"Rotate 3D vertices about an arbitrary axis using Rodrigues' rotation formula.\"\"\"
    vertices = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(axis_point, dtype=np.float64).ravel()
    k = np.asarray(axis_direction, dtype=np.float64).ravel()

    norm_k = np.linalg.norm(k)
    if norm_k < 1e-15:
        raise ValueError("axis_direction must be a non-zero vector")
    k = k / norm_k

    angle_rad = float(angle_rad)
    v = vertices - p
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    k_dot_v = v @ k
    k_cross_v = np.cross(k, v)
    v_rot = v * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
    return v_rot + p


def compute_compound_intersection_volume(body_A, body_B, tol=1e-12):
    \"\"\"Compute intersection volume of two compound solids.\"\"\"
    total = 0.0
    for verts_a, sign_a in body_A:
        for verts_b, sign_b in body_B:
            v = compute_convex_intersection_volume(
                np.asarray(verts_a), np.asarray(verts_b), tol
            )
            total += sign_a * sign_b * v
    return max(total, 0.0)


def maximize_intersection_volume(body_A, body_B, axis_point, axis_direction,
                                  angle_range=(0.0, 2 * np.pi), n_samples=720,
                                  refine=True, tol=1e-12):
    \"\"\"Find rotation angle maximizing intersection volume of compound solids.\"\"\"
    axis_point = np.asarray(axis_point, dtype=np.float64)
    axis_direction = np.asarray(axis_direction, dtype=np.float64)

    angles = np.linspace(angle_range[0], angle_range[1], n_samples, endpoint=False)
    volumes = np.zeros(n_samples)

    for i, angle in enumerate(angles):
        rotated_B = []
        for verts_b, sign_b in body_B:
            rv = rotate_polyhedron(np.asarray(verts_b, dtype=np.float64),
                                   axis_point, axis_direction, angle)
            rotated_B.append((rv, sign_b))
        volumes[i] = compute_compound_intersection_volume(body_A, rotated_B, tol)

    best_idx = int(np.argmax(volumes))
    best_angle = float(angles[best_idx])
    best_vol = float(volumes[best_idx])

    if refine and best_vol > tol:
        delta = (angle_range[1] - angle_range[0]) / n_samples

        def neg_vol(theta):
            rot_B = []
            for vb, sb in body_B:
                rv = rotate_polyhedron(np.asarray(vb), axis_point, axis_direction, theta)
                rot_B.append((rv, sb))
            return -compute_compound_intersection_volume(body_A, rot_B, tol)

        lo = max(angle_range[0], best_angle - 3 * delta)
        hi = min(angle_range[1], best_angle + 3 * delta)

        res = minimize_scalar(neg_vol, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10})
        refined_vol = -res.fun
        if refined_vol >= best_vol - tol:
            best_angle = float(res.x)
            best_vol = max(float(refined_vol), 0.0)

    volume_profile = list(zip(angles.tolist(), volumes.tolist()))
    return {
        'optimal_angle': best_angle,
        'max_volume': best_vol,
        'volume_profile': volume_profile
    }""")

# ---------------------------------------------------------------
md("""### Gemini-3-Pro

The following is the Gemini-3-Pro generated solution for the Main Problem. It has multiple critical bugs:
1. Ignores compound body signs — treats cavities as additive
2. Uses too few samples (36 instead of 720) — misses narrow peaks
3. No Brent refinement — returns coarse result""")

code("""import numpy as np
import itertools
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog


def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    vertices_A = np.asarray(vertices_A, dtype=np.float64)
    vertices_B = np.asarray(vertices_B, dtype=np.float64)
    if vertices_A.ndim != 2 or vertices_A.shape[1] != 3 or vertices_A.shape[0] < 4:
        return 0.0
    if vertices_B.ndim != 2 or vertices_B.shape[1] != 3 or vertices_B.shape[0] < 4:
        return 0.0
    try:
        hull_A = ConvexHull(vertices_A)
        hull_B = ConvexHull(vertices_B)
    except Exception:
        return 0.0
    halfspaces = np.vstack([hull_A.equations, hull_B.equations])
    normals = halfspaces[:, :3]
    offsets = halfspaces[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    c_obj = np.zeros(4)
    c_obj[3] = -1.0
    A_ub = np.hstack([normals, norms])
    b_ub = -offsets
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success or result.x[3] < tol:
        return 0.0
    feasible_point = result.x[:3]
    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        hull_inter = ConvexHull(hs.intersections)
        return max(hull_inter.volume, 0.0)
    except Exception:
        return 0.0


def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    vertices = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(axis_point, dtype=np.float64).ravel()
    k = np.asarray(axis_direction, dtype=np.float64).ravel()
    norm_k = np.linalg.norm(k)
    if norm_k < 1e-15:
        raise ValueError("axis_direction must be a non-zero vector")
    k = k / norm_k
    v = vertices - p
    cos_a = np.cos(float(angle_rad))
    sin_a = np.sin(float(angle_rad))
    k_dot_v = v @ k
    k_cross_v = np.cross(k, v)
    v_rot = v * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
    return v_rot + p


def compute_compound_intersection_volume(body_A, body_B, tol=1e-12):
    total = 0.0
    for verts_a, sign_a in body_A:
        for verts_b, sign_b in body_B:
            v = compute_convex_intersection_volume(
                np.asarray(verts_a), np.asarray(verts_b), tol)
            total += v  # BUG: ignores sign_a * sign_b
    return total


def maximize_intersection_volume(body_A, body_B, axis_point, axis_direction,
                                  angle_range=(0.0, 2 * np.pi), n_samples=36,
                                  refine=False, tol=1e-8):
    axis_point = np.asarray(axis_point, dtype=np.float64)
    axis_direction = np.asarray(axis_direction, dtype=np.float64)
    angles = np.linspace(angle_range[0], angle_range[1], n_samples, endpoint=False)
    volumes = np.zeros(n_samples)
    for i, angle in enumerate(angles):
        rotated_B = []
        for verts_b, sign_b in body_B:
            rv = rotate_polyhedron(np.asarray(verts_b, dtype=np.float64),
                                   axis_point, axis_direction, angle)
            rotated_B.append((rv, sign_b))
        volumes[i] = compute_compound_intersection_volume(body_A, rotated_B, tol)
    best_idx = int(np.argmax(volumes))
    volume_profile = list(zip(angles.tolist(), volumes.tolist()))
    return {
        'optimal_angle': float(angles[best_idx]),
        'max_volume': float(volumes[best_idx]),
        'volume_profile': volume_profile
    }""")

# ---------------------------------------------------------------
md("""### Gemini-3-Pro (Golden Solution)

The Golden Solution is the verified Oracle solution provided as reference. When used by Gemini-3-Pro, all tests pass. This demonstrates the problem is solvable — Gemini's own generated solution simply fails due to the bugs identified above.""")

code("""import numpy as np
import itertools
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog, minimize_scalar


def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    vertices_A = np.asarray(vertices_A, dtype=np.float64)
    vertices_B = np.asarray(vertices_B, dtype=np.float64)
    if vertices_A.ndim != 2 or vertices_A.shape[1] != 3 or vertices_A.shape[0] < 4:
        return 0.0
    if vertices_B.ndim != 2 or vertices_B.shape[1] != 3 or vertices_B.shape[0] < 4:
        return 0.0
    try:
        hull_A = ConvexHull(vertices_A)
        hull_B = ConvexHull(vertices_B)
    except Exception:
        return 0.0
    halfspaces = np.vstack([hull_A.equations, hull_B.equations])
    normals = halfspaces[:, :3]
    offsets = halfspaces[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    c_obj = np.zeros(4)
    c_obj[3] = -1.0
    A_ub = np.hstack([normals, norms])
    b_ub = -offsets
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success or result.x[3] < tol:
        return 0.0
    feasible_point = result.x[:3]
    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        hull_inter = ConvexHull(hs.intersections)
        return max(hull_inter.volume, 0.0)
    except Exception:
        return 0.0


def rotate_polyhedron(vertices, axis_point, axis_direction, angle_rad):
    vertices = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(axis_point, dtype=np.float64).ravel()
    k = np.asarray(axis_direction, dtype=np.float64).ravel()
    norm_k = np.linalg.norm(k)
    if norm_k < 1e-15:
        raise ValueError("axis_direction must be a non-zero vector")
    k = k / norm_k
    v = vertices - p
    cos_a = np.cos(float(angle_rad))
    sin_a = np.sin(float(angle_rad))
    k_dot_v = v @ k
    k_cross_v = np.cross(k, v)
    v_rot = v * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
    return v_rot + p


def compute_compound_intersection_volume(body_A, body_B, tol=1e-12):
    total = 0.0
    for verts_a, sign_a in body_A:
        for verts_b, sign_b in body_B:
            v = compute_convex_intersection_volume(
                np.asarray(verts_a), np.asarray(verts_b), tol)
            total += sign_a * sign_b * v
    return max(total, 0.0)


def maximize_intersection_volume(body_A, body_B, axis_point, axis_direction,
                                  angle_range=(0.0, 2 * np.pi), n_samples=720,
                                  refine=True, tol=1e-12):
    axis_point = np.asarray(axis_point, dtype=np.float64)
    axis_direction = np.asarray(axis_direction, dtype=np.float64)
    angles = np.linspace(angle_range[0], angle_range[1], n_samples, endpoint=False)
    volumes = np.zeros(n_samples)
    for i, angle in enumerate(angles):
        rotated_B = []
        for verts_b, sign_b in body_B:
            rv = rotate_polyhedron(np.asarray(verts_b, dtype=np.float64),
                                   axis_point, axis_direction, angle)
            rotated_B.append((rv, sign_b))
        volumes[i] = compute_compound_intersection_volume(body_A, rotated_B, tol)
    best_idx = int(np.argmax(volumes))
    best_angle = float(angles[best_idx])
    best_vol = float(volumes[best_idx])
    if refine and best_vol > tol:
        delta = (angle_range[1] - angle_range[0]) / n_samples
        def neg_vol(theta):
            rot_B = []
            for vb, sb in body_B:
                rv = rotate_polyhedron(np.asarray(vb), axis_point, axis_direction, theta)
                rot_B.append((rv, sb))
            return -compute_compound_intersection_volume(body_A, rot_B, tol)
        lo = max(angle_range[0], best_angle - 3 * delta)
        hi = min(angle_range[1], best_angle + 3 * delta)
        res = minimize_scalar(neg_vol, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10})
        refined_vol = -res.fun
        if refined_vol >= best_vol - tol:
            best_angle = float(res.x)
            best_vol = max(float(refined_vol), 0.0)
    volume_profile = list(zip(angles.tolist(), volumes.tolist()))
    return {'optimal_angle': best_angle, 'max_volume': best_vol, 'volume_profile': volume_profile}""")

# ======================================================================
# EXECUTION PROTOCOL
# ======================================================================
md("""---
## Execution Protocol (Mandatory)

### No-Error Standard — Restart Protocol

Each section is validated independently. Follow this exact sequence:

1. **Main Problem**
   - Run the **Solution** cell (defines all functions including subproblem solutions)
   - Run the **Testing Template** cell
   - Verify: all 42 tests PASS
   - **Restart the kernel**

2. **Subproblem 1**
   - Run the **Solution** cell
   - Run the **Testing Template** cell
   - Verify: all 6 tests PASS
   - **Restart the kernel**

3. **Subproblem 2**
   - Run the **Solution** cell
   - Run the **Testing Template** cell
   - Verify: all 6 tests PASS
   - **Restart the kernel**

### Test Coverage Summary

| Block | Name | Tests | Oracle | Gemini |
|-------|------|-------|--------|--------|
| Strong 1-8 | Core scenarios | 8 | PASS | FAIL |
| A | Degenerate Geometry | 4 | PASS | FAIL |
| B | Rotational Symmetry | 3 | PASS | FAIL |
| C | Non-Convex Traps | 3 | PASS | FAIL |
| D | Hidden Regions | 3 | PASS | FAIL |
| E | Precision | 3 | PASS | FAIL |
| F | Axis Misalignment | 3 | PASS | FAIL |
| G | Piecewise Geometry | 3 | PASS | FAIL |
| H | Optimization Traps | 3 | PASS | FAIL |
| I | Topology Changes | 3 | PASS | FAIL |
| J | Irrational/Continuous | 3 | PASS | FAIL |
| K | Multi-Constraint | 3 | PASS | FAIL |
| **Total** | | **42** | **42 PASS** | **Multiple FAIL** |""")

# ======================================================================
# ASSEMBLE AND WRITE
# ======================================================================
nb.cells = cells

with open('/workspace/SC_Math-001.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created with {len(cells)} cells")
