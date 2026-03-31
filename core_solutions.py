import numpy as np
import itertools
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog, minimize_scalar


def compute_convex_intersection_volume(vertices_A, vertices_B, tol=1e-12):
    """Compute the volume of intersection of two 3D convex polyhedra.

    Uses the half-space intersection method with Chebyshev center computation
    via linear programming.

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
    """
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
    """Rotate 3D vertices about an arbitrary axis using Rodrigues' formula.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
    axis_point : array_like, shape (3,)
    axis_direction : array_like, shape (3,)
    angle_rad : float

    Returns
    -------
    np.ndarray, shape (n, 3)
    """
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
    """Compute intersection volume of two compound solids.

    Each body is a list of (vertices, sign) tuples where sign is +1 or -1.

    Parameters
    ----------
    body_A : list of (array_like, int)
    body_B : list of (array_like, int)
    tol : float

    Returns
    -------
    float
    """
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
    """Find rotation angle maximizing intersection volume of compound solids.

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
    dict with 'optimal_angle', 'max_volume', 'volume_profile'
    """
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
    }


def make_cube(center, size):
    c = np.array(center, dtype=float)
    h = size / 2.0
    offsets = np.array(list(itertools.product([-h, h], repeat=3)))
    return offsets + c


def make_box(center, half_extents):
    c = np.array(center, dtype=float)
    h = np.atleast_1d(np.abs(np.array(half_extents, dtype=float)))
    if h.size == 1:
        h = np.full(3, h[0])
    offsets = np.array(list(itertools.product(*[[-hi, hi] for hi in h])))
    return offsets + c


if __name__ == '__main__':
    # Quick smoke tests
    A = make_cube((0, 0, 0), 1.0)
    B = make_cube((0, 0, 0), 1.0)
    v = compute_convex_intersection_volume(A, B)
    print(f"Identical cubes: {v:.6f} (expected 1.0)")

    B2 = make_cube((0.5, 0, 0), 1.0)
    v2 = compute_convex_intersection_volume(A, B2)
    print(f"Half-overlap cubes: {v2:.6f} (expected 0.5)")

    B3 = make_cube((1, 0, 0), 1.0)
    v3 = compute_convex_intersection_volume(A, B3)
    print(f"Face-touching cubes: {v3:.10f} (expected ~0)")

    B4 = make_cube((1, 1, 1), 1.0)
    v4 = compute_convex_intersection_volume(A, B4)
    print(f"Vertex-touching cubes: {v4:.10f} (expected ~0)")

    rot = rotate_polyhedron(np.array([[1, 0, 0]]), [0, 0, 0], [0, 0, 1], np.pi / 2)
    print(f"Rotate (1,0,0) by pi/2 about z: {rot[0]} (expected [0,1,0])")

    rot2 = rotate_polyhedron(np.array([[1, 0, 0]]), [0.5, 0, 0], [0, 0, 1], np.pi)
    print(f"Rotate (1,0,0) by pi about z through (0.5,0,0): {rot2[0]} (expected [0,0,0])")

    # Strong case 2: touching cubes, rotate about shared face center
    B_touch = make_cube((1, 0, 0), 1.0)
    rot_B = rotate_polyhedron(B_touch, np.array([0.5, 0, 0]), np.array([0, 0, 1]), 0.1)
    v_rot = compute_convex_intersection_volume(A, rot_B)
    print(f"Strong case 2 (rotated touching): {v_rot:.6f} (expected small > 0)")

    # Compound body test: cube with cavity
    outer = make_cube((0, 0, 0), 3.0)
    inner = make_cube((0, 0, 0), 1.0)
    B_comp = make_cube((0, 0, 0), 2.0)
    body_A = [(outer, +1), (inner, -1)]
    body_B = [(B_comp, +1)]
    v_comp = compute_compound_intersection_volume(body_A, body_B)
    print(f"Cube with cavity: {v_comp:.6f} (expected 7.0)")

    # Nested: small cube inside cavity
    B_small = make_cube((0, 0, 0), 0.5)
    body_B2 = [(B_small, +1)]
    v_nested = compute_compound_intersection_volume(body_A, body_B2)
    print(f"Small cube inside cavity: {v_nested:.6f} (expected 0.0)")
