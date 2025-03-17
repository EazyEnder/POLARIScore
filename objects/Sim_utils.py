from scipy.spatial import cKDTree
import numpy as np

def trilinear_interpolation(p, neighbors):
    """Interpolate density at point p using 8 nearest neighbors."""
    if len(neighbors) < 8:  # If fewer than 8 points, return average density
        return np.mean([d for _, d in neighbors]) if neighbors else 0.0

    # Extract positions and densities
    positions = np.array([n[0] for n in neighbors])
    densities = np.array([n[1] for n in neighbors])

    # Find bounding box
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)

    # Normalize p within the unit cube
    tx, ty, tz = (p - min_corner) / (max_corner - min_corner + 1e-6)

    # Densities at the corners
    c000, c001, c010, c011, c100, c101, c110, c111 = densities[:8]

    # Interpolation formula
    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz

def get_density(tree, points, densities, x, y, z):
    """Query density at any (x, y, z) using KD-Tree for fast search."""
    p = np.array([x, y, z])
    
    # Find 8 nearest neighbors
    dists, idxs = tree.query(p, k=8)  
    neighbors = [(points[i], densities[i]) for i in idxs]
    
    return trilinear_interpolation(p, neighbors)

def get_column_density_along_ray(tree, points, densities, sizes, ray_origin, ray_direction, max_distance=1, step_size=0.1):
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    current_position = ray_origin
    column_density = 0.0
    
    while np.linalg.norm(current_position - ray_origin) < max_distance:
        search_radius = np.min(sizes) * 2 
        idxs = tree.query_ball_point(current_position, search_radius)
        
        for idx in idxs:
            point_position = points[idx]
            density = densities[idx]
            size = sizes[idx]
            
            path_length = np.linalg.norm(current_position - point_position)
            
            column_density += density * size
        
        # Move along the ray by the specified step size
        current_position += ray_direction * step_size
        print(column_density)
    
    return column_density