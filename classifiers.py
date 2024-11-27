# classifiers.py

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.hbonds import HydrogenBondAnalysis

def check_hydrogen_bond(u, params):
    h = HydrogenBondAnalysis(u,
                             donors_sel='resname A and (name N or name O)',
                             acceptors_sel='resname B and (name N or name O)',
                             distance=params['distance'],
                             angle=params['angle'])
    h.run()
    return len(h.timeseries) > 0

def check_pi_pi_stacking(u, params):
    ring_A = u.select_atoms('resname A and (name C1 C2 C3 C4 C5 C6)')
    ring_B = u.select_atoms('resname B and (name C1 C2 C3 C4 C5 C6)')
    if len(ring_A) < 3 or len(ring_B) < 3:
        return False

    def calculate_ring_properties(ring):
        positions = ring.positions
        center = np.mean(positions, axis=0)
        vec1 = positions[1] - positions[0]
        vec2 = positions[2] - positions[0]
        normal = np.cross(vec1, vec2)
        normal /= np.linalg.norm(normal)
        return center, normal

    center_A, normal_A = calculate_ring_properties(ring_A)
    center_B, normal_B = calculate_ring_properties(ring_B)

    center_distance = np.linalg.norm(center_A - center_B)
    cos_theta = np.dot(normal_A, normal_B)
    angle = np.degrees(np.arccos(np.abs(cos_theta)))
    displacement = np.linalg.norm(np.dot((center_A - center_B), normal_A))

    if (center_distance <= params['center_distance'] and
        angle <= params['angle_threshold'] and
        displacement <= params['displacement_threshold']):
        return True
    else:
        return False

def check_simple_proximity(u, params):
    sel1 = u.select_atoms('resname A')
    sel2 = u.select_atoms('resname B')
    min_dist = distances.distance_array(sel1.positions, sel2.positions).min()
    return min_dist <= params['distance_threshold']
