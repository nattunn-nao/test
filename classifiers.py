# classifiers.py

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.hbonds import HydrogenBondAnalysis
import networkx as nx
from MDAnalysis.analysis.graph import build_bond_list

def load_ring_indices(filename):
    """
    ファイルから原子インデックスのリストを読み込む
    """
    rings = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                indices = [int(idx) for idx in line.strip().split()]
                rings.append(indices)
    return rings

def calculate_ring_properties(ring_atoms):
    positions = ring_atoms.positions
    center = np.mean(positions, axis=0)
    vec1 = positions[1] - positions[0]
    vec2 = positions[2] - positions[0]
    normal = np.cross(vec1, vec2)
    normal /= np.linalg.norm(normal)
    return center, normal

def check_hydrogen_bond(u, params):
    h = HydrogenBondAnalysis(u,
                             donors_sel='resname A and (name N or name O)',
                             acceptors_sel='resname B and (name N or name O)',
                             distance=params['distance'],
                             angle=params['angle'])
    h.run()
    return len(h.timeseries) > 0

def check_pi_pi_stacking(u, params):
    # 環原子インデックスをファイルから読み込む
    ring_indices_list1 = load_ring_indices(params['ring_indices_file1'])
    ring_indices_list2 = load_ring_indices(params.get('ring_indices_file2', params['ring_indices_file1']))  # 分子Bのインデックスファイルがない場合は分子Aと同じものを使用

    # 全ての環の組み合わせでスタッキングをチェック
    for indices_A in ring_indices_list1:
        ring_A_atoms = u.atoms[indices_A]
        center_A, normal_A = calculate_ring_properties(ring_A_atoms)

        for indices_B in ring_indices_list2:
            ring_B_atoms = u.atoms[indices_B]
            center_B, normal_B = calculate_ring_properties(ring_B_atoms)

            # スタッキングの判定
            center_distance = np.linalg.norm(center_A - center_B)
            cos_theta = np.dot(normal_A, normal_B)
            angle = np.degrees(np.arccos(np.abs(cos_theta)))
            displacement = np.linalg.norm(np.dot((center_A - center_B), normal_A))

            if (center_distance <= params['center_distance'] and
                angle <= params['angle_threshold'] and
                displacement <= params['displacement_threshold']):
                return True  # スタッキングが見つかった場合

    return False  # すべての組み合わせでスタッキングが見つからなかった場合

def check_simple_proximity(u, params):
    sel1 = u.select_atoms('resname A')
    sel2 = u.select_atoms('resname B')
    min_dist = distances.distance_array(sel1.positions, sel2.positions).min()
    return min_dist <= params['distance_threshold']
