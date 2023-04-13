import numpy as np
from openbabel import pybel

class BindingPocket:
    def __init__(self, occupied_cells):
        self.occupied_cells = occupied_cells

def create_3d_grid(pocket, resolution):
    min_coords = np.min(pocket.occupied_cells, axis=0)
    shifted_coords = pocket.occupied_cells - min_coords
    
    max_coords = np.max(shifted_coords, axis=0)
    grid_shape = np.ceil((max_coords) / resolution).astype(int) + 1
    grid = np.zeros(grid_shape, dtype=bool)

    for cell in shifted_coords:
        cell_idx = np.floor(cell / resolution).astype(int)
        grid[tuple(cell_idx)] = True
    return grid

def intersection_over_union(pocket1, pocket2, resolution):
    grid1= create_3d_grid(pocket1, resolution)
    grid2 = create_3d_grid(pocket2, resolution)
    
    common_grid_shape = np.maximum(grid1.shape, grid2.shape)
    
    grid1_padded = np.pad(
        grid1,
        [(0, int(common_grid_shape[i] - grid1.shape[i])) for i in range(3)],
        mode='constant',
        constant_values=False
    )
    
    grid2_padded = np.pad(
        grid2,
        [(0, int(common_grid_shape[i] - grid2.shape[i])) for i in range(3)],
        mode='constant',
        constant_values=False
    )
    
    intersection_grid = np.logical_and(grid1_padded, grid2_padded)
    union_grid = np.logical_or(grid1_padded, grid2_padded)
    
    intersection_volume = np.sum(intersection_grid) * resolution**3
    union_volume = np.sum(union_grid) * resolution**3
    
    return intersection_volume / union_volume
def intersection_over_lig(lig, pocket, resolution):
    grid1= create_3d_grid(lig, resolution)
    grid2 = create_3d_grid(pocket, resolution)
    
    common_grid_shape = np.maximum(grid1.shape, grid2.shape)
    
    grid1_padded = np.pad(
        grid1,
        [(0, int(common_grid_shape[i] - grid1.shape[i])) for i in range(3)],
        mode='constant',
        constant_values=False
    )
    
    grid2_padded = np.pad(
        grid2,
        [(0, int(common_grid_shape[i] - grid2.shape[i])) for i in range(3)],
        mode='constant',
        constant_values=False
    )
    
    intersection_grid = np.logical_and(grid1_padded, grid2_padded)

    intersection_volume = np.sum(intersection_grid) * resolution**3
    lig_volume = np.sum(grid1_padded) * resolution**3
    
    return intersection_volume / lig_volume
def coordinates(pdb_file):
    molecule = next(pybel.readfile(pdb_file.split('.')[-1], pdb_file))
    ligand_coords = [atom.coords for atom in molecule.atoms]
    return np.array(ligand_coords)

def get_DVO(pkt1,pkt2,resolution = 1):
    pocket1_coords = coordinates(pkt1)
    pocket2_coords = coordinates(pkt2)
    pocket1 = BindingPocket(pocket1_coords)
    pocket2 = BindingPocket(pocket2_coords)
    return intersection_over_union(pocket1, pocket2, resolution)

def get_PLI(lig,pkt,resolution = 1):
    lig_coords=coordinates(lig)
    pkt_coords=coordinates(pkt)
    ligand=BindingPocket(lig_coords)
    pocket=BindingPocket(pkt_coords)
    return intersection_over_lig(ligand,pocket,resolution)
