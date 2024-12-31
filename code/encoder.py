import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

#%% DeepNeo
# 아미노산 간의 영향을 주는 matrix (생략)
impact_matrix = np.array([
    [-0.274, -0.092, -0.181, -0.116, -0.214, -0.051, -0.164, -0.142, -0.012, -0.263, 0.028, 0.092, 0.245, -0.092, -0.077, 0.02, 0.139, 0.016, 0.169, 0.043],
    [-0.092, -0.26, -0.126, -0.153, -0.149, -0.151, -0.272, -0.191, -0.059, -0.109, -0.033, -0.019, 0.139, 0.06, 0.054, -0.076, 0.039, 0.128, 0.15, 0.024],
    [-0.181, -0.126, -0.333, -0.188, -0.292, 0.007, -0.266, -0.172, -0.027, -0.211, -0.05, 0.046, 0.319, 0.035, 0.009, -0.025, 0.043, 0.049, 0.123, -0.014],
    [-0.116, -0.153, -0.188, -0.617, -0.206, -0.124, -0.371, -0.177, -0.079, -0.207, -0.075, -0.034, 0.179, 0.007, -0.041, 0, 0.08, 0.036, 0.136, 0.121],
    [-0.214, -0.149, -0.292, -0.206, -0.404, -0.135, -0.239, -0.153, -0.01, -0.31, -0.062, -0.036, 0.197, -0.035, 0.043, 0.021, 0.139, 0.096, 0.148, 0.057],
    [-0.051, -0.151, 0.007, -0.124, -0.135, -0.416, -0.26, -0.253, -0.132, -0.056, 0.017, -0.034, 0.11, -0.111, -0.053, -0.004, 0.028, 0.058, 0.167, 0.017],
    [-0.164, -0.272, -0.266, -0.371, -0.239, -0.26, -2.066, -0.315, -0.192, -0.187, -0.108, -0.207, 0.018, 0.001, -0.036, -0.191, -0.038, 0.202, 0.097, 0.156],
    [-0.142, -0.191, -0.172, -0.177, -0.153, -0.253, -0.315, -0.238, -0.062, -0.117, -0.027, 0.019, 0.121, -0.06, -0.003, -0.023, -0.003, 0.145, 0.165, -0.013],
    [-0.012, -0.059, -0.027, -0.079, -0.01, -0.132, -0.192, -0.062, -0.4, -0.032, -0.026, -0.006, 0.192, 0.038, 0.121, 0.032, 0.133, 0.232, 0.083, 0.373],
    [-0.263, -0.109, -0.211, -0.207, -0.31, -0.056, -0.187, -0.117, -0.032, -0.559, -0.05, -0.105, 0.139, -0.128, -0.07, -0.031, 0.046, -0.012, 0.073, 0.075],
    [0.028, -0.033, -0.05, -0.075, -0.062, 0.017, -0.108, -0.027, -0.026, -0.05, -0.112, -0.027, 0.227, 0.131, 0.032, -0.017, 0.033, 0.134, 0.087, 0.165],
    [0.092, -0.019, 0.046, -0.034, -0.036, -0.034, -0.207, 0.019, -0.006, -0.105, -0.027, -0.138, 0.067, 0.084, 0.134, -0.013, 0.048, 0.24, 0.137, 0.179],
    [0.245, 0.139, 0.319, 0.179, 0.197, 0.11, 0.018, 0.121, 0.192, 0.139, 0.227, 0.067, 0.247, 0.299, 0.276, 0.201, 0.288, 0.428, 0.366, 0.496],
    [-0.092, 0.06, 0.035, 0.007, -0.035, -0.111, 0.001, -0.06, 0.038, -0.128, 0.131, 0.084, 0.299, 0.02, 0.077, 0.127, 0.209, -0.157, -0.038, 0.35],
    [-0.077, 0.054, 0.009, -0.041, 0.043, -0.053, -0.036, -0.003, 0.121, -0.07, 0.032, 0.134, 0.276, 0.077, -0.105, 0.09, 0.053, 0.106, 0.13, 0.121],
    [0.02, -0.076, -0.025, 0, 0.021, -0.004, -0.191, -0.023, 0.032, -0.031, -0.017, -0.013, 0.201, 0.127, 0.09, -0.108, 0.024, 0.196, 0.06, 0.178],
    [0.139, 0.039, 0.043, 0.08, 0.139, 0.028, -0.038, -0.003, 0.133, 0.046, 0.033, 0.048, 0.288, 0.209, 0.053, 0.024, -0.205, 0.245, 0.036, 0.117],
    [0.016, 0.128, 0.049, 0.036, 0.096, 0.058, 0.202, 0.145, 0.232, -0.012, 0.134, 0.24, 0.428, -0.157, 0.106, 0.196, 0.245, 0.181, 0.396, -0.182],
    [0.169, 0.15, 0.123, 0.136, 0.148, 0.167, 0.097, 0.165, 0.083, 0.073, 0.087, 0.137, 0.366, -0.038, 0.13, 0.06, 0.036, 0.396, 0.301, -0.006],
    [0.043, 0.024, -0.014, 0.121, 0.057, 0.017, 0.156, -0.013, 0.373, 0.075, 0.165, 0.179, 0.496, 0.35, 0.121, 0.178, 0.117, -0.182, -0.006, 0.205]
])

# 아미노산 코드와 인덱스를 매칭하는 딕셔너리
aa_to_index = {aa: i for i, aa in enumerate("LFIMVWCYHATGPRQSNEDK")}

# encoder.py


def deepneo_single_data(data, matrix_size=(9, 369)):
    hla_name, epi_seq, target, hla_seq = data
    encoded_matrix = np.zeros(matrix_size)
    for i, epi_aa in enumerate(epi_seq):
        for j, hla_aa in enumerate(hla_seq):
            if epi_aa == "*" or hla_aa == "*":
                encoded_matrix[i, j] = 0
            elif epi_aa == "X" or hla_aa == "X":
                encoded_matrix[i, j] = 0
            elif epi_aa == "U":
                encoded_matrix[i, j] = impact_matrix[aa_to_index["C"], aa_to_index[hla_aa]]
            elif epi_aa == "J":
                encoded_matrix[i, j] = max(impact_matrix[aa_to_index["L"], aa_to_index[hla_aa]], impact_matrix[aa_to_index["I"], aa_to_index[hla_aa]])
            elif epi_aa == "Z":
                encoded_matrix[i, j] = max(impact_matrix[aa_to_index["Q"], aa_to_index[hla_aa]], impact_matrix[aa_to_index["E"], aa_to_index[hla_aa]])
            elif epi_aa == "B":
                encoded_matrix[i, j] = max(impact_matrix[aa_to_index["D"], aa_to_index[hla_aa]], impact_matrix[aa_to_index["N"], aa_to_index[hla_aa]])
            else:
                encoded_matrix[i, j] = impact_matrix[aa_to_index[epi_aa], aa_to_index[hla_aa]]
    return encoded_matrix, target

class deepneo(Dataset):
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        data = self.data_provider[idx]  # hla_name, epi_seq, target, hla_seq
        encoded_matrix, target = deepneo_single_data(data, matrix_size=(9, 369))
        # Convert to tensors
        encoded_matrix = torch.tensor(encoded_matrix, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return encoded_matrix, target

class deepneo_2(Dataset):
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        data = self.data_provider[idx]  # hla_name, epi_seq, target, hla_seq
        encoded_matrix, target = deepneo_single_data(data, matrix_size=(15, 269))
        # Convert to tensors
        encoded_matrix = torch.tensor(encoded_matrix, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return encoded_matrix, target

#%% ESM
blosum62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, -1, -1, -4],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, -2, 0, -1, -4],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 4, -3, 0, -1, -4],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, -3, 1, -1, -4],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -1, -3, -1, -4],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, -2, 4, -1, -4],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, -3, 4, -1, -4],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -4, -2, -1, -4],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, -3, 0, -1, -4],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, 3, -3, -1, -4],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, 3, -3, -1, -4],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, -3, 1, -1, -4],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, 2, -1, -1, -4],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, 0, -3, -1, -4],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -3, -1, -1, -4],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, -2, 0, -1, -4],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, -1, -1, -4],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -2, -2, -1, -4],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -1, -2, -1, -4],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, 2, -2, -1, -4],
    'B': [-2, -1, 4, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, -3, 0, -1, -4],
    'J': [-2, -3, -3, -1, -2, -3, -4, -3, 3, 3, -3, 2, 0, -3, -2, -1, -2, -1, 2, -3, 3, -3, -1, -4],
    'Z': [-1, 0, 0, 1, -3, 4, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -2, -2, -2, 0, -3, 4, -1, -4],
    'X': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -4],
    '*': [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1]
}

def blosum62_encode(epitope, blosum62):
    if not isinstance(epitope, str):
        return torch.zeros((1, 25), dtype=torch.float32)
    try:
        encoded = [blosum62[aa] for aa in epitope]
    except KeyError as e:
        print(f"Invalid character in epitope sequence: {e}")
        encoded = [[0] * len(blosum62['A'])] * len(epitope)
    return torch.tensor(encoded, dtype=torch.float32)

class plm_blosum(Dataset):
    def __init__(self, data_provider, hla_emb_path):
        self.data_provider = data_provider
        self.hdf5_path = hla_emb_path

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        
        with h5py.File(self.hdf5_path, 'r') as hla_embeddings:
            embedding = np.squeeze(hla_embeddings[hla_name][()])
            hla_embedding = torch.tensor(embedding, dtype=torch.float32)
        epi_encoding = blosum62_encode(epi_seq, blosum62)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
        return hla_embedding, epi_encoding, target
    
class plm_plm(Dataset):
    def __init__(self, data_provider, hla_emb_path, epi_emb_path):
        self.data_provider = data_provider
        self.hdf5_path_1 = hla_emb_path
        self.hdf5_path_2 = epi_emb_path

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        
        with h5py.File(self.hdf5_path_1, 'r') as hla_embeddings:
            embedding = np.squeeze(hla_embeddings[hla_name][()])
            hla_embedding = torch.tensor(embedding, dtype=torch.float32)
        with h5py.File(self.hdf5_path_2, 'r') as epi_embeddings:
            embedding = np.squeeze(epi_embeddings[epi_seq][()])
            epi_embedding = torch.tensor(embedding, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
        return hla_embedding, epi_embedding, target