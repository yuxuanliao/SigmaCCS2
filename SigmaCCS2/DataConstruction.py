# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:06:51 2024

@author: yxliao
"""

import torch
import networkx as nx
from torch_geometric.data import Data

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import BondType as BT

import pandas as pd
import numpy as np
from tqdm import tqdm
import SigmaCCS2.Parameter as parameter


def read_data(filename,):
    data = pd.read_csv(filename)
    smiles = np.array(data['SMILES'])
    adduct = np.array(data['Adduct'])
    ccs    = np.array(data['True CCS'])
    return smiles, adduct, ccs


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def GetSmilesAtomSet(smiles):
    All_Atoms = []
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        All_Atoms += [atom.GetSymbol() for atom in mol.GetAtoms()]
        All_Atoms = list(set(All_Atoms))
    All_Atoms.sort()
    return All_Atoms


def Standardization(data):
    data_list = [data[i] for i in data]
    Max_data, Min_data = np.max(data_list), np.min(data_list)
    for i in data:
        data[i] = (data[i] - Min_data) / (Max_data - Min_data)
    return data


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
        
    return G


def atom_feature_oneHot(atom, All_Atoms, Atom_radius, Atom_mass):
    return np.array(
        # Atomic Type (One-Hot)
        one_of_k_encoding_unk(atom.GetSymbol() ,All_Atoms) +
        # Atomic Degree (One-Hot)
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
        # Atomic radius  Atomic mass (float)
        [Atom_radius[atom.GetSymbol()],Atom_mass[atom.GetSymbol()]] +
        # Atomic is in Ring ? (One-Hot)
        one_of_k_encoding_unk(atom.IsInRing(), [0, 1])
    )


def smiles2LineGraph(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    G = mol_to_nx(mol)
    Line_G = nx.line_graph(G)

    ps = AllChem.ETKDGv3()
    ps.randomSeed = -1
    ps.maxAttempts = 1
    ps.numThreads = 0
    ps.useRandomCoords = True
    re = AllChem.EmbedMultipleConfs(mol, numConfs = 1, params = ps)
    re = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads = 0)
    conf = mol.GetConformer()
    
    bond_indx, bond_type = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_type = bond.GetBondTypeAsDouble()
        bond_indx.append(set([start, end]))
        bond_type.append(edge_type)

    Line_G_nodes_name = []
    Line_G_node_features = []
    Line_G_edge_features = []
    Line_G_edge_indexs = []

    for lg_node in Line_G.nodes():
        edge_type = bond_type[bond_indx.index(set([lg_node[0], lg_node[1]]))]
        edge_type_oneHot = one_of_k_encoding_unk(edge_type, [1.0, 1.5 ,2.0, 3.0])
        edge_length = rdMolTransforms.GetBondLength(conf, lg_node[0], lg_node[1])

        Line_G_node_features.append(
            [edge_length] + 
            list(np.array(edge_type_oneHot).astype(int))
        )
        Line_G_nodes_name.append(set([lg_node[0], lg_node[1]]))

    for lg_edge in Line_G.edges():
        x, y = lg_edge[0]
        z, t = lg_edge[1]

        if z in [x,y]:
            if z == x: A = y; C = t
            if z == y: A = x; C = t
        else :
            if t == x: A = y; C = z
            if t == y: A = x; C = z

        AB = rdMolTransforms.GetBondLength(conf, x, y)
        BC = rdMolTransforms.GetBondLength(conf, z, t)
        AC = rdMolTransforms.GetBondLength(conf, A, C)

        cosABC = (AB**2 + BC**2 - AC**2) / (2 * AB * BC)
        angle = np.arccos(cosABC)
        angle = np.pi - angle

        Line_G_edge_features.append([angle])
        Line_G_edge_features.append([angle])
        Line_G_edge_indexs.append([Line_G_nodes_name.index(set([x,y])), Line_G_nodes_name.index(set([z,t]))])
        Line_G_edge_indexs.append([Line_G_nodes_name.index(set([z,t])), Line_G_nodes_name.index(set([x,y]))])
    
    Line_G_node_features = np.array(Line_G_node_features)
    Line_G_edge_features = np.array(Line_G_edge_features)
    Line_G_edge_indexs = np.array(Line_G_edge_indexs)
    
    return Line_G_node_features.astype(np.float32), Line_G_edge_features.astype(np.float32), Line_G_edge_indexs.T.astype(np.int64)


def smiles2Graph(smi, All_Atoms):
    Atom_radius = Standardization(parameter.Atom_radius)
    Atom_mass = Standardization(parameter.Atom_mass)
    
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveHs(mol)
    N = mol.GetNumAtoms()

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_feature_oneHot(atom, All_Atoms, Atom_radius, Atom_mass))
    
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_attr += 2 * [one_of_k_encoding_unk(bond.GetBondTypeAsDouble(),[1,1.5,2,3])]
        
    x = torch.tensor(np.array(x), dtype=torch.float32)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    
    
    return x, edge_attr, edge_index


def load_data(smiles, adduct_one_hot, ccs, All_Atoms):
    graph_adduct_data = []
    Index = 0
    for smi in tqdm(smiles):
        
        g_x, g_e, g_i = smiles2Graph(smi, All_Atoms)
        
        lg_x, lg_e, lg_i = smiles2LineGraph(smi)
    
        # label: true CCS
        y = torch.tensor([ccs[Index]], dtype=torch.float)
        
        one_graph = {}
        one_graph['x'] = g_x
        one_graph['edge_index'] = g_i
        one_graph['edge_attr'] = g_e
        
        one_graph['lg_x'] = lg_x
        one_graph['lg_edge_index'] = lg_i
        one_graph['lg_edge_attr'] = lg_e
        
        one_graph['y'] = y
        one_graph['adduct'] = adduct_one_hot[Index]
        graph_adduct_data.append(one_graph)
        
        Index += 1
    return graph_adduct_data
