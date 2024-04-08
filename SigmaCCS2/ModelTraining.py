# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:01:44 2024

@author: yxliao
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.data import Data
from torch_geometric.nn import NNConv

import numpy as np
from tqdm import tqdm

from SigmaCCS2.DataConstruction import *


class SigmaCCS2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        '''ECC on molecular graph'''
        self.lin0 = torch.nn.Linear(20, 64)
        nn = Sequential(Linear(4, 64), ReLU(), Linear(64, 64 * 64))
        self.conv = NNConv(64, 64, nn, aggr='mean')
        self.graph_conv_N = 3
        
        '''ECC on Line graph'''
        self.lin0_lg = torch.nn.Linear(5, 64)
        nn_lg = Sequential(Linear(1, 64), ReLU(), Linear(64, 64 * 64))
        self.conv_lg = NNConv(64, 64, nn_lg, aggr='mean') #add
        self.linegraph_conv_N = 3
        
        '''MLP'''
        self.bottelneck = torch.nn.Linear(64 + 64 + 3, 384)
        self.lin1 = torch.nn.Linear(384, 384)
        self.lin2 = torch.nn.Linear(384, 1)
        self.MLP_N = 6

    def forward(self, graph, linegraph, adduct):
        out_g = F.relu(self.lin0(graph.x))
        for i in range(self.graph_conv_N):
            out_g = F.relu(self.conv(out_g, graph.edge_index, graph.edge_attr))
            out_g = out_g.squeeze(0)            
        out_g = torch.sum(out_g, dim=0)
        
        out_lg = F.relu(self.lin0_lg(linegraph.x))
        for i in range(self.linegraph_conv_N):
            out_lg = F.relu(self.conv_lg(out_lg, linegraph.edge_index, linegraph.edge_attr))
            out_lg = out_lg.squeeze(0)            
        out_lg = torch.sum(out_lg, dim=0)
        
        out = torch.cat([out_g, out_lg, adduct], dim=-1)
        out = F.relu(self.bottelneck(out))
        for j in range(self.MLP_N):
            out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out
    
    
def SigmaCCS2_train(ifilepath, epochs, batchsize, savepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    smiles, adduct, ccs = read_data(ifilepath)
    print('## Read data : ',len(smiles))
    
    All_Atoms = GetSmilesAtomSet(smiles)
    print('## All Atoms : ', All_Atoms)
    
    adduct_SET = list(set(adduct))
    adduct_SET.sort()
    print('## Adduct set order : ', adduct_SET) # important        
    adduct_one_hot = [(one_of_k_encoding_unk(adduct[_], adduct_SET)) for _ in range(len(adduct))]
    adduct_one_hot = list(np.array(adduct_one_hot).astype(int))
    
    graph_adduct_data = load_data(smiles, adduct_one_hot, ccs, All_Atoms)
    
    split_line = int(len(ccs)*0.9)
    train_graph_adduct_data = graph_adduct_data[:split_line]
    valid_graph_adduct_data = graph_adduct_data[split_line:]
    print('## The size of the training set : ', len(train_graph_adduct_data))
    print('## The size of the validation set : ', len(valid_graph_adduct_data))

    train_loader = torch.utils.data.DataLoader(
        train_graph_adduct_data,
        shuffle=True,
        num_workers=0,
        batch_size=1
    )

    val_loader = torch.utils.data.DataLoader(
        valid_graph_adduct_data,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    
    model = SigmaCCS2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(0, epochs):
        batch_loss = 0
        Index = 0
        
        loss_all = []
        with tqdm(total=len(train_loader)) as p_bar:
            for data in train_loader:
                Index += 1
    
                graph = Data(
                    x=data['x'][0], 
                    edge_index=data['edge_index'][0],
                    edge_attr=data['edge_attr'][0], 
                    y=data['y'][0]
                ).to(device)
    
                line_graph = Data(
                    x=data['lg_x'][0], 
                    edge_index=data['lg_edge_index'][0],
                    edge_attr=data['lg_edge_attr'][0], 
                ).to(device)
    
                adduct = data['adduct'][0].to(device)
    
                pred = model(graph, line_graph, adduct)
                loss = F.huber_loss(pred, graph.y)
                loss_all.append(loss.cpu().detach().numpy())
    
                batch_loss += loss
                if Index % batchsize == 0:
                    optimizer.zero_grad()
                    batch_loss = batch_loss / batchsize
                    batch_loss.backward()
                    optimizer.step()
                    
                    p_bar.update(batchsize)
                    p_bar.set_description("Training-Loss {:.2f}".format(batch_loss))
                    batch_loss = 0
        train_loss = np.mean(loss_all)
            
        loss_all = []
        with torch.no_grad():
            for data in tqdm(val_loader):
                graph = Data(
                    x=data['x'][0], 
                    edge_index=data['edge_index'][0],
                    edge_attr=data['edge_attr'][0], 
                    y=data['y'][0]
                ).to(device)
    
                line_graph = Data(
                    x=data['lg_x'][0], 
                    edge_index=data['lg_edge_index'][0],
                    edge_attr=data['lg_edge_attr'][0], 
                ).to(device)
    
                adduct = data['adduct'][0].to(device)
    
                pred = model(graph, line_graph, adduct)
                loss = F.huber_loss(pred, graph.y)
                loss_all.append(loss.cpu().detach().numpy())
        val_loss = np.mean(loss_all)
        
        print('train-loss', train_loss, 'val-loss', val_loss)
        
    torch.save(model.state_dict(), savepath)