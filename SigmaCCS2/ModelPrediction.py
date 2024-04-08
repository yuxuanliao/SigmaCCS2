# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:00:51 2024

@author: yxliao
"""

from SigmaCCS2.DataConstruction import *
from SigmaCCS2.ModelTraining import *
import SigmaCCS2.Parameter as parameter
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean, median, abs, sum, cumsum, histogram, sqrt
np.set_printoptions(suppress=True)


def metrics(y_true, y_pred):   #r2_score(y_true, y_pred)
    RelativeError = [abs(y_pred[i]-y_true[i])/y_true[i] for i in range(len(y_true))]
    R2_Score = r2_score(y_true,y_pred)
    abs_y_err = [abs(y_pred[i]-y_true[i]) for i in range(len(y_true))]
    mae = mean(abs_y_err)
    mdae = median(abs_y_err)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    print("R2 Score :", R2_Score)
    print("Mean Absolute Error :", mae)
    print("Median Absolute Error :", mdae)
    print("Median Relative Error :", np.median(RelativeError) * 100, '%')
    print("Mean Relative Error :", np.mean(RelativeError) * 100, '%')
    print("Root Mean Squared Error :", RMSE)
    print("Mean Squared Error :", MSE)
    return R2_Score, np.median(RelativeError) * 100


def SigmaCCS2_predict(ifilepath, mfilepath, ofilepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    smiles, adduct, ccs = read_data(ifilepath)
    print('## Read data : ',len(smiles))
    print('## All Atoms : ', parameter.All_Atoms)
    print('## All Adduct : ', parameter.adduct_SET)
    
    smiles2 = list(smiles)
    adduct2 = list(adduct)
 
    adduct_one_hot = [(one_of_k_encoding_unk(adduct[_], parameter.adduct_SET)) for _ in range(len(adduct))]
    adduct_one_hot = list(np.array(adduct_one_hot).astype(int))
    
    graph_adduct_data = load_data(smiles, adduct_one_hot, ccs, parameter.All_Atoms)
    
    test_loader = torch.utils.data.DataLoader(
        graph_adduct_data,
        shuffle=False,
        num_workers=0,
        batch_size=1
    )
    print('## Molecular graph & Line graph & Adduct dataset completed')
    
    pred_ccs = []
    true_ccs = []
    '''Test'''
    with torch.no_grad():
        for data in tqdm(test_loader):
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
            
            m_state_dict = torch.load(mfilepath)
            new_m = SigmaCCS2().to(device)
            new_m.load_state_dict(m_state_dict)
            predict_test = new_m(graph, line_graph, adduct)
            pred_ccs.append(predict_test[0].cpu().numpy().tolist())
            true_ccs.append(data['y'][0][0].numpy().tolist())
    
    data2 = {'SMILES' : smiles2,
             'Adduct' : adduct2,
             'Ture CCS': true_ccs,
             'Predicted CCS': pred_ccs}
    
    df = DataFrame(data2)
    df.to_csv(ofilepath, index=False)
    print('## CCS predicted completed')
    re_Metrics = metrics(true_ccs, pred_ccs)
    # return re_Metrics