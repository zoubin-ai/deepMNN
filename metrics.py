#!/usr/bin/env python3

#Compute metrics over AnnData objects

import pandas as pd
import scanpy as sc
import numpy as np
from math import log
from sklearn.metrics import silhouette_score

def shannon_entropy (x, b_vec, N_b):
    
    tabled_values = b_vec[x > 0].value_counts()/ len(b_vec[x >0]) #class 'pandas.core.series.Series'

    tabled_val = tabled_values.tolist() 
    
    entropy = 0.0
    for element in tabled_val:
        if element != 0:
            entropy += element * log(element)
            
    entropy /= log(N_b)

    return(-entropy) #the entropy formula is the -sum, this is why we include the minus sign

def compute_entropy(adata, output_entropy=None, batch_key='batch', celltype_key='celltype'):
    print("Calculating entropy ...")
    kwargs = {}
    #batch vector(batch id of each cell)
    kwargs['batch_vector'] = adata.obs[batch_key]
    #modify index of batch vector so it coincides with matrix's index
    kwargs['batch_vector'].index = range(0,len(kwargs['batch_vector']))
    #number of batches
    kwargs['N_batches'] = len(adata.obs[batch_key].astype('category').cat.categories)

    #cell_type vector( betch id of each cell)
    kwargs['cell_type_vector'] = adata.obs[celltype_key]
    #modify index of cell_type vector so it coincides with matrix's index
    kwargs['cell_type_vector'].index = range(0,len(kwargs['cell_type_vector']))
    #number of cell_types
    kwargs['N_cell_types'] = len(adata.obs[celltype_key].astype('category').cat.categories)    

    try:
        knn_graph = adata.uns['neighbors']
        print('use exist neighbors')
    except KeyError:
        #compute neighbors
        print('compute neighbors')
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)

    #knn graph
    knn_graph = adata.uns['neighbors']['connectivities']
    #transforming csr_matrix to dataframe
    df = pd.DataFrame(knn_graph.toarray())
    
    #apply function
    batch_entropy = df.apply(shannon_entropy, axis=0, args=(kwargs['batch_vector'],kwargs['N_batches']))
    cell_type_entropy = df.apply(shannon_entropy, axis=0, args=(kwargs['cell_type_vector'] ,kwargs['N_cell_types']))
    print("Entropy calculated!")
    
    results = {'batch': batch_entropy, "cell_type":cell_type_entropy}
    results = pd.concat(results, axis = 1, keys = ['batch', 'cell_type'])
    
    if output_entropy:
        results.to_csv(output_entropy, header = True, index = False)
    
    return results

def silhouette_coeff_ASW(adata, method_use='raw',save_dir='', save_fn='', percent_extract=0.8, batch_key='batch', celltype_key='celltype'):
    asw_fscore = []
    asw_bn = []
    asw_bn_sub = []
    asw_ctn = []
    iters = []
    for i in range(20):
        iters.append('iteration_'+str(i+1))
        rand_cidx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
        adata_ext = adata[rand_cidx,:]
        asw_batch = silhouette_score(adata_ext.obsm['X_pca'], adata_ext.obs[batch_key])
        asw_celltype = silhouette_score(adata_ext.obsm['X_pca'], adata_ext.obs[celltype_key])
        min_val = -1
        max_val = 1
        asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
        asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)
        
        fscoreASW = (2 * (1 - asw_batch_norm)*(asw_celltype_norm))/(1 - asw_batch_norm + asw_celltype_norm)
        asw_fscore.append(fscoreASW)
        asw_bn.append(asw_batch_norm)
        asw_bn_sub.append(1-asw_batch_norm)
        asw_ctn.append(asw_celltype_norm)
    

    df = pd.DataFrame({'asw_batch_norm':asw_bn, 'asw_batch_norm_sub': asw_bn_sub,
                       'asw_celltype_norm': asw_ctn, 'fscore':asw_fscore,
                       'method_use':np.repeat(method_use, len(asw_fscore))})
#     df.to_csv(save_dir + save_fn + '.csv')
#     print('Save output of pca in: ',save_dir)
    return df