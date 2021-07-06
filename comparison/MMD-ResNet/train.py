# -*- coding: utf-8 -*-
"""
Created on Sat Apr 6 00:17:31 2020

@author: urixs
"""

import numpy as np
from sklearn import decomposition
import argparse
from itertools import count
import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import scatterHist as sh
import utils
# ==============================================================================
# =                                Input arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_blocks', 
                    type=int, 
                    default=2, 
                    help='Number of resNet blocks')
parser.add_argument('--file1', 
                    type=str, 
                    default="Figure3_dataset5_0_time.csv",
                    help='path to file 1')
parser.add_argument('--file2', 
                    type=str, 
                    default="Figure3_dataset5_1_time.csv",
                    help='path to file 2')
parser.add_argument("--scale_k",
                    type=int,
                    default=5,
                    help="Number of neighbors for determining the RBF scale")
parser.add_argument('--batch_size', 
                    type=int, 
                    default=256,
                    help='Batch size (default=128)')
parser.add_argument('--lr', 
                    type=float, 
                    default=1e-5,
                    help='learning_rate (default=1e-3)')
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-6,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=10,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-4,
                    help="l_2 weight penalty")
parser.add_argument("--epochs_wo_im",
                    type=int,
                    default=5,
                    help="Number of epochs without improvement before stopping")
parser.add_argument("--save_dir",
                    type=str,
                    default='/media/bgi/zhouruilong/batchEffectRemoval2020-master/Figure3_result',
                    help="Directory for calibrated data")

args = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# =                                   Dataset                                  =
# ==============================================================================

sample1 = np.loadtxt(args.file1, delimiter=',')
sample2 = np.loadtxt(args.file2, delimiter=',')

sample1_tensor = torch.Tensor(sample1)
sample1_dataset = torch.utils.data.TensorDataset(sample1_tensor)

sample2_tensor = torch.Tensor(sample2)
sample2_dataset = torch.utils.data.TensorDataset(sample2_tensor)

sample1_loader = torch.utils.data.DataLoader(sample1_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

sample2_loader = torch.utils.data.DataLoader(sample2_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

input_dim1 = sample1.shape[1]
input_dim2 = sample2.shape[1]
assert input_dim1 == input_dim2, "samples are of different dimensions"
input_dim = input_dim1

# ==============================================================================
# =                                    Model                                   =
# ==============================================================================

class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    
    def __init__(self, 
                 dim,
                 use_dropout=False):
        """Initialize the Resnet block"""
        
        super(ResnetBlock, self).__init__()
        self.block = self.build_resnet_block(dim,
                                             use_dropout)

    def build_resnet_block(self,
                           dim,
                           use_dropout=False):
    
        block = [torch.nn.Linear(dim, dim),
                 torch.nn.BatchNorm1d(dim),
                 torch.nn.PReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        block += [torch.nn.Linear(dim, dim),
                 torch.nn.BatchNorm1d(dim),
                 torch.nn.PReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        return nn.Sequential(*block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out
        

class Mmd_resnet(nn.Module):   
    def __init__(self, 
                 input_dim,
                 n_blocks,
                 use_dropout=False):
        super(Mmd_resnet, self).__init__()
        
        model = []
        for i in range(n_blocks):  # add resnet blocks layers
            model += [ResnetBlock(input_dim,
                                  use_dropout)]
            
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Forward function (with skip connections)"""
        out = input + self.model(input)  # add skip connection
        return out

        
mmd_resnet = Mmd_resnet(input_dim,
                        args.n_blocks).to(device=device)  
    
# ==============================================================================
# =                         Optimizer and Learning rate                        =
# ==============================================================================    

optim = torch.optim.SGD(mmd_resnet.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)    

def lambda_rule(epoch) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((epoch + 1) / args.decay_step_size))
    return np.power(args.lr_decay_factor, exponent)

scheduler = lr_scheduler.LambdaLR(optim, 
                                        lr_lambda=lambda_rule)   

def update_lr():
        """ Learning rate updater """
        
        scheduler.step()
        lr = optim.param_groups[0]['lr']
        if lr < args.min_lr:
            optim.param_groups[0]['lr'] = args.min_lr
            lr = optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 

# ==============================================================================
# =                              Training procedure                            =
# ==============================================================================    

def training_step(batch1, batch2):
    
    mmd_resnet.train(True)
    
    calibrated_batch2 = mmd_resnet(batch2)
    
    # Compute distance matrices
    D1 = utils.compute_dist_mat(batch1, device=device)
    D2 = utils.compute_dist_mat(calibrated_batch2, device=device)
    D12 = utils.compute_dist_mat(batch1, calibrated_batch2)
    
    # Compute scale
    Dis, _ = utils.nn_search(batch1, k=args.scale_k)
    scale = utils.compute_scale(Dis, k=args.scale_k)
    
    # Compute kernel matrices
    K1 = utils.compute_kernel_mat(D1, scale)   
    K2 = utils.compute_kernel_mat(D2, scale) 
    K12 = utils.compute_kernel_mat(D12, scale)
    
    # Loss function and backprop
    mmd = torch.mean(K1) - 2 * torch.mean(K12) + torch.mean(K2)    
    loss = torch.sqrt(mmd)  

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()
    
# ==============================================================================
# =                                     Main                                   =
# ==============================================================================    
     
def main():
    
    best_loss = 100
    eps = 1e-4
    epoch_counter = 0
    for epoch in count(1):
        batch_losses = []
        
        samp2_batches = enumerate(sample2_loader)
        for batch_idx, batch1 in enumerate(sample1_loader):
            try:
                _, batch2 = next(samp2_batches)
            except:
                samp2_batches = enumerate(sample2_loader)
                _, batch2 = next(samp2_batches)
                
            batch1 = batch1[0].to(device=device)
            batch2 = batch2[0].to(device=device)
            
            batch_loss = training_step(batch1, batch2)
            batch_losses.append(batch_loss)
        
        epoch_loss = np.mean(batch_losses)
        
        if epoch_loss < best_loss - eps:
            best_loss = epoch_loss
            epoch_counter = 0
        else:
            epoch_counter += 1
            
        print('Epoch {}, loss: {:.3f}, counter: {}'.format(epoch, 
                                              epoch_loss,
                                              epoch_counter)
        )
   
        update_lr()
        
        if epoch_counter == args.epochs_wo_im:
            break
    print('Finished training')
    
    # calibrate sample2 -> batch 1    
    
    mmd_resnet.train(False)
    
    calibrated_sample2 = []
    for batch_idx, batch2 in enumerate(sample2_loader):
        batch2 = batch2[0].to(device=device)
        calibrated_batch = mmd_resnet(batch2)
        calibrated_sample2 += [calibrated_batch.detach().cpu().numpy()]
        
    calibrated_sample2 = np.concatenate(calibrated_sample2)
               
    # ==============================================================================
    # =                         visualize calibration                              =
    # ==============================================================================
    
    # PCA
    pca = decomposition.PCA()
    pca.fit(sample1)
    pc1 = 0
    pc2 = 1
    axis1 = 'PC'+str(pc1)
    axis2 = 'PC'+str(pc2)
    
    # plot data before calibration
    sample1_pca = pca.transform(sample1)
    sample2_pca = pca.transform(sample2)
    sh.scatterHist(sample1_pca[:,pc1], 
                   sample1_pca[:,pc2], 
                   sample2_pca[:,pc1], 
                   sample2_pca[:,pc2], 
                   axis1, 
                   axis2, 
                   title="Data before calibration",
                   name1='sample1', 
                   name2='sample2')
    
    # plot data after calibration
    calibrated_sample2_pca = pca.transform(calibrated_sample2)
    sh.scatterHist(sample1_pca[:,pc1], 
                   sample1_pca[:,pc2], 
                   calibrated_sample2_pca[:,pc1], 
                   calibrated_sample2_pca[:,pc2], 
                   axis1, 
                   axis2, 
                   title="Data after calibration",
                   name1='sample1', 
                   name2='sample2')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(args.save_dir + '/sample1_time.csv', sample1)
    np.save(args.save_dir + '/calibrated_sample2_time.csv', calibrated_sample2)
    
    
if __name__ == '__main__':
    main()