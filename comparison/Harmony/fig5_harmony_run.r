library(harmony)
library(cowplot)
library(Seurat)  
library(magrittr)
library(SeuratDisk)
rm(list=ls())

########################
#settings

filter_genes = T
filter_cells = T
normData = T
Datascaling = T
regressUMI = F
min_cells = 10
min_genes = 300
norm_method = "LogNormalize"
scale_factor = 10000
b_x_low_cutoff = 0.0125
b_x_high_cutoff = 3
b_y_cutoff = 0.5
numVG = 300
npcs = 50
visualize = T
outfile_prefix = "Dataset5"
save_obj = T

src_dir = "./"
working_dir = "./"

read_dir = "./"

expr_mat_filename = "myData_pancreatic_5batches.txt"
metadata_filename = "mySample_pancreatic_5batches.txt"

batch_label = "batchlb"
celltype_label = "CellType"

########################
# read data

expr_mat <- read.table(file = paste0(read_dir,expr_mat_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata <- read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'celltype'] <- "CellType"

expr_mat <- expr_mat[, rownames(metadata)]

########################
# run pipeline

source('call_harmony.R')
#setwd(working_dir)

b_seurat = harmony_preprocess(expr_mat, metadata, 
                filter_genes = filter_genes, filter_cells = filter_cells,
                normData = normData, Datascaling = Datascaling, regressUMI = regressUMI, 
                min_cells = min_cells, min_genes = min_genes, 
                norm_method = norm_method, scale_factor = scale_factor, 
                b_x_low_cutoff = b_x_low_cutoff, b_x_high_cutoff = b_x_high_cutoff, b_y_cutoff = b_y_cutoff, 
                numVG = numVG, npcs = npcs, 
                batch_label = batch_label, celltype_label = celltype_label)

b_seurat = call_harmony(b_seurat, batch_label, celltype_label, npcs, plotout_dir = working_dir, saveout_dir = working_dir, outfilename_prefix = outfile_prefix, visualize = visualize, save_obj = save_obj)

SaveH5Seurat(b_seurat, filename = "batches.h5Seurat")
Convert("batches.h5Seurat", dest = "batches.h5ad")


