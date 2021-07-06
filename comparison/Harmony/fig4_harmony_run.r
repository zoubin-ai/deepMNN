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

batch_label = "batch"
celltype_label = "celltype"
fig4data <- LoadH5Seurat("fig4.h5seurat")
metadata = fig4data[[]]
expr_mat = GetAssayData(object = fig4data)

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


