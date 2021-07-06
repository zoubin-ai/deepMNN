
library(Seurat)
library(magrittr)
library(cowplot)
library(SeuratDisk)
normData = T
Datascaling = T
regressUMI = F
min_cells = 10
min_genes = 300
norm_method = "LogNormalize"
scale_factor = 10000
numVG = 300
nhvg = 2000
npcs = 50 ##20
visualize = T
outfile_prefix = "Dataset4"
save_obj = F

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


source('call_seurat_3.R')
batch_list = seurat3_preprocess(
                expr_mat, metadata,
                normData = normData, Datascaling = Datascaling, regressUMI = regressUMI,
                min_cells = min_cells, min_genes = min_genes,
                norm_method = norm_method, scale_factor = scale_factor,
                numVG = numVG, nhvg = nhvg,
                batch_label = batch_label, celltype_label = celltype_label)
batches = call_seurat3(batch_list, batch_label, celltype_label, npcs, plotout_dir = working_dir, saveout_dir = working_dir, outfilename_prefix = outfile_prefix, visualize = visualize, save_obj = save_obj)

SaveH5Seurat(batches, filename = "batches.h5Seurat")
Convert("batches.h5Seurat", dest = "batches.h5ad")
