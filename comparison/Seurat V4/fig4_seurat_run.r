library(Seurat)
library(magrittr)
library(cowplot)
library(SeuratDisk)

normData = T
Datascaling = T
regressUMI = F
min_cells = 0
min_genes = 0
norm_method = "LogNormalize"
scale_factor = 10000
numVG = 300
nhvg = 2000
npcs = 50
visualize = T
outfile_prefix = "Dataset5"
save_obj = F
src_dir = "./"
working_dir = "./"


batch_label = "batch"
celltype_label = "celltype"

Convert("fig4.h5ad", dest = "./fig4.h5seurat")
fig4data <- LoadH5Seurat("fig4.h5seurat") 
metadata = fig4data[[]]
expr_mat = GetAssayData(object = fig4data)


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

