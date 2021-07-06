
seurat3_preprocess <- function(expr_mat_seurat, metadata, 
                              normData = T, Datascaling = T, regressUMI = F, 
                              min_cells = 10, min_genes = 300, 
                              norm_method = "LogNormalize", 
                              scale_factor = 10000, 
                              numVG = 300, nhvg = 2000, 
                              batch_label = "batchlb", celltype_label = "CellType",
                              hvg = T)
{

  ##########################################################
  # preprocessing

  batches <- CreateSeuratObject(expr_mat_seurat, meta.data = metadata, 
                                min.cells = min_cells, min.features = min_genes, project = "Seurat3_benchmark")

  batch_list <- SplitObject(batches, split.by = batch_label)
  for(i in 1:length(batch_list)) {
    if (normData) {
      batch_list[[i]] <- NormalizeData(batch_list[[i]],normalization.method = norm_method, scale.factor = scale_factor)
    }
    if(hvg){
      batch_list[[i]] <- FindVariableFeatures(batch_list[[i]], selection.method = "vst", nfeatures = nhvg, 
                                              verbose = FALSE)
      #print(dim(batch_list[[i]]))
    } else {
      batch_list[[i]]@assays$RNA@var.features <- rownames(expr_mat_seurat)
    }
  }
  return(batch_list)
}

call_seurat3 <- function(batch_list, batch_label, celltype_label, npcs = 20, 
                         plotout_dir = "", saveout_dir = "", 
                         outfilename_prefix = "", 
                         visualize = T, save_obj = T)
{

  #plotting
  k_seed = 10
  tsne_perplex = 30
  tsneplot_filename = '_seurat3_tsne'
  umapplot_filename = '_seurat3_umap'
  obj_filename = "_seurat3_sobj"
  cca_filename = "_seurat3_cca"
  pca_filename = "_seurat3_pca"

  ##########################################################
  #run

  t1 = Sys.time()

  cell_anchors <- FindIntegrationAnchors(object.list = batch_list, dims = 1:npcs)
  batches <- IntegrateData(anchorset = cell_anchors, dims = 1:npcs)
  dim(batches)

  t2 = Sys.time()
  print(t2-t1)

  DefaultAssay(batches) <- "integrated"

  if(regressUMI && Datascaling) {
    batches <- ScaleData(object = batches, vars.to.regress = c("nUMI"))  # in case of read count data
  } else if (Datascaling) { # default option
    batches <- ScaleData(object = batches)
  }

  ##########################################################
  #post processing and save

  batches <- RunPCA(object = batches, npcs = npcs, verbose = FALSE)

  seurat3_res <- as.data.frame(batches@reductions$pca@cell.embeddings)
  cells_use <- rownames(seurat3_res)
  seurat3_res$batchlb <- batches@meta.data[, batch_label]
#  seurat3_res$batch <- batches@meta.data[, ]
  seurat3_res$celltype <- batches@meta.data[, celltype_label]
  write.table(seurat3_res, file=paste0(saveout_dir,outfilename_prefix,pca_filename,".txt"), quote=F, sep='\t', row.names = T, col.names = NA)

  if(save_obj) {
    saveRDS(batches, file=paste0(saveout_dir,outfilename_prefix,obj_filename,".RDS"))
  }
  saveRDS(batches, file="save_batches.RDS")
  if (visualize) {

    ##########################################################
    #preparing plots

    batches <- RunTSNE(batches, reduction = "pca", dims = 1:npcs, do.fast = T, k.seed = k_seed, check_duplicates = FALSE, perplexity = tsne_perplex)
    batches <- RunUMAP(batches, reduction = "pca", dims = 1:npcs, do.fast = T, k.seed = k_seed)

    ##########################################################
    #tSNE plot
    p11 <- DimPlot(object = batches, reduction = 'tsne', group.by = batch_label)
    p12 <- DimPlot(object = batches, reduction = 'tsne', group.by = celltype_label)

    png(paste0(plotout_dir,outfilename_prefix,tsneplot_filename,".png"),width = 2*1000, height = 800, res = 2*72)
    print(plot_grid(p11, p12))
    dev.off()

    pdf(paste0(plotout_dir,outfilename_prefix,tsneplot_filename,".pdf"),width=15,height=7,paper='special') 
    print(plot_grid(p11, p12))
    dev.off()

    ##########################################################
    #UMAP plot

    p21 <- DimPlot(object = batches, reduction = 'umap', group.by = batch_label)
    p22 <- DimPlot(object = batches, reduction = 'umap', group.by = celltype_label)

    png(paste0(plotout_dir,outfilename_prefix,umapplot_filename,".png"),width = 2*1000, height = 800, res = 2*72)
    print(plot_grid(p21, p22))
    dev.off()

    pdf(paste0(plotout_dir,outfilename_prefix,umapplot_filename,".pdf"),width=15,height=7,paper='special') 
    print(plot_grid(p21, p22))
    dev.off()
  }

  return(batches)
}







