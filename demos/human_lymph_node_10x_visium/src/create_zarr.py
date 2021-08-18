import argparse

from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.cluster

def create_zarr(output_path):
    adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node", include_hires_tiff=True)

    # Reference: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
    # Calculate QC metrics
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Perform basic filtering
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    # Perform normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    # Determine the top 2000 highly variable genes.
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=300)

    # Dimensionality reduction and clustering
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters")


    # Hierarchical clustering of genes for optimal gene ordering
    X_hvg_arr = adata[:, adata.var['highly_variable']].X.toarray()
    X_hvg_index = adata[:, adata.var['highly_variable']].var.copy().index

    Z = scipy.cluster.hierarchy.linkage(X_hvg_arr.T, method="average", optimal_ordering=True)

    # Get the hierarchy-based ordering of genes.
    num_genes = adata.var.shape[0]
    highly_var_index_ordering = scipy.cluster.hierarchy.leaves_list(Z)
    highly_var_genes = X_hvg_index.values[highly_var_index_ordering].tolist()

    all_genes = adata.var.index.values.tolist()
    not_var_genes = adata.var.loc[~adata.var['highly_variable']].index.values.tolist()

    def get_orig_index(gene_id):
        return all_genes.index(gene_id)
    
    var_index_ordering = list(map(get_orig_index, highly_var_genes)) + list(map(get_orig_index, not_var_genes))

    # Create a new *ordered* gene expression dataframe.
    adata = adata[:, var_index_ordering].copy()
    adata.obsm["X_hvg"] = adata[:, adata.var['highly_variable']].X.copy()

    adata.obsm['spatial'] = adata.obsm['spatial'].astype('uint16')
    adata.obsm['xy'] = np.stack((adata.obs['array_col'].values, adata.obs['array_row'].values), axis=-1).astype('uint16')

    # Need to convert images from interleaved to non-interleaved (color axis should be first).
    img_hires = adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires']
    img_lowres = adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowres']

    adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'] = np.transpose(img_hires, (2, 0, 1))
    adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowres'] = np.transpose(img_lowres, (2, 0, 1))

    adata.write_zarr(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='Output Zarr store'
    )
    args = parser.parse_args()
    create_zarr(
        args.output
    )
