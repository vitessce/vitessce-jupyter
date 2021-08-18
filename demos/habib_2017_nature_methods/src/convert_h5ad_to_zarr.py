import argparse

import json
from anndata import read_h5ad

def convert_h5ad_to_zarr(input_path, output_path):
    adata = read_h5ad(input_path)
    # Store an additional expression matrix with only the highly variable genes.
    adata.obsm['X_hvg'] = adata[:, adata.var['highly_variable']].X.copy()
    adata.write_zarr(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help='Input H5AD file'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='Output Zarr store'
    )
    args = parser.parse_args()
    convert_h5ad_to_zarr(
        args.input,
        args.output
    )
