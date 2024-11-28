import argparse
from anndata import read_h5ad
import scipy
import numpy as np
import pandas as pd
import platform
import os
import zarr
import math

def convert_h5ad_to_zarr(input_path, output_path):
    adata = read_h5ad(input_path)

    # Clear X so that we can write it ourselves manually
    X = adata.X #.copy()
    
    adata.X = None
    adata.write_zarr(output_path)

    assert isinstance(X, scipy.sparse.spmatrix)

    print(output_path)

    store = zarr.DirectoryStore(output_path)
    z = zarr.zeros(shape=X.shape, chunks=(X.shape[0], 10), dtype=X.dtype, store = store, path = "/X", overwrite=True)

    chunk_shape = (10000, 10000)
    x_chunks = math.ceil(X.shape[0] / chunk_shape[0])
    y_chunks = math.ceil(X.shape[1] / chunk_shape[1])


    for i in range(x_chunks):
        for j in range(y_chunks):
            x_start = i * chunk_shape[0]
            x_end = min((i + 1) * chunk_shape[0], X.shape[0])
            y_start = j * chunk_shape[1]
            y_end = min((j + 1) * chunk_shape[1], X.shape[1])

            X_chunk = X[x_start:x_end, y_start:y_end].tocoo(copy=False)
            z.set_coordinate_selection(
                # Add x_start and y_start as offsets to the row/chunk coordinates
                ([cx+x_start for cx in X_chunk.row], [cy+y_start for cy in X_chunk.col]),
                X_chunk.data
            )

    print("done")

if __name__ == '__main__':
    # Argparse
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
        args.output,
    )
