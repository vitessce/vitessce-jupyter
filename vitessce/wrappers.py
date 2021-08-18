import os
from os.path import join
import tempfile
import shutil
import math
import json

import numpy as np
import pandas as pd
import zarr
from numcodecs import Zlib
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from generate_tiff_offsets import get_offsets

from starlette.responses import JSONResponse, UJSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from .constants import (
    CoordinationType as ct,
    Component as cm,
    DataType as dt,
    FileType as ft,
)
from .entities import Cells, CellSets, GenomicProfiles
from .routes import range_repsonse

VAR_CHUNK_SIZE = 10

class JsonRoute(Route):
    def __init__(self, path, endpoint, data_json):
        super().__init__(path, endpoint)
        self.data_json = data_json
    

class AbstractWrapper:
    """
    An abstract class that can be extended when
    implementing custom dataset object wrapper classes. 
    """

    def __init__(self, **kwargs):
        """
        Abstract constructor to be inherited by dataset wrapper classes.

        :param str out_dir: The path to a local directory used for data processing outputs. By default, uses a temp. directory.
        """
        self.out_dir = kwargs['out_dir'] if 'out_dir' in kwargs else tempfile.mkdtemp()
        self.routes = []
        self.is_remote = False
        self.file_def_creators = []

    def __repr__(self):
        return self._repr

    def convert_and_save(self, dataset_uid, obj_i):
        """
        Fill in the file_def_creators array.
        Each function added to this list should take in a base URL and generate a Vitessce file definition.
        If this wrapper is wrapping local data, then create routes and fill in the routes array.
        This method is void, should not return anything.

        :param str dataset_uid: A unique identifier for this dataset.
        :param int obj_i: Within the dataset, the index of this data wrapper object.
        """
        os.makedirs(self._get_out_dir(dataset_uid, obj_i), exist_ok=True)

    def get_routes(self):
        """
        Obtain the routes that have been created for this wrapper class.

        :returns: A list of server routes.
        :rtype: list[starlette.routing.Route]
        """
        return self.routes

    def get_file_defs(self, base_url):
        """
        Obtain the file definitions for this wrapper class.

        :param str base_url: A base URL to prepend to relative URLs.

        :returns: A list of file definitions.
        :rtype: list[dict]
        """
        file_defs_with_base_url = []
        for file_def_creator in self.file_def_creators:
            file_def = file_def_creator(base_url)
            if file_def is not None:
                file_defs_with_base_url.append(file_def)
        return file_defs_with_base_url
    
        
    def get_out_dir_route(self, dataset_uid, obj_i):
        """
        Obtain the Mount for the `out_dir`

        :param str dataset_uid: A dataset unique identifier for the Mount
        :param str obj_i: A index of the current vitessce.wrappers.AbstractWrapper among all other wrappers in the view config

        :returns: A starlette Mount of the the `out_dir`
        :rtype: list[starlette.routing.Mount]
        """
        if not self.is_remote:
            out_dir = self._get_out_dir(dataset_uid, obj_i)
            return [Mount(self._get_route_str(dataset_uid, obj_i),
                            app=StaticFiles(directory=out_dir, html=False))]
        return []

    def _get_url(self, base_url, dataset_uid, obj_i, *args):
        return base_url + self._get_route_str(dataset_uid, obj_i, *args)

    def _get_route_str(self, dataset_uid, obj_i, *args):
        return "/" + "/".join(map(str, [dataset_uid, obj_i, *args]))
    
    def _get_out_dir(self, dataset_uid, obj_i, *args):
        return join(self.out_dir, dataset_uid, str(obj_i), *args)

    def auto_view_config(self, vc):
        """
        Auto view configuration is intended to be used internally by the `VitessceConfig.from_object` method.
        Each subclass of `AbstractWrapper` may implement this method which takes in a `VitessceConfig` instance
        and modifies it by adding datasets, visualization components, and view coordinations.
        Implementations of this method may create an opinionated view config based on inferred use cases.

        :param vc: The view config instance.
        :type vc: VitessceConfig
        """
        raise NotImplementedError("Auto view configuration has not yet been implemented for this data object wrapper class.")


def _make_repr(init_locals):
    '''
    >>> orig = MultiImageWrapper('IMAGE_WRAPPERS', foo='bar')
    >>> orig_repr = repr(orig)
    >>> print(orig_repr)
    MultiImageWrapper(image_wrappers='IMAGE_WRAPPERS', use_physical_size_scaling=False, foo='bar')
    >>> evalled = eval(orig_repr)
    >>> assert orig_repr == repr(evalled)
    '''
    del init_locals['self']
    clazz = init_locals.pop('__class__')  # Requires superclass to be initialized.
    kwargs = init_locals.pop('kwargs')
    args = {
        **init_locals,
        **kwargs
    }
    params = ', '.join([f'{k}={repr(v)}' for k, v in args.items()])
    return f'{clazz.__name__}({params})'


class MultiImageWrapper(AbstractWrapper):
    """
    Wrap multiple imaging datasets by creating an instance of the ``MultiImageWrapper`` class.

    :param list image_wrappers: A list of imaging wrapper classes (only :class:`~vitessce.wrappers.OmeTiffWrapper` supported now)
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """
    def __init__(self, image_wrappers, use_physical_size_scaling=False, **kwargs):
        super().__init__(**kwargs)
        self._repr = _make_repr(locals())
        self.image_wrappers = image_wrappers
        self.use_physical_size_scaling = use_physical_size_scaling
    
    def convert_and_save(self, dataset_uid, obj_i):
        for image in self.image_wrappers:
            image.convert_and_save(dataset_uid, obj_i)
        file_def_creator = self.make_raster_file_def_creator(dataset_uid, obj_i)
        routes = self.make_raster_routes()
        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_raster_routes(self):
        obj_routes = []
        for num, image in enumerate(self.image_wrappers):
            obj_routes = obj_routes + image.get_routes()
        return obj_routes
    
    def make_raster_file_def_creator(self, dataset_uid, obj_i):

        def raster_file_def_creator(base_url):
            raster_json = {
                "schemaVersion": "0.0.2",
                "usePhysicalSizeScaling": self.use_physical_size_scaling,
                "images": [],
                "renderLayers": []
            }
            for image in self.image_wrappers:
                image_json = image.make_image_def(dataset_uid, obj_i, base_url)
                raster_json['images'].append(image_json)
                raster_json['renderLayers'].append(image.name)
            
            return {
                "type": dt.RASTER.value,
                "fileType": ft.RASTER_JSON.value,
                "options": raster_json
            }

        return raster_file_def_creator

class OmeTiffWrapper(AbstractWrapper):

    """
    Wrap an OME-TIFF File by creating an instance of the ``OmeTiffWrapper`` class.

    :param str img_path: A local filepath to an OME-TIFF file.
    :param str offsets_path: A local filepath to an offsets.json file.
    :param str img_url: A remote URL of an OME-TIFF file.
    :param str offsets_url: A remote URL of an offsets.json file.
    :param str name: The display name for this OME-TIFF within Vitessce.
    :param list[number] transformation_matrix: A column-major ordered matrix for transforming this image (see http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#homogeneous-coordinates for more information).
    :param bool is_bitmask: Whether or not this image is a bitmask.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, offsets_path=None, img_url=None, offsets_url=None, name="", transformation_matrix=None, is_bitmask=False,
     **kwargs):
        super().__init__(**kwargs)
        self._repr = _make_repr(locals())
        self.name = name
        self._img_path = img_path
        self._img_url = img_url
        self._offsets_url = offsets_url
        self._transformation_matrix = transformation_matrix
        self.is_remote = img_url is not None
        self.is_bitmask = is_bitmask
        if img_url is not None and (img_path is not None or offsets_path is not None):
            raise ValueError("Did not expect img_path or offsets_path to be provided with img_url")
    
    def convert_and_save(self, dataset_uid, obj_i):
        # Only create out-directory if needed
        if not self.is_remote:   
            super().convert_and_save(dataset_uid, obj_i)
        
        file_def_creator = self.make_raster_file_def_creator(dataset_uid, obj_i)
        routes = self.make_raster_routes(dataset_uid, obj_i)
        
        self.file_def_creators.append(file_def_creator)
        self.routes += routes
    
    def make_raster_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            offsets = get_offsets(self._img_path)
            async def response_func(req):
                return UJSONResponse(offsets)
            routes = [
                Route(self._get_route_str(dataset_uid, obj_i, self._get_img_filename()), lambda req: range_repsonse(req, self._img_path)),
                JsonRoute(self._get_route_str(dataset_uid, obj_i, self.get_offsets_path_name()), response_func, offsets)
            ]
            return routes
    
    def make_image_def(self, dataset_uid, obj_i, base_url):
        img_url = self.get_img_url(base_url, dataset_uid, obj_i)
        offsets_url = self.get_offsets_url(base_url, dataset_uid, obj_i)
        return self.create_image_json(img_url, offsets_url)

    def make_raster_file_def_creator(self, dataset_uid, obj_i):
        def raster_file_def_creator(base_url):
            raster_json = {
                "schemaVersion": "0.0.2",
                "images": [self.make_image_def(dataset_uid, obj_i, base_url)],
            }

            return {
                "type": dt.RASTER.value,
                "fileType": ft.RASTER_JSON.value,
                "options": raster_json
            }
        return raster_file_def_creator
    
    def create_image_json(self, img_url, offsets_url=None):
        metadata = {}
        image = {
            "name": self.name,
            "type": "ome-tiff",
            "url": img_url,
        }
        if offsets_url is not None:
            metadata["omeTiffOffsetsUrl"] = offsets_url
        if self._transformation_matrix is not None:
            metadata["transform"] = {
                "matrix": self._transformation_matrix
            }
        metadata["isBitmask"] = self.is_bitmask
        # Only attach metadata if there is some - otherwise schema validation fails.
        if len(metadata.keys()) > 0:
            image["metadata"] = metadata
        return image
    
    def _get_image_dir(self):
        return os.path.dirname(self._img_path)
    
    def _get_img_filename(self):
        return os.path.basename(self._img_path)

    def get_img_url(self, base_url="", dataset_uid="", obj_i=""):
        if self._img_url is not None:
            return self._img_url
        img_url = self._get_url(base_url, dataset_uid, obj_i, self._get_img_filename())
        return img_url

    def get_offsets_path_name(self):
        return f"{self._get_img_filename().split('ome.tif')[0]}offsets.json"
    
    def get_offsets_url(self, base_url="", dataset_uid="", obj_i=""):
        if self._offsets_url is not None or self._img_url is not None:
            return self._offsets_url
        offsets_url = self._get_url(base_url, dataset_uid, obj_i, self.get_offsets_path_name())
        return offsets_url


# class OmeZarrWrapper(AbstractWrapper):

#     def __init__(self, z, name="", **kwargs):
#         super().__init__(**kwargs)
#         self.z = z
#         self.name = name

#     def create_raster_json(self, img_url):
#         raster_json = {
#             "schemaVersion": "0.0.2",
#             "images": [
#                 {
#                     "name": self.name,
#                     "type": "zarr",
#                     "url": img_url,
#                     "metadata": {
#                         "dimensions": [
#                             {
#                                 "field": "channel",
#                                 "type": "nominal",
#                                 "values": [
#                                     "DAPI - Hoechst (nuclei)",
#                                     "FITC - Laminin (basement membrane)",
#                                     "Cy3 - Synaptopodin (glomerular)",
#                                     "Cy5 - THP (thick limb)"
#                                 ]
#                             },
#                             {
#                                 "field": "y",
#                                 "type": "quantitative",
#                                 "values": None
#                             },
#                             {
#                                 "field": "x",
#                                 "type": "quantitative",
#                                 "values": None
#                             }
#                         ],
#                         "isPyramid": True,
#                         "transform": {
#                             "scale": 1,
#                             "translate": {
#                                 "x": 0,
#                                 "y": 0,
#                             }
#                         }
#                     }
#                 }
#             ],
#         }
#         return raster_json

#     def get_raster(self, base_url, dataset_uid, obj_i):
#         obj_routes = []
#         obj_file_defs = []

#         if type(self.z) == zarr.hierarchy.Group:
#             img_dir_path = self.z.store.path

#             raster_json = self.create_raster_json(
#                 self._get_url(base_url, dataset_uid, obj_i, "raster_img"),
#             )

#             obj_routes = [
#                 Mount(self._get_route_str(dataset_uid, obj_i, "raster_img"),
#                         app=StaticFiles(directory=img_dir_path, html=False)),
#                 JsonRoute(self._get_route_str(dataset_uid, obj_i, "raster"),
#                         self._create_response_json(raster_json), raster_json)
#             ]
#             obj_file_defs = [
#                 {
#                     "type": dt.RASTER.value,
#                     "fileType": ft.RASTER_JSON.value,
#                     "url": self._get_url(base_url, dataset_uid, obj_i, "raster")
#                 }
#             ]

#         return obj_file_defs, obj_routes


class AnnDataWrapper(AbstractWrapper):
    def __init__(self, adata=None, adata_url=None, expression_matrix=None, matrix_gene_var_filter=None, gene_var_filter=None, cell_set_obs=None, cell_set_obs_names=None, spatial_centroid_obsm=None, spatial_polygon_obsm=None, mappings_obsm=None, mappings_obsm_names=None, mappings_obsm_dims=None, request_init=None, factors_obs=None, **kwargs):
        """
        Wrap an AnnData object by creating an instance of the ``AnnDataWrapper`` class.

        :param adata: An AnnData object containing single-cell experiment data.
        :type adata: anndata.AnnData
        :param str adata_url: A remote url pointing to a zarr-backed AnnData store.
        :param str expression_matrix: Location of the expression (cell x gene) matrix, like `X` or `obsm/highly_variable_genes_subset`
        :param str gene_var_filter: A string like `highly_variable` (from `var` in the AnnData stored) used in conjunction with expression_matrix if expression_matrix points to a subset of `X` of the full `var` list.
        :param str matrix_gene_var_filter: A string like `highly_variable` (from `var` in the AnnData stored) used in conjunction with expression_matrix if expression_matrix points to a subset of `X` of the full `var` list.
        :param list[str] factors_obs: Column names like `['top_marker_gene', 'sex']` for showing factors when cells are hovered over
        :param list[str] cell_set_obs: Column names like `['louvain', 'cellType']` for showing cell sets from `obs`
        :param list[str] cell_set_obs_names: Names to display in place of those in `cell_set_obs`, like `['Louvain', 'Cell Type']
        :param str spatial_centroid_obsm: Column name in `obsm` that contains centroid coordinates for displaying centroids in the spatial viewer
        :param str spatial_polygon_obsm: Column name in `obsm` that contains polygonal coordinates for displaying outlines in the spatial viewer
        :param list[str] mappings_obsm: Column names like `['X_umap', 'X_pca']` for showing scatterplots from `obsm`
        :param list[str] mappings_obsm_names: Overriding names like `['UMAP', 'PCA'] for displaying above scatterplots
        :param list[str] mappings_obsm_dims: Dimensions along which to get data for the scatterplot, like [[0, 1], [4, 5]] where [0, 1] is just the normal x and y but [4, 5] could be comparing the third and fourth principal components, for example.
        :param dict request_init: options to be passed along with every fetch request from the browser, like { "header": { "Authorization": "Bearer dsfjalsdfa1431" } }
        :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
        """
        super().__init__(**kwargs)
        self._repr = _make_repr(locals())
        self._adata = adata
        self._adata_url = adata_url
        if adata is not None:
            self.is_remote = False
            self.zarr_folder = 'anndata.zarr'
        else:
            self.is_remote = True
            self.zarr_folder = None
        self._expression_matrix = expression_matrix
        self._cell_set_obs_names = cell_set_obs_names
        self._mappings_obsm_names = mappings_obsm_names
        self._gene_var_filter = "var/" + gene_var_filter if gene_var_filter is not None else gene_var_filter
        self._matrix_gene_var_filter = "var/" + matrix_gene_var_filter if matrix_gene_var_filter is not None else matrix_gene_var_filter
        self._cell_set_obs = ["obs/" + i for i in cell_set_obs] if cell_set_obs is not None else cell_set_obs
        self._factors_obs = ["obs/" + i for i in factors_obs] if factors_obs is not None else factors_obs
        self._spatial_centroid_obsm = "obsm/" + spatial_centroid_obsm if spatial_centroid_obsm is not None else spatial_centroid_obsm
        self._spatial_polygon_obsm = "obsm/" + spatial_polygon_obsm if spatial_polygon_obsm is not None else spatial_polygon_obsm
        self._mappings_obsm = ["obsm/" + i for i in mappings_obsm] if mappings_obsm is not None else mappings_obsm
        self._mappings_obsm_dims = mappings_obsm_dims
        self._request_init = request_init

    def convert_and_save(self, dataset_uid, obj_i):
        # Only create out-directory if needed
        if not self.is_remote:   
            super().convert_and_save(dataset_uid, obj_i)
            zarr_filepath = self.get_zarr_path(dataset_uid, obj_i)
            # In the future, we can use sparse matrices with equal performance:
            # https://github.com/theislab/anndata/issues/524
            if isinstance(self._adata.X, sparse.spmatrix):
                self._adata.X = self._adata.X.todense()
            self._adata.write_zarr(zarr_filepath, chunks=[self._adata.shape[0], VAR_CHUNK_SIZE])
        
        cells_file_creator = self.make_cells_file_def_creator(dataset_uid, obj_i)
        cell_sets_file_creator = self.make_cell_sets_file_def_creator(dataset_uid, obj_i)
        expression_matrix_file_creator = self.make_expression_matrix_file_def_creator(dataset_uid, obj_i)
        
        self.file_def_creators += [cells_file_creator, cell_sets_file_creator, expression_matrix_file_creator] 
        self.routes += self.get_out_dir_route(dataset_uid, obj_i)
    
    def get_zarr_path(self, dataset_uid, obj_i):
        out_dir = self._get_out_dir(dataset_uid, obj_i)
        zarr_filepath = join(out_dir, self.zarr_folder)
        return zarr_filepath
        

    def get_zarr_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._adata_url
        else:
            return self._get_url(base_url, dataset_uid, obj_i, self.zarr_folder)
    
    def make_cells_file_def_creator(self, dataset_uid, obj_i):
        def get_cells(base_url):
            options = {}
            if self._spatial_centroid_obsm is not None:
                options["xy"] = self._spatial_centroid_obsm
            if self._spatial_polygon_obsm is not None:
                options["poly"] = self._spatial_polygon_obsm
            if self._mappings_obsm is not None:
                options["mappings"] = {}
                if self._mappings_obsm_names is not None:
                    for key, mapping in zip(self._mappings_obsm_names, self._mappings_obsm): 
                        options["mappings"][key] = {
                            "key": mapping,
                            "dims": [0, 1]
                        }
                else:
                    for mapping in self._mappings_obsm:
                        mapping_key = mapping.split('/')[-1]
                        self._mappings_obsm_names = mapping_key
                        options["mappings"][mapping_key] = {
                            "key": mapping,
                            "dims": [0, 1]
                        }
                if self._mappings_obsm_dims is not None:
                    for dim, key in zip(self._mappings_obsm_dims, self._mappings_obsm_names):
                        options["mappings"][key]['dims'] = dim
            if self._factors_obs is not None:
                options["factors"] = []
                for obs in self._factors_obs:
                    options["factors"].append(obs)
            if len(options.keys()) > 0:
                obj_file_def = {
                    "type": dt.CELLS.value,
                    "fileType": ft.ANNDATA_CELLS_ZARR.value,
                    "url": self.get_zarr_url(base_url, dataset_uid, obj_i),
                    "options": options
                }
                if self._request_init is not None:
                    obj_file_def['requestInit'] = self._request_init
                return obj_file_def
            return None
        return get_cells

    def make_cell_sets_file_def_creator(self, dataset_uid, obj_i):
        def get_cell_sets(base_url):
            if self._cell_set_obs is not None:
                options = []
                if self._cell_set_obs_names is not None:
                    names = self._cell_set_obs_names
                else:
                    names = [obs.split('/')[-1] for obs in self._cell_set_obs]
                for obs, name in zip(self._cell_set_obs, names):
                    options.append({
                        "groupName": name,
                        "setName": obs
                    })

                obj_file_def = {
                    "type": dt.CELL_SETS.value,
                    "fileType": ft.ANNDATA_CELL_SETS_ZARR.value,
                    "url": self.get_zarr_url(base_url, dataset_uid, obj_i),
                    "options": options
                }
                if self._request_init is not None:
                    obj_file_def['requestInit'] = self._request_init
                
                return obj_file_def
            return None
        return get_cell_sets

    def make_expression_matrix_file_def_creator(self, dataset_uid, obj_i):
        def get_expression_matrix(base_url):
            options = {}
            if self._expression_matrix is not None:
                options["matrix"] = self._expression_matrix
                if self._gene_var_filter is not None:
                    options["geneFilter"] = self._gene_var_filter
                if self._matrix_gene_var_filter is not None:
                    options["matrixGeneFilter"] = self._matrix_gene_var_filter
                obj_file_def = {
                    "type": dt.EXPRESSION_MATRIX.value,
                    "fileType": ft.ANNDATA_EXPRESSION_MATRIX_ZARR.value,
                    "url": self.get_zarr_url(base_url, dataset_uid, obj_i),
                    "options": options
                }
                if self._request_init is not None:
                    obj_file_def['requestInit'] = self._request_init
                
                return obj_file_def
            return None
        return get_expression_matrix
    
    def auto_view_config(self, vc):
        dataset = vc.add_dataset().add_object(self)
        mapping_name = self._mappings_obsm_names[0] if (self._mappings_obsm_names is not None) else self._mappings_obsm[0].split('/')[-1]
        scatterplot = vc.add_view(dataset, cm.SCATTERPLOT, mapping=mapping_name)
        cell_sets = vc.add_view(dataset, cm.CELL_SETS)
        genes = vc.add_view(dataset, cm.GENES)
        heatmap = vc.add_view(dataset, cm.HEATMAP)
        if self._spatial_polygon_obsm  is not None or self._spatial_centroid_obsm is not None:
            spatial = vc.add_view(dataset, cm.SPATIAL)
            vc.layout((scatterplot | spatial) / (heatmap | (cell_sets / genes)))
        else:
            vc.layout((scatterplot | (cell_sets / genes)) / heatmap)

class SnapWrapper(AbstractWrapper):

    # The Snap file is difficult to work with.
    # For now we can use the processed cell-by-bin MTX file
    # However, the HuBMAP pipeline currently computes this with resolution 5000
    # TODO: Make a PR to sc-atac-seq-pipeline to output this at a higher resolution (e.g. 200)
    # https://github.com/hubmapconsortium/sc-atac-seq-pipeline/blob/develop/bin/snapAnalysis.R#L93

    def __init__(self, in_mtx, in_barcodes_df, in_bins_df, in_clusters_df, starting_resolution=5000, **kwargs):
        super().__init__(**kwargs)
        self._repr = _make_repr(locals())
        self.in_mtx = in_mtx # scipy.sparse.coo.coo_matrix (filtered_cell_by_bin.mtx)
        self.in_barcodes_df = in_barcodes_df # pandas dataframe (barcodes.txt)
        self.in_bins_df = in_bins_df # pandas dataframe (bins.txt)
        self.in_clusters_df = in_clusters_df # pandas dataframe (umap_coords_clusters.csv)
        self.zarr_folder = 'profiles.zarr'

        self.starting_resolution = starting_resolution

        # Convert to dense matrix if sparse.
        if type(in_mtx) == coo_matrix:
            self.in_mtx = in_mtx.toarray()
    
    def convert_and_save(self, dataset_uid, obj_i):
        super().convert_and_save(dataset_uid, obj_i)
        out_dir = self._get_out_dir(dataset_uid, obj_i)
        zarr_filepath = join(out_dir, self.zarr_folder)

        self.create_genomic_multivec_zarr(zarr_filepath)
        with open(join(out_dir, 'cell-sets'), 'w') as f:
            f.write(json.dumps(self.create_cell_sets_json()))
        with open(join(out_dir, 'cells'), 'w') as f:
            f.write(json.dumps(self.create_cells_json()))
        
        cells_file_creator = self.make_cells_file_def_creator(dataset_uid, obj_i)
        cell_sets_file_creator = self.make_cell_sets_file_def_creator(dataset_uid, obj_i)
        genomic_profiles_file_creator = self.make_genomic_profiles_file_def_creator(dataset_uid, obj_i)
        
        self.file_def_creators += [cells_file_creator, cell_sets_file_creator, genomic_profiles_file_creator] 
        self.routes += self.get_out_dir_route(dataset_uid, obj_i)
    
    def create_genomic_multivec_zarr(self, zarr_filepath):
        in_mtx = self.in_mtx
        in_clusters_df = self.in_clusters_df
        in_barcodes_df = self.in_barcodes_df
        in_bins_df = self.in_bins_df

        starting_resolution = self.starting_resolution

        # The bin datafram consists of one column like chrName:binStart-binEnd
        def convert_bin_name_to_chr_name(bin_name):
            try:
                return bin_name[:bin_name.index(':')]
            except ValueError:
                return np.nan
        def convert_bin_name_to_chr_start(bin_name):
            try:
                return int(bin_name[bin_name.index(':')+1:bin_name.index('-')])
            except ValueError:
                return np.nan
        def convert_bin_name_to_chr_end(bin_name):
            try:
                return int(bin_name[bin_name.index('-')+1:])
            except ValueError:
                return np.nan
        
        # The genome assembly is GRCh38 but the chromosome names in the bin names do not start with the "chr" prefix.
        # This is incompatible with the chromosome names from `negspy`, so we need to append the prefix.
        in_bins_df[0] = in_bins_df[0].apply(lambda x: "chr" + x)
        
        in_bins_df["chr_name"] = in_bins_df[0].apply(convert_bin_name_to_chr_name)
        in_bins_df["chr_start"] = in_bins_df[0].apply(convert_bin_name_to_chr_start)
        in_bins_df["chr_end"] = in_bins_df[0].apply(convert_bin_name_to_chr_end)

        # Drop any rows that had incorrect bin strings (missing a chromosome name, bin start, or bin end value).
        in_bins_df = in_bins_df.dropna(subset=["chr_name", "chr_start", "chr_end"]).copy()

        # Ensure that the columns have the expect types.
        in_bins_df["chr_name"] = in_bins_df["chr_name"].astype(str)
        in_bins_df["chr_start"] = in_bins_df["chr_start"].astype(int)
        in_bins_df["chr_end"] = in_bins_df["chr_end"].astype(int)

        # Create the Zarr store for the outputs.
        out_f = zarr.open(zarr_filepath, mode='w')

        # Get a list of clusters.
        in_clusters_df["cluster"] = in_clusters_df["cluster"].astype(str)
        cluster_ids = in_clusters_df["cluster"].unique().tolist()
        cluster_ids.sort(key=int)

        cluster_paths = [ [ "Clusters", cluster_id ] for cluster_id in cluster_ids ]
        
        # "SnapTools performs quantification using a specified aligner, and HuBMAP has standardized on BWA with the GRCh38 reference genome"
        # Reference: https://github.com/hubmapconsortium/sc-atac-seq-pipeline/blob/bb023f95ca3330128bfef41cc719ffcb2ee6a190/README.md
        genomic_profiles = GenomicProfiles(out_f, profile_paths=cluster_paths, assembly='hg38', starting_resolution=starting_resolution)
        chrom_name_to_length = genomic_profiles.chrom_name_to_length

        # Create each chromosome dataset.
        for chr_name, chr_len in chrom_name_to_length.items():
            # The bins dataframe frustratingly does not contain every bin.
            # We need to figure out which bins are missing.

            # We want to check for missing bins in each chromosome separately,
            # otherwise too much memory is used during the join step.
            chr_bins_in_df = in_bins_df.loc[in_bins_df["chr_name"] == chr_name]
            if chr_bins_in_df.shape[0] == 0:
                # No processing or output is necessary if there is no data for this chromosome.
                # Continue on through all resolutions of this chromosome to the next chromosome.
                continue
            
            # Determine the indices of the matrix at which the bins for this chromosome start and end.
            chr_bin_i_start = int(chr_bins_in_df.head(1).iloc[0].name)
            chr_bin_i_end = int(chr_bins_in_df.tail(1).iloc[0].name) + 1
            
            # Extract the part of the matrix corresponding to the current chromosome.
            chr_mtx = in_mtx[:,chr_bin_i_start:chr_bin_i_end]

            # Create a list of the "ground truth" bins (all bins from position 0 to the end of the chromosome).
            # We will join the input bins onto this dataframe to determine which bins are missing.
            chr_bins_gt_df = pd.DataFrame()
            chr_bins_gt_df["chr_start"] = np.arange(0, math.ceil(chr_len/starting_resolution)) * starting_resolution
            chr_bins_gt_df["chr_end"] = chr_bins_gt_df["chr_start"] + starting_resolution
            chr_bins_gt_df["chr_start"] = chr_bins_gt_df["chr_start"] + 1
            chr_bins_gt_df["chr_start"] = chr_bins_gt_df["chr_start"].astype(int)
            chr_bins_gt_df["chr_end"] = chr_bins_gt_df["chr_end"].astype(int)
            chr_bins_gt_df["chr_name"] = chr_name
            chr_bins_gt_df[0] = chr_bins_gt_df.apply(lambda r: f"{r['chr_name']}:{r['chr_start']}-{r['chr_end']}", axis='columns')
            
            # We will add a new column "i", which should match the _old_ index, so that we will be able join with the data matrix on the original indices.
            # For the new rows, we will add values for the "i" column that are greater than any of the original indices,
            # to prevent any joining with the incoming data matrix onto these bins for which the data is missing.
            chr_bins_in_df = chr_bins_in_df.reset_index(drop=True)
            chr_bins_in_df["i"] = chr_bins_in_df.index.values
            chr_bins_gt_df["i"] = chr_bins_gt_df.index.values + (in_mtx.shape[1] + 1)
            
            # Set the full bin string column as the index of both data frames.
            chr_bins_gt_df = chr_bins_gt_df.set_index(0)
            chr_bins_in_df = chr_bins_in_df.set_index(0)
            
            # Join the input bin subset dataframe right onto the full bin ground truth dataframe.
            chr_bins_in_join_df = chr_bins_in_df.join(chr_bins_gt_df, how='right', lsuffix="", rsuffix="_gt")
            # The bins which were not present in the input will have NaN values in the "i" column.
            # For these rows, we replace the NaN values with the much higher "i_gt" values which will not match to any index of the data matrix.
            chr_bins_in_join_df["i"] = chr_bins_in_join_df.apply(lambda r: r['i'] if pd.notna(r['i']) else r['i_gt'], axis='columns').astype(int)

            # Clean up the joined data frame by removing unnecessary columns.
            chr_bins_in_join_df = chr_bins_in_join_df.drop(columns=['chr_name', 'chr_start', 'chr_end', 'i_gt'])
            chr_bins_in_join_df = chr_bins_in_join_df.rename(columns={'chr_name_gt': 'chr_name', 'chr_start_gt': 'chr_start', 'chr_end_gt': 'chr_end'})
            
            # Create a dataframe from the data matrix, so that we can join to the joined bins dataframe.
            chr_mtx_df = pd.DataFrame(data=chr_mtx.T)
            
            chr_bins_i_df = chr_bins_in_join_df.drop(columns=['chr_name', 'chr_start', 'chr_end'])

            # Join the data matrix dataframe and the bins dataframe.
            # Bins that are missing from the data matrix will have "i" values higher than any of the data matrix dataframe row indices,
            # and therefore the data values for these bins in the resulting joined dataframe will all be NaN.
            chr_mtx_join_df = chr_bins_i_df.join(chr_mtx_df, how='left', on='i')
            # We fill in these NaN values with 0.
            chr_mtx_join_df = chr_mtx_join_df.fillna(value=0.0)
            
            # Drop the "i" column, since it is not necessary now that we have done the join.
            chr_mtx_join_df = chr_mtx_join_df.drop(columns=['i'])
            # Obtain the new full data matrix, which contains values for all bins of the chromosome.
            chr_mtx = chr_mtx_join_df.values.T

            # Fill in the Zarr store with data for each cluster.
            for cluster_index, cluster_id in enumerate(cluster_ids):
                # Get the list of cells in the current cluster.
                cluster_df = in_clusters_df.loc[in_clusters_df["cluster"] == cluster_id]
                cluster_cell_ids = cluster_df.index.values.tolist()
                cluster_num_cells = len(cluster_cell_ids)
                cluster_cells_tf = (in_barcodes_df[0].isin(cluster_cell_ids)).values

                # Get the rows of the data matrix corresponding to the cells in this cluster.
                cluster_cell_by_bin_mtx = chr_mtx[cluster_cells_tf,:]
                # Take the sum of this cluster along the cells axis.
                cluster_profile = cluster_cell_by_bin_mtx.sum(axis=0)

                genomic_profiles.add_profile(cluster_profile, chr_name, cluster_index)
        
        return
    
    def make_genomic_profiles_file_def_creator(self, dataset_uid, obj_i):
        def get_genomic_profiles(base_url):
            return {
                "type": dt.GENOMIC_PROFILES.value,
                "fileType": ft.GENOMIC_PROFILES_ZARR.value,
                "url": self._get_url(base_url, dataset_uid, obj_i, self.zarr_folder)
            }
        
        return get_genomic_profiles
    

    def create_cell_sets_json(self):
        in_clusters_df = self.in_clusters_df
        cell_sets = CellSets()
        cell_sets.add_level_zero_node('Clusters')

        cell_ids = in_clusters_df.index.values.tolist()
        in_clusters_df['cluster'] = in_clusters_df['cluster'].astype(str)
        cluster_ids = in_clusters_df['cluster'].unique().tolist()
        cluster_ids.sort(key=int)
        cell_cluster_ids = in_clusters_df['cluster'].values.tolist()

        cell_cluster_tuples = list(zip(cell_ids, cell_cluster_ids))

        for cluster_id in cluster_ids:
            cell_set = [
                str(cell_id)
                for cell_id, cell_cluster_id in cell_cluster_tuples
                if cell_cluster_id == cluster_id
            ]
            cell_sets.add_node(str(cluster_id), ['Clusters'], cell_set)

        return cell_sets.json
    
    def make_cell_sets_file_def_creator(self, dataset_uid, obj_i):
        def get_cell_sets(base_url):
            return {
                "type": dt.CELL_SETS.value,
                "fileType": ft.CELL_SETS_JSON.value,
                "url": self._get_url(base_url, dataset_uid, obj_i, "cell-sets")
            }
        return get_cell_sets
    
    def create_cells_json(self):
        in_clusters_df = self.in_clusters_df

        cell_ids = in_clusters_df.index.tolist()
        cells = Cells(cell_ids=cell_ids)
        mapping = in_clusters_df[["umap.1", "umap.2"]].values.tolist()
        cells.add_mapping("UMAP", mapping)
        return cells.json
    
    def make_cells_file_def_creator(self, dataset_uid, obj_i):
        def get_cells(base_url):

            return {
                "type": dt.CELLS.value,
                "fileType": ft.CELLS_JSON.value,
                "url": self._get_url(base_url, dataset_uid, obj_i, "cells")
            }
        return get_cells

    def auto_view_config(self, vc):
        dataset = vc.add_dataset().add_object(self)
        genomic_profiles = vc.add_view(dataset, cm.GENOMIC_PROFILES)
        scatter = vc.add_view(dataset, cm.SCATTERPLOT, mapping = "UMAP")
        cell_sets = vc.add_view(dataset, cm.CELL_SETS)

        vc.layout(genomic_profiles / (scatter | cell_sets))
