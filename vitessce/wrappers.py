import os
from os.path import join
import tempfile
from uuid import uuid4
from pathlib import PurePath, PurePosixPath

from .constants import (
    ViewType as cm,
    FileType as ft,
)
from .repr import make_repr


def file_path_to_url_path(local_path, prepend_slash=True, path_class=None):
    # force_windows is used in tests
    url_path = str(PurePosixPath(PurePath(local_path) if path_class is None else path_class(local_path)))
    if prepend_slash and not url_path.startswith("/"):
        url_path = f"/{url_path}"
    return url_path


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
        self.out_dir = kwargs['out_dir'] if 'out_dir' in kwargs else tempfile.mkdtemp(
        )
        self.routes = []
        self.is_remote = False
        self.file_def_creators = []
        self.base_dir = None

    def __repr__(self):
        return self._repr

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        """
        Fill in the file_def_creators array.
        Each function added to this list should take in a base URL and generate a Vitessce file definition.
        If this wrapper is wrapping local data, then create routes and fill in the routes array.
        This method is void, should not return anything.

        :param str dataset_uid: A unique identifier for this dataset.
        :param int obj_i: Within the dataset, the index of this data wrapper object.
        """
        os.makedirs(self._get_out_dir(dataset_uid, obj_i), exist_ok=True)
        self.base_dir = base_dir

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
            # TODO: Move imports back to top when this is factored out.
            from starlette.staticfiles import StaticFiles
            from starlette.routing import Mount
            return [Mount(self._get_route_str(dataset_uid, obj_i),
                          app=StaticFiles(directory=out_dir, html=False))]
        return []

    def get_local_dir_url(self, base_url, dataset_uid, obj_i, local_dir_path, local_dir_uid):
        if not self.is_remote and self.base_dir is not None:
            return self._get_url_simple(base_url, file_path_to_url_path(local_dir_path, prepend_slash=False))
        return self._get_url(base_url, dataset_uid, obj_i, local_dir_uid)

    def get_local_dir_route(self, dataset_uid, obj_i, local_dir_path, local_dir_uid):
        """
        Obtain the Mount for some local directory

        :param str dataset_uid: A dataset unique identifier for the Mount
        :param str obj_i: A index of the current vitessce.wrappers.AbstractWrapper among all other wrappers in the view config
        :param str local_dir_path: The path to the local directory to serve.
        :param str local_dir_uid: The UID to include as the route path suffix.

        :returns: A starlette Mount of the the `local_dir_path`
        :rtype: list[starlette.routing.Mount]
        """
        if not self.is_remote:
            if self.base_dir is None:
                route_path = self._get_route_str(dataset_uid, obj_i, local_dir_uid)
            else:
                route_path = file_path_to_url_path(local_dir_path)
                local_dir_path = join(self.base_dir, local_dir_path)
            # TODO: Move imports back to top when this is factored out.
            from starlette.staticfiles import StaticFiles
            from starlette.routing import Mount
            return [Mount(route_path,
                          app=StaticFiles(directory=local_dir_path, html=False))]
        return []

    def _get_url(self, base_url, dataset_uid, obj_i, *args):
        return base_url + self._get_route_str(dataset_uid, obj_i, *args)

    def _get_url_simple(self, base_url, suffix):
        return base_url + "/" + suffix

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
        raise NotImplementedError(
            "Auto view configuration has not yet been implemented for this data object wrapper class.")


class MultiImageWrapper(AbstractWrapper):
    """
    Wrap multiple imaging datasets by creating an instance of the ``MultiImageWrapper`` class.

    :param list image_wrappers: A list of imaging wrapper classes (only :class:`~vitessce.wrappers.OmeTiffWrapper` supported now)
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, image_wrappers, use_physical_size_scaling=False, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        self.image_wrappers = image_wrappers
        self.use_physical_size_scaling = use_physical_size_scaling

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        for image in self.image_wrappers:
            image.convert_and_save(dataset_uid, obj_i, base_dir=base_dir)
        file_def_creator = self.make_raster_file_def_creator(
            dataset_uid, obj_i)
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
        self._repr = make_repr(locals())
        self.name = name
        self._img_path = img_path
        self._img_url = img_url
        self._offsets_url = offsets_url
        self._transformation_matrix = transformation_matrix
        self.is_remote = img_url is not None
        self.is_bitmask = is_bitmask
        self.local_img_uid = str(uuid4())
        self.local_offsets_uid = str(uuid4())
        if img_url is not None and (img_path is not None or offsets_path is not None):
            raise ValueError(
                "Did not expect img_path or offsets_path to be provided with img_url")

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_raster_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_raster_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_raster_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            # TODO: Move imports back to top when this is factored out.
            from .routes import range_repsonse, JsonRoute, FileRoute
            from generate_tiff_offsets import get_offsets
            from starlette.responses import UJSONResponse

            offsets = get_offsets(self._img_path)

            async def response_func(req):
                return UJSONResponse(offsets)
            if self.base_dir is None:
                local_img_path = self._img_path
                local_img_route_path = self._get_route_str(dataset_uid, obj_i, self.local_img_uid)
                local_offsets_route_path = self._get_route_str(dataset_uid, obj_i, self.local_offsets_uid)
            else:
                local_img_path = join(self.base_dir, self._img_path)
                local_img_route_path = file_path_to_url_path(self._img_path)
                # Do not include offsets in base_dir mode.
                local_offsets_route_path = None

            routes = [
                FileRoute(local_img_route_path, lambda req: range_repsonse(req, local_img_path), local_img_path),
            ]
            if local_offsets_route_path is not None:
                # Do not include offsets in base_dir mode.
                routes.append(JsonRoute(local_offsets_route_path, response_func, offsets))

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
        if offsets_url is not None and self.base_dir is None:
            # Do not include offsets in base_dir mode.
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

    def get_img_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._img_url
        if self.base_dir is not None:
            return self._get_url_simple(base_url, file_path_to_url_path(self._img_path, prepend_slash=False))
        return self._get_url(base_url, dataset_uid,
                             obj_i, self.local_img_uid)

    def get_offsets_url(self, base_url="", dataset_uid="", obj_i=""):
        if self._offsets_url is not None or self.is_remote:
            return self._offsets_url
        offsets_url = self._get_url(
            base_url, dataset_uid, obj_i, self.local_offsets_uid)
        return offsets_url


class CsvWrapper(AbstractWrapper):

    """
    Wrap a CSV file by creating an instance of the ``CsvWrapper`` class.

    :param str data_type: The data type of the information contained in the file.
    :param str csv_path: A local filepath to a CSV file.
    :param str csv_url: A remote URL of a CSV file.
    :param dict options: The file options.
    :param dict coordination_values: The coordination values.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, csv_path=None, csv_url=None, data_type=None, options=None, coordination_values=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        self._csv_path = csv_path
        self._csv_url = csv_url
        self._data_type = data_type
        self._options = options
        self._coordination_values = coordination_values
        self.is_remote = csv_url is not None
        self.local_csv_uid = str(uuid4())
        if data_type is None:
            raise ValueError("Expected data_type to be provided")
        if csv_url is not None and csv_path is not None:
            raise ValueError(
                "Did not expect csv_url to be provided with csv_path")
        if csv_url is None and csv_path is None:
            raise ValueError(
                "Expected csv_url or csv_path to be provided")

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_csv_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_csv_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_csv_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            # TODO: Move imports back to top when this is factored out.
            from .routes import FileRoute
            from starlette.responses import FileResponse

            if self.base_dir is not None:
                local_csv_path = join(self.base_dir, self._csv_path)
                local_csv_route_path = file_path_to_url_path(self._csv_path)
            else:
                local_csv_path = self._csv_path
                local_csv_route_path = self._get_route_str(dataset_uid, obj_i, self.local_csv_uid)

            async def response_func(req):
                return FileResponse(local_csv_path, filename=os.path.basename(self._csv_path))
            routes = [
                FileRoute(local_csv_route_path, response_func, local_csv_path),
            ]
            return routes

    def make_csv_file_def_creator(self, dataset_uid, obj_i):
        def csv_file_def_creator(base_url):
            file_def = {
                "fileType": f"{self._data_type}.csv",
                "url": self.get_csv_url(base_url, dataset_uid, obj_i),
            }
            if self._options is not None:
                file_def["options"] = self._options
            if self._coordination_values is not None:
                file_def["coordinationValues"] = self._coordination_values
            return file_def
        return csv_file_def_creator

    def get_csv_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._csv_url
        if self.base_dir is not None:
            return self._get_url_simple(base_url, file_path_to_url_path(self._csv_path, prepend_slash=False))
        return self._get_url(base_url, dataset_uid,
                             obj_i, self.local_csv_uid)


class OmeZarrWrapper(AbstractWrapper):

    """
    Wrap an OME-NGFF Zarr store by creating an instance of the ``OmeZarrWrapper`` class.

    :param str img_path: A local filepath to an OME-NGFF Zarr store.
    :param str img_url: A remote URL of an OME-NGFF Zarr store.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, img_url=None, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        if img_url is not None and img_path is not None:
            raise ValueError(
                "Did not expect img_path to be provided with img_url")
        if img_url is None and img_path is None:
            raise ValueError(
                "Expected either img_url or img_path to be provided")
        self._img_path = img_path
        self._img_url = img_url
        if self._img_path is not None:
            self.is_remote = False
        else:
            self.is_remote = True
        self.local_dir_uid = str(uuid4())

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_image_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_image_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_image_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            return self.get_local_dir_route(dataset_uid, obj_i, self._img_path, self.local_dir_uid)

    def get_img_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._img_url
        return self.get_local_dir_url(base_url, dataset_uid, obj_i, self._img_path, self.local_dir_uid)

    def make_image_file_def_creator(self, dataset_uid, obj_i):
        def image_file_def_creator(base_url):
            return {
                "fileType": "image.ome-zarr",
                "url": self.get_img_url(base_url, dataset_uid, obj_i)
            }
        return image_file_def_creator


class AnnDataWrapper(AbstractWrapper):
    def __init__(self, adata_path=None, adata_url=None, obs_feature_matrix_path=None, feature_filter_path=None, initial_feature_filter_path=None, obs_set_paths=None, obs_set_names=None, obs_locations_path=None, obs_segmentations_path=None, obs_embedding_paths=None, obs_embedding_names=None, obs_embedding_dims=None, request_init=None, feature_labels_path=None, convert_to_dense=True, coordination_values=None, **kwargs):
        """
        Wrap an AnnData object by creating an instance of the ``AnnDataWrapper`` class.

        :param str adata_path: A path to an AnnData object written to a Zarr store containing single-cell experiment data.
        :param str adata_url: A remote url pointing to a zarr-backed AnnData store.
        :param str obs_feature_matrix_path: Location of the expression (cell x gene) matrix, like `X` or `obsm/highly_variable_genes_subset`
        :param str feature_filter_path: A string like `var/highly_variable` used in conjunction with `obs_feature_matrix_path` if obs_feature_matrix_path points to a subset of `X` of the full `var` list.
        :param str initial_feature_filter_path: A string like `var/highly_variable` used in conjunction with `obs_feature_matrix_path` if obs_feature_matrix_path points to a subset of `X` of the full `var` list.
        :param list[str] obs_set_paths: Column names like `['obs/louvain', 'obs/cellType']` for showing cell sets
        :param list[str] obs_set_names: Names to display in place of those in `obs_set_paths`, like `['Louvain', 'Cell Type']
        :param str obs_locations_path: Column name in `obsm` that contains centroid coordinates for displaying centroids in the spatial viewer
        :param str obs_segmentations_path: Column name in `obsm` that contains polygonal coordinates for displaying outlines in the spatial viewer
        :param list[str] obs_embedding_paths: Column names like `['obsm/X_umap', 'obsm/X_pca']` for showing scatterplots
        :param list[str] obs_embedding_names: Overriding names like `['UMAP', 'PCA'] for displaying above scatterplots
        :param list[str] obs_embedding_dims: Dimensions along which to get data for the scatterplot, like [[0, 1], [4, 5]] where [0, 1] is just the normal x and y but [4, 5] could be comparing the third and fourth principal components, for example.
        :param dict request_init: options to be passed along with every fetch request from the browser, like { "header": { "Authorization": "Bearer dsfjalsdfa1431" } }
        :param str feature_labels_path: The name of a column containing gene names, instead of the default index in `var` of the AnnData store.
        :param bool convert_to_dense: Whether or not to convert `X` to dense the zarr store (dense is faster but takes more disk space).
        :param coordination_values: Coordination values for the file definition.
        :type coordination_values: dict or None
        :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
        """
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        self._adata_path = adata_path
        self._adata_url = adata_url
        if adata_url is not None and (adata_path is not None):
            raise ValueError(
                "Did not expect adata_url to be provided with adata_path")
        if adata_url is None and (adata_path is None):
            raise ValueError(
                "Expected either adata_url or adata_path to be provided")
        if adata_path is not None:
            self.is_remote = False
            self.zarr_folder = 'anndata.zarr'
        else:
            self.is_remote = True
            self.zarr_folder = None
        self.local_dir_uid = str(uuid4())
        self._expression_matrix = obs_feature_matrix_path
        self._cell_set_obs_names = obs_set_names
        self._mappings_obsm_names = obs_embedding_names
        self._gene_var_filter = feature_filter_path
        self._matrix_gene_var_filter = initial_feature_filter_path
        self._cell_set_obs = obs_set_paths
        self._spatial_centroid_obsm = obs_locations_path
        self._spatial_polygon_obsm = obs_segmentations_path
        self._mappings_obsm = obs_embedding_paths
        self._mappings_obsm_dims = obs_embedding_dims
        self._request_init = request_init
        self._gene_alias = feature_labels_path
        self._convert_to_dense = convert_to_dense
        self._coordination_values = coordination_values

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_anndata_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_anndata_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            return self.get_local_dir_route(dataset_uid, obj_i, self._adata_path, self.local_dir_uid)

    def get_zarr_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._adata_url
        else:
            return self.get_local_dir_url(base_url, dataset_uid, obj_i, self._adata_path, self.local_dir_uid)

    def make_file_def_creator(self, dataset_uid, obj_i):
        def get_anndata_zarr(base_url):
            options = {}
            if self._spatial_centroid_obsm is not None:
                options["obsLocations"] = {
                    "path": self._spatial_centroid_obsm
                }
            if self._spatial_polygon_obsm is not None:
                options["obsSegmentations"] = {
                    "path": self._spatial_polygon_obsm
                }
            if self._mappings_obsm is not None:
                options["obsEmbedding"] = []
                if self._mappings_obsm_names is not None:
                    for key, mapping in zip(self._mappings_obsm_names, self._mappings_obsm):
                        options["obsEmbedding"].append({
                            "path": mapping,
                            "dims": [0, 1],
                            "embeddingType": key
                        })
                else:
                    for mapping in self._mappings_obsm:
                        mapping_key = mapping.split('/')[-1]
                        self._mappings_obsm_names = mapping_key
                        options["obsEmbedding"].append({
                            "path": mapping,
                            "dims": [0, 1],
                            "embeddingType": mapping_key
                        })
                if self._mappings_obsm_dims is not None:
                    for dim_i, dim in enumerate(self._mappings_obsm_dims):
                        options["obsEmbedding"][dim_i]['dims'] = dim
            if self._cell_set_obs is not None:
                options["obsSets"] = []
                if self._cell_set_obs_names is not None:
                    names = self._cell_set_obs_names
                else:
                    names = [obs.split('/')[-1] for obs in self._cell_set_obs]
                for obs, name in zip(self._cell_set_obs, names):
                    options["obsSets"].append({
                        "name": name,
                        "path": obs
                    })
            if self._expression_matrix is not None:
                options["obsFeatureMatrix"] = {
                    "path": self._expression_matrix
                }
                if self._gene_var_filter is not None:
                    options["obsFeatureMatrix"]["featureFilterPath"] = self._gene_var_filter
                if self._matrix_gene_var_filter is not None:
                    options["obsFeatureMatrix"]["initialFeatureFilterPath"] = self._matrix_gene_var_filter
                if self._gene_alias is not None:
                    options["featureLabels"] = {
                        "path": self._gene_alias
                    }
            if len(options.keys()) > 0:
                obj_file_def = {
                    "fileType": ft.ANNDATA_ZARR.value,
                    "url": self.get_zarr_url(base_url, dataset_uid, obj_i),
                    "options": options
                }
                if self._request_init is not None:
                    obj_file_def['requestInit'] = self._request_init
                if self._coordination_values is not None:
                    obj_file_def['coordinationValues'] = self._coordination_values
                return obj_file_def
            return None
        return get_anndata_zarr

    def auto_view_config(self, vc):
        dataset = vc.add_dataset().add_object(self)
        mapping_name = self._mappings_obsm_names[0] if (
            self._mappings_obsm_names is not None) else self._mappings_obsm[0].split('/')[-1]
        scatterplot = vc.add_view(
            cm.SCATTERPLOT, dataset=dataset, mapping=mapping_name)
        cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset)
        genes = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
        heatmap = vc.add_view(cm.HEATMAP, dataset=dataset)
        if self._spatial_polygon_obsm is not None or self._spatial_centroid_obsm is not None:
            spatial = vc.add_view(cm.SPATIAL, dataset=dataset)
            vc.layout((scatterplot | spatial)
                      / (heatmap | (cell_sets / genes)))
        else:
            vc.layout((scatterplot | (cell_sets / genes))
                      / heatmap)


class MultivecZarrWrapper(AbstractWrapper):

    def __init__(self, zarr_path=None, zarr_url=None, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        if zarr_url is not None and zarr_path is not None:
            raise ValueError(
                "Did not expect zarr_path to be provided with zarr_url")
        if zarr_url is None and zarr_path is None:
            raise ValueError(
                "Expected either zarr_url or zarr_path to be provided")
        self._zarr_path = zarr_path
        self._zarr_url = zarr_url
        if self._zarr_path is not None:
            self.is_remote = False
        else:
            self.is_remote = True
        self.local_dir_uid = str(uuid4())

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_genomic_profiles_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_genomic_profiles_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_genomic_profiles_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            return self.get_local_dir_route(dataset_uid, obj_i, self._zarr_path, self.local_dir_uid)

    def get_zarr_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._zarr_url
        return self.get_local_dir_url(base_url, dataset_uid, obj_i, self._zarr_path, self.local_dir_uid)

    def make_genomic_profiles_file_def_creator(self, dataset_uid, obj_i):
        def genomic_profiles_file_def_creator(base_url):
            return {
                "fileType": "genomic-profiles.zarr",
                "url": self.get_zarr_url(base_url, dataset_uid, obj_i)
            }
        return genomic_profiles_file_def_creator
