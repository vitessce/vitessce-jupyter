from __future__ import annotations

from typing import TYPE_CHECKING

from collections import defaultdict
import os
from os.path import join
import tempfile
from typing import Callable, Optional, Type, TypeVar, Union
from uuid import uuid4
from pathlib import PurePath, PurePosixPath
import warnings
import zarr

import numpy as np

if TYPE_CHECKING:
    import lamindb as ln
    from spatialdata import SpatialData

from vitessce.file_def_utils import (
    gen_obs_locations_schema,
    gen_obs_segmentations_schema,
    gen_obs_spots_schema,
    gen_obs_points_schema,
    gen_obs_embedding_schema,
    gen_feature_labels_schema,
    gen_obs_feature_matrix_schema,
    gen_obs_labels_schema,
    gen_obs_sets_schema,
    gen_sdata_image_schema,
    gen_sdata_labels_schema,
    gen_sdata_obs_spots_schema,
    gen_sdata_obs_sets_schema,
    gen_sdata_obs_feature_matrix_schema,
)

from .constants import (
    norm_enum,
    ViewType as cm,
    FileType as ft,
    DataType as dt,
)
from .repr import make_repr


def make_unique_filename(file_ext):
    return f"{str(uuid4())}{file_ext}"


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
        :param dict request_init: options to be passed along with every fetch request from the browser, like `{ "header": { "Authorization": "Bearer dsfjalsdfa1431" } }`
        """
        self.out_dir = kwargs['out_dir'] if 'out_dir' in kwargs else tempfile.mkdtemp(
        )
        self.routes = []
        self.is_remote = False  # TODO: change to needs_localhost_serving for clarity
        self.is_store = False  # TODO: change to needs_store_registration for clarity
        self.file_def_creators = []
        self.base_dir = None
        self.stores = {}
        self.artifacts = {}
        self._request_init = kwargs['request_init'] if 'request_init' in kwargs else None

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

    def register_artifact(self, artifact):
        """
        Register an artifact.

        :param artifact: The artifact object to register.
        :type artifact: lamindb.Artifact
        :returns: The artifact URL.
        :rtype: str
        """
        artifact_url = artifact.path.to_url()
        self.artifacts[artifact_url] = artifact
        return artifact_url

    def get_artifacts(self):
        """
        Obtain the dictionary that maps artifact URLs to artifact objects.

        :returns: A dictionary that maps artifact URLs to Artifact objects.
        :rtype: dict[str, lamindb.Artifact]
        """
        return self.artifacts
    
    def try_getting_artifact_stores(self):
        artifact_stores = dict()
        for artifact_url, artifact in self.artifacts.items():
            local_path = artifact._local_filepath
            if local_path is not None and os.path.isdir(local_path):
                store = zarr.DirectoryStore(local_path)
                artifact_stores[artifact_url] = store
        return artifact_stores

    def get_stores(self, base_url, prefer_local=False):
        """
        Obtain the stores that have been created for this wrapper class.

        :returns: A dictionary that maps file URLs to Zarr Store objects.
        :rtype: dict[str, zarr.Store]
        """
        relative_stores = self.stores
        absolute_stores = {}
        for relative_url, store in relative_stores.items():
            absolute_url = base_url + relative_url
            absolute_stores[absolute_url] = store
        if prefer_local:
            absolute_stores.update(self.try_getting_artifact_stores())
        return absolute_stores

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

    def get_local_file_url(self, base_url, dataset_uid, obj_i, local_file_path, local_file_uid):
        if not self.is_remote and self.base_dir is not None:
            return self._get_url_simple(base_url, file_path_to_url_path(local_file_path, prepend_slash=False))
        return self._get_url(base_url, dataset_uid, obj_i, local_file_uid)

    def get_local_dir_url(self, base_url, dataset_uid, obj_i, local_dir_path, local_dir_uid):
        # Logic for files and directories is the same for this function.
        return self.get_local_file_url(base_url, dataset_uid, obj_i, local_dir_path, local_dir_uid)

    def register_zarr_store(self, dataset_uid, obj_i, store_or_local_dir_path, local_dir_uid):
        if not self.is_remote and self.is_store:
            # Set up `store` and `local_dir_path` variables.
            if isinstance(store_or_local_dir_path, str):
                # TODO: use zarr.FSStore if fsspec is installed?
                store = zarr.DirectoryStore(store_or_local_dir_path)
                local_dir_path = store_or_local_dir_path
            else:
                # TODO: check that store_or_local_dir_path is a zarr.Store or StoreLike?
                store = store_or_local_dir_path
                # A store instance was passed directly, so there is no local directory path.
                # Instead we just make one up using _get_route_str but it could be any string.
                local_dir_path = self._get_route_str(dataset_uid, obj_i, local_dir_uid)

            # Register the store on the same route path
            # that will be used for the "url" field in the file definition.
            if self.base_dir is None:
                route_path = self._get_route_str(dataset_uid, obj_i, local_dir_uid)
            else:
                route_path = file_path_to_url_path(local_dir_path)
                local_dir_path = join(self.base_dir, local_dir_path)

            self.stores[route_path] = store

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

    def get_local_file_route(self, dataset_uid, obj_i, local_file_path, local_file_uid):
        if not self.is_remote:
            from .routes import range_repsonse, FileRoute

            if self.base_dir is None:
                route_path = self._get_route_str(dataset_uid, obj_i, local_file_uid)
            else:
                route_path = file_path_to_url_path(local_file_path)
                local_file_path = join(self.base_dir, local_file_path)

            return [
                FileRoute(route_path, lambda req: range_repsonse(req, local_file_path), local_file_path),
            ]
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
        self.local_img_uid = make_unique_filename(".ome.tif")
        self.local_offsets_uid = make_unique_filename(".offsets.json")
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


class ImageOmeTiffWrapper(AbstractWrapper):

    """
    Wrap an OME-TIFF File by creating an instance of the ``ImageOmeTiffWrapper`` class. Intended to be used with the spatialBeta and layerControllerBeta views.

    :param str img_path: A local filepath to an OME-TIFF file.
    :param str offsets_path: A local filepath to an offsets.json file.
    :param str img_url: A remote URL of an OME-TIFF file.
    :param str offsets_url: A remote URL of an offsets.json file.
    :param list coordinate_transformations: A column-major ordered matrix for transforming this image (see http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#homogeneous-coordinates for more information).
    :param dict coordination_values: Optional coordinationValues to be passed in the file definition.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, img_url=None, img_artifact=None, offsets_path=None, offsets_url=None, offsets_artifact=None, coordinate_transformations=None, coordination_values=None, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        num_inputs = sum([1 for x in [img_path, img_url, img_artifact] if x is not None])
        if num_inputs != 1:
            raise ValueError(
                "Expected one of img_path, img_url, or img_artifact to be provided")

        num_inputs = sum([1 for x in [offsets_path, offsets_url, offsets_artifact] if x is not None])
        if num_inputs > 1:
            raise ValueError(
                "Expected zero or one of offsets_path, offsets_url, or offsets_artifact to be provided")

        self._img_path = img_path
        self._img_url = img_url
        self._img_artifact = img_artifact
        self._offsets_path = offsets_path
        self._offsets_url = offsets_url
        self._offsets_artifact = offsets_artifact
        self._coordinate_transformations = coordinate_transformations
        self._coordination_values = coordination_values
        self.is_remote = img_url is not None or img_artifact is not None
        self.local_img_uid = make_unique_filename(".ome.tif")
        self.local_offsets_uid = make_unique_filename(".offsets.json")

        if img_artifact is not None:
            self._img_url = self.register_artifact(img_artifact)

        if offsets_artifact is not None:
            self._offsets_url = self.register_artifact(offsets_artifact)

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

    def make_raster_file_def_creator(self, dataset_uid, obj_i):
        def raster_file_def_creator(base_url):
            options = {}
            if self._coordinate_transformations is not None:
                options["coordinateTransformations"] = self._coordinate_transformations

            offsets_url = self.get_offsets_url(base_url, dataset_uid, obj_i)
            if offsets_url is not None and self.base_dir is None:
                # Do not include offsets in base_dir mode.
                options["offsetsUrl"] = offsets_url

            file_def = {
                "fileType": "image.ome-tiff",
                "url": self.get_img_url(base_url, dataset_uid, obj_i),
            }
            if len(options.keys()) > 0:
                file_def["options"] = options
            if self._coordination_values is not None:
                file_def["coordinationValues"] = self._coordination_values
            return file_def
        return raster_file_def_creator

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


class ObsSegmentationsOmeTiffWrapper(AbstractWrapper):

    """
    Wrap an OME-TIFF File by creating an instance of the ``ObsSegmentationsOmeTiffWrapper`` class. Intended to be used with the spatialBeta and layerControllerBeta views.

    :param str img_path: A local filepath to an OME-TIFF file.
    :param str img_url: A remote URL of an OME-TIFF file.
    :param img_artifact: A lamindb Artifact corresponding to the image.
    :type img_artifact: lamindb.Artifact
    :param str offsets_path: A local filepath to an offsets.json file.
    :param str offsets_url: A remote URL of an offsets.json file.
    :param offsets_artifact: A lamindb Artifact corresponding to the offsets JSON.
    :type offsets_artifact: lamindb.Artifact
    :param list coordinate_transformations: A column-major ordered matrix for transforming this image (see http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#homogeneous-coordinates for more information).
    :param bool obs_types_from_channel_names: Whether to use the channel names to determine the obs types. Optional.
    :param dict coordination_values: Optional coordinationValues to be passed in the file definition.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, img_url=None, img_artifact=None, offsets_path=None, offsets_url=None, offsets_artifact=None, coordinate_transformations=None, obs_types_from_channel_names=None, coordination_values=None, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        num_inputs = sum([1 for x in [img_path, img_url, img_artifact] if x is not None])
        if num_inputs != 1:
            raise ValueError(
                "Expected one of img_path, img_url, or img_artifact to be provided")

        num_inputs = sum([1 for x in [offsets_path, offsets_url, offsets_artifact] if x is not None])
        if num_inputs > 1:
            raise ValueError(
                "Expected zero or one of offsets_path, offsets_url, or offsets_artifact to be provided")

        self._img_path = img_path
        self._img_url = img_url
        self._img_artifact = img_artifact
        self._offsets_path = offsets_path
        self._offsets_url = offsets_url
        self._offsets_artifact = offsets_artifact
        self._coordinate_transformations = coordinate_transformations
        self._obs_types_from_channel_names = obs_types_from_channel_names
        self._coordination_values = coordination_values
        self.is_remote = img_url is not None or img_artifact is not None
        self.local_img_uid = make_unique_filename(".ome.tif")
        self.local_offsets_uid = make_unique_filename(".offsets.json")

        if img_artifact is not None:
            self._img_url = self.register_artifact(img_artifact)

        if offsets_artifact is not None:
            self._offsets_url = self.register_artifact(offsets_artifact)

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

    def make_raster_file_def_creator(self, dataset_uid, obj_i):
        def raster_file_def_creator(base_url):
            options = {}
            if self._coordinate_transformations is not None:
                options["coordinateTransformations"] = self._coordinate_transformations

            if self._obs_types_from_channel_names is not None:
                options["obsTypesFromChannelNames"] = self._obs_types_from_channel_names

            offsets_url = self.get_offsets_url(base_url, dataset_uid, obj_i)
            if offsets_url is not None and self.base_dir is None:
                # Do not include offsets in base_dir mode.
                options["offsetsUrl"] = offsets_url

            file_def = {
                "fileType": "obsSegmentations.ome-tiff",
                "url": self.get_img_url(base_url, dataset_uid, obj_i),
            }
            if len(options.keys()) > 0:
                file_def["options"] = options
            if self._coordination_values is not None:
                file_def["coordinationValues"] = self._coordination_values
            return file_def
        return raster_file_def_creator

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
        self._data_type = norm_enum(data_type, dt)
        self._options = options
        self._coordination_values = coordination_values
        self.is_remote = csv_url is not None
        self.local_csv_uid = make_unique_filename(".csv")
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


class JsonWrapper(AbstractWrapper):

    """
    Wrap a JSON file by creating an instance of the ``JsonWrapper`` class.

    :param str data_type: The data type of the information contained in the file.
    :param str json_path: A local filepath JSON a JSON file.
    :param str json_url: A remote URL of a CSV file.
    :param dict options: The file options.
    :param dict coordination_values: The coordination values.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, json_path=None, json_url=None, data_type=None, options=None, coordination_values=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        self._json_path = json_path
        self._json_url = json_url
        self._data_type = norm_enum(data_type, dt)
        self._options = options
        self._coordination_values = coordination_values
        self.is_remote = json_url is not None
        self.local_json_uid = make_unique_filename(".json")
        if data_type is None:
            raise ValueError("Expected data_type to be provided")
        if json_url is not None and json_path is not None:
            raise ValueError(
                "Did not expect json_url to be provided with json_path")
        if json_url is None and json_path is None:
            raise ValueError(
                "Expected json_url or json_path to be provided")

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_json_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_json_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_json_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        else:
            # TODO: Move imports back to top when this is factored out.
            from .routes import FileRoute
            from starlette.responses import FileResponse

            if self.base_dir is not None:
                local_json_path = join(self.base_dir, self._json_path)
                local_json_route_path = file_path_to_url_path(self._json_path)
            else:
                local_json_path = self._json_path
                local_json_route_path = self._get_route_str(dataset_uid, obj_i, self.local_json_uid)

            async def response_func(req):
                return FileResponse(local_json_path, filename=os.path.basename(self._json_path))
            routes = [
                FileRoute(local_json_route_path, response_func, local_json_path),
            ]
            return routes

    def make_json_file_def_creator(self, dataset_uid, obj_i):
        def json_file_def_creator(base_url):
            file_def = {
                "fileType": f"{self._data_type}.json",
                "url": self.get_json_url(base_url, dataset_uid, obj_i),
            }
            if self._options is not None:
                file_def["options"] = self._options
            if self._coordination_values is not None:
                file_def["coordinationValues"] = self._coordination_values
            return file_def
        return json_file_def_creator

    def get_json_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._json_url
        if self.base_dir is not None:
            return self._get_url_simple(base_url, file_path_to_url_path(self._json_path, prepend_slash=False))
        return self._get_url(base_url, dataset_uid,
                             obj_i, self.local_json_uid)


class OmeZarrWrapper(AbstractWrapper):

    """
    Wrap an OME-NGFF Zarr store by creating an instance of the ``OmeZarrWrapper`` class.

    :param str img_path: A local filepath to an OME-NGFF Zarr store.
    :param str img_url: A remote URL of an OME-NGFF Zarr store.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, img_url=None, name="", is_bitmask=False, **kwargs):
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
        self.name = name
        self.is_bitmask = is_bitmask
        if self._img_path is not None:
            self.is_remote = False
        else:
            self.is_remote = True
        self.local_dir_uid = make_unique_filename(".ome.zarr")

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

    # The following two functions will be used when OmeZarrWrapper
    # is used within MultiImageWrapper.
    def make_image_def(self, dataset_uid, obj_i, base_url):
        img_url = self.get_img_url(base_url, dataset_uid, obj_i)
        return self.create_image_json(img_url)

    def create_image_json(self, img_url):
        metadata = {}
        image = {
            "name": self.name,
            "type": "ome-zarr",
            "url": img_url,
        }
        if self.is_bitmask:
            metadata["isBitmask"] = self.is_bitmask
        # Only attach metadata if there is some - otherwise schema validation fails.
        if len(metadata.keys()) > 0:
            image["metadata"] = metadata
        return image


class ImageOmeZarrWrapper(AbstractWrapper):

    """
    Wrap an OME-NGFF Zarr store by creating an instance of the ``ImageOmeZarrWrapper`` class. Intended to be used with the spatialBeta and layerControllerBeta views.

    :param str img_path: A local filepath to an OME-NGFF Zarr store.
    :param str img_url: A remote URL of an OME-NGFF Zarr store.
    :param img_artifact: A lamindb Artifact corresponding to the image.
    :type img_artifact: lamindb.Artifact
    :param list coordinate_transformations: A column-major ordered matrix for transforming this image (see http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#homogeneous-coordinates for more information).
    :param dict coordination_values: Optional coordinationValues to be passed in the file definition.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, img_url=None, img_artifact=None, coordinate_transformations=None, coordination_values=None, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())

        num_inputs = sum([1 for x in [img_path, img_url, img_artifact] if x is not None])
        if num_inputs != 1:
            raise ValueError(
                "Expected one of img_path, img_url, or img_artifact to be provided")

        self._img_path = img_path
        self._img_url = img_url
        self._img_artifact = img_artifact
        self._coordinate_transformations = coordinate_transformations
        self._coordination_values = coordination_values
        if self._img_path is not None:
            self.is_remote = False
        else:
            self.is_remote = True

        if self._img_artifact is not None:
            # To serve as a placeholder in the config JSON URL field
            self._img_url = self.register_artifact(img_artifact)

        self.local_dir_uid = make_unique_filename(".ome.zarr")

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
            options = {}
            if self._coordinate_transformations is not None:
                options["coordinateTransformations"] = self._coordinate_transformations

            file_def = {
                "fileType": "image.ome-zarr",
                "url": self.get_img_url(base_url, dataset_uid, obj_i)
            }

            if len(options.keys()) > 0:
                file_def["options"] = options
            if self._coordination_values is not None:
                file_def["coordinationValues"] = self._coordination_values
            return file_def

        return image_file_def_creator


class ObsSegmentationsOmeZarrWrapper(AbstractWrapper):

    """
    Wrap an OME-NGFF Zarr store by creating an instance of the ``ObsSegmentationsOmeZarrWrapper`` class. Intended to be used with the spatialBeta and layerControllerBeta views.

    :param str img_path: A local filepath to an OME-NGFF Zarr store.
    :param str img_url: A remote URL of an OME-NGFF Zarr store.
    :param img_artifact: A lamindb Artifact corresponding to the image.
    :type img_artifact: lamindb.Artifact
    :param list coordinate_transformations: A column-major ordered matrix for transforming this image (see http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#homogeneous-coordinates for more information).
    :param dict coordination_values: Optional coordinationValues to be passed in the file definition.
    :param bool obs_types_from_channel_names: Whether to use the channel names to determine the obs types. Optional.
    :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
    """

    def __init__(self, img_path=None, img_url=None, img_artifact=None, coordinate_transformations=None, coordination_values=None, obs_types_from_channel_names=None, **kwargs):
        super().__init__(**kwargs)
        self._repr = make_repr(locals())

        num_inputs = sum([1 for x in [img_path, img_url, img_artifact] if x is not None])
        if num_inputs != 1:
            raise ValueError(
                "Expected one of img_path, img_url, or img_artifact to be provided")
        self._img_path = img_path
        self._img_url = img_url
        self._img_artifact = img_artifact
        self._coordinate_transformations = coordinate_transformations
        self._obs_types_from_channel_names = obs_types_from_channel_names
        self._coordination_values = coordination_values
        if self._img_path is not None:
            self.is_remote = False
        else:
            self.is_remote = True

        if self._img_artifact is not None:
            # To serve as a placeholder in the config JSON URL field
            self._img_url = self.register_artifact(img_artifact)

        self.local_dir_uid = make_unique_filename(".ome.zarr")

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
            options = {}
            if self._coordinate_transformations is not None:
                options["coordinateTransformations"] = self._coordinate_transformations

            if self._obs_types_from_channel_names is not None:
                options["obsTypesFromChannelNames"] = self._obs_types_from_channel_names

            file_def = {
                "fileType": "obsSegmentations.ome-zarr",
                "url": self.get_img_url(base_url, dataset_uid, obj_i)
            }

            if len(options.keys()) > 0:
                file_def["options"] = options
            if self._coordination_values is not None:
                file_def["coordinationValues"] = self._coordination_values
            return file_def

        return image_file_def_creator


def raise_error_if_zero_or_more_than_one(inputs):
    num_inputs = sum([1 for x in inputs if x is not None])
    if num_inputs > 1:
        raise ValueError(
            "Expected only one type of data input parameter to be provided (_url, _path, _store, etc.), but received more than one."
        )
    if num_inputs == 0:
        raise ValueError(
            "Expected one type of data input parameter to be provided (_url, _path, _store, etc.), but received none."
        )
    return True


def raise_error_if_any(inputs):
    num_inputs = sum([1 for x in inputs if x is not None])
    if num_inputs > 0:
        raise ValueError(
            "Did not expect any of these parameters to be provided, but received one or more: " + str(inputs)
        )
    return True


def raise_error_if_more_than_one(inputs):
    num_inputs = sum([1 for x in inputs if x is not None])
    if num_inputs > 1:
        raise ValueError(
            "Expected only one of these parameters to be provided, but received more than one: " + str(inputs)
        )
    return True


class AnnDataWrapper(AbstractWrapper):
    def __init__(self, adata_path=None, adata_url=None, adata_store=None, adata_artifact=None, ref_path=None, ref_url=None, ref_artifact=None, obs_feature_matrix_path=None, feature_filter_path=None, initial_feature_filter_path=None, obs_set_paths=None, obs_set_names=None, obs_locations_path=None, obs_segmentations_path=None, obs_embedding_paths=None, obs_embedding_names=None, obs_embedding_dims=None, obs_spots_path=None, obs_points_path=None, feature_labels_path=None, obs_labels_path=None, convert_to_dense=True, coordination_values=None, obs_labels_paths=None, obs_labels_names=None, **kwargs):
        """
        Wrap an AnnData object by creating an instance of the ``AnnDataWrapper`` class.

        :param str adata_path: A path to an AnnData object written to a Zarr store containing single-cell experiment data.
        :param str adata_url: A remote url pointing to a zarr-backed AnnData store.
        :param adata_store: A path to pass to zarr.DirectoryStore, or an existing store instance.
        :type adata_store: str or zarr.Storage
        :param adata_artifact: A lamindb Artifact corresponding to the AnnData object.
        :type adata_artifact: lamindb.Artifact
        :param str obs_feature_matrix_path: Location of the expression (cell x gene) matrix, like `X` or `obsm/highly_variable_genes_subset`
        :param str feature_filter_path: A string like `var/highly_variable` used in conjunction with `obs_feature_matrix_path` if obs_feature_matrix_path points to a subset of `X` of the full `var` list.
        :param str initial_feature_filter_path: A string like `var/highly_variable` used in conjunction with `obs_feature_matrix_path` if obs_feature_matrix_path points to a subset of `X` of the full `var` list.
        :param list[str] obs_set_paths: Column names like `['obs/louvain', 'obs/cellType']` for showing cell sets
        :param list[str] obs_set_names: Names to display in place of those in `obs_set_paths`, like `['Louvain', 'Cell Type']`
        :param str obs_locations_path: Column name in `obsm` that contains centroid coordinates for displaying centroids in the spatial viewer
        :param str obs_segmentations_path: Column name in `obsm` that contains polygonal coordinates for displaying outlines in the spatial viewer
        :param list[str] obs_embedding_paths: Column names like `['obsm/X_umap', 'obsm/X_pca']` for showing scatterplots
        :param list[str] obs_embedding_names: Overriding names like `['UMAP', 'PCA']` for displaying above scatterplots
        :param list[str] obs_embedding_dims: Dimensions along which to get data for the scatterplot, like `[[0, 1], [4, 5]]` where `[0, 1]` is just the normal x and y but `[4, 5]` could be comparing the third and fourth principal components, for example.
        :param str obs_spots_path: Column name in `obsm` that contains centroid coordinates for displaying spots in the spatial viewer
        :param str obs_points_path: Column name in `obsm` that contains centroid coordinates for displaying points in the spatial viewer
        :param str feature_labels_path: The name of a column containing feature labels (e.g., alternate gene symbols), instead of the default index in `var` of the AnnData store.
        :param str obs_labels_path: (DEPRECATED) The name of a column containing observation labels (e.g., alternate cell IDs), instead of the default index in `obs` of the AnnData store. Use `obs_labels_paths` and `obs_labels_names` instead. This arg will be removed in a future release.
        :param list[str] obs_labels_paths: The names of columns containing observation labels (e.g., alternate cell IDs), instead of the default index in `obs` of the AnnData store.
        :param list[str] obs_labels_names: The optional display names of columns containing observation labels (e.g., alternate cell IDs), instead of the default index in `obs` of the AnnData store.
        :param bool convert_to_dense: Whether or not to convert `X` to dense the zarr store (dense is faster but takes more disk space).
        :param coordination_values: Coordination values for the file definition.
        :type coordination_values: dict or None
        :param \\*\\*kwargs: Keyword arguments inherited from :class:`~vitessce.wrappers.AbstractWrapper`
        """
        super().__init__(**kwargs)
        self._repr = make_repr(locals())
        self._adata_path = adata_path
        self._adata_url = adata_url
        self._adata_store = adata_store
        self._adata_artifact = adata_artifact

        # For reference spec JSON with .h5ad files
        self._ref_path = ref_path
        self._ref_url = ref_url
        self._ref_artifact = ref_artifact

        if ref_path is not None or ref_url is not None or ref_artifact is not None:
            self.is_h5ad = True
        else:
            self.is_h5ad = False

        if adata_store is not None and (ref_path is not None or ref_url is not None or ref_artifact is not None):
            raise ValueError(
                "Did not expect reference JSON to be provided with adata_store")

        raise_error_if_zero_or_more_than_one([adata_path, adata_url, adata_store, adata_artifact])

        if adata_path is not None:
            self.is_remote = False
            self.is_store = False
            self.zarr_folder = 'anndata.zarr'
        elif adata_url is not None or adata_artifact is not None:
            self.is_remote = True
            self.is_store = False
            self.zarr_folder = None

            # Store artifacts on AbstractWrapper.artifacts for downstream access,
            # e.g. in lamindb.save_vitessce_config
            if adata_artifact is not None:
                self._adata_url = self.register_artifact(adata_artifact)
            if ref_artifact is not None:
                self._ref_url = self.register_artifact(ref_artifact)
        else:
            # Store case
            self.is_remote = False
            self.is_store = True
            self.zarr_folder = None

        self.local_dir_uid = make_unique_filename(".adata.zarr")
        self.local_file_uid = make_unique_filename(".h5ad")
        self.local_ref_uid = make_unique_filename(".ref.json")

        self._expression_matrix = obs_feature_matrix_path
        self._obs_set_names = obs_set_names
        self._mappings_obsm_names = obs_embedding_names
        self._gene_var_filter = feature_filter_path
        self._matrix_gene_var_filter = initial_feature_filter_path
        self._obs_set_elems = obs_set_paths
        self._spatial_centroid_obsm = obs_locations_path
        self._spatial_polygon_obsm = obs_segmentations_path
        self._mappings_obsm = obs_embedding_paths
        self._mappings_obsm_dims = obs_embedding_dims
        self._spatial_spots_obsm = obs_spots_path
        self._spatial_points_obsm = obs_points_path
        self._feature_labels = feature_labels_path
        # Support legacy provision of single obs labels path
        if (obs_labels_path is not None):
            warnings.warn("`obs_labels_path` will be deprecated in a future release.", DeprecationWarning)
            self._obs_labels_elems = [obs_labels_path]
            self._obs_labels_names = [obs_labels_path.split('/')[-1]]
        else:
            self._obs_labels_elems = obs_labels_paths
            self._obs_labels_names = obs_labels_names
        self._convert_to_dense = convert_to_dense
        self._coordination_values = coordination_values

    def convert_and_save(self, dataset_uid, obj_i, base_dir=None):
        # Only create out-directory if needed
        if not self.is_remote:
            super().convert_and_save(dataset_uid, obj_i, base_dir=base_dir)

        file_def_creator = self.make_file_def_creator(
            dataset_uid, obj_i)
        routes = self.make_routes(dataset_uid, obj_i)

        self.file_def_creators.append(file_def_creator)
        self.routes += routes

    def make_routes(self, dataset_uid, obj_i):
        if self.is_remote:
            return []
        elif self.is_store:
            self.register_zarr_store(dataset_uid, obj_i, self._adata_store, self.local_dir_uid)
            return []
        else:
            if self.is_h5ad:
                return [
                    *self.get_local_file_route(dataset_uid, obj_i, self._adata_path, self.local_file_uid),
                    *self.get_local_file_route(dataset_uid, obj_i, self._ref_path, self.local_ref_uid)
                ]
            else:
                return self.get_local_dir_route(dataset_uid, obj_i, self._adata_path, self.local_dir_uid)

    def get_zarr_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._adata_url
        else:
            return self.get_local_dir_url(base_url, dataset_uid, obj_i, self._adata_path, self.local_dir_uid)

    def get_h5ad_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._adata_url
        else:
            return self.get_local_file_url(base_url, dataset_uid, obj_i, self._adata_path, self.local_file_uid)

    def get_ref_url(self, base_url="", dataset_uid="", obj_i=""):
        if self.is_remote:
            return self._ref_url
        else:
            return self.get_local_file_url(base_url, dataset_uid, obj_i, self._ref_path, self.local_ref_uid)

    def make_file_def_creator(self, dataset_uid, obj_i):
        def get_anndata_zarr(base_url):
            options = {}
            options = gen_obs_locations_schema(self._spatial_centroid_obsm, options)
            options = gen_obs_segmentations_schema(self._spatial_polygon_obsm, options)
            options = gen_obs_spots_schema(self._spatial_spots_obsm, options)
            options = gen_obs_points_schema(self._spatial_points_obsm, options)
            options = gen_obs_embedding_schema(options, self._mappings_obsm, self._mappings_obsm_names, self._mappings_obsm_dims)
            options = gen_obs_sets_schema(options, self._obs_set_elems, self._obs_set_names,)
            options = gen_obs_feature_matrix_schema(options, self._expression_matrix, self._gene_var_filter, self._matrix_gene_var_filter)
            options = gen_feature_labels_schema(self._feature_labels, options)
            options = gen_obs_labels_schema(options, self._obs_labels_elems, self._obs_labels_names)

            if len(options.keys()) > 0:
                if self.is_h5ad:
                    options["refSpecUrl"] = self.get_ref_url(base_url, dataset_uid, obj_i)

                obj_file_def = {
                    "fileType": ft.ANNDATA_ZARR.value if not self.is_h5ad else ft.ANNDATA_H5AD.value,
                    "url": self.get_zarr_url(base_url, dataset_uid, obj_i) if not self.is_h5ad else self.get_h5ad_url(base_url, dataset_uid, obj_i),
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


SpatialDataWrapperType = TypeVar('SpatialDataWrapperType', bound='SpatialDataWrapper')


class SpatialDataWrapper(AnnDataWrapper):

    def __init__(self, sdata_path: Optional[str] = None, sdata_url: Optional[str] = None, sdata_store: Optional[Union[str, zarr.storage.StoreLike]] = None, sdata_artifact: Optional[ln.Artifact] = None, image_path: Optional[str] = None, region: Optional[str] = None, coordinate_system: Optional[str] = None, affine_transformation: Optional[np.ndarray] = None, obs_spots_path: Optional[str] = None, labels_path: Optional[str] = None, table_path: str = "tables/table", **kwargs):
        """
        Wrap a SpatialData object.

        :param sdata_path: SpatialData path, exclusive with other `{sdata,adata}_xxxx` arguments, by default None
        :type sdata_path: Optional[str]
        :param sdata_url: SpatialData url, exclusive with other `{sdata,adata}_xxxx` arguments, by default None
        :type sdata_url: Optional[str]
        :param sdata_store: SpatialData store, exclusive with other `{spatialdata,adata}_xxxx` arguments, by default None
        :type sdata_store: Optional[Union[str, zarr.storage.StoreLike]]
        :param sdata_artifact: Artifact that corresponds to a SpatialData object.
        :type sdata_artifact: Optional[ln.Artifact]
        :param image_path: Path to the image element of interest. By default, None.
        :type image_path: Optional[str]
        :param coordinate_system: Name of a target coordinate system.
        :type coordinate_system: Optional[str]
        :param affine_transformation: Transformation to be applied to the image. By default, None. Prefer coordinate_system.
        :type affine_transformation: Optional[np.ndarray]
        :param obs_spots_path: Location of shapes that should be interpreted as spot observations, by default None
        :type obs_spots_path: Optional[str]
        :param labels_path: Location of the labels (segmentation bitmask image), by default None
        :type labels_path: Optional[str]
        """
        raise_error_if_zero_or_more_than_one([
            sdata_path,
            sdata_url,
            sdata_store,
            sdata_artifact,
        ])
        raise_error_if_any([
            kwargs.get('adata_path', None),
            kwargs.get('adata_url', None),
            kwargs.get('adata_store', None),
            kwargs.get('adata_artifact', None)
        ])
        super().__init__(adata_path=sdata_path, adata_url=sdata_url, adata_store=sdata_store, adata_artifact=sdata_artifact, **kwargs)
        self.local_dir_uid = make_unique_filename(".sdata.zarr")
        self._image_path = image_path
        self._region = region
        self._coordinate_system = coordinate_system
        self._affine_transformation = affine_transformation
        self._kwargs = kwargs
        self._obs_spots_path = obs_spots_path
        self._labels_path = labels_path
        if self._adata_path is not None:
            self.zarr_folder = 'spatialdata.zarr'
        self.obs_type_label = None
        if self._coordination_values is not None and "obsType" in self._coordination_values:
            self.obs_type_label = self._coordination_values["obsType"]
        self._table_path = table_path

    @classmethod
    def from_object(cls: Type[SpatialDataWrapperType], sdata: SpatialData, table_keys_to_image_elems: dict[str, Union[str, None]] = defaultdict(type(None)), table_keys_to_regions: dict[str, Union[str, None]] = defaultdict(type(None)), obs_type_label: str = "spot") -> list[SpatialDataWrapperType]:
        """Instantiate a wrapper for SpatialData stores, one per table, directly from the SpatialData object.
        By default, we "show everything" that can reasonable be inferred given the information.  If you wish to have more control,
        consider instantiating the object directly.  This function will error if something cannot be inferred i.e., the user does not present
        regions explicitly but there is more than one for a given table.


        Parameters
        ----------
        cls : Type[SpatialDataWrapperType]
            _description_
        spatialdata : SpatialData
            _description_
        table_keys_to_image_elems : dict[str, str], optional
            which image paths to use for a given table for the visualization, by default None for each table key.
        table_keys_to_regions : dict[str, str], optional
            which regions to use for a given table for the visualization, by default None for each table key.

        Returns
        -------
        list[SpatialDataWrapperType]

        Raises
        ------
        ValueError
        """
        wrappers = []
        parent_table_key = "table" if (sdata.path / "table").exists() else "tables"
        for table_key, table in sdata.tables.items():
            spot_shapes_elem = None
            image_elem = table_keys_to_image_elems[table_key]
            labels_elem = None
            spatialdata_attr = table.uns['spatialdata_attrs']
            region = table_keys_to_regions[table_key]
            if region is not None:
                assert region in spatialdata_attr['region']
            else:
                region = spatialdata_attr['region']
            if isinstance(region, list):
                if len(region) > 1:
                    raise ValueError("Vitessce cannot subset AnnData objects on the fly.  Please provide an explicit region")
                region = region[0]
            if region in sdata.shapes:
                spot_shapes_elem = f"shapes/{region}"
                # Currently, only circle shapes are supported.
                # TODO: add if statement to check that this region contains spot shapes rather than other types of shapes
            if region in sdata.labels:
                labels_elem = f"labels/{region}"
            obs_feature_matrix_elem = f"{parent_table_key}/{table_key}/X"
            if 'highly_variable' in table.var:
                # TODO: fix first key needing to be "table" in vitessce-js
                initial_feature_filter_elem = 'highly_variable'
            else:
                initial_feature_filter_elem = None
            obs_set_elems = [f"{parent_table_key}/{table_key}/obs/{elem}" for elem in table.obs if table.obs[elem].dtype == 'category']
            wrappers += [
                cls(
                    sdata_path=str(sdata.path),
                    image_path=str(image_elem) if image_elem is not None else None,
                    labels_path=str(labels_elem) if labels_elem is not None else None,
                    obs_feature_matrix_path=str(obs_feature_matrix_elem),
                    obs_spots_path=str(spot_shapes_elem) if spot_shapes_elem is not None else None,
                    initial_feature_filter_path=initial_feature_filter_elem,
                    obs_set_paths=obs_set_elems,
                    coordination_values={"obsType": "spot"}  # TODO: should we remove?
                )
            ]
        return wrappers

    def make_file_def_creator(self, dataset_uid: str, obj_i: str) -> Optional[Callable]:
        def generator(base_url):
            options = {}
            options = gen_obs_labels_schema(options, self._obs_labels_elems, self._obs_labels_names)
            options = gen_sdata_obs_feature_matrix_schema(options, self._expression_matrix, self._gene_var_filter, self._matrix_gene_var_filter, self._region)
            options = gen_sdata_obs_sets_schema(options, self._obs_set_elems, self._obs_set_names, self._table_path, self._region)
            options = gen_sdata_obs_spots_schema(options, self._obs_spots_path, self._table_path, self._region, self._coordinate_system)
            options = gen_sdata_image_schema(options, self._image_path, self._coordinate_system, self._affine_transformation)
            options = gen_sdata_labels_schema(options, self._labels_path, self._table_path, self._coordinate_system, self._affine_transformation)
            options = gen_feature_labels_schema(self._feature_labels, options)
            if len(options.keys()) > 0:
                obj_file_def = {
                    "fileType": ft.SPATIALDATA_ZARR.value,
                    "url": self.get_zarr_url(base_url, dataset_uid, obj_i),
                    "options": options
                }
                if self._request_init is not None:
                    obj_file_def['requestInit'] = self._request_init
                if self._coordination_values is not None:
                    obj_file_def['coordinationValues'] = self._coordination_values
                return obj_file_def
            return None

        return generator


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
        self.local_dir_uid = make_unique_filename(".multivec.zarr")

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
            obj_file_def = {
                "fileType": "genomic-profiles.zarr",
                "url": self.get_zarr_url(base_url, dataset_uid, obj_i)
            }
            if self._request_init is not None:
                obj_file_def['requestInit'] = self._request_init
            return obj_file_def
        return genomic_profiles_file_def_creator
