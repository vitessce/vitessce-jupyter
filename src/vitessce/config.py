import sys
import inspect
import copy as copy_module
import black
from collections import OrderedDict

from .constants import (
    norm_enum,
    CoordinationType as ct,
    ViewType as cm,  # TODO: change to vt
    FileType as ft
)

from .repr import make_repr, make_params_repr
from .utils import create_prefixed_get_next_scope_numeric


def _get_next_scope(prev_scopes):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    next_char_indices = [0]

    def next():
        r = []
        for char_index in next_char_indices:
            r = [chars[char_index]] + r
        increment = True
        for i in range(len(next_char_indices)):
            next_char_indices[i] += 1
            val = next_char_indices[i]
            if val >= len(chars):
                next_char_indices[i] = 0
            else:
                increment = False
                break

        if increment:
            next_char_indices.append(0)

        return "".join([str(j) for j in r])

    next_scope = next()
    while next_scope in prev_scopes:
        next_scope = next()

    return next_scope


class VitessceConfigDatasetFile:
    """
    A class to represent a file (described by a URL, data type, and file type) in a Vitessce view config dataset.
    """

    def __init__(self, file_type, url=None, coordination_values=None, options=None, data_type=None, request_init=None):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``VitessceConfigDataset.add_file()`` method.

        :param str file_type: A file type.
        :param url: A URL to this file. Can be a localhost URL or a remote URL.
        :type url: str or None
        :param coordination_values: Coordination values to pass to the file loader class.
        :type coordination_values: dict or None
        :param options: Extra options to pass to the file loader class.
        :type options: dict or list or None
        :param data_type: Deprecated / not used. Only included for backwards compatibility with the old API.
        :param request_init: Optional request init object to pass to the fetch API.
        :type request_init: dict or None
        """
        self.file = {
            "fileType": file_type
        }
        if url:
            self.file["url"] = url
        if options:
            self.file["options"] = options
        if coordination_values:
            self.file["coordinationValues"] = coordination_values
        if request_init:
            self.file["requestInit"] = request_init

    def __repr__(self):
        repr_dict = {
            "file_type": self.file["fileType"],
        }
        if "url" in self.file:
            repr_dict["url"] = self.file["url"]
        if "coordinationValues" in self.file:
            repr_dict["coordination_values"] = self.file["coordinationValues"]
        if "options" in self.file:
            repr_dict["options"] = self.file["options"]
        if "requestInit" in self.file:
            repr_dict["request_init"] = self.file["requestInit"]

        return make_repr(repr_dict, class_def=self.__class__)

    def to_dict(self):
        return self.file


class VitessceConfigDataset:
    """
    A class to represent a dataset (i.e. list of files containing common biological entities) in the Vitessce view config.
    """

    def __init__(self, uid, name, base_dir=None):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``VitessceConfig.add_dataset()`` method.

        :param str uid: A unique identifier for this dataset.
        :param str name: A name for this dataset.
        """
        self.dataset = {
            "uid": uid,
            "name": name,
            "files": [],
        }
        self.objs = []

        self.base_dir = base_dir

    def _to_py_params(self):
        return {
            "uid": self.dataset["uid"],
            "name": self.dataset["name"]
        }

    def get_name(self):
        """
        Get the name for this dataset.

        :returns: The name.
        :rtype: str
        """
        return self.dataset["name"]

    def get_uid(self):
        """
        Get the uid value for this dataset.

        :returns: The uid.
        :rtype: str
        """
        return self.dataset["uid"]

    def add_file(self, file_type, url=None, coordination_values=None, options=None, data_type=None, request_init=None):
        """
        Add a new file definition to this dataset instance.

        :param file_type: The file type. Must be compatible with the specified data type.
        :type file_type: str or vitessce.constants.FileType
        :param url: The URL for the file, pointing to either a local or remote location.
        :type url: str or None
        :param coordination_values: Coordination values to pass to the file loader class. Optional.
        :type coordination_values: dict or None
        :param options: Extra options to pass to the file loader class. Optional.
        :type options: dict or list or None
        :param data_type: Deprecated / not used. Only included for backwards compatibility with the old API.
        :param request_init: Optional request init object to pass to the fetch API.
        :type request_init: dict or None

        :returns: Self, to allow function chaining.
        :rtype: VitessceConfigDataset

        .. code-block:: python
            :emphasize-lines: 6-10

            from vitessce import VitessceConfig, DataType as dt, FileType as ft

            vc = VitessceConfig(schema_version="1.0.15", name='My Config')
            my_dataset = (
                vc.add_dataset(name='My Dataset')
                .add_file(
                    url="http://example.com/cells.json",
                    data_type=dt.CELLS,
                    file_type=ft.CELLS_JSON,
                )
            )
        """

        file_type_str = norm_enum(file_type, ft)

        self._add_file(VitessceConfigDatasetFile(
            url=url, file_type=file_type_str, coordination_values=coordination_values, options=options, request_init=request_init))
        return self

    def _add_file(self, obj):
        self.dataset["files"].append(obj)
        return self

    def add_object(self, obj):
        """
        Add a data object to this dataset instance.

        :param obj: A data object that can be served locally or which points to a remote storage provider. Typically, a subclass of AbstractWrapper.
        :type obj: vitessce.AbstractWrapper

        :returns: Self, to allow function chaining.
        :rtype: VitessceConfigDataset
        """
        obj.convert_and_save(self.dataset["uid"], len(self.objs), base_dir=self.base_dir)
        self.objs.append(obj)
        return self

    def _get_files(self):
        """
        Get a list of files and data objects associated with this dataset.

        :returns: The list of files and datasets.
        :rtype: list of VitessceConfigDatasetFile
        """
        return self.dataset["files"]

    def _get_objects(self):
        """
        Get a list of data objects associated with this dataset.

        :returns: The list of data objects.
        :rtype: list of AbstractWrapper instances
        """
        return self.objs

    def to_dict(self, base_url=None):
        obj_file_defs = []
        for obj in self.objs:
            obj_file_defs += obj.get_file_defs(base_url)

        return {
            **self.dataset,
            "files": [f.to_dict() for f in self.dataset["files"]] + obj_file_defs,
        }

    def get_routes(self):
        routes = []
        for obj in self.objs:
            routes += obj.get_routes()

        return routes

    def get_artifacts(self):
        artifacts = {}
        for obj in self.objs:
            artifacts.update(obj.get_artifacts())

        return artifacts

    def get_stores(self, base_url=None, prefer_local=False):
        stores = {}
        for obj in self.objs:
            stores = {
                **stores,
                **obj.get_stores(base_url, prefer_local=prefer_local)
            }

        return stores


class VitessceConfigViewHConcat:
    """
    A class to represent a horizontal concatenation of view instances.
    """

    def __init__(self, views, split=None):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``hconcat`` helper function.
        """
        self.views = views
        self.split = split

    def __or__(self, other):
        return hconcat(self, other)

    def __truediv__(self, other):
        return vconcat(self, other)


def hconcat(*views, split=None):
    """
    A helper function to create a ``VitessceConfigViewHConcat`` instance.

    :param \\*views: A variable number of views to concatenate horizontally.
    :type \\*views: VitessceConfigView or VitessceConfigViewVConcat or VitessceConfigViewHConcat
    :param split: Optional list of relative width fractions for each view.
    :type split: list of int

    :returns: The concatenated view instance.
    :rtype: VitessceConfigViewHConcat

    .. code-block:: python
        :emphasize-lines: 8

        from vitessce import VitessceConfig, ViewType as vt, hconcat, vconcat

        vc = VitessceConfig(schema_version="1.0.15")
        my_dataset = vc.add_dataset(name='My Dataset')
        v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
        v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
        v3 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
        vc.layout(hconcat(v1, vconcat(v2, v3), split = [1,2]))
        split = [1,2] would give v1 (1/3) width, and vconcat(v2, v3) 2/3 width
    """

    return VitessceConfigViewHConcat(views, split=split)


class VitessceConfigViewVConcat:
    """
    A class to represent a vertical concatenation of view instances.
    """

    def __init__(self, views, split=None):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``vconcat`` helper function.
        """
        self.views = views
        self.split = split

    def __or__(self, other):
        return hconcat(self, other)

    def __truediv__(self, other):
        return vconcat(self, other)


def vconcat(*views, split=None):
    """
    A helper function to create a ``VitessceConfigViewVConcat`` instance.

    :param \\*views: A variable number of views to concatenate vertically.
    :type \\*views: VitessceConfigView or VitessceConfigViewVConcat or VitessceConfigViewHConcat
    :param split: Optional list of relative height fractions for each view.
    :type split: list of int

    :returns: The concatenated view instance.
    :rtype: VitessceConfigViewVConcat

    .. code-block:: python
        :emphasize-lines: 8

        from vitessce import VitessceConfig, ViewType as vt, hconcat, vconcat

        vc = VitessceConfig(schema_version="1.0.15")
        my_dataset = vc.add_dataset(name='My Dataset')
        v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
        v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
        v3 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
        vc.layout(hconcat(v1, vconcat(v2, v3, [1,2])))
        split = [1,2] would give v2 (1/3) height, and v3, 2/3 height
    """
    return VitessceConfigViewVConcat(views, split=split)


def _use_coordination_by_dict_helper(scopes, coordination_scopes, coordination_scopes_by):
    """
    // Set this.coordinationScopes and this.coordinationScopesBy by recursion on `scopes`.
    /*
        // Destructured, `scopes` might look like:
        const {
        [CoordinationType.SPATIAL_IMAGE_LAYER]: [
            {
            scope: imageLayerScope,
            children: {
                [CoordinationType.IMAGE]: { scope: imageScope },
                [CoordinationType.SPATIAL_LAYER_VISIBLE]: { scope: imageVisibleScope },
                [CoordinationType.SPATIAL_LAYER_OPACITY]: { scope: imageOpacityScope },
                [CoordinationType.SPATIAL_IMAGE_CHANNEL]: [
                {
                    scope: imageChannelScopeR,
                    children: {
                    [CoordinationType.SPATIAL_TARGET_C]: { scope: rTargetScope },
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: { scope: rColorScope },
                    },
                },
                {
                    scope: imageChannelScopeG,
                    children: {
                    [CoordinationType.SPATIAL_TARGET_C]: { scope: gTargetScope },
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: { scope: gColorScope },
                    },
                },
                ],
            },
            },
        ],
        // ...
        } = scopes;
        // This would set the values to:
        this.coordinationScopes = {
        // Add the top-level coordination types to `coordinationScopes`.
        [CoordinationType.SPATIAL_IMAGE_LAYER]: [imageLayerScope.cScope],
        };
        this.coordinationScopesBy = {
        [CoordinationType.SPATIAL_IMAGE_LAYER]: {
            [CoordinationType.IMAGE]: {
            [imageLayerScope.cScope]: imageScope.cScope,
            },
            [CoordinationType.SPATIAL_LAYER_VISIBLE]: {
            [imageLayerScope.cScope]: imageVisibleScope.cScope,
            },
            [CoordinationType.SPATIAL_LAYER_OPACITY]: {
            [imageLayerScope.cScope]: imageOpacityScope.cScope,
            },
            [CoordinationType.SPATIAL_IMAGE_CHANNEL]: {
            [imageLayerScope.cScope]: [imageChannelScopeR.cScope, imageChannelScopeG.cScope],
            },
        },
        [CoordinationType.SPATIAL_IMAGE_CHANNEL]: {
            [CoordinationType.SPATIAL_TARGET_C]: {
            [imageChannelScopeR.cScope]: rTargetScope.cScope,
            [imageChannelScopeG.cScope]: gTargetScope.cScope,
            },
            [CoordinationType.SPATIAL_CHANNEL_COLOR]: {
            [imageChannelScopeR.cScope]: rColorScope.cScope,
            [imageChannelScopeG.cScope]: gColorScope.cScope,
            },
        },
        };
    */
    """

    # Recursive inner function.
    def process_level(parent_type, parent_scope, level_type, level_val):
        parent_type = norm_enum(parent_type, ct)
        level_type = norm_enum(level_type, ct)

        if isinstance(level_val, list):
            coordination_scopes_by[parent_type] = {
                **coordination_scopes_by.get(parent_type, {}),
                level_type: {
                    **coordination_scopes_by.get(parent_type, {}).get(level_type, {}),
                    parent_scope.c_scope: [child_val["scope"].c_scope for child_val in level_val],
                },
            }
            for child_val in level_val:
                if "children" in child_val:
                    # Continue recursion.
                    for next_level_type, next_level_val in child_val["children"].items():
                        process_level(level_type, child_val["scope"], next_level_type, next_level_val)
                # Else is the base case: no children
        else:
            coordination_scopes_by[parent_type] = {
                **coordination_scopes_by.get(parent_type, {}),
                level_type: {
                    **coordination_scopes_by.get(parent_type, {}).get(level_type, {}),
                    parent_scope.c_scope: level_val["scope"].c_scope,
                },
            }

            if "children" in level_val:
                # Continue recursion.
                for next_level_type, next_level_val in level_val["children"].items():
                    process_level(level_type, level_val["scope"], next_level_type, next_level_val)
            # Else is the base case: no children
    # End process_level inner function

    for top_level_type, top_level_val in scopes.items():
        top_level_type = norm_enum(top_level_type, ct)
        if isinstance(top_level_val, list):
            coordination_scopes[top_level_type] = [level_val["scope"].c_scope for level_val in top_level_val]

            for level_val in top_level_val:
                if "children" in level_val:
                    # Begin recursion.
                    for next_level_type, next_level_val in level_val["children"].items():
                        process_level(top_level_type, level_val["scope"], next_level_type, next_level_val)

        else:
            coordination_scopes[top_level_type] = top_level_val["scope"].c_scope
            if "children" in top_level_val:
                # Begin recursion.
                for next_level_type, next_level_val in top_level_val["children"].items():
                    next_level_type = norm_enum(next_level_type, ct)
                    process_level(top_level_type, top_level_val["scope"], next_level_type, next_level_val)

    return (coordination_scopes, coordination_scopes_by)


class VitessceConfigView:
    """
    A class to represent a view (i.e. visualization component) in the Vitessce view config layout.
    """

    def __init__(self, component, coordination_scopes, x, y, w, h):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``VitessceConfig.add_view()`` method.

        :param str component: The name of the component used for this view.
        :param dict coordination_scopes: A mapping of coordination types to coordination scopes.
        :param int x: An x-coordinate for this view in the grid.
        :param int y: A y-coordinate for this view in the grid.
        :param int w: A width for this view in the grid.
        :param int h: A height for this view in the grid.
        """
        self.view = {
            "component": component,
            "coordinationScopes": coordination_scopes,
            # "coordinationScopesBy": None, # TODO: initialize from parameter?
            "x": x,
            "y": y,
            "w": w,
            "h": h
        }

    def _to_py_params(self):
        params_dict = {
            "component": self.view["component"],
            "x": self.view["x"],
            "y": self.view["y"],
            "w": self.view["w"],
            "h": self.view["h"]
        }
        # Only include coordination_scopes if there are coordination scopes other than
        # the coorindation scope for the 'dataset' coordination type.
        non_dataset_coordination_scopes = {
            c_type: c_scope
            for c_type, c_scope in self.view["coordinationScopes"].items()
            if c_type != ct.DATASET.value
        }
        if len(non_dataset_coordination_scopes) > 0:
            params_dict["coordination_scopes"] = non_dataset_coordination_scopes
        return params_dict

    def get_coordination_scope(self, c_type):
        """
        Get the coordination scope name for a particular coordination type.

        :param str c_type: The coordination type of interest.

        :returns: The coordination scope name.
        :rtype: str or None
        """
        return self.view["coordinationScopes"].get(norm_enum(c_type, ct))

    def use_coordination(self, *c_scopes, allow_multiple_scopes_per_type=False):
        """
        Attach a coordination scope to this view instance. All views using the same coordination scope for a particular coordination type will effectively be linked together.

        :param \\*c_scopes: A variable number of coordination scope instances can be passed.
        :type \\*c_scopes: VitessceConfigCoordinationScope
        :param bool allow_multiple_scopes_per_type: Whether to allow multiple coordination scopes per coordination type. If true, multiple values for the same coordination type are treated as a list. If false, latest value for same coordination type is used. Defaults to False.

        :returns: Self, to allow chaining.
        :rtype: VitessceConfigView

        .. code-block:: python
            :emphasize-lines: 12-13

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            zoom_scope, x_scope, y_scope = vc.add_coordination(
                ct.SPATIAL_ZOOM,
                ct.SPATIAL_TARGET_X,
                ct.SPATIAL_TARGET_Y,
            )
            v1.use_coordination(zoom_scope, x_scope, y_scope)
            v2.use_coordination(zoom_scope, x_scope, y_scope)
            zoom_scope.set_value(2)
            x_scope.set_value(0)
            y_scope.set_value(0)
        """
        for c_scope in c_scopes:
            assert isinstance(c_scope, VitessceConfigCoordinationScope)
            existing_value = self.view["coordinationScopes"].get(c_scope.c_type)
            new_value = c_scope.c_scope
            if (existing_value is not None and allow_multiple_scopes_per_type):
                if (isinstance(existing_value, list)):
                    self.view["coordinationScopes"][c_scope.c_type] = existing_value + [new_value]
                else:
                    self.view["coordinationScopes"][c_scope.c_type] = [existing_value, new_value]
            else:
                self.view["coordinationScopes"][c_scope.c_type] = new_value
        return self

    def use_coordination_by_dict(self, scopes):
        """
        Attach potentially multi-level coordination scopes to this view.

        :param scopes: A value returned by ``VitessceConfig.add_coordination_by_dict``. Not intended to be a manually-constructed object.
        :type scopes: dict

        :returns: Self, to allow chaining.
        :rtype: VitessceConfigView

        .. code-block:: python
            :emphasize-lines: 11

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            spatial_view = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            scopes = vc.add_coordination_by_dict({
                ct.SPATIAL_ZOOM: 2,
                ct.SPATIAL_TARGET_X: 0,
                ct.SPATIAL_TARGET_Y: 0,
            })
            spatial_view.use_coordination_by_dict(scopes)
        """
        if "coordinationScopes" not in self.view["coordinationScopes"] or self.view["coordinationScopes"] is None:
            self.view["coordinationScopes"] = {}

        if "coordinationScopesBy" not in self.view or self.view["coordinationScopesBy"] is None:
            self.view["coordinationScopesBy"] = {}

        (next_coordination_scopes, next_coordination_scopes_by) = _use_coordination_by_dict_helper(
            scopes,
            self.view["coordinationScopes"],
            self.view["coordinationScopesBy"],
        )
        self.view["coordinationScopes"] = next_coordination_scopes
        self.view["coordinationScopesBy"] = next_coordination_scopes_by
        return self

    def use_meta_coordination(self, meta_scope):
        """
        Attach meta coordination scopes to this view.
        :param meta_scope: A meta coordination scope instance.
        :type meta_scope: VitessceConfigMetaCoordinationScope
        :returns: Self, to allow chaining.
        :rtype: VitessceConfigView

        .. code-block:: python
            :emphasize-lines: 16-17

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            spatial_view = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            lc_view = vc.add_view(vt.LAYER_CONTROLLER, dataset=my_dataset)
            scopes = vc.add_coordination_by_dict({
                ct.SPATIAL_ZOOM: 2,
                ct.SPATIAL_TARGET_X: 0,
                ct.SPATIAL_TARGET_Y: 0,
            })

            meta_scopes = vc.add_meta_coordination()
            meta_scopes.use_coordination_by_dict(scopes)

            spatial_view.use_meta_coordination(meta_scopes)
            lc_view.use_meta_coordination(meta_scopes)
        """
        if self.view["coordinationScopes"] is None:
            self.view["coordinationScopes"] = {}

        self.view["coordinationScopes"][ct.META_COORDINATION_SCOPES.value] = [
            *self.view["coordinationScopes"].get(ct.META_COORDINATION_SCOPES.value, []),
            meta_scope.meta_scope.c_scope,
        ]
        self.view["coordinationScopes"][ct.META_COORDINATION_SCOPES_BY.value] = [
            *self.view["coordinationScopes"].get(ct.META_COORDINATION_SCOPES_BY.value, []),
            meta_scope.meta_by_scope.c_scope,
        ]
        return self

    def set_xywh(self, x, y, w, h):
        """
        Set the dimensions for this view.

        :param int x: The horizontal position.
        :param int y: The vertical position.
        :param int w: The width.
        :param int h: The height.

        :returns: Self, to allow chaining.
        :rtype: VitessceConfigView
        """
        self.view["x"] = x
        self.view["y"] = y
        self.view["w"] = w
        self.view["h"] = h
        return self

    def set_props(self, **kwargs):
        """
        Set the props for this view.

        :param \\*\\*kwargs: A variable number of named props.

        :returns: Self, to allow chaining.
        :rtype: VitessceConfigView
        """
        if "props" in self.view:
            self.view["props"].update(kwargs)
        else:
            self.view["props"] = kwargs
        return self

    def get_props(self):
        """
        Get the props for this view.

        :returns: The props.
        :rtype: dict or None
        """
        return self.view.get("props")

    def to_dict(self):
        return self.view

    def __or__(self, other):
        return hconcat(self, other)

    def __truediv__(self, other):
        return vconcat(self, other)

# would import as CL for convenience


class CoordinationLevel:
    def __init__(self, value):
        self.value = value
        self.cached_value = None

    def set_cached(self, processed_level):
        self.cached_value = processed_level

    def get_cached(self):
        return self.cached_value

    def is_cached(self):
        return self.cached_value is not None


class VitessceConfigCoordinationScope:
    """
    A class to represent a coordination scope in the Vitessce view config coordination space.
    """

    def __init__(self, c_type, c_scope, c_value=None):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``VitessceConfig.add_coordination()`` method.

        :param str c_type: The coordination type for this coordination scope.
        :param str c_scope: The coordination scope name.
        :param c_value: The value for the coordination scope. Optional.
        """
        self.c_type = norm_enum(c_type, ct)
        self.c_scope = c_scope
        self.c_value = c_value

    def _to_py_params(self):
        return {
            "c_type": self.c_type,
            "c_scope": self.c_scope,
            "c_value": self.c_value,
        }

    def set_value(self, c_value):
        """
        Set the value of the coordination scope.

        :param any c_value: The coordination value to be set. Can be any value that is valid for the coordination type. Must be serializable to JSON.

        :returns: Self, to allow chaining.
        :rtype: VitessceConfigCoordinationScope

        .. code-block:: python
            :emphasize-lines: 14-16

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            zoom_scope, x_scope, y_scope = vc.add_coordination(
                ct.SPATIAL_ZOOM,
                ct.SPATIAL_TARGET_X,
                ct.SPATIAL_TARGET_Y,
            )
            v1.use_coordination(zoom_scope, x_scope, y_scope)
            v2.use_coordination(zoom_scope, x_scope, y_scope)
            zoom_scope.set_value(2)
            x_scope.set_value(0)
            y_scope.set_value(0)
        """
        self.c_value = c_value
        return self


class VitessceConfigMetaCoordinationScope:
    """
    A class representing a pair of coordination scopes, for metaCoordinationScopes and metaCoordinationScopesBy, respectively, in the coordination space.
    """

    def __init__(self, meta_scope, meta_by_scope):
        """
        Not meant to be instantiated directly, but instead created and returned by the ``VitessceConfig.add_meta_coordination()`` method.

        :param str meta_scope: The name of the coordination scope for metaCoordinationScopes.
        :param str meta_by_scope: The name of the coordination scope for metaCoordinationScopesBy.
        """
        self.meta_scope = VitessceConfigCoordinationScope(
            ct.META_COORDINATION_SCOPES.value,
            meta_scope,
        )
        self.meta_by_scope = VitessceConfigCoordinationScope(
            ct.META_COORDINATION_SCOPES_BY.value,
            meta_by_scope,
        )

    def use_coordination(self, *c_scopes):
        """
        Attach coordination scopes to this meta scope.

        :param  \\*c_scopes: A variable number of coordination scope instances.
        :type \\*c_scopes: VitessceConfigCoordinationScope
        :returns: Self, to allow chaining.
        :rtype: VitessceConfigMetaCoordinationScope
        """
        if self.meta_scope.c_value is None:
            self.meta_scope.set_value({})
        meta_scopes_val = self.meta_scope.c_value
        for c_scope in c_scopes:
            meta_scopes_val[c_scope.c_type] = c_scope.c_scope
        self.meta_scope.set_value(meta_scopes_val)
        return self

    def use_coordination_by_dict(self, scopes):
        """
        Attach potentially multi-level coordination scopes to this meta-scopes instance.

        :param scopes: A value returned by ``VitessceConfig.add_coordination_by_dict``. Not intended to be a manually-constructed object.
        :type scopes: dict

        :returns: Self, to allow chaining.
        :rtype: VitessceConfigMetaCoordinationScope

        .. code-block:: python
            :emphasize-lines: 14

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            spatial_view = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            lc_view = vc.add_view(vt.LAYER_CONTROLLER, dataset=my_dataset)
            scopes = vc.add_coordination_by_dict({
                ct.SPATIAL_ZOOM: 2,
                ct.SPATIAL_TARGET_X: 0,
                ct.SPATIAL_TARGET_Y: 0,
            })

            meta_scopes = vc.add_meta_coordination()
            meta_scopes.use_coordination_by_dict(scopes)

            spatial_view.use_meta_coordination(meta_scopes)
            lc_view.use_meta_coordination(meta_scopes)
        """
        if self.meta_scope.c_value is None:
            self.meta_scope.set_value({})

        if self.meta_by_scope.c_value is None:
            self.meta_by_scope.set_value({})

        (meta_scopes_val, meta_by_scopes_val) = _use_coordination_by_dict_helper(
            scopes,
            self.meta_scope.c_value,
            self.meta_by_scope.c_value,
        )
        self.meta_scope.set_value(meta_scopes_val)
        self.meta_by_scope.set_value(meta_by_scopes_val)
        return self


class VitessceConfig:
    """
    A class to represent a Vitessce view config.
    """

    def __init__(self, schema_version, name=None, description=None, base_dir=None):
        """
        Construct a Vitessce view config object.

        :param str schema_version: The view config schema version.
        :param str name: A name for the view config. Optional.
        :param str description: A description for the view config. Optional.
        :param str base_dir: A local path to a directory to be served. If provided, local data objects will be configured relative to this directory. Optional.

        .. code-block:: python
            :emphasize-lines: 3

            from vitessce import VitessceConfig

            vc = VitessceConfig(schema_version="1.0.15", name='My Config')
        """
        self.config = {
            "version": schema_version,
            "name": name,
            "description": description,
            "datasets": [],
            "coordinationSpace": {},
            "layout": [],
            "initStrategy": "auto"
        }

        self.get_next_scope = _get_next_scope

        if name is None:
            self.config["name"] = ""
        else:
            self.config["name"] = name
        if description is None:
            self.config["description"] = ""
        else:
            self.config["description"] = description

        self.background_servers = {}

        self.base_dir = base_dir

    def register_server(self, port, server):
        self.background_servers[port] = server

    def stop_server(self, port):
        if port in self.background_servers:
            self.background_servers[port].stop()
            del self.background_servers[port]

    def stop_all_servers(self):
        for server in self.background_servers.values():
            server.stop()
        self.background_servers = {}

    def has_server(self, port):
        return port in self.background_servers

    def _to_py_params(self):
        return {
            "schema_version": self.config["version"],
            "name": self.config["name"],
            "description": self.config["description"],
        }

    def add_dataset(self, name="", uid=None, files=None, objs=None):
        """
        Add a dataset to the config.

        :param str name: A name for this dataset.
        :param str uid: A unique identifier for this dataset. Optional. If None, one will be automatically generated.
        :param files: A list of VitessceConfigDatasetFile instances. optional.
        :type files: list or None
        :param objs: A list of AbstractWrapper instances. Optional.
        :type objs: list or None

        :returns: The instance for the new dataset.
        :rtype: VitessceConfigDataset

        .. code-block:: python
            :emphasize-lines: 5

            from vitessce import VitessceConfig, DataType as dt, FileType as ft

            vc = VitessceConfig(schema_version="1.0.15", name='My Config')
            my_dataset = (
                vc.add_dataset(name='My Dataset')
                .add_file(
                    url="http://example.com/cells.json",
                    file_type=ft.CELLS_JSON,
                )
            )
        """
        uid = uid if uid is not None else self.get_next_scope(
            [d.dataset['uid'] for d in self.config["datasets"]])
        assert isinstance(uid, str)
        vcd = VitessceConfigDataset(uid, name, base_dir=self.base_dir)
        self.config["datasets"].append(vcd)
        [d_scope] = self.add_coordination(ct.DATASET)
        d_scope.set_value(uid)

        if isinstance(files, list):
            for obj in files:
                vcd._add_file(obj)
        if isinstance(objs, list):
            for obj in objs:
                vcd.add_object(obj)

        return vcd

    def get_dataset_by_uid(self, uid):
        """
        Get a dataset associated with this configuration based on its uid.

        :param str uid: The unique identifier for the dataset of interest.

        :returns: The dataset object.
        :rtype: VitessceConfigDataset or None
        """
        for dataset in self.config["datasets"]:
            if dataset.get_uid() == uid:
                return dataset
        return None

    def get_dataset_by_coordination_scope_name(self, query_scope_name):
        """
        Get a dataset associated with this configuration based on a coordination scope.

        :param str query_scope_name: The unique identifier for the dataset coordination scope of interest.

        :returns: The dataset object.
        :rtype: VitessceConfigDataset or None
        """
        if ct.DATASET.value in self.config["coordinationSpace"]:
            for scope_name, dataset_scope in self.config["coordinationSpace"][ct.DATASET.value].items():
                if scope_name == query_scope_name:
                    return self.get_dataset_by_uid(dataset_scope.c_value)
        return None

    def get_datasets(self):
        """
        Get the datasets associated with this configuration.

        :returns: The list of dataset objects.
        :rtype: list of VitessceConfigDataset
        """
        return self.config["datasets"]

    def add_view(self, view_type, dataset=None, dataset_uid=None, x=0, y=0, w=1, h=1, mapping=None, coordination_scopes=None, props=None):
        """
        Add a view to the config.

        :param view_type: A component name, either as a string or using the ViewType enum values.
        :type view_type: str or vitessce.constants.ViewType
        :param dataset: A dataset instance to be used for the data visualized in this view. Must provide dataset or dataset_uid, but not both.
        :type dataset: VitessceConfigDataset or None
        :param dataset_uid: A unique ID for a dataset to be used for the data visualized in this view. Must provide dataset or dataset_uid, but not both.
        :type dataset_uid: str or None

        :param str mapping: An optional convenience parameter for setting the EMBEDDING_TYPE coordination scope value. This parameter is only applicable to the SCATTERPLOT component.
        :param int x: The horizontal position of the view. Must be an integer between 0 and 11. Optional. This will be ignored if you call the `layout` method of this class using `VitessceConfigViewHConcat` and `VitessceConfigViewVConcat` objects.
        :param int y: The vertical position of the view. Must be an integer between 0 and 11. Optional. This will be ignored if you call the `layout` method of this class using `VitessceConfigViewHConcat` and `VitessceConfigViewVConcat` objects.
        :param int w: The width of the view. Must be an integer between 1 and 12. Optional. This will be ignored if you call the `layout` method of this class using `VitessceConfigViewHConcat` and `VitessceConfigViewVConcat` objects.
        :param int h: The height of the view. Must be an integer between 1 and 12. Optional. This will be ignored if you call the `layout` method of this class using `VitessceConfigViewHConcat` and `VitessceConfigViewVConcat` objects.
        :param coordination_scopes: A mapping from coordination types to coordination scope names for this view.
        :type coordination_scopes: dict or None
        :param props: Props to set for the view using the VitessceConfigView.set_props method.
        :type props: dict or None

        :returns: The instance for the new view.
        :rtype: VitessceConfigView

        .. code-block:: python
            :emphasize-lines: 5-6

            from vitessce import VitessceConfig, ViewType as vt

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v2 = vc.add_view(vt.SCATTERPLOT, dataset=my_dataset, mapping="X_umap")
        """
        # User should only provide dataset or dataset_uid, but not both.
        assert isinstance(dataset, VitessceConfigDataset) or isinstance(
            dataset_uid, str)
        assert dataset is None or dataset_uid is None
        component = view_type
        # assert type(component) in [str, cm]

        if dataset is None:
            dataset = self.get_dataset_by_uid(dataset_uid)
            if dataset is None:
                raise ValueError(
                    "A dataset with the provided dataset_uid could not be found.")

        component_str = norm_enum(component, cm)

        # Find the coordination scope name associated with the dataset
        dataset_matches = [
            scope_name
            for scope_name, dataset_scope in self.config["coordinationSpace"][ct.DATASET.value].items()
            if dataset_scope.c_value == dataset.dataset["uid"]
        ] if ct.DATASET.value in self.config["coordinationSpace"].keys() else []
        if len(dataset_matches) == 1:
            dataset_scope = dataset_matches[0]
        else:
            raise ValueError(
                "No coordination scope matching the dataset parameter could be found in the coordination space.")

        # Set up the view's dataset coordination scope based on the dataset parameter.
        internal_coordination_scopes = {
            ct.DATASET.value: dataset_scope
        }
        if coordination_scopes is not None:
            internal_coordination_scopes.update(coordination_scopes)
        vcv = VitessceConfigView(
            component_str, internal_coordination_scopes, x, y, w, h)

        # Use the mapping parameter if component is scatterplot and the mapping is not None
        if mapping is not None:
            [et_scope] = self.add_coordination(ct.EMBEDDING_TYPE)
            et_scope.set_value(mapping)
            vcv.use_coordination(et_scope)

        if isinstance(props, dict):
            vcv.set_props(**props)

        self.config["layout"].append(vcv)
        return vcv

    def add_coordination(self, *c_types):
        """
        Add scope(s) for new coordination type(s) to the config.

        :param \\*c_types: A variable number of coordination types.
        :type \\*c_types: str or vitessce.constants.CoordinationType

        :returns: The instances for the new scope objects corresponding to each coordination type. These can be linked to views via the ``VitessceConfigView.use_coordination()`` method.
        :rtype: list[VitessceConfigCoordinationScope]

        .. code-block:: python
            :emphasize-lines: 7-11

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            zoom_scope, x_scope, y_scope = vc.add_coordination(
                ct.SPATIAL_ZOOM,
                ct.SPATIAL_TARGET_X,
                ct.SPATIAL_TARGET_Y,
            )
            v1.use_coordination(zoom_scope, x_scope, y_scope)
            v2.use_coordination(zoom_scope, x_scope, y_scope)
            zoom_scope.set_value(2)
            x_scope.set_value(0)
            y_scope.set_value(0)
        """
        result = []
        for c_type in c_types:
            c_type_str = norm_enum(c_type, ct)
            prev_scopes = list(self.config["coordinationSpace"][c_type_str].keys(
            )) if c_type_str in self.config["coordinationSpace"].keys() else []
            scope = VitessceConfigCoordinationScope(
                c_type_str, self.get_next_scope(prev_scopes))
            if scope.c_type not in self.config["coordinationSpace"]:
                self.config["coordinationSpace"][scope.c_type] = {}
            self.config["coordinationSpace"][scope.c_type][scope.c_scope] = scope
            result.append(scope)
        return result

    def add_meta_coordination(self):
        """
        Initialize a new meta coordination scope in the coordination space, and get a reference to it in the form of a meta coordination scope instance.

        :returns: A new meta coordination scope instance.
        :rtype: VitessceConfigMetaCoordinationScope

        .. code-block:: python
            :emphasize-lines: 13

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            spatial_view = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            lc_view = vc.add_view(vt.LAYER_CONTROLLER, dataset=my_dataset)
            scopes = vc.add_coordination_by_dict({
                ct.SPATIAL_ZOOM: 2,
                ct.SPATIAL_TARGET_X: 0,
                ct.SPATIAL_TARGET_Y: 0,
            })

            meta_scopes = vc.add_meta_coordination()
            meta_scopes.use_coordination_by_dict(scopes)

            spatial_view.use_meta_coordination(meta_scopes)
            lc_view.use_meta_coordination(meta_scopes)
        """
        prev_meta_scopes = self.config["coordinationSpace"].get(ct.META_COORDINATION_SCOPES.value, {}).keys()
        prev_meta_by_scopes = self.config["coordinationSpace"].get(ct.META_COORDINATION_SCOPES_BY.value, {}).keys()

        meta_container = VitessceConfigMetaCoordinationScope(
            self.get_next_scope(prev_meta_scopes),
            self.get_next_scope(prev_meta_by_scopes),
        )
        if ct.META_COORDINATION_SCOPES.value not in self.config["coordinationSpace"]:
            self.config["coordinationSpace"][ct.META_COORDINATION_SCOPES.value] = {}
        if ct.META_COORDINATION_SCOPES_BY.value not in self.config["coordinationSpace"]:
            self.config["coordinationSpace"][ct.META_COORDINATION_SCOPES_BY.value] = {}
        self.config["coordinationSpace"][ct.META_COORDINATION_SCOPES.value][meta_container.meta_scope.c_scope] = meta_container.meta_scope
        self.config["coordinationSpace"][ct.META_COORDINATION_SCOPES_BY.value][meta_container.meta_by_scope.c_scope] = meta_container.meta_by_scope
        return meta_container

    def add_coordination_by_dict(self, input_val):
        """
        Set up the initial values for multi-level coordination in the coordination space. Get a reference to these values to pass to the ``useCoordinationByObject`` method of either view or meta coordination scope instances.

        :param input_val: A (potentially nested) object with coordination types as keys and values being either the initial coordination value, a ``VitessceConfigCoordinationScope`` instance, or a ``CoordinationLevel`` instance. The CoordinationLevel constructor takes an array of objects as its argument to support nesting.
        :type input_val: dict
        :returns: A (potentially nested) object with coordination types as keys and values being either ``{ scope }``, ``{ scope, children }``, or an array of these. Not intended to be manipulated before being passed to a ``useCoordinationByObject`` function.
        :rtype: dict

        .. code-block:: python
            :emphasize-lines: 7-11

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            spatial_view = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            lc_view = vc.add_view(vt.LAYER_CONTROLLER, dataset=my_dataset)
            scopes = vc.add_coordination_by_dict({
                ct.SPATIAL_ZOOM: 2,
                ct.SPATIAL_TARGET_X: 0,
                ct.SPATIAL_TARGET_Y: 0,
            })
        """

        # Developer notes
        """
        /*
        // The value for `input` might look like:
        {
            [CoordinationType.SPATIAL_IMAGE_LAYER]: CL([
            {
                [CoordinationType.IMAGE]: 'S-1905-017737_bf',
                [CoordinationType.SPATIAL_LAYER_VISIBLE]: true,
                [CoordinationType.SPATIAL_LAYER_OPACITY]: 1,
                [CoordinationType.SPATIAL_IMAGE_CHANNEL]: CL([
                {
                    [CoordinationType.SPATIAL_TARGET_C]: 0,
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: [255, 0, 0],
                },
                {
                    [CoordinationType.SPATIAL_TARGET_C]: 1,
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: [0, 255, 0],
                },
                ]),
            },
            ]),
            [CoordinationType.SPATIAL_SEGMENTATION_LAYER]: CL([
            {
                [CoordinationType.IMAGE]: 'S-1905-017737',
                [CoordinationType.SPATIAL_LAYER_VISIBLE]: true,
                [CoordinationType.SPATIAL_LAYER_OPACITY]: 1,
                [CoordinationType.SPATIAL_SEGMENTATION_CHANNEL]: CL([
                {
                    [CoordinationType.OBS_TYPE]: 'Cortical Interstitia',
                    [CoordinationType.SPATIAL_TARGET_C]: 0,
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: [255, 0, 0],
                },
                {
                    [CoordinationType.OBS_TYPE]: 'Non-Globally Sclerotic Glomeruli',
                    [CoordinationType.SPATIAL_TARGET_C]: 1,
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: [255, 0, 0],
                },
                {
                    [CoordinationType.OBS_TYPE]: 'Globally Sclerotic Glomeruli',
                    [CoordinationType.SPATIAL_TARGET_C]: 2,
                    [CoordinationType.SPATIAL_CHANNEL_COLOR]: [255, 0, 0],
                },
                ]),
            },
            ]),
        }
        // Which would correspond to this `output`,
        // a valid input for `VitessceConfigMetaCoordinationScope.useComplexCoordination()`:
        {
            [CoordinationType.SPATIAL_IMAGE_LAYER]: [
            {
                scope: imageLayerScope,
                children: {
                [CoordinationType.IMAGE]: { scope: imageScope },
                [CoordinationType.SPATIAL_LAYER_VISIBLE]: { scope: imageVisibleScope },
                [CoordinationType.SPATIAL_LAYER_OPACITY]: { scope: imageOpacityScope },
                [CoordinationType.SPATIAL_IMAGE_CHANNEL]: [
                    {
                    scope: imageChannelScopeR,
                    children: {
                        [CoordinationType.SPATIAL_TARGET_C]: { scope: rTargetScope },
                        [CoordinationType.SPATIAL_CHANNEL_COLOR]: { scope: rColorScope },
                    },
                    },
                    {
                    scope: imageChannelScopeG,
                    children: {
                        [CoordinationType.SPATIAL_TARGET_C]: { scope: gTargetScope },
                        [CoordinationType.SPATIAL_CHANNEL_COLOR]: { scope: gColorScope },
                    },
                    },
                ],
                },
            },
            ],
            // ...
        }
        */
        """
        def process_level(level):
            result = {}
            if level is None:
                return result
            for c_type, next_level_or_initial_value in level.items():
                c_type_str = norm_enum(c_type, ct)
                # Check if value of object is instanceof CoordinationLevel
                # (otherwise assume it is the coordination value).
                if isinstance(next_level_or_initial_value, CoordinationLevel):
                    next_level = next_level_or_initial_value.value
                    if isinstance(next_level, list):
                        if next_level_or_initial_value.is_cached():
                            result[c_type_str] = next_level_or_initial_value.get_cached()
                        else:
                            def map_func(next_el):
                                (dummy_scope, ) = self.add_coordination(c_type_str)
                                # TODO: set a better initial value for dummy cases.
                                dummy_scope.set_value('__dummy__')
                                return {
                                    "scope": dummy_scope,
                                    "children": process_level(next_el),
                                }
                            processed_level = list(map(map_func, next_level))
                            next_level_or_initial_value.set_cached(processed_level)
                            result[c_type_str] = processed_level
                    else:
                        raise ValueError('Expected CoordinationLevel.value to be an array.')
                else:
                    # Base case.
                    initial_value = next_level_or_initial_value
                    if isinstance(initial_value, VitessceConfigCoordinationScope):
                        result[c_type_str] = {"scope": initial_value}
                    else:
                        (scope, ) = self.add_coordination(c_type_str)
                        scope.set_value(initial_value)
                        result[c_type_str] = {"scope": scope}
            return result
        # End process_level function

        # Begin recursion.
        output_val = process_level(input_val)
        return output_val

    def link_views_by_dict(self, views, input_val, meta=True, scope_prefix=None):
        """
        A convenience function for setting up multi-level and meta-coordination scopes across a set of views.

        :param views: An array of view objects to link together.
        :type views: list[VitessceConfigView]
        :param input_val: A (potentially nested) object with coordination types as keys and values being either the initial coordination value, a ``VitessceConfigCoordinationScope`` instance, or a ``CoordinationLevel`` instance. The CoordinationLevel constructor takes an array of objects as its argument to support nesting.
        :type input_val: dict
        :param bool meta: Whether or not to use meta-coordination to link the views. Optional.
        :returns: Self, to allow chaining.
        :rtype: VitessceConfig

        .. code-block:: python
            :emphasize-lines: 7-11

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            spatial_view = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            lc_view = vc.add_view(vt.LAYER_CONTROLLER, dataset=my_dataset)
            scopes = vc.link_views_by_dict([spatial_view, lc_view], {
                ct.SPATIAL_ZOOM: 2,
                ct.SPATIAL_TARGET_X: 0,
                ct.SPATIAL_TARGET_Y: 0,
            })
        """
        if scope_prefix:
            self.get_next_scope = create_prefixed_get_next_scope_numeric(scope_prefix)
        scopes = self.add_coordination_by_dict(input_val)
        if meta:
            meta_scope = self.add_meta_coordination()
            meta_scope.use_coordination_by_dict(scopes)

            for view in views:
                view.use_meta_coordination(meta_scope)
        else:
            for view in views:
                view.use_coordination_by_dict(scopes)
        if scope_prefix:
            self.get_next_scope = _get_next_scope
        return self

    def set_coordination_value(self, c_type, c_scope, c_value):
        """
        Set the value for a coordination scope. If a coordination object for the coordination type does not yet exist in the coordination space, it will be created.

        :param str c_type: The coordination type for this coordination scope.
        :param str c_scope: The coordination scope name.
        :param any c_value: The value for the coordination scope. Optional.

        :returns: The coordination scope instance.
        :rtype: VitessceConfigCoordinationScope
        """
        scope = VitessceConfigCoordinationScope(c_type, c_scope, c_value)
        if scope.c_type not in self.config["coordinationSpace"]:
            self.config["coordinationSpace"][scope.c_type] = {}
        self.config["coordinationSpace"][scope.c_type][scope.c_scope] = scope
        return scope

    def link_views(self, views, c_types, c_values=None, allow_multiple_scopes_per_type=False):
        """
        A convenience function for setting up new coordination scopes across a set of views.

        :param views: views An array of view objects to link together.
        :type views: list of VitessceConfigView
        :param c_types: The coordination types on which to coordinate the views.
        :type c_types: list of str or list of vitessce.constants.CoordinationType
        :param list c_values: Initial values corresponding to each coordination type. Should have the same length as the c_types array. Optional.
        :param bool allow_multiple_scopes_per_type: Whether to allow multiple coordination scopes per coordination type. If true, multiple values for the same coordination type are treated as a list. If false, latest value for same coordination type is used. Defaults to False.

        :returns: Self, to allow chaining.
        :rtype: VitessceConfig
        """
        c_scopes = self.add_coordination(*c_types)
        for view in views:
            for c_scope in c_scopes:
                view.use_coordination(c_scope, allow_multiple_scopes_per_type=allow_multiple_scopes_per_type)

        if c_values is not None and len(c_values) == len(c_types):
            for i, c_scope in enumerate(c_scopes):
                c_scope.set_value(c_values[i])

        return self

    def layout(self, view_concat):
        """
        Create a multi-view layout based on (potentially recursive) view concatenations.

        :param view_concat: Views arranged by concatenating vertically or horizontally. Alternatively, a single view can be passed.
        :type view_concat: VitessceConfigViewHConcat or VitessceConfigViewVConcat or VitessceConfigView

        :returns: Self, to allow chaining.
        :rtype: VitessceConfig

        .. code-block:: python
            :emphasize-lines: 8

            from vitessce import VitessceConfig, ViewType as vt, hconcat, vconcat

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v3 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            vc.layout(hconcat(v1, vconcat(v2, v3)))

        .. code-block:: python
            :emphasize-lines: 8

            from vitessce import VitessceConfig, ViewType as vt

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v2 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            v3 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            vc.layout(v1 | (v2 / v3)) # * magic * (alternative syntax)
        """

        def _layout(obj, x_min, x_max, y_min, y_max):
            w = x_max - x_min
            h = y_max - y_min
            if isinstance(obj, VitessceConfigView):
                obj.set_xywh(x_min, y_min, w, h)
            else:
                views = obj.views
                # If the split parameter is provided, it must have the same length as the views.
                assert obj.split is None or len(obj.split) == len(views)
                split = obj.split or [1] * len(views)  # Default to equal split if not provided
                total = sum(split)

                if isinstance(obj, VitessceConfigViewHConcat):
                    widths = [int(s / total * w) for s in split]
                    x_pos = x_min
                    for view, width in zip(views, widths):
                        _layout(view, x_pos, x_pos + width, y_min, y_max)
                        x_pos += width
                if isinstance(obj, VitessceConfigViewVConcat):
                    heights = [int(s / total * h) for s in split]
                    y_pos = y_min
                    for view, height in zip(views, heights):
                        _layout(view, x_min, x_max, y_pos, y_pos + height)
                        y_pos += height

        # Recursively set the values (x,y,w,h) for each view.
        _layout(view_concat, 0, 12, 0, 12)

        # TODO: decide how to handle views that were omitted from the `view_concat` parameter
        # TODO: decide how to handle .add_view() being called after .layout() has been called

        return self

    def to_dict(self, base_url=None):
        """
        Convert the view config instance to a dict object.

        :param str base_url: Optional parameter for non-remote data to specify the url from which the data will be served.

        :returns: The view config as a dict. Useful for serializing to JSON.
        :rtype: dict
        """
        return {
            **self.config,
            "datasets": [d.to_dict(base_url) for d in self.config["datasets"]],
            "coordinationSpace": dict([
                (c_type, dict([
                    (c_scope_name, c_scope.c_value) for c_scope_name, c_scope in c_scopes.items()
                ])) for c_type, c_scopes in self.config["coordinationSpace"].items()
            ]),
            # TODO: compute the x,y,w,h values if not explicitly defined
            "layout": [c.to_dict() for c in self.config["layout"]]
        }

    def get_views(self):
        """
        Provides all the views in the config.layout object list

        :returns: A list of VitessceConfigView objects.

        """
        return self.config["layout"]

    def get_view_by_index(self, index):
        """
        Get a view from the layout by the index specified by the 'index' parameter.

        :param index: Index (int) of the view in the Layout array.
        :type index: int

        :returns: The view corresponding to the provided index
        :rtype: VitessceConfigView or None if not found
        """
        if isinstance(index, int):
            if 0 <= index < len(self.config["layout"]):
                return self.config["layout"][index]
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError("index must be an integer")

    def get_first_view_by_type(self, view_type):
        """
        Get a view from the layout by view type (component) specified by the 'view_type' parameter.

        :param view_type: The view type (str) of the view in the Layout array.
        :type view_type:  str

        :returns: The view corresponding to the provided view_type.
        :rtype: VitessceConfigView or None if not found
        """
        if isinstance(view_type, str):
            for view in self.config["layout"]:
                if view.view["component"].lower() == view_type.lower():
                    return view
            raise ValueError(f"No view found with component view_type: {view_type}")
        else:
            raise TypeError("view_type must be a string representing the view type")

    def remove_view_by_index(self, index):
        """
        Removes a view from the layout by the index specified by the 'index' parameter.

        :param index: the index (int) of the view
        :type index: int

        :returns: The layout component of the config corresponding to the specified index
        :rtype: VitessceConfigView or None if not found

        """
        if isinstance(index, int):
            if 0 <= index < len(self.config["layout"]):
                return self.config["layout"].pop(index)
            else:
                raise IndexError("Index out of range")
        else:
            raise TypeError("index must be an integer")

    def remove_first_view_by_type(self, view_type):
        """
        Removes a view from the layout by the view type (component) specified by the 'view_type' parameter.

        :param view_by: A component view_type (str).
        :type view_by: str

        :returns: The layout component  of the config corresponding to the specified view_type
        :rtype: VitessceConfigView or None if not found

        """
        if isinstance(view_type, str):
            for i, view in enumerate(self.config["layout"]):
                if view.view["component"].lower() == view_type.lower():
                    return self.config["layout"].pop(i)
            raise ValueError(f"No view found with component type: {view_type}")
        else:
            raise TypeError("view_by must a string representing component type")

    def get_routes(self):
        """
        Convert the routes for this view config from the datasets.

        :returns: A list of server routes.
        :rtype: list[starlette.routing.Route]
        """
        routes = []
        for d in self.config["datasets"]:
            routes += d.get_routes()
        return routes

    def get_artifacts(self):
        """
        Get all artifacts for this view config from the datasets.

        :returns: A dict mapping artifact URLs to corresponding artifact objects.
        :rtype: dict[str, lamindb.Artifact]
        """
        artifacts = {}
        for d in self.config["datasets"]:
            artifacts.update(d.get_artifacts())
        return artifacts

    def get_stores(self, base_url=None, prefer_local=False):
        """
        Convert the routes for this view config from the datasets.

        :returns: A list of server routes.
        :rtype: list[starlette.routing.Route]
        """
        stores = {}
        for d in self.config["datasets"]:
            stores = {
                **stores,
                **d.get_stores(base_url, prefer_local=prefer_local)
            }
        return stores

    def to_python(self):
        """
        Convert the VitessceConfig instance to a one-line Python code snippet that can be used to generate it.

        :returns: (A list of classes from the vitessce package used in the code block, The formatted code block)
        :rtype: (list[str], str)
        """
        classes_to_import = OrderedDict()
        classes_to_import[VitessceChainableConfig.__name__] = True
        code_block = f'{VitessceChainableConfig.__name__}({make_params_repr(self._to_py_params())})'

        for vcd in self.config["datasets"]:
            vcd_file_list_contents = ', '.join(
                [repr(f) for f in vcd._get_files()])
            vcd_obj_list_contents = ', '.join(
                [repr(f) for f in vcd._get_objects()])
            add_dataset_func = self.add_dataset.__name__
            add_dataset_params_list = [
                make_params_repr(vcd._to_py_params()),
            ]
            if len(vcd._get_files()) > 0:
                add_dataset_params_list.append(
                    f'files=[{vcd_file_list_contents}]')
                classes_to_import[VitessceConfigDatasetFile.__name__] = True
            if len(vcd._get_objects()) > 0:
                add_dataset_params_list.append(
                    f'objs=[{vcd_obj_list_contents}]')
            add_dataset_params = ', '.join(add_dataset_params_list)
            code_block += f'.{add_dataset_func}({add_dataset_params})'
            for obj in vcd._get_objects():
                if "vitessce" in sys.modules and obj.__class__.__name__ in dict(inspect.getmembers(sys.modules["vitessce"])):
                    classes_to_import[obj.__class__.__name__] = True
        for c_type, c_obj in self.config["coordinationSpace"].items():
            if c_type != ct.DATASET.value:
                for c_scope_name, c_scope in c_obj.items():
                    set_coordination_func = self.set_coordination_value.__name__
                    set_coordination_params = make_params_repr(
                        c_scope._to_py_params())
                    code_block += f'.{set_coordination_func}({set_coordination_params})'

        for vcv in self.config["layout"]:
            dataset_for_view = self.get_dataset_by_coordination_scope_name(
                vcv.get_coordination_scope(ct.DATASET.value))
            if dataset_for_view is not None:
                dataset_uid = dataset_for_view.get_uid()
            elif len(self.config["datasets"]) == 1:
                # If there is only one dataset available, assume it is the dataset for this view.
                dataset_uid = self.config["datasets"][0].get_uid()
            else:
                raise ValueError(
                    "At least one dataset must be present in the config before adding a view.")
            add_view_params_dict = {
                "dataset_uid": dataset_uid,
            }
            add_view_params_dict.update(vcv._to_py_params())
            if vcv.get_props() is not None:
                add_view_params_dict["props"] = vcv.get_props()
            add_view_func = self.add_view.__name__
            add_view_params = make_params_repr(add_view_params_dict)
            code_block += f'.{add_view_func}({add_view_params})'
        formatted_code_block = black.format_str(
            code_block, mode=black.FileMode())
        return list(classes_to_import), formatted_code_block

    @staticmethod
    def from_dict(config):
        """
        Helper function to construct a Vitessce view config object from an existing config.

        :param dict config: An existing Vitessce view config as a dict to allow manipulation through the ``VitessceConfig`` API.

        :returns: The config instance.
        :rtype: VitessceConfig

        .. code-block:: python
            :emphasize-lines: 3

            from vitessce import VitessceConfig

            vc = VitessceConfig.from_dict(my_existing_config)
        """
        vc = VitessceConfig(
            schema_version=config["version"], name=config["name"], description=config["description"])

        # Add each dataset from the incoming config.
        for d in config["datasets"]:
            new_dataset = vc.add_dataset(uid=d["uid"], name=d["name"])
            for f in d["files"]:
                new_dataset.add_file(
                    file_type=f["fileType"],
                    url=f.get("url"),
                    coordination_values=f.get("coordinationValues"),
                    options=f.get("options"),
                    request_init=f.get("requestInit")
                )
        if 'coordinationSpace' in config:
            for c_type in config['coordinationSpace'].keys():
                if c_type != ct.DATASET.value:
                    c_obj = config['coordinationSpace'][c_type]
                    vc.config['coordinationSpace'][c_type] = {}
                    for c_scope_name, c_scope_value in c_obj.items():
                        scope = VitessceConfigCoordinationScope(
                            c_type, c_scope_name)
                        scope.set_value(c_scope_value)
                        vc.config['coordinationSpace'][c_type][c_scope_name] = scope

        for c in config['layout']:
            c_coord_scopes = c['coordinationScopes'] if 'coordinationScopes' in c.keys() else {
            }
            if len(config["datasets"]) > 1 and ct.DATASET.value not in c_coord_scopes:
                raise ValueError(
                    "Multiple datasets are present, so every view must have an explicit dataset coordination scope.")
            new_view = VitessceConfigView(
                c['component'], c_coord_scopes, c['x'], c['y'], c['w'], c['h'])
            if 'props' in c.keys():
                new_view.set_props(**c['props'])
            vc.config['layout'].append(new_view)

        return vc

    @staticmethod
    def from_object(obj, schema_version, name=None, description=None):
        """
        Helper function to automatically construct a Vitessce view config object from a single-cell dataset object.
        Particularly helpful when using the ``VitessceWidget`` Jupyter widget.

        :param obj: A single-cell dataset in the format of a commonly-used single-cell or imaging data analysis package.
        :type obj: anndata.AnnData or loompy.LoomConnection or zarr.Store
        :param str schema_version: The schema version to pass to the VitessceConfig constructor.

        :returns: The config instance.
        :rtype: VitessceConfig

        .. code-block:: python
            :emphasize-lines: 3

            from vitessce import VitessceConfig

            vc = VitessceConfig.from_object(my_scanpy_object, schema_version="1.0.15")
        """
        vc = VitessceConfig(schema_version=schema_version, name=name, description=description)

        # The data object may modify the view config if it implements the auto_view_config() method.
        obj.auto_view_config(vc)

        return vc

    def widget(self, **kwargs):
        """
        Instantiate a VitessceWidget object based on this config.

        :param str theme: The theme name, either "light" or "dark". By default, "auto", which selects light or dark based on operating system preferences.
        :param int height: The height of the widget, in pixels. By default, 600.
        :param int port: The port to use when serving data objects on localhost. By default, 8000.
        :param bool proxy: Is this widget being served through a proxy, for example with a cloud notebook (e.g. Binder)?

        :returns: The Jupyter widget.
        :rtype: VitessceWidget

        .. code-block:: python
            :emphasize-lines: 6-7

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            vc.layout(v1)
            vw = vc.widget()
            vw
        """
        from .widget import VitessceWidget  # TODO: Move import back to top when this is factored out.
        return VitessceWidget(self, **kwargs)

    def web_app(self, **kwargs):
        """
        Launch the http://vitessce.io web app using this config.

        :param str theme: The theme name, either "light" or "dark". By default, "auto", which selects light or dark based on operating system preferences.
        :param int port: The port to use when serving data objects on localhost. By default, 8000.
        :param base_url: If the web app is being accessed remotely (i.e. the data is being served from a remote machine), specify the base URL here. If serving and accessing the data on the same machine, keep as None to use a localhost URL.
        :type base_url: str or None
        :param host_name: The host name where the Jupyter server is running, e.g. "http://localhost:8888". By default, None.
        :type host_name: str or None
        :param bool proxy: Is this widget being served through a proxy, for example with a cloud notebook? If True, host_name should be provided.

        :param bool open: Should the browser be opened to the web app URL? By default, True.

        :returns: The URL of the web app (containing the Vitessce configuration as URL-encoded JSON).
        :rtype: str

        .. code-block:: python
            :emphasize-lines: 6

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            vc.layout(v1)
            vc.web_app()
        """
        from .widget import launch_vitessce_io  # TODO: Move import back to top when this is factored out.
        return launch_vitessce_io(self, **kwargs)

    def display(self, **kwargs):
        """
        As a fallback to widget, render Vitessce using functions from IPython.display. This method does not support bi-directional communication (i.e., user interactions in Vitessce cannot be sent back to Python).

        :param str theme: The theme name, either "light" or "dark". By default, "auto", which selects light or dark based on operating system preferences.
        :param int port: The port to use when serving data objects on localhost. By default, 8000.
        :param base_url: If the web app is being accessed remotely (i.e. the data is being served from a remote machine), specify the base URL here. If serving and accessing the data on the same machine, keep as None to use a localhost URL.
        :type base_url: str or None
        :param host_name: The host name where the Jupyter server is running, e.g. "http://localhost:8888". By default, None.
        :type host_name: str or None
        :param bool proxy: Is this widget being served through a proxy, for example with a cloud notebook? If True, host_name should be provided.

        .. code-block:: python
            :emphasize-lines: 6

            from vitessce import VitessceConfig, ViewType as vt, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            vc.layout(v1)
            vc.display()
        """
        from .widget import ipython_display
        return ipython_display(self, **kwargs)

    def export(self, to, *args, **kwargs):
        """
        Export this config's data objects to the local file system or a cloud storage system and get the resulting view config.

        :param str to: The export destination. Valid values include "S3" and "files".
        :param \\*\\*kwargs: Keyword arguments to pass to the export function.
        :returns: The config as a dict, with URLs for the bucket filled in.
        :rtype: dict

        .. code-block:: python
            :emphasize-lines: 8

            from vitessce import VitessceConfig, ViewType as cvtm, CoordinationType as ct

            vc = VitessceConfig(schema_version="1.0.15")
            my_dataset = vc.add_dataset(name='My Dataset')
            v1 = vc.add_view(vt.SPATIAL, dataset=my_dataset)
            vc.layout(v1)

            config_dict = vc.export(to="S3")
        """
        from .export import (export_to_s3, export_to_files)  # TODO: Move import back to top when this is factored out.
        if to == "S3":
            return export_to_s3(self, *args, **kwargs)
        elif to == "files":
            return export_to_files(self, *args, **kwargs)
        else:
            raise ValueError("Unknown export destination.")


class VitessceChainableConfig(VitessceConfig):
    """
    A class to represent a Vitessce view config, where the methods ``add_dataset``, ``add_view``, and ``set_coordination_value`` return self (the config instance). This class inherits from ``VitessceConfig``.
    """

    def __init__(self, schema_version, **kwargs):
        """
        Construct a Vitessce view config object.

        :param \\*\\*kwargs:  Takes the same arguments as the constructor on the ``VitessceConfig`` class.

        .. code-block:: python
            :emphasize-lines: 3

            from vitessce import VitessceChainableConfig

            vc = VitessceChainableConfig(schema_version='1.0.15', name='My Config')
        """
        super().__init__(schema_version, **kwargs)

    def __copy__(self):
        new_vc = VitessceChainableConfig(
            schema_version=self.config["version"], name=self.config["name"], description=self.config["description"])
        new_vc.config = self.config.copy()
        return new_vc

    def add_dataset(self, copy=True, **kwargs):
        """
        Add a dataset to this config.

        :param \\*\\*kwargs: Takes the same arguments as the ``add_dataset`` method on the ``VitessceConfig`` class.

        :returns: The config instance.
        :rtype: VitessceChainableConfig
        """
        if copy:
            new_vc = copy_module.copy(self)
            return new_vc.add_dataset(copy=False, **kwargs)
        super().add_dataset(**kwargs)
        return self

    def add_view(self, component, copy=True, **kwargs):
        """
        Add a view to this config.

        :param component: Takes the same arguments as the ``add_view`` method on the ``VitessceConfig`` class.
        :param \\*\\*kwargs: Takes the same arguments as the ``add_view`` method on the ``VitessceConfig`` class.

        :returns: The config instance.
        :rtype: VitessceChainableConfig
        """
        if copy:
            new_vc = copy_module.copy(self)
            return new_vc.add_view(component, copy=False, **kwargs)
        super().add_view(component, **kwargs)
        return self

    def set_coordination_value(self, c_type, c_scope, c_value, copy=True):
        """
        Add a coordination value to this config.

        :param c_type: Takes the same arguments as the ``set_coordination_value`` method on the ``VitessceConfig`` class.
        :param c_scope: Takes the same arguments as the ``set_coordination_value`` method on the ``VitessceConfig`` class.
        :param c_value: Takes the same arguments as the ``set_coordination_value`` method on the ``VitessceConfig`` class.

        :returns: The config instance.
        :rtype: VitessceChainableConfig
        """
        if copy:
            new_vc = copy_module.copy(self)
            return new_vc.set_coordination_value(c_type, c_scope, c_value, copy=False)
        super().set_coordination_value(c_type, c_scope, c_value)
        return self
