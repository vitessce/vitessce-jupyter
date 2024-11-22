import importlib.util
from urllib.parse import quote_plus
import json

# Widget dependencies
import anywidget
from traitlets import Unicode, Dict, List, Int, Bool
import time
import uuid

# Server dependencies
from uvicorn import Config, Server

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from threading import Thread
import socket

MAX_PORT_TRIES = 1000
DEFAULT_PORT = 8000


class BackgroundServer:
    # Reference: https://github.com/gosling-lang/gos/blob/main/gosling/data/_background_server.py#L10
    def __init__(self, routes):
        middleware = [
            Middleware(CORSMiddleware, allow_origins=[
                       '*'], allow_methods=["OPTIONS", "GET"], allow_headers=['Range'])
        ]
        self.app = Starlette(debug=True, routes=routes, middleware=middleware)
        self.port = None
        self.thread = None
        self.server = None

    def stop(self):
        if self.thread is None:
            return self
        assert self.server is not None
        try:
            self.server.should_exit = True
            self.thread.join()
        finally:
            self.server = None
            self.thread = None
        return self

    def start(self, port=None, timeout=1, daemon=True, log_level="warning"):
        if self.thread is not None:
            return self

        config = Config(
            app=self.app,
            port=port,
            timeout_keep_alive=timeout,
            log_level=log_level
        )
        self.port = config.port
        self.server = Server(config=config)
        self.thread = Thread(target=self.server.run, daemon=daemon)
        self.thread.start()

        while not self.server.started:
            time.sleep(1e-3)

        return self


class VitessceDataServer:
    def __init__(self):
        self.served_configs = []

    def stop_all(self):
        for config in self.served_configs:
            config.stop_all_servers()
        self.served_configs = []

    def register(self, config):
        if config not in self.served_configs:
            self.served_configs.append(config)


# Create a singleton to have a global way to stop all servers
# attached to configs.
data_server = VitessceDataServer()


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_base_url_and_port(port, next_port, proxy=False, base_url=None, host_name=None):
    if port is None:
        use_port = next_port
        next_port += 1
        port_tries = 1
        while is_port_in_use(use_port) and port_tries < MAX_PORT_TRIES:
            use_port = next_port
            next_port += 1
            port_tries += 1
    else:
        use_port = port

    if base_url is None:
        if proxy:
            if importlib.util.find_spec('jupyter_server_proxy') is None:
                raise ValueError(
                    "To use the widget through a proxy, jupyter-server-proxy must be installed.")
            if host_name is None:
                base_url = f"proxy/{use_port}"
            else:
                base_url = f"{host_name}/proxy/{use_port}"
        else:
            base_url = f"http://localhost:{use_port}"

    return base_url, use_port, next_port


def serve_routes(config, routes, use_port):
    if not config.has_server(use_port) and len(routes) > 0:
        server = BackgroundServer(routes)
        config.register_server(use_port, server)
        data_server.register(config)
        server.start(port=use_port)


def launch_vitessce_io(config, theme='light', port=None, base_url=None, host_name=None, proxy=False, open=True):
    import webbrowser
    base_url, use_port, _ = get_base_url_and_port(
        port, DEFAULT_PORT, proxy=proxy, base_url=base_url, host_name=host_name)
    config_dict = config.to_dict(base_url=base_url)
    routes = config.get_routes()
    serve_routes(config, routes, use_port)
    vitessce_url = f"http://vitessce.io/#?theme={theme}&url=data:," + quote_plus(
        json.dumps(config_dict))
    if open:
        webbrowser.open(vitessce_url)
    return vitessce_url


def get_uid_str(uid):
    if uid is None or not str(uid).isalnum():
        uid_str = str(uuid.uuid4())[:4]
    else:
        uid_str = uid
    return uid_str


ESM = """
import { importWithMap } from 'https://unpkg.com/dynamic-importmap@0.1.0';
const importMap = {
  imports: {
    "react": "https://esm.sh/react@18.2.0?dev",
    "react-dom": "https://esm.sh/react-dom@18.2.0?dev",
    "react-dom/client": "https://esm.sh/react-dom@18.2.0/client?dev",
  },
};

const React = await importWithMap("react", importMap);
const { createRoot } = await importWithMap("react-dom/client", importMap);

const e = React.createElement;

const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

// The jupyter server may be running through a proxy,
// which means that the client needs to prepend the part of the URL before /proxy/8000 such as
// https://hub.gke2.mybinder.org/user/vitessce-vitessce-python-swi31vcv/proxy/8000/A/0/cells
function prependBaseUrl(config, proxy, hasHostName) {
  if(!proxy || hasHostName) {
    return config;
  }
  const { origin } = new URL(window.location.href);
  let baseUrl;
  const jupyterLabConfigEl = document.getElementById('jupyter-config-data');

  if (jupyterLabConfigEl) {
    // This is jupyter lab
    baseUrl = JSON.parse(jupyterLabConfigEl.textContent || '').baseUrl;
  } else {
    // This is jupyter notebook
    baseUrl = document.getElementsByTagName('body')[0].getAttribute('data-base-url');
  }
  return {
    ...config,
    datasets: config.datasets.map(d => ({
      ...d,
      files: d.files.map(f => ({
        ...f,
        url: `${origin}${baseUrl}${f.url}`,
      })),
    })),
  };
}

async function render(view) {
    const cssUid = view.model.get('uid');
    const jsDevMode = view.model.get('js_dev_mode');
    const jsPackageVersion = view.model.get('js_package_version');
    const customJsUrl = view.model.get('custom_js_url');
    const pluginEsmArr = view.model.get('plugin_esm');
    const remountOnUidChange = view.model.get('remount_on_uid_change');
    const storeUrls = view.model.get('store_urls');
    const invokeTimeout = view.model.get('invoke_timeout');

    const pageMode = view.model.get('page_mode');
    const pageEsm = view.model.get('page_esm');

    const pkgName = (jsDevMode ? "@vitessce/dev" : "vitessce");

    importMap.imports["vitessce"] = (customJsUrl.length > 0
        ? customJsUrl
        : `https://unpkg.com/${pkgName}@${jsPackageVersion}`
    );

    const {
        Vitessce,
        PluginFileType,
        PluginViewType,
        PluginCoordinationType,
        PluginJointFileType,
        PluginAsyncFunction,
        z,
        useCoordination,
        usePageModeView,
        useGridItemSize,
        // TODO: names and function signatures are subject to change for the following functions
        // Reference: https://github.com/keller-mark/use-coordination/issues/37#issuecomment-1946226827
        useComplexCoordination,
        useMultiCoordinationScopesNonNull,
        useMultiCoordinationScopesSecondaryNonNull,
        useComplexCoordinationSecondary,
        useCoordinationScopes,
        useCoordinationScopesBy,
    } = await importWithMap("vitessce", importMap);

    let pluginViewTypes = [];
    let pluginCoordinationTypes = [];
    let pluginFileTypes = [];
    let pluginJointFileTypes = [];
    let pluginAsyncFunctions = [];

    const stores = Object.fromEntries(
        storeUrls.map(storeUrl => ([
            storeUrl,
            {
                async get(key) {
                    const [data, buffers] = await view.experimental.invoke("_zarr_get", [storeUrl, key], {
                        signal: AbortSignal.timeout(invokeTimeout),
                    });
                    if (!data.success) return undefined;
                    return buffers[0].buffer;
                },
            }
        ])),
    );

    function invokePluginCommand(commandName, commandParams, commandBuffers) {
        return view.experimental.invoke("_plugin_command", [commandName, commandParams], {
            signal: AbortSignal.timeout(invokeTimeout),
            ...(commandBuffers ? { buffers: commandBuffers } : {}),
        });
    }

    for (const pluginEsm of pluginEsmArr) {
        try {
            const pluginEsmUrl = URL.createObjectURL(new Blob([pluginEsm], { type: "text/javascript" }));
            const pluginModule = (await import(pluginEsmUrl)).default;
            URL.revokeObjectURL(pluginEsmUrl);

            const pluginDeps = {
                React,
                PluginFileType,
                PluginViewType,
                PluginCoordinationType,
                PluginJointFileType,
                PluginAsyncFunction,
                z,
                invokeCommand: invokePluginCommand,
                useCoordination,
                useGridItemSize,
                useComplexCoordination,
                useMultiCoordinationScopesNonNull,
                useMultiCoordinationScopesSecondaryNonNull,
                useComplexCoordinationSecondary,
                useCoordinationScopes,
                useCoordinationScopesBy,
            };
            const pluginsObj = await pluginModule.createPlugins(pluginDeps);
            if(Array.isArray(pluginsObj.pluginViewTypes)) {
                pluginViewTypes = [...pluginViewTypes, ...pluginsObj.pluginViewTypes];
            }
            if(Array.isArray(pluginsObj.pluginCoordinationTypes)) {
                pluginCoordinationTypes = [...pluginCoordinationTypes, ...pluginsObj.pluginCoordinationTypes];
            }
            if(Array.isArray(pluginsObj.pluginFileTypes)) {
                pluginFileTypes = [...pluginFileTypes, ...pluginsObj.pluginFileTypes];
            }
            if(Array.isArray(pluginsObj.pluginJointFileTypes)) {
                pluginJointFileTypes = [...pluginJointFileTypes, ...pluginsObj.pluginJointFileTypes];
            }
            if(Array.isArray(pluginsObj.pluginAsyncFunctions)) {
                pluginAsyncFunctions = [...pluginAsyncFunctions, ...pluginsObj.pluginAsyncFunctions];
            }
        } catch(e) {
            console.error("Error loading plugin ESM or executing createPlugins function.");
            console.error(e);
        }
    }

    let PageComponent;
    if(pageMode && pageEsm.length > 0) {
        try {
            const pageEsmUrl = URL.createObjectURL(new Blob([pageEsm], { type: "text/javascript" }));
            const pageModule = (await import(pageEsmUrl)).default;
            URL.revokeObjectURL(pageEsmUrl);

            const pageDeps = {
                usePageModeView,
            };
            PageComponent = await pageModule.createPage(pageDeps);
        } catch(e) {
            console.error("Error loading page ESM or executing createPage function.")
            console.error(e);
        }
    }

    function VitessceWidget(props) {
        const { model } = props;

        const [config, setConfig] = React.useState(prependBaseUrl(model.get('config'), model.get('proxy'), model.get('has_host_name')));
        const [validateConfig, setValidateConfig] = React.useState(true);
        const height = model.get('height');
        const theme = model.get('theme') === 'auto' ? (prefersDark ? 'dark' : 'light') : model.get('theme');

        const divRef = React.useRef();

        React.useEffect(() => {
            if(!divRef.current) {
                return () => {};
            }

            function handleMouseEnter() {
                const jpn = divRef.current.closest('.jp-Notebook');
                if(jpn) {
                    jpn.style.overflow = "hidden";
                }
            }
            function handleMouseLeave(event) {
                if(event.relatedTarget === null || (event.relatedTarget && event.relatedTarget.closest('.jp-Notebook')?.length)) return;
                const jpn = divRef.current.closest('.jp-Notebook');
                if(jpn) {
                    jpn.style.overflow = "auto";
                }
            }
            divRef.current.addEventListener("mouseenter", handleMouseEnter);
            divRef.current.addEventListener("mouseleave", handleMouseLeave);

            return () => {
                if(divRef.current) {
                    divRef.current.removeEventListener("mouseenter", handleMouseEnter);
                    divRef.current.removeEventListener("mouseleave", handleMouseLeave);
                }
            };
        }, [divRef]);

        // Config changed on JS side (from within <Vitessce/>),
        // send updated config to Python side.
        const onConfigChange = React.useCallback((config) => {
            model.set('config', config);
            setValidateConfig(false);
            model.save_changes();
        }, [model]);

        // Config changed on Python side,
        // pass to <Vitessce/> component to it is updated on JS side.
        React.useEffect(() => {
            model.on('change:config', () => {
                const newConfig = prependBaseUrl(model.get('config'), model.get('proxy'), model.get('has_host_name'));

                // Force a re-render and re-validation by setting a new config.uid value.
                // TODO: make this conditional on a parameter from Python.
                //newConfig.uid = `random-${Math.random()}`;
                //console.log('newConfig', newConfig);
                setConfig(newConfig);
            });
        }, []);

        const vitessceProps = {
            height, theme, config, onConfigChange, validateConfig,
            pluginViewTypes, pluginCoordinationTypes,
            pluginFileTypes,pluginJointFileTypes, pluginAsyncFunctions,
            remountOnUidChange, stores, pageMode,
        };

        return e('div', { ref: divRef, style: { height: height + 'px' } },
            e(React.Suspense, { fallback: e('div', {}, 'Loading...') },
                e(React.StrictMode, {},
                    e(Vitessce, vitessceProps,
                        (pageMode ? e(PageComponent, {}) : null)
                    ),
                ),
            ),
        );
    }

    const root = createRoot(view.el);
    root.render(e(VitessceWidget, { model: view.model }));

    return () => {
        // Re-enable scrolling.
        const jpn = view.el.closest('.jp-Notebook');
        if(jpn) {
            jpn.style.overflow = "auto";
        }

        // Clean up React and DOM state.
        root.unmount();
        if(view._isFromDisplay) {
            view.el.remove();
        }
    };
}
export default { render };
"""

DEFAULT_PLUGIN_ESM = """
function createPlugins(utilsForPlugins) {
    const {
        React,
        PluginFileType,
        PluginViewType,
        PluginCoordinationType,
        PluginJointFileType,
        PluginAsyncFunction,
        z,
        useCoordination,
        invokeCommand,
    } = utilsForPlugins;
    return {
        pluginViewTypes: undefined,
        pluginFileTypes: undefined,
        pluginCoordinationTypes: undefined,
        pluginJointFileTypes: undefined,
        pluginAsyncFunctions: undefined,
    };
}
export default { createPlugins };
"""

DEFAULT_PAGE_ESM = """
function createPage(utilsForPage) {
    const {
        usePageModeView,
    } = utilsForPage;

    function PageComponent(props) {
        const BiomarkerSelect = usePageModeView('biomarker-select');

        return (
            <div>
                <BiomarkerSelect />
            </div>
        );
    }
    return PageComponent;
}
export default { createPage };
"""


class VitesscePlugin:
    """
    A class that represents a Vitessce widget plugin. Custom plugins can be created by subclassing this class.
    """
    plugin_esm = DEFAULT_PLUGIN_ESM
    commands = {}

    def on_config_change(self, new_config):
        """
        Config change handler.

        :param dict new_config: The new config object.

        :returns: config (likely with new "uid" property) or None
        :rtype: dict or None
        """
        raise NotImplementedError("on_config_change may optionally be implemented by subclasses.")


class VitessceWidget(anywidget.AnyWidget):
    """
    A class to represent a Jupyter widget for Vitessce.
    """
    _esm = ESM

    # Widget specific property.
    # Widget properties are defined as traitlets. Any property tagged with `sync=True`
    # is automatically synced to the frontend *any* time it changes in Python.
    # It is synced back to Python from the frontend *any* time the model is touched.
    config = Dict({}).tag(sync=True)
    height = Int(600).tag(sync=True)
    theme = Unicode('auto').tag(sync=True)
    proxy = Bool(False).tag(sync=True)
    uid = Unicode('').tag(sync=True)
    has_host_name = Bool(False).tag(sync=True)

    next_port = DEFAULT_PORT

    js_package_version = Unicode('3.5.0').tag(sync=True)
    js_dev_mode = Bool(False).tag(sync=True)
    custom_js_url = Unicode('').tag(sync=True)
    plugin_esm = List(trait=Unicode(''), default_value=[]).tag(sync=True)
    remount_on_uid_change = Bool(True).tag(sync=True)
    page_mode = Bool(False).tag(sync=True)
    page_esm = Unicode('').tag(sync=True)

    store_urls = List(trait=Unicode(''), default_value=[]).tag(sync=True)
    invoke_timeout = Int(30000).tag(sync=True)

    def __init__(self, config, height=600, theme='auto', uid=None, port=None, proxy=False, js_package_version='3.5.0', js_dev_mode=False, custom_js_url='', plugins=None, remount_on_uid_change=True, invoke_timeout=30000, page_mode=False, page_esm=None):
        """
        Construct a new Vitessce widget.

        :param config: A view config instance.
        :type config: VitessceConfig
        :param str theme: The theme name, either "light" or "dark". By default, "auto", which selects light or dark based on operating system preferences.
        :param int height: The height of the widget, in pixels. By default, 600.
        :param int port: The port to use when serving data objects on localhost. By default, 8000.
        :param bool proxy: Is this widget being served through a proxy, for example with a cloud notebook (e.g. Binder)?
        :param str js_package_version: The version of the NPM package ('vitessce' if not js_dev_mode else '@vitessce/dev').
        :param bool js_dev_mode: Should @vitessce/dev be used (typically for debugging purposes)? By default, False.
        :param str custom_js_url: A URL to a JavaScript file to use (instead of 'vitessce' or '@vitessce/dev' NPM package).
        :param list[VitesscePlugin] plugins: A list of subclasses of VitesscePlugin. Optional.
        :param bool remount_on_uid_change: Passed to the remountOnUidChange prop of the <Vitessce/> React component. By default, True.
        :param int invoke_timeout: The timeout in milliseconds for invoking Python functions from JavaScript. By default, 30000.
        :param bool page_mode: Whether to render the <Vitessce/> component in grid-mode or page-mode. By default, False.
        :param str page_esm: The ES module string for the page component creation function. Optional.

        .. code-block:: python
            :emphasize-lines: 4

            from vitessce import VitessceConfig, VitessceWidget

            vc = VitessceConfig.from_object(my_scanpy_object)
            vw = vc.widget()
            vw
        """

        base_url, use_port, VitessceWidget.next_port = get_base_url_and_port(
            port, VitessceWidget.next_port, proxy=proxy)
        self.config_obj = config
        self.port = use_port
        config_dict = config.to_dict(base_url=base_url)
        routes = config.get_routes()

        self._stores = config.get_stores(base_url=base_url)
        self._plugins = plugins or []

        plugin_esm = [p.plugin_esm for p in self._plugins]
        self._plugin_commands = {}
        for plugin in self._plugins:
            self._plugin_commands.update(plugin.commands)

        uid_str = get_uid_str(uid)

        super(VitessceWidget, self).__init__(
            config=config_dict, height=height, theme=theme, proxy=proxy,
            js_package_version=js_package_version, js_dev_mode=js_dev_mode, custom_js_url=custom_js_url,
            plugin_esm=plugin_esm, remount_on_uid_change=remount_on_uid_change,
            page_mode=page_mode, page_esm=('' if page_esm is None else page_esm),
            invoke_timeout=invoke_timeout,
            uid=uid_str, store_urls=list(self._stores.keys())
        )

        # Register chained plugin on_config_change functions with a change observer.
        def handle_config_change(change):
            new_config = change.new
            for plugin in self._plugins:
                try:
                    new_config = plugin.on_config_change(new_config)
                except NotImplementedError:
                    # It is optional for plugins to implement on_config_change.
                    pass
            if new_config is not None:
                self.config = new_config

        self.observe(handle_config_change, names=['config'])

        serve_routes(config, routes, use_port)

    def _get_coordination_value(self, coordination_type, coordination_scope):
        obj = self.config['coordinationSpace'][coordination_type]
        obj_scopes = list(obj.keys())
        if coordination_scope is not None:
            if coordination_scope in obj_scopes:
                return obj[coordination_scope]
            else:
                raise ValueError(
                    f"The specified coordination scope '{coordination_scope}' could not be found for the coordination type '{coordination_type}'. Known coordination scopes are {obj_scopes}")
        else:
            if len(obj_scopes) == 1:
                auto_coordination_scope = obj_scopes[0]
                return obj[auto_coordination_scope]
            elif len(obj_scopes) > 1:
                raise ValueError(
                    f"The coordination scope could not be automatically determined because multiple coordination scopes exist for the coordination type '{coordination_type}'. Please specify one of {obj_scopes} using the scope parameter.")
            else:
                raise ValueError(
                    f"No coordination scopes were found for the coordination type '{coordination_type}'.")

    def get_cell_selection(self, scope=None):
        return self._get_coordination_value('cellSelection', scope)

    def close(self):
        self.config_obj.stop_server(self.port)
        super().close()

    @anywidget.experimental.command
    def _zarr_get(self, params, buffers):
        [store_url, key] = params
        store = self._stores[store_url]
        try:
            buffers = [store[key.lstrip("/")]]
        except KeyError:
            buffers = []
        return {"success": len(buffers) == 1}, buffers

    @anywidget.experimental.command
    def _plugin_command(self, params, buffers):
        [command_name, command_params] = params
        command_func = self._plugin_commands[command_name]
        return command_func(command_params, buffers)

# Launch Vitessce using plain HTML representation (no ipywidgets)


def ipython_display(config, height=600, theme='auto', base_url=None, host_name=None, uid=None, port=None, proxy=False, js_package_version='3.5.0', js_dev_mode=False, custom_js_url='', plugin_esm=DEFAULT_PLUGIN_ESM, remount_on_uid_change=True, page_mode=False, page_esm=None):
    from IPython.display import display, HTML
    uid_str = "vitessce" + get_uid_str(uid)

    base_url, use_port, _ = get_base_url_and_port(
        port, DEFAULT_PORT, proxy=proxy, base_url=base_url, host_name=host_name)
    config_dict = config.to_dict(base_url=base_url)
    routes = config.get_routes()
    serve_routes(config, routes, use_port)

    plugins = plugins or []
    plugin_esm = [p.plugin_esm for p in plugins]

    model_vals = {
        "uid": uid_str,
        "js_package_version": js_package_version,
        "js_dev_mode": js_dev_mode,
        "custom_js_url": custom_js_url,
        "plugin_esm": plugin_esm,
        "page_mode": page_mode,
        "page_esm": ('' if page_esm is None else page_esm),
        "remount_on_uid_change": remount_on_uid_change,
        "invoke_timeout": 30000,
        "proxy": proxy,
        "has_host_name": host_name is not None,
        "height": height,
        "theme": theme,
        "config": config_dict,
        "store_urls": [],
    }

    # We need to clean up the React and DOM state in any case in which
    # .display() is being run in the same cell as a previous .display().
    # Otherwise, React tries to diff the virtual DOM,
    # causing the browser to become unresponsive.

    # In the .widget() case, we return a cleanup function that anywidget runs for us.
    # However, in the .display() case, we need to run do cleanup ourselves.
    # However, using IPython.display, there is not bidirectional communication,
    # so we cannot simply sent "events" or "messages" to previously rendered widgets
    # to tell them to clean up. Instead, we need to store a reference to the wrapper
    # div element on the window, scoped to the cell, for cleanup.

    # There are two edge cases in which the user runs .display(), then .widget()
    # or vice-versa, in the same cell -- we do not currently handle those,
    # as this would require using the hack-y cleanup over the AnyWidget cleanup
    # in all cases.

    CLEANUP_STR = """
        // Initialize the global Map if not yet defined.
        if (!window.__VITESSCE_DISPLAY_CELLS__) {
            window.__VITESSCE_DISPLAY_CELLS__ = new Map();
        }

        // Rename for readability.
        const CELL_MAP = window.__VITESSCE_DISPLAY_CELLS__;

        // Need cases for getting parent in plain notebook, colab, etc.
        const potentialParents = [
            nextWidgetEl.closest('.jp-Notebook-cell'), // JupyterLab
            nextWidgetEl.closest('.cell'), // JupyterLab Classic Notebook
        ];

        // Get the cell's parent element.
        const parentEl = potentialParents.find(potentialEl => potentialEl !== null);

        if (parentEl) {
            const prevCleanupFunctionPromise = CELL_MAP.get(parentEl);

            if (prevCleanupFunctionPromise) {
                const prevCleanupFunction = await prevCleanupFunctionPromise;
                prevCleanupFunction();
                CELL_MAP.set(parentEl, null);
            }
        }
    """

    HTML_STR = f"""
        <div id="{uid_str}"></div>

        <script type="module">

            const nextWidgetEl = document.getElementById("{uid_str}");

            """ + ESM + """

            """ + CLEANUP_STR + """

            const nextCleanupFunction = render({
                model: {
                    get: (key) => {
                        const vals = """ + json.dumps(model_vals) + """;
                        return vals[key];
                    },
                    set: () => {},
                    save_changes: () => {},
                    on: () => {},
                },
                el: nextWidgetEl,
                _isFromDisplay: true,
            });

            // Store a reference to the widget div element on the window, scoped to the cell,
            // for future cleanups to use.
            if (parentEl) {
                CELL_MAP.set(parentEl, nextCleanupFunction);
            }
        </script>
    """

    display(HTML(HTML_STR))
