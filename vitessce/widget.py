import importlib.util
from urllib.parse import quote_plus
import json

# Widget dependencies
import anywidget
from traitlets import Unicode, Dict, Int, Bool
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

function asEsModule(component) {
  return {
    __esModule: true,
    default: component,
  };
}

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

export async function render(view) {
    const cssUid = view.model.get('uid');
    const jsDevMode = view.model.get('js_dev_mode');
    const jsPackageVersion = view.model.get('js_package_version');
    const customJsUrl = view.model.get('custom_js_url');

    const pkgName = (jsDevMode ? "@vitessce/dev" : "vitessce");

    importMap.imports["vitessce"] = (customJsUrl.length > 0
        ? customJsUrl
        : `https://unpkg.com/${pkgName}@${jsPackageVersion}`
    );

    const { Vitessce } = await importWithMap("vitessce", importMap);

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

        const vitessceProps = { height, theme, config, onConfigChange, validateConfig };

        return e('div', { ref: divRef, style: { height: height + 'px' } },
            e(React.Suspense, { fallback: e('div', {}, 'Loading...') },
                e(React.StrictMode, {},
                    e(Vitessce, vitessceProps)
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
"""


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

    js_package_version = Unicode('3.2.1').tag(sync=True)
    js_dev_mode = Bool(False).tag(sync=True)
    custom_js_url = Unicode('').tag(sync=True)

    def __init__(self, config, height=600, theme='auto', uid=None, port=None, proxy=False, js_package_version='3.2.1', js_dev_mode=False, custom_js_url=''):
        """
        Construct a new Vitessce widget.

        :param config: A view config instance.
        :type config: VitessceConfig
        :param str theme: The theme name, either "light" or "dark". By default, "auto", which selects light or dark based on operating system preferences.
        :param int height: The height of the widget, in pixels. By default, 600.
        :param int port: The port to use when serving data objects on localhost. By default, 8000.
        :param bool proxy: Is this widget being served through a proxy, for example with a cloud notebook (e.g. Binder)?

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

        uid_str = get_uid_str(uid)

        super(VitessceWidget, self).__init__(
            config=config_dict, height=height, theme=theme, proxy=proxy,
            js_package_version=js_package_version, js_dev_mode=js_dev_mode, custom_js_url=custom_js_url,
            uid=uid_str,
        )

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

# Launch Vitessce using plain HTML representation (no ipywidgets)


def ipython_display(config, height=600, theme='auto', base_url=None, host_name=None, uid=None, port=None, proxy=False, js_package_version='3.2.1', js_dev_mode=False, custom_js_url=''):
    from IPython.display import display, HTML
    uid_str = "vitessce" + get_uid_str(uid)

    base_url, use_port, _ = get_base_url_and_port(
        port, DEFAULT_PORT, proxy=proxy, base_url=base_url, host_name=host_name)
    config_dict = config.to_dict(base_url=base_url)
    routes = config.get_routes()
    serve_routes(config, routes, use_port)

    model_vals = {
        "uid": uid_str,
        "js_package_version": js_package_version,
        "js_dev_mode": js_dev_mode,
        "custom_js_url": custom_js_url,
        "proxy": proxy,
        "has_host_name": host_name is not None,
        "height": height,
        "theme": theme,
        "config": config_dict,
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
