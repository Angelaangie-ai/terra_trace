from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class ScriptSpec:
    name: str
    path: str


@dataclass
class PluginSpec:
    name: str
    version: str
    storage: dict[str, str]
    modules: list[str]
    tool_header: list[str] = field(default_factory=list)
    data_scripts: list[ScriptSpec] = field(default_factory=list)

    def __post_init__(self):
        self.data_scripts = [
            ScriptSpec(**script) if isinstance(script, dict) else script
            for script in self.data_scripts
        ]

    @classmethod
    def load(cls, filepath: str):
        with open(filepath) as f:
            plugin_spec = cls(**yaml.safe_load(f))
        return plugin_spec


@dataclass
class AppManifest:
    name: str
    resource_group: str


@dataclass
class ContainerAppManifest(AppManifest):
    """Container app information for the CLI"""

    environment: str
    web_apps: list[AppManifest] = field(default_factory=list)
    outbound_ips: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.web_apps = [
            AppManifest(**web_app) if isinstance(web_app, dict) else web_app
            for web_app in self.web_apps
        ]


@dataclass
class PluginManifest:
    """Plugin information for the CLI"""

    name: str
    directory: str
    registry: str | None = None
    container_app: ContainerAppManifest | None = None

    def __post_init__(self):
        if isinstance(self.container_app, dict):
            self.container_app = ContainerAppManifest(**self.container_app)


@dataclass
class Manifest:
    """Manifest file for the CLI"""

    default_registry: str | None = None
    default_environment: str | None = None
    default_resource_group: str | None = None
    plugins: list[PluginManifest] = field(default_factory=list)

    def __post_init__(self):
        self.plugins = [
            PluginManifest(**plugin) if isinstance(plugin, dict) else plugin
            for plugin in self.plugins
        ]

    @classmethod
    def _load_file(cls, filepath: str):
        with open(filepath, "r") as file:
            return cls(**yaml.safe_load(file))

    def _write_file(self, filepath: str):
        with open(filepath, "w") as file:
            yaml.safe_dump(asdict(self), file)

    def _find_plugin(self, plugin_name: str):
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                return plugin
        raise ValueError(f"Plugin {plugin_name} not found")
