import os

import yaml

from ..constants import CONFIG_DIR, PLUGIN_FILENAME, PLUGINS_DIR
from ..spec import ContainerAppManifest, Manifest, PluginManifest, PluginSpec

# Path to manifest file
FILENAME = "plugin_manifest.yaml"
MANIFEST_FILE_PATH = os.path.join(CONFIG_DIR, FILENAME)


def create_manifest_file():
    """Create the manifest file. Include existing plugins"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    manifest_filepath = os.path.join(CONFIG_DIR, FILENAME)
    if os.path.exists(manifest_filepath):
        return
    manifest = Manifest()
    manifest._write_file(manifest_filepath)
    for dir in os.listdir(PLUGINS_DIR):
        plugin_yaml = os.path.join(PLUGINS_DIR, dir, "plugin.yaml")
        if os.path.exists(plugin_yaml):
            with open(plugin_yaml, "r") as file:
                data = yaml.safe_load(file)
            name = data["name"]
            add_plugin(name, dir)


def load_manifest():
    """Load the manifest file"""
    create_manifest_file()
    return Manifest._load_file(MANIFEST_FILE_PATH)


def save_manifest(manifest: Manifest):
    """Save the manifest file"""
    manifest._write_file(MANIFEST_FILE_PATH)


def add_plugin(plugin_name: str, plugin_dir: str | None = None):
    """Create a new plugin"""
    manifest = load_manifest()
    if plugin_dir is None:
        plugin_dir = plugin_name.lower()
    manifest.plugins.append(PluginManifest(plugin_name, plugin_dir))
    save_manifest(manifest)


def get_plugin(plugin_name: str) -> PluginManifest:
    """Get a plugin by name"""
    manifest = load_manifest()
    return manifest._find_plugin(plugin_name)


def delete_plugin(plugin_name: str):
    """Delete a plugin by name"""
    manifest = load_manifest()
    for plugin in manifest.plugins:
        if plugin.name == plugin_name:
            manifest.plugins.remove(plugin)
            break
    else:
        raise ValueError(f"Plugin {plugin_name} not found")
    save_manifest(manifest)


def get_plugin_dir(plugin_name: str) -> str:
    """Get a plugin directory by name"""
    plugin = get_plugin(plugin_name)
    return os.path.join(PLUGINS_DIR, plugin.directory)


def get_plugin_spec(plugin_name: str) -> PluginSpec:
    """Get the plugin spec for a plugin"""
    plugin_dir = get_plugin_dir(plugin_name)
    return PluginSpec.load(os.path.join(plugin_dir, PLUGIN_FILENAME))


def get_plugin_registry(plugin_name: str) -> str | None:
    """Get the ACR for a plugin"""
    plugin = get_plugin(plugin_name)
    return plugin.registry


def add_plugin_registry(plugin_name: str, registry: str):
    """Set the ACR for a plugin"""
    manifest = load_manifest()
    plugin = manifest._find_plugin(plugin_name)
    plugin.registry = registry
    save_manifest(manifest)


def get_container_app(plugin_name: str) -> ContainerAppManifest:
    """Get the container app for a plugin"""
    plugin = get_plugin(plugin_name)
    container_app = plugin.container_app
    if container_app is None:
        raise ValueError(f"Container app not found for plugin {plugin_name}")
    return container_app


def add_container_app(plugin_name: str, container_app: ContainerAppManifest):
    """Set the container app for a plugin"""
    manifest = load_manifest()
    plugin = manifest._find_plugin(plugin_name)
    plugin.container_app = container_app
    save_manifest(manifest)


def delete_container_app(plugin_name: str):
    """Delete the container app for a plugin"""
    manifest = load_manifest()
    plugin = manifest._find_plugin(plugin_name)
    plugin.container_app = None
    save_manifest(manifest)


# Default options
def get_default_environment():
    """Get the default container app environment"""
    manifest = load_manifest()
    return manifest.default_environment


def set_default_environment(environment: str):
    """Set the default container app environment"""
    manifest = load_manifest()
    manifest.default_environment = environment
    save_manifest(manifest)


def get_default_registry():
    """Get the default registry for a plugin"""
    manifest = load_manifest()
    return manifest.default_registry


def set_default_registry(registry: str):
    """Set the default registry for a plugin"""
    manifest = load_manifest()
    manifest.default_registry = registry
    save_manifest(manifest)


def get_default_resource_group():
    """Get the default resource group for a container app"""
    manifest = load_manifest()
    return manifest.default_resource_group


def set_default_resource_group(resource_group: str):
    """ "Set the default resource group for a container app"""
    manifest = load_manifest()
    manifest.default_resource_group = resource_group
    save_manifest(manifest)
