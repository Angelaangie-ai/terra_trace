import logging
import os
import shutil
from pathlib import Path

from ..constants import PLUGINS_DIR
from . import manifest
from .helper import verify_to_proceed

logger = logging.getLogger()

CURRENT_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = CURRENT_DIR / "templates"


def list_plugins():
    """List all plugins"""
    m = manifest.load_manifest()
    for plugin in m.plugins:
        print(plugin.name)


def create_plugin(plugin_name: str):
    """Create a new plugin"""
    try:
        manifest.get_plugin(plugin_name)
        logger.error(f"Plugin {plugin_name} already exists")
        return
    except ValueError:
        pass

    plugin_path = Path(PLUGINS_DIR, plugin_name)
    create_plugin_stub(plugin_name, plugin_path)
    manifest.add_plugin(plugin_name)
    logger.info(f"Plugin stub created successfully at {plugin_path}")


def create_plugin_stub(plugin_name: str, plugin_path: Path):
    os.makedirs(plugin_path, exist_ok=True)

    # Copy all template files into the plugin directory
    for filename in os.listdir(TEMPLATES_DIR):
        source = TEMPLATES_DIR / filename
        destination = plugin_path / filename
        if filename == "plugin.yaml":
            with open(source, "r") as f:
                data = f.read()
            filled_data = data.format(name=plugin_name)
            with open(destination, "w") as f:
                f.write(filled_data)
        elif filename == "module.py":
            # Create a template module file in the modules dir
            modules_dir = plugin_path / "modules"
            os.makedirs(modules_dir, exist_ok=True)
            module_filepath = modules_dir / f"{plugin_name}.py"
            shutil.copy(TEMPLATES_DIR / filename, module_filepath)
        else:
            shutil.copy(source, destination)


def set_env(environment: str):
    """Set the default environment for a container app"""
    default_env = manifest.get_default_environment()
    if default_env and not verify_to_proceed(f"Override default environment {default_env}?"):
        return
    manifest.set_default_environment(environment)
    logger.info(f"Default environment set to {environment}")


def set_registry(registry: str):
    """Set the default registry for a plugin"""
    default_registry = manifest.get_default_registry()
    if default_registry and not verify_to_proceed(f"Override default registry {default_registry}?"):
        return
    manifest.set_default_registry(registry)
    logger.info(f"Default registry set to {registry}")


def set_rg(resource_group: str):
    """Set the default resource group for a container app"""
    default_rg = manifest.get_default_resource_group()
    if default_rg and not verify_to_proceed(f"Override default resource group {default_rg}?"):
        return
    manifest.set_default_resource_group(resource_group)
    logger.info(f"Default resource group set to {resource_group}")
