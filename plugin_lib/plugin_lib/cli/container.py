import logging
import os
from pathlib import Path

from ..constants import PLUGIN_FILENAME
from ..spec import PluginSpec
from . import manifest
from .helper import verify_to_proceed
from .wrappers import AzWrapper, DockerWrapper

logger = logging.getLogger()

CURRENT_DIR = Path(__file__).parent.resolve()
PLUGIN_LIB_DIR = CURRENT_DIR.parent.parent


def build_plugin(plugin_name: str) -> str:
    dw = DockerWrapper()
    """Build the plugin locally"""
    plugin_dir = manifest.get_plugin_dir(plugin_name)

    # Get the plugin name and version
    plugin_spec = PluginSpec.load(os.path.join(plugin_dir, PLUGIN_FILENAME))
    name = plugin_spec.name.lower()
    version = plugin_spec.version.lower()

    # Build the plugin using the appropriate tags
    tag = f"{name}:{version}"
    dw.build(plugin_dir, tag, build_context=f"plugin_lib={PLUGIN_LIB_DIR}")
    return tag


def get_registry_for_plugin(plugin_name: str) -> str:
    registry = manifest.get_plugin_registry(plugin_name)
    if registry:
        return registry
    # Try the default registry, if it exists
    default_registry = manifest.get_default_registry()
    if default_registry and verify_to_proceed(f"Use default registry {default_registry}?"):
        manifest.add_plugin_registry(plugin_name, default_registry)
        return default_registry
    logger.info(f"Registry not found for plugin {plugin_name}")
    if not verify_to_proceed("Would you like to add an ACR for this plugin?"):
        raise ValueError("Cannot upload plugin without a registry")
    registry = input("Please enter the ACR name: ")
    manifest.add_plugin_registry(plugin_name, registry)
    return registry


def upload_plugin_to_acr(plugin_name: str, registry: str | None = None) -> str:
    """Upload the given image to the provided registry"""
    azw = AzWrapper()
    dw = DockerWrapper()
    if registry is None:
        registry = get_registry_for_plugin(plugin_name)

    # Log into the ACR
    logger.info(f"Logging into ACR {registry}")

    azw.acr_login(registry)

    # Build the image
    image = build_plugin(plugin_name)

    # Get login server
    login_server = azw.get_acr_login_server(registry)

    # Tag the image with the registry name
    new_image_name = f"{login_server}/{image}"

    dw.tag(image, new_image_name)

    # Push the image to the registry
    logger.info(f"Uploading plugin to {registry}")
    dw.push(new_image_name)
    logger.info("Plugin uploaded successfully")

    # After everything works, offer to set as default registry, if we don't have one
    default_registry = manifest.get_default_registry()
    if not default_registry and verify_to_proceed(f"Set {registry} as default registry?"):
        manifest.set_default_registry(registry)
    return new_image_name
