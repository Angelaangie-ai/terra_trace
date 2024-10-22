import os
from importlib import util as importlib_util

from plugin_lib.constants import APP_PATH, PLUGIN_FILENAME
from plugin_lib.spec import PluginSpec


def load_plugin_spec():
    filepath = os.path.join(APP_PATH, PLUGIN_FILENAME)
    return PluginSpec.load(filepath)


def load_module(filepath: str):
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    module_filepath = os.path.join(APP_PATH, filepath)
    spec = importlib_util.spec_from_file_location(module_name, module_filepath)
    assert spec is not None
    assert spec.loader is not None
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
