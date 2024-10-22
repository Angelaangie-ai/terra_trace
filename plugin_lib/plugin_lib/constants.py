import os
from pathlib import Path

DATA_PATH_ENV_VAR = "PLUGIN_DATA_PATH"
APP_PATH_ENV_VAR = "PLUGIN_APP_PATH"
PLUGIN_DIR_ENV_VAR = "PLUGIN_DIR"
CONFIG_DIR_ENV_VAR = "PLUGIN_CONFIG_DIR"
CPU_ENV_VAR = "PLUGIN_DEFAULT_CPU"
MEM_ENV_VAR = "PLUGIN_DEFAULT_MEMORY"
DEFAULT_VERIFY_ENV_VAR = "PLUGIN_DEFAULT_VERIFY"

DATA_PATH = os.environ.get(DATA_PATH_ENV_VAR, "/mnt/data")
CONFIG_PATH = os.path.join(DATA_PATH, "config.yaml")
APP_PATH = os.environ.get(APP_PATH_ENV_VAR, "/app")
DATA_SCRIPTS_PATH = os.path.join(APP_PATH, "data_scripts")
PLUGIN_FILENAME = "plugin.yaml"

PLUGINS_DIR = os.environ.get(PLUGIN_DIR_ENV_VAR, "plugins")
CONFIG_DIR = os.environ.get(CONFIG_DIR_ENV_VAR, str(Path("~/.config/plugin_lib").expanduser()))

DEFAULT_CPU = float(os.environ.get(CPU_ENV_VAR, 2.0))
DEFAULT_MEMORY = float(os.environ.get(MEM_ENV_VAR, 4.0))

DEFAULT_VERIFY = os.environ.get(DEFAULT_VERIFY_ENV_VAR, "").lower() == "y"
