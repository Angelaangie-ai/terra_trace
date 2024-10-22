from fastapi import FastAPI

from plugin_lib.config import config
from plugin_lib.tool import router as tool_router
from plugin_lib.utils import load_module, load_plugin_spec

plugin_spec = load_plugin_spec()


def load_modules():
    tool_modules = plugin_spec.modules
    for tool_module in tool_modules:
        load_module(tool_module)


load_modules()
app = FastAPI(title=plugin_spec.name, version=plugin_spec.version)
app.include_router(config)
app.include_router(tool_router)
