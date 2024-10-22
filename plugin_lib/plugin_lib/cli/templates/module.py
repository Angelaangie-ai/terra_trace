from typing import Annotated

from plugin_lib.config import config
from plugin_lib.tool import tool


# You can implement tools here
# Use the @tool decorator to define a tool based on a function
# The function should have type hints for arguments
# Each argument should be annotated with Annotated[arg_type, "description"]
# Tools will be exposed endpoints in the API in the path /tools/<tool_name>
# Example:
@tool("This should contain the tool description")
def tool_name(arg1: Annotated[str, "This is the description for arg1"]):
    # This is the tool implementation
    return f"Hello, {arg1}!"


# You can also implement configuration endpoints here
# Use the config router to define configuration endpoints with the desired methods
# Example:
@config.get("/custom_config")
def get_custom_config():
    return {"custom": "config"}
