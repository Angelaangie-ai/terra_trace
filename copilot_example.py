import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import requests
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from prance import ResolvingParser
from termcolor import colored
from termcolor._types import Color

SYSTEM_PROMPT = (
    "You are an AI assistant capable of using external tools. "
    f"Today is {datetime.now().strftime('%Y/%m/%d - %A')}."
)

DEFAUL_BASE_URL = os.environ.get("PLUGIN_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("COPILOT_MODEL", "gpt-4")
DEFAULT_ENDPOINT = os.environ.get("COPILOT_ENPOINT")
DEFAULT_USER_ID = os.environ.get("COPILOT_USER_ID", "USER1")


def get_plugin_specs(base_url: str):
    parser = ResolvingParser(f"{base_url}/openapi.json")
    spec = parser.specification
    assert spec is not None
    # Assume one spec per path using post and with all parameters in the body with json-type
    tool_specs = [
        get_tool_spec(name, path)
        for name, path in spec["paths"].items()
        if name.startswith("/tools")
    ]
    return tool_specs


def get_tool_spec(name: str, path: Dict[str, Any]):
    assert "post" in path
    operation = path["post"]
    body = operation.get("requestBody")
    if body is None:
        parameters = {"type": "object", "properties": {}}
    else:
        content = body["content"]["application/json"]
        parameters = content["schema"]
    tool_spec = {
        "type": "function",
        "function": {
            "name": name.replace("/tools/", "").strip("/").replace("/", "_"),
            "description": operation["description"],
            "parameters": parameters,
        },
    }
    return tool_spec


def pretty_print_conversation(message: Any):
    role_to_color: Dict[str, Color] = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }
    if message["role"] == "system":
        print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
    elif message["role"] == "user":
        print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
    elif message["role"] == "assistant" and message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            print(colored(f"assistant: {tool_call}\n", role_to_color[message["role"]]))
    elif message["role"] == "assistant" and not message.get("function_call"):
        print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
    elif message["role"] == "tool":
        print(
            colored(
                f"function ({message['name']}): {message['content']}\n",
                role_to_color[message["role"]],
            )
        )


def call_tool(tool_call: ChatCompletionMessageToolCall, user_id: str):
    func = tool_call.function
    response = requests.post(
        f"{DEFAUL_BASE_URL}/tools/{func.name}",
        headers={"User-ID": user_id},
        json=json.loads(func.arguments),
    )
    return response.text


def call_tools_if_necessary(message: ChatCompletionMessage, user_id: str):
    tool_responses = []
    if message.tool_calls:
        # Can do multiple parallel calls, loop through them
        for tool_call in message.tool_calls:
            func = tool_call.function
            tool_response = call_tool(tool_call, user_id)
            tool_responses.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func.name,
                    "content": tool_response,
                }
            )
    return tool_responses


def run_conversation(client: AzureOpenAI, model: str, tools: List[Dict[str, Any]], user_id: str):
    user_input = input("User: ")
    user_message = {
        "role": "user",
        "content": user_input,
    }
    pretty_print_conversation(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, user_message]
    loop = True
    while loop:
        response = client.chat.completions.create(
            messages=messages, model=model, tools=tools, max_tokens=1024  # type:ignore
        )
        message = response.choices[0].message
        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content if message.content else "",
        }

        if message.tool_calls:
            message_dict["tool_calls"] = message.tool_calls
        messages.append(message_dict)
        pretty_print_conversation(message_dict)
        tool_responses = call_tools_if_necessary(message, user_id)
        for tool_response in tool_responses:
            messages.append(tool_response)
            pretty_print_conversation(tool_response)
        if not tool_responses:
            user_input = input("User: ")
            if user_input == "exit":
                loop = False
            else:
                user_message = {
                    "role": "user",
                    "content": user_input,
                }
                pretty_print_conversation(user_message)
                messages.append(user_message)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--user-id", type=str, default=DEFAULT_USER_ID)
    arg_parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT)
    arg_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    arg_parser.add_argument("--plugin-url", type=str, default=DEFAUL_BASE_URL)

    args = arg_parser.parse_args()
    api_key = os.environ.get("COPILOT_API_KEY")
    if not args.endpoint:
        raise ValueError("Endpoint not provided")
    if not api_key:
        raise ValueError("API key not provided")
    plugin_urls = args.plugin_url.split(",")

    tool_specs = []
    for plugin_url in plugin_urls:
        tool_specs.extend(get_plugin_specs(plugin_url))
    [print(i) for i in tool_specs]

    client = AzureOpenAI(
        api_version="2023-08-01-preview",
        api_key=api_key,
        azure_endpoint=args.endpoint,
    )

    run_conversation(client, args.model, tool_specs, args.user_id)


if __name__ == "__main__":
    main()
