import argparse
from collections import namedtuple
from enum import Enum
from typing import Any, Callable

from .container import build_plugin, upload_plugin_to_acr
from .container_app import deploy_plugin_to_container_app
from .local import create_plugin, list_plugins, set_env, set_registry, set_rg

CLIAction = namedtuple("CLIAction", ["action", "help_text"])


class CLIActions(Enum):
    LIST = CLIAction("list", "List all plugins")
    CREATE = CLIAction("create", "Create a new plugin")
    BUILD = CLIAction("build", "Build the plugin locally")
    PUSH = CLIAction("push", "Push the plugin to a container registry")
    DEPLOY = CLIAction("deploy", "Deploy the plugin to a container app")
    GET_DEPLOYMENT = CLIAction("get-deployment", "Get the deployment for a plugin")
    SET_ENV = CLIAction("set-env", "Set the default environment for a container app")
    SET_REGISTRY = CLIAction("set-registry", "Set the default registry for a plugin")
    SET_RG = CLIAction("set-rg", "Set the default resource group for a container app")

    @staticmethod
    def from_action(action: str) -> "CLIActions":
        for item in CLIActions:
            if item.value.action == action:
                return item
        raise ValueError(f"Invalid action: {action}")


class CliParser:
    dispatch_mapping: dict[CLIActions, Callable[..., Any]] = {
        CLIActions.LIST: list_plugins,
        CLIActions.CREATE: create_plugin,
        CLIActions.BUILD: build_plugin,
        CLIActions.PUSH: upload_plugin_to_acr,
        CLIActions.DEPLOY: deploy_plugin_to_container_app,
        CLIActions.SET_ENV: set_env,
        CLIActions.SET_REGISTRY: set_registry,
        CLIActions.SET_RG: set_rg,
    }

    PLUGIN_COMMANDS = (
        CLIActions.CREATE,
        CLIActions.BUILD,
        CLIActions.PUSH,
        CLIActions.DEPLOY,
        CLIActions.GET_DEPLOYMENT,
    )

    DEFAULT_COMMANDS = (CLIActions.SET_ENV, CLIActions.SET_REGISTRY, CLIActions.SET_RG)

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        subparsers = self.parser.add_subparsers(
            dest="action", help="Action to perform", required=True
        )
        self.command_parsers = {}
        for command in CLIActions:
            self.command_parsers[command] = subparsers.add_parser(
                command.value.action, help=command.value.help_text
            )
        self._add_positional_args()
        self._add_optional_args()

    def _add_positional_args(self):
        for command in self.PLUGIN_COMMANDS:
            self.command_parsers[command].add_argument("plugin_name", help="Name of the plugin")
        for command in self.DEFAULT_COMMANDS:
            self.command_parsers[command].add_argument("default", help="New default value")

    def _add_optional_args(self):
        deploy_parser = self.command_parsers[CLIActions.DEPLOY]
        deploy_parser.add_argument(
            "--cpu", type=float, help="CPU units for the container app", default=None
        )
        deploy_parser.add_argument(
            "--memory",
            type=float,
            help="Memory (in GB) for the container app",
            default=None,
        )

    def parse_args(self):
        args = self.parser.parse_args()
        if args.action not in [action.value.action for action in CLIActions]:
            raise ValueError(f"Invalid action: {args.action}")

        return args

    def dispatch(self, args: argparse.Namespace):
        fun = self.dispatch_mapping[CLIActions.from_action(args.action)]

        if CLIActions.from_action(args.action) == CLIActions.DEPLOY:
            return fun(args.plugin_name, args.cpu, args.memory)
        if CLIActions.from_action(args.action) in self.PLUGIN_COMMANDS:
            return fun(args.plugin_name)
        if CLIActions.from_action(args.action) in self.DEFAULT_COMMANDS:
            return fun(args.default)
        fun()
