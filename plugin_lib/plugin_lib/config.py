from typing import Any, Dict

from fastapi import APIRouter

from .storage import get_storage

config = APIRouter(prefix="/config")


def get_config():
    storage = get_storage()
    return storage.retrieve_config()


def set_config(config: Dict[str, Any]) -> Dict[str, Any]:
    storage = get_storage()
    storage.store_config(config)
    return config


config.get("/")(get_config)
config.post("/")(set_config)
