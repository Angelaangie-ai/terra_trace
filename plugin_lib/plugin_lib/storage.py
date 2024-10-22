import io
import json
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Union

import pandas as pd
import yaml
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.storage.blob import ContainerClient
from hydra.utils import instantiate

from .constants import APP_PATH
from .utils import load_plugin_spec


class ResourceNotFoundError(Exception):
    pass


JSON = Any


class Storage(ABC):
    @abstractmethod
    def store_dataframe(self, id: str, df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def retrieve_dataframe(self, id: str):
        raise NotImplementedError

    @abstractmethod
    def remove_dataframe(self, id: str, not_found_ok: bool = False):
        raise NotImplementedError

    @abstractmethod
    def store_metadata(self, key: str, value: JSON):
        raise NotImplementedError

    @abstractmethod
    def retrieve_metadata(self, key: str) -> JSON:
        raise NotImplementedError

    @abstractmethod
    def remove_metadata(self, key: str, not_found_ok: bool = False):
        raise NotImplementedError

    @abstractmethod
    def store_config(self, config: JSON):
        raise NotImplementedError

    @abstractmethod
    def retrieve_config(self) -> JSON:
        raise NotImplementedError


class LocalStorage(Storage):
    """
    Local storage implementation. Save files to a directory in disk.
    """

    METADATA_FILENAME = "metadata.json"
    CONFIG_FILENAME = "config.yaml"

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        metadata_filepath = os.path.join(root_dir, self.METADATA_FILENAME)
        if not os.path.exists(metadata_filepath):
            # Create empty metadata file
            with open(metadata_filepath, "w") as f:
                json.dump({}, f)
        config_filepath = os.path.join(root_dir, self.CONFIG_FILENAME)
        if not os.path.exists(config_filepath):
            # Copy over default config
            shutil.copy(os.path.join(APP_PATH, self.CONFIG_FILENAME), config_filepath)

    def store_dataframe(self, id: str, df: pd.DataFrame):
        filepath = os.path.join(self.root_dir, f"{id}.parquet")
        df.to_parquet(filepath)

    def retrieve_dataframe(self, id: str):
        filepath = os.path.join(self.root_dir, f"{id}.parquet")
        if not os.path.exists(filepath):
            raise ResourceNotFoundError(f"Resource {id} not found")
        return pd.read_parquet(filepath)

    def remove_dataframe(self, id: str, not_found_ok: bool = False):
        filepath = os.path.join(self.root_dir, f"{id}.parquet")
        if not os.path.exists(filepath):
            if not_found_ok:
                return
            raise ResourceNotFoundError(f"Resource {id} not found")
        os.remove(filepath)

    def _load_metadata(self):
        filepath = os.path.join(self.root_dir, self.METADATA_FILENAME)
        if os.path.exists(filepath):
            with open(filepath) as f:
                metadata = json.load(f)
            return metadata
        return {}

    def store_metadata(self, key: str, value: JSON):
        metadata = self._load_metadata()
        metadata[key] = value
        filepath = os.path.join(self.root_dir, self.METADATA_FILENAME)
        with open(filepath, "w") as f:
            json.dump(metadata, f)

    def retrieve_metadata(self, key: str) -> JSON:
        metadata = self._load_metadata()
        try:
            return metadata[key]
        except KeyError:
            raise ResourceNotFoundError(f"Metadata for '{key}' not found")

    def remove_metadata(self, key: str, not_found_ok: bool = False):
        metadata = self._load_metadata()
        try:
            del metadata[key]
        except KeyError:
            if not_found_ok:
                return
            raise ResourceNotFoundError(f"Metadata for '{key}' not found")
        filepath = os.path.join(self.root_dir, self.METADATA_FILENAME)
        with open(filepath, "w") as f:
            json.dump(metadata, f)

    def store_config(self, config: JSON):
        filepath = os.path.join(self.root_dir, self.CONFIG_FILENAME)
        with open(filepath, "w") as f:
            yaml.dump(config, f)

    def retrieve_config(self) -> JSON:
        filepath = os.path.join(self.root_dir, self.CONFIG_FILENAME)
        with open(filepath) as f:
            config = yaml.safe_load(f)
        return config


class BlobStorage(Storage):
    """
    Blob storage implementation. Save files to a container in a blob storage.
    """

    METADATA_FILENAME = "metadata.json"
    CONFIG_FILENAME = "config.yaml"

    def __init__(self, account_url: str, container_name: str):
        self.account_url = account_url
        self.container_name = container_name
        for cred in (DefaultAzureCredential(), AzureCliCredential()):
            try:
                self.container_client = ContainerClient(
                    account_url, container_name, credential=cred
                )
                if not self.container_client.exists():
                    self.container_client.create_container()
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not authenticate to Azure Blob Storage")

        # Check if metadata file exists, if not create an empty one
        metadata_blob_client = self._get_blob(self.METADATA_FILENAME)
        if not metadata_blob_client.exists():
            self.container_client.upload_blob(self.METADATA_FILENAME, "{}")

        # Check if config file exists, if not create an empty one
        config_blob_client = self._get_blob(self.CONFIG_FILENAME)
        if not config_blob_client.exists():
            with open(os.path.join(APP_PATH, self.CONFIG_FILENAME)) as f:
                config = f.read()
            self.container_client.upload_blob(self.CONFIG_FILENAME, config)

    def _get_blob(self, blob_name: str):
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client

    def _load_blob(self, blob_name: str):
        blob_client = self._get_blob(blob_name)
        if not blob_client.exists():
            raise ResourceNotFoundError(f"Resource {blob_name} not found")
        blob = blob_client.download_blob()
        return blob.readall()

    def _store_blob(self, blob_name: str, blob: Union[bytes, str]):
        blob_client = self._get_blob(blob_name)
        blob_client.upload_blob(blob, overwrite=True)

    def store_dataframe(self, id: str, df: pd.DataFrame):
        with io.BytesIO() as f:
            df.to_parquet(f)
            self._store_blob(f"{id}.parquet", f.getvalue())

    def retrieve_dataframe(self, id: str):
        with io.BytesIO(self._load_blob(f"{id}.parquet")) as f:
            return pd.read_parquet(f)

    def remove_dataframe(self, id: str, not_found_ok: bool = False):
        blob_client = self._get_blob(f"{id}.parquet")
        if not blob_client.exists():
            if not_found_ok:
                return
            raise ResourceNotFoundError(f"Resource {id} not found")
        blob_client.delete_blob()

    def _load_metadata(self):
        metadata = self._load_blob(self.METADATA_FILENAME)
        return json.loads(metadata)

    def store_metadata(self, key: str, value: JSON):
        metadata = self._load_metadata()
        metadata[key] = value
        self._store_blob(self.METADATA_FILENAME, json.dumps(metadata))

    def retrieve_metadata(self, key: str) -> JSON:
        metadata = self._load_metadata()
        try:
            return metadata[key]
        except KeyError:
            raise ResourceNotFoundError(f"Metadata for '{key}' not found")

    def remove_metadata(self, key: str, not_found_ok: bool = False):
        metadata = self._load_metadata()
        if key not in metadata:
            if not_found_ok:
                return
            raise ResourceNotFoundError(f"Metadata for '{key}' not found")
        del metadata[key]
        self._store_blob(self.METADATA_FILENAME, json.dumps(metadata))

    def store_config(self, config: JSON):
        self._store_blob(self.CONFIG_FILENAME, yaml.dump(config))

    def retrieve_config(self) -> JSON:
        return yaml.safe_load(self._load_blob(self.CONFIG_FILENAME))


def get_storage() -> Storage:
    """
    Instantiates the storage according to the plugin specification.
    """
    plugin_spec = load_plugin_spec()
    storage_config = plugin_spec.storage
    storage = instantiate(storage_config)
    return storage
