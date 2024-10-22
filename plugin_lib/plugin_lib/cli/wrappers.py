import json
import logging
import time
from typing import Any, overload

from .helper import run_subprocess

logger = logging.getLogger()


class DockerWrapper:

    def build(self, context_dir: str, tag: str, build_context: str | None = None):
        build_command = f"docker build -t {tag} {context_dir}"
        if build_context:
            build_command += f" --build-context {build_context}"
        run_subprocess(build_command)
        logger.info("Image built successfully")

    def push(self, tag: str):
        push_command = f"docker push {tag}"
        run_subprocess(push_command)
        logger.info("Image pushed successfully")

    def tag(self, image: str, tag: str):
        tag_command = f"docker tag {image} {tag}"
        run_subprocess(tag_command)
        logger.info(f"Image {image} tagged as {tag} successfully")


class AzWrapper:
    MAX_TRIES = 3
    WAIT_S = 10

    def acr_login(self, registry: str):
        login_command = f"az acr login --name {registry}"
        run_subprocess(login_command)
        logger.info(f"Logged into ACR '{registry}' successfully")

    def get_acr_login_server(self, registry: str):
        get_server_command = f"az acr show --name {registry} --query loginServer --output tsv"
        return run_subprocess(get_server_command)

    def get_rg_location(self, resource_group: str):
        get_location_command = (
            f"az group show --name {resource_group} --query location --output tsv"
        )
        return run_subprocess(get_location_command)

    def get_container_app(self, name: str, resource_group: str):
        command = f"az containerapp show -n {name} --resource-group {resource_group}"
        o = run_subprocess(command)
        return json.loads(o)

    def create_container_app(
        self,
        name: str,
        container_image: str,
        resource_group: str,
        environment: str,
        cpu: float | None = None,
        memory: float | None = None,
    ):
        create_command = (
            f"az containerapp up -n {name} -g {resource_group}  --image {container_image} "
            f"--environment {environment} --ingress external --target-port 8000"
        )
        logger.info(f"Creating the container app {name}")
        run_subprocess(create_command)
        if cpu is None and memory is None:
            # If we don't specify the resources, we can return
            return
        # If we specify the resources, we need to update the app after creation
        time.sleep(self.WAIT_S)
        logger.info("Updating container app resources")
        for _ in range(self.MAX_TRIES):
            try:
                self.update_container_app(name, resource_group, cpu=cpu, memory=memory)
                logger.info("Container app created successfully")
                break
            except RuntimeError:
                logger.warning(
                    f"Failed to update container app resources, retrying in {self.WAIT_S}s..."
                )
                time.sleep(self.WAIT_S)
        else:
            raise RuntimeError(
                f"Failed to update container app resources after {self.MAX_TRIES} tries"
            )

    def update_container_app(
        self,
        name: str,
        resource_group: str,
        container_image: str | None = None,
        cpu: float | None = None,
        memory: float | None = None,
    ):
        logger.info(f"Updating the container app {name}")

        # Update the app with the new image
        update_app_cmd = f"az containerapp update --name {name} --resource-group {resource_group}"
        if container_image:
            update_app_cmd += f" --image {container_image}"
        if cpu is not None:
            update_app_cmd += f" --cpu {cpu:g}"
        if memory is not None:
            update_app_cmd += f" --memory {memory:g}Gi"
        run_subprocess(update_app_cmd)
        logger.info("Container app updated successfully")

    @overload
    def get_environment(
        self, *, environment: str, resource_group: str, env_id: None = None
    ) -> dict[str, Any]: ...

    @overload
    def get_environment(
        self, *, environment: None = None, resource_group: None = None, env_id: str
    ) -> dict[str, Any]: ...

    def get_environment(
        self,
        *,
        environment: str | None = None,
        resource_group: str | None = None,
        env_id: str | None = None,
    ):
        if (environment is None or resource_group is None) and env_id is None:
            raise ValueError("Either environment and resource_group or env_id must be provided")
        command = "az containerapp env show"
        if environment and resource_group:
            command += f" -n {environment} --resource-group {resource_group}"
        elif env_id:
            command += f" --ids {env_id}"
        else:
            raise ValueError("Either environment and resource_group or env_id must be provided")

        o = run_subprocess(command)
        return json.loads(o)

    def create_environment(self, name: str, resource_group: str, location: str | None):
        create_command = (
            f"az containerapp env create --name {name} " f"--resource-group {resource_group}"
        )
        if location:
            create_command += f" --location {location}"
        run_subprocess(create_command)
        logger.info("Container app environment created successfully")
