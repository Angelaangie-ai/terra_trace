import logging

from ..constants import DEFAULT_CPU, DEFAULT_MEMORY
from . import manifest
from .container import upload_plugin_to_acr
from .helper import run_subprocess, verify_to_proceed
from .wrappers import AzWrapper

logger = logging.getLogger()


def container_app_for_plugin(plugin_name: str):
    """Get the container app for a plugin"""
    try:
        container_app = manifest.get_container_app(plugin_name)
    except ValueError:
        manifest.get_plugin(plugin_name)
        logger.error(f"Container app not found for plugin {plugin_name}")
        container_app = None
    if container_app:
        return container_app
    if not verify_to_proceed("Would you like to create a new container app?"):
        raise ValueError(f"Container app not found for plugin {plugin_name}")
    name, resource_group, environment = get_new_app_info()
    container_app = manifest.ContainerAppManifest(name, resource_group, environment)

    return container_app


def get_new_app_info():
    """Get the container app information from the user"""
    confirmed = False
    while not confirmed:
        name = input("Please enter the container app name: ")
        default_rg = manifest.get_default_resource_group()
        if default_rg and verify_to_proceed(f"Use default resource group {default_rg}?"):
            rg = default_rg
        else:
            rg = input("Please enter the resource group: ")
        default_environment = manifest.get_default_environment()
        if default_environment and verify_to_proceed(
            f"Use default container app environment {default_environment}?"
        ):
            environment = default_environment
        else:
            environment = input("Please enter the container app environment: ")
        confirmed = verify_to_proceed(
            f"Create container app '{name}' in '{rg}' using environment '{environment}'?"
        )
        if not default_rg and verify_to_proceed(f"Set '{rg}' as the default resource group?"):
            manifest.set_default_resource_group(rg)

    return name, rg, environment


def prepare_container_env(container_app: manifest.ContainerAppManifest):
    azw = AzWrapper()
    default_environment = manifest.get_default_environment()
    # Check if env exists
    env = container_app.environment
    rg = container_app.resource_group
    try:
        azw.get_environment(environment=env, resource_group=rg)
        logger.info(f"Container app environment {env} found")
        return
    except RuntimeError:
        pass
    logger.info(f"Container app environment {env} not found")
    logger.info(f"Creating the container app environment {env}")
    region = azw.get_rg_location(rg)
    azw.create_environment(env, rg, region)
    if not default_environment and verify_to_proceed(
        f"Would you like to set '{env}' as the default container app environment?"
    ):
        manifest.set_default_environment(env)


def deploy_plugin_to_container_app(plugin_name: str, cpu: float | None, memory: float | None):
    """Update the container app with the new image"""
    azw = AzWrapper()
    container_app = container_app_for_plugin(plugin_name)
    name, rg, env = container_app.name, container_app.resource_group, container_app.environment
    # Check if the containerapp exists
    try:
        o = azw.get_container_app(name, rg)
        logger.info(f"Container app {name} already exists")
        app_exists = True
        env_id = o["properties"]["environmentId"]
        env = azw.get_environment(env_id=env_id)["name"]
        container_app.environment = env
    except RuntimeError:
        logger.info(f"Container app {name} not found")
        app_exists = False

    verify_text = "Update" if app_exists else "Create"
    verify = verify_to_proceed(f"{verify_text} container app {name}?")
    if not verify:
        logger.info("Exiting...")
        return
    if app_exists:
        remote_tag = upload_plugin_to_acr(plugin_name)
        # Update the app with the new image and optionally the new resources
        azw.update_container_app(name, rg, container_image=remote_tag, cpu=cpu, memory=memory)
    else:
        prepare_container_env(container_app)
        remote_tag = upload_plugin_to_acr(plugin_name)
        cpu = cpu or DEFAULT_CPU
        memory = memory or DEFAULT_MEMORY
        azw.create_container_app(name, remote_tag, rg, env, cpu=cpu, memory=memory)

    manifest.add_container_app(plugin_name, container_app)


def update_container_access(
    application_name: str, resource_group: str, webapp_name: str, webapp_resource_group: str
):
    """Update the container app to allow connections from the webapp"""
    logger.info(f"Updating the network policy of {application_name}")

    # Get the webapp's outbound IP
    list_ips_cmd = (
        f"az webapp show --resource-group {webapp_resource_group} --name {webapp_name}"
        " --query outboundIpAddresses --output tsv"
    )
    webapp_ips = run_subprocess(list_ips_cmd).split(",")

    # Update the app with the new image
    rule_name_prefix = f"allow_{webapp_name}_webapp"
    for count, ip_address in enumerate(webapp_ips):
        update_app_cmd = (
            f"az containerapp ingress access-restriction set --name {application_name} "
            f"--resource-group {resource_group} --rule-name {rule_name_prefix}_{count} "
            f"--action Allow --ip-address ${ip_address}"
        )
        run_subprocess(update_app_cmd)
    logger.info("Container app updated successfully")
