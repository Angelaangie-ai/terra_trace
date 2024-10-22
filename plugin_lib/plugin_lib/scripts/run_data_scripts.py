# PREPARE WEATHER DATA FOR THE LLM
# Download data from Herbie for each field and store it in a database
# Data can be stored on a CSV for now (one CSV per field?)
# Herbie will use multiple threads to download HRRR GRIB files that can be used by all fields
# Ideally we download the array and get all nearest points in one go
# Do we keep the GRIB files around? I'm tempted to discard them
##
# The jobs will take a long time to run, so we might need to keep track of the state
# Keep track of each field separately? State store with status for each field?
# Group date ranges and download for all fields that need it? <- Annoying to do...
# Run download script every day?
# Config: start date, end date, frequency, fields?
# Add field to config using an endpoint?
# Write state to disk using simple stuff like JSON for now (mount point in docker)

import asyncio
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

from croniter import croniter

from plugin_lib.spec import ScriptSpec
from plugin_lib.storage import get_storage
from plugin_lib.utils import load_module, load_plugin_spec

LOGGER = logging.getLogger(__name__)


def load_entrypoint(script_path: str):
    module = load_module(script_path)
    return module.main


async def run_job_at(script_spec: ScriptSpec):
    storage = get_storage()
    config = storage.retrieve_config()
    script_config = config[script_spec.name]

    schedule_str = script_config["schedule"]
    iter = croniter(schedule_str)
    executor = ThreadPoolExecutor(max_workers=1)
    LOGGER.info(f"{script_spec.name} - running job  with schedule '{schedule_str}'")
    while True:
        now = datetime.datetime.now()
        next_run = iter.get_next(datetime.datetime)
        time_to_next_run = (next_run - now).total_seconds()
        if time_to_next_run < 0:
            continue
        LOGGER.info(f"{script_spec.name} - sleeping for {time_to_next_run} seconds")
        await asyncio.sleep(time_to_next_run)
        # Run script in a separate process
        entrypoint = load_entrypoint(script_spec.path)
        # Reload config to make sure it's up to date
        script_config = storage.retrieve_config()[script_spec.name]
        params = script_config.get("params", {})
        p = Process(target=entrypoint, kwargs=params if params is not None else {})
        p.start()
        loop = asyncio.get_event_loop()
        # Wait for the job to finish
        await loop.run_in_executor(executor, p.join)

        # Reload config to make sure it's up to date
        script_config = storage.retrieve_config()[script_spec.name]
        # If the schedule has changed, let's recreate the iterator
        if script_config["schedule"] != schedule_str:
            LOGGER.info(
                f"{script_spec.name} - upating schedule from '{schedule_str}' "
                f"to '{script_config['schedule']}'"
            )
            schedule_str = script_config["schedule"]
            iter = croniter(schedule_str)


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    plugin_spec = load_plugin_spec()

    data_scripts = plugin_spec.data_scripts
    jobs = []
    for script in data_scripts:
        jobs.append(run_job_at(script))
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*jobs))


if __name__ == "__main__":
    main()
