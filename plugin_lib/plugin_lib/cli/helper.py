import logging
import subprocess

from ..constants import DEFAULT_VERIFY

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def verify_to_proceed(message: str, default: bool = DEFAULT_VERIFY) -> bool:
    """Ask the user for confirmation before proceeding"""
    opt_text = " (Y/n): " if default else " (y/N): "
    confirmation = input(f"{message}{opt_text}")
    if not confirmation:
        return default
    if confirmation.lower() == "y":
        return True
    return False


def run_subprocess(command: str):
    """Run a subprocess command and return the output"""
    logger.debug(f"Running command: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Failed to run command: {error}")
    logger.debug(f"Done running command: {command}")
    return output.decode("utf-8").strip()
