# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import os
import click
from ..common.config import DATA_PATH
from loguru import logger


@click.group()
def cli():
    pass


@click.command(help="Display help")
def help():
    click.echo("Help")


def _prepare_folder():
    # check if data directory exists
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Data directory is created.")
    else:
        logger.info("Data directory already exists. Skipping creating directory.")


# TODO: Add more data source
@click.command(help="Download data")
def download():
    logger.info("Preparing Data for the first time ...")
    _prepare_folder()

    # Check if gz file is exist
    raw_data_path = DATA_PATH / "AStockData.tar.gz"
    if raw_data_path.exists():
        logger.info("Data already exists, Remove Data First...")
        # ensure the file is removed before downloading again
        raw_data_path.unlink(missing_ok=True)  # remove the file
        logger.info("Data Removed.")
    else:
        logger.info("Downloading data ...")
        # Download data from file server
        request_url = (
            "https://github.com/fire-institute/fire/releases/download/marketdata/AStockData.tar.gz"
        )

        os.system(f"wget {request_url} -O {raw_data_path}")
        os.system(f'tar -xvf {raw_data_path} -C {DATA_PATH} --strip-components=1')

        data_file =DATA_PATH / "index.feather"
        if not data_file.exists():
            raise Exception(f"Failed to download data, If you are behind a proxy, please set it up. or download the data manually from {request_url} and run fire load <file_path> to load the data.")

    logger.info("Data is downloaded and saved to local storage. Done!!")


@click.command(help="Prepare data")
@click.argument("file_path", type=click.Path(exists=True))
def load(file_path: str = None):
    logger.info("Preparing Data for the first time ...")
    _prepare_folder()

    # tar unzip file, print progress
    try:
        os.system(
            f'tar -xvf {file_path} -C {DATA_PATH} --strip-components=1'
        )
    except Exception as e:
        logger.error(f"Failed to unzip file: {e}")


cli.add_command(help)
cli.add_command(download)
cli.add_command(load)
