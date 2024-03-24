# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

import os

import click
from loguru import logger


@click.group()
def cli():
    pass


@click.command(help="Display help")
def help():
    click.echo("Help")


def _prepare_folder():
    # check if data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), "../data/raw")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info("Data directory is created.")
    else:
        logger.info("Data directory already exists. Skipping creating directory.")


# TODO: Add more data source
@click.command(help="Download data")
def download():
    logger.info("Preparing Data for the first time ...")
    _prepare_folder()

    # Check if gz file is exist
    raw_data_path = os.path.join(os.path.dirname(__file__), "../data/raw/AStockData.tar.gz")
    if os.path.exists(raw_data_path):
        logger.info("Data already exists, Remove Data First...")
        os.remove(raw_data_path)
    else:
        logger.info("Downloading data ...")
        # Download data from file server
        request_url = (
            "https://github.com/auderson/FactorInvestmentResearchEngine/releases/download/marketdata/AStockData.tar.gz"
        )

        os.system(f"wget {request_url} -O {raw_data_path}")
        os.system(f'tar -xvf {raw_data_path} -C {os.path.join(os.path.dirname(__file__), "../data/raw")} --strip-components=1')

        data_file = os.path.join(os.path.dirname(__file__), "../data/raw/close.feather")
        if not os.path.exists(data_file):
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
            f'tar -xvf {file_path} -C {os.path.join(os.path.dirname(__file__), "../data/raw")} --strip-components=1'
        )
    except Exception as e:
        logger.error(f"Failed to unzip file: {e}")


cli.add_command(help)
cli.add_command(download)
cli.add_command(load)
