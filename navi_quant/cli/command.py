import click
import os
from ..common.config import logger

@click.group()
def cli():
    pass


@click.command(help="Display help")
def help():
    click.echo('Help')

# TODO: Add more data source
@click.command(help="Download data")
def download():
    logger.info('Preparing Data for the first time ...')
    # Check if gz file is exist
    raw_data_path = os.path.join(os.path.dirname(__file__), '../data/raw/AStockData.tar.gz')
    if os.path.exists(raw_data_path):
        logger.info('Data is already downloaded. Skip downloading.')
    else:
        logger.infoo('Downloading data ...')
        # Download data from file server
        request_url = 'https://github.com/auderson/FactorInvestmentResearchEngine/releases/download/marketdata/AStockData.tar.gz'
        try:
            os.system(f'wget {request_url} -O {raw_data_path}')
        except Exception as e:
            logger.error(f"Failed to download file: {e}")

    # tar unzip file, print progress
    try:
        os.system(f'tar -xvf {raw_data_path} -C {os.path.join(os.path.dirname(__file__), "../data/raw")} --strip-components=1')
    except Exception as e:
        logger.error(f"Failed to unzip file: {e}")
    
    logger.info('Data is downloaded and saved to local storage. Done.!!')
    
@click.command(help="Prepare data")
@click.argument('file_path', type=click.Path(exists=True))
def load(file_path: str = None):
    logger.info('Preparing Data for the first time ...')
    # tar unzip file, print progress
    try:
        os.system(f'tar -xvf {file_path} -C {os.path.join(os.path.dirname(__file__), "../data/raw")} --strip-components=1')
    except Exception as e:
        logger.error(f"Failed to unzip file: {e}")



cli.add_command(help)
cli.add_command(download)
cli.add_command(load)