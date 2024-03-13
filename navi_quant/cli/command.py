import click
from ..common.config import logger

@click.group()
def cli():
    pass


@click.command(help="Display help")
def help():
    click.echo('Help')

@click.command(help="Download data")
def download():
    logger.warning('Not implemented yet! Need setup a file sever for file download.')
    click.echo('Downloading data ...')



cli.add_command(help)
cli.add_command(download)