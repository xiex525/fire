import configparser
from loguru import logger

config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get(section='data', option="DATA_PATH", fallback='data')