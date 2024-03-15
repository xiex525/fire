# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get(section='data', option="DATA_PATH", fallback='data')