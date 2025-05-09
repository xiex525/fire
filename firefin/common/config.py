# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt
import os
import pathlib
import yaml
with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as stream: 
    config = yaml.safe_load(stream) 

# Load configuration from YAML file 

# Define DATA_PATH based on configuration, expanding user and resolving path 
DATA_PATH = pathlib.Path(config['DATA_PATH']).expanduser().resolve() 
