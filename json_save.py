# -*- encoding: utf-8 -*-

# We use simplejson instead of the json package in the Python standard package
# library, as the latter does not save namedtuples correctly
import simplejson as json
import logging as log
from typing import Any, Dict


def save_parameters_to_json(params: Dict[str, Any], output_file_name: str):
    'Save the content of "params" in a JSON file'

    with open(output_file_name, "wt") as json_file:
        json.dump(params, json_file, indent=4, sort_keys=True)
    log.info('Parameters saved into file "{0}"'.format(output_file_name))
