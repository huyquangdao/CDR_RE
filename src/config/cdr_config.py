"""This module indicates the detailed configurations of our framework"""

from __future__ import annotations

import json

KEY_NAMES_LIST = ["train_file_path", "dev_file_path", "test_file_path", "use_title", "mesh_filtering", "use_full"]
VALUE_TYPES_DICT = {
    "train_file_path": str,
    "dev_file_path": str,
    "test_file_path": str,
    "use_title": bool,
    "mesh_filtering": bool,
    "use_full": bool,
}


class CDRConfig:

    """The CDR Configuration class"""

    chemical_string = "Chemical"
    disease_string = "Disease"
    adjacency_rel = "node"
    root_rel = "root"

    @staticmethod
    def from_json_file(json_file_path: str) -> CDRConfig:
        """load the our method\'s configurations from a json config file

        Args:
            json_file_path (str): path to the json config file

        Returns:
            CDRConfig: an instance of class CDRConfig
        """
        with open(json_file_path) as f_json:
            json_data = json.load(f_json)
            CDRConfig.validate_json_data(json_data)
            config = CDRConfig()
            for attr, value in json_data.items():
                setattr(config, attr, value)
            return config

    @staticmethod
    def validate_json_data(json_data: dict) -> None:
        """validate the json data dictionary

        Args:
            json_data (dict): [description]

        Returns:
            bool: [description]
        """

        if len(json_data) < len(KEY_NAMES_LIST):
            missing_params = [key for key in KEY_NAMES_LIST if key not in list(json_data.keys())]
            raise Exception(f"params: {missing_params} must be defined")
        for key, value in json_data.items():
            if key not in KEY_NAMES_LIST:
                raise Exception(f"all config params must be in the pre-defined list: {KEY_NAMES_LIST}")
            if not isinstance(value, VALUE_TYPES_DICT[key]):
                raise Exception(f"Param's type not match. given:{type(value)}, expected:{VALUE_TYPES_DICT[key]}")
