"""This module indicates the detailed configurations of our framework"""

from __future__ import annotations

import json

KEY_NAMES_LIST = [
    "train_file_path",
    "dev_file_path",
    "test_file_path",
    "mesh_path",
    "mesh_filtering",
    "use_title",
    "use_full",
    "train_elmo_path",
    "train_flair_path",
    "dev_elmo_path",
    "dev_flair_path",
    "test_elmo_path",
    "test_flair_path",
    "word_vocab_path",
    "rel_vocab_path",
    "pos_vocab_path",
    "char_vocab_path",
    "hypernym_vocab_path",
    "synonym_vocab_path",
]
VALUE_TYPES_DICT = {
    "train_file_path": str,
    "dev_file_path": str,
    "test_file_path": str,
    "mesh_path": str,
    "use_title": bool,
    "mesh_filtering": bool,
    "use_full": bool,
    "train_elmo_path": str,
    "train_flair_path": str,
    "dev_elmo_path": str,
    "dev_flair_path": str,
    "test_elmo_path": str,
    "test_flair_path": str,
    "word_vocab_path": str,
    "rel_vocab_path": str,
    "pos_vocab_path": str,
    "char_vocab_path": str,
    "hypernym_vocab_path": str,
    "synonym_vocab_path": str,
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
        """validate the json data

        Args:
            json_data (dict): the dictionary that contains param, value pairs after loading the json data.

        Raises:
            Exception: there are some compulsory params which were not defined.
            Exception: contain any param name which not in KEY_NAMES_LIST
            Exception: param type doesn't match. eg given: int and expected: str.
        """
        if len(json_data) < len(KEY_NAMES_LIST):
            missing_params = [key for key in KEY_NAMES_LIST if key not in list(json_data.keys())]
            raise Exception(f"params: {missing_params} must be defined")
        for key, value in json_data.items():
            if key not in KEY_NAMES_LIST:
                raise Exception(f"all config params must be in the pre-defined list: {KEY_NAMES_LIST}")
            if not isinstance(value, VALUE_TYPES_DICT[key]):
                raise Exception(f"Param's type not match. given:{type(value)}, expected:{VALUE_TYPES_DICT[key]}")
