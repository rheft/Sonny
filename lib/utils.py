# Import Libraries
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv


class LoadData():
    def __init__(self, dataset, sub_dir="", **kwargs):
        envs = load_dotenv(find_dotenv())
        self._file = os.getenv("data_file")
        full_path = self._file + sub_dir + "/" + dataset
        self._header = kwargs.get('header', False)
        self._seperator = kwargs.get('seperator', ',')
        self._quoting = kwargs.get('quoting', 0)
        print(self._seperator)
        if self._header is not None:
            self._data = pd.read_csv(
                full_path,
                sep=self._seperator,
                quoting=self._quoting
            )
        else:
            self._data = pd.read_csv(
                full_path,
                header=self._header,
                sep=self._seperator,
                quoting=self._quoting
            )

    @property
    def data(self):
        return self._data
