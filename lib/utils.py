# Import Libraries
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv


class LoadData():
    def __init__(self, dataset, sub_dir=""):
        envs = load_dotenv(find_dotenv())
        self._file = os.getenv("data_file")
        full_path = self._file + sub_dir + "/" + dataset
        self._data = pd.read_csv(full_path)

    @property
    def data(self):
        return self._data
