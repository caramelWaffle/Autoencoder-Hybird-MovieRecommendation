import pandas as pd
import os


class CSVHelper:

    def read_csv(self, path):
        return pd.read_csv(path, encoding="utf-8")

    def write_csv(self, save_path, file_name, dataframe):
        dataframe.to_csv(os.path.join(save_path, f'{file_name}.csv'), index=False)