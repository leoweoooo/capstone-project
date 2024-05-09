from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

import json
import glob
import os

@dataclass
class CapstoneData:
    envr: str
    medium: str
    species: str
    time: int
    trial: int
    data: pd.DataFrame
    
    def min_quality(self, min_threshold:int=70) -> None:
        """
        Discards compounds that are below a certain quality index.
        """
        self.data = self.data[self.data['Quality'] >= min_threshold]
        
    def no_siloxane(self) -> None:
        """
        Discards compounds that contains `siloxane` in its name. 
        They are assumed to be non-mVOCs. 
        """
        self.data = self.data[~self.data['Compound'].str.contains("siloxane", case=False)]
    
    def sig_filter(self, sig_compounds:list[str]) -> None:
        """
        Discards compounds that are not found in the `sig_compounds` list.
        """
        self.data = self.data[self.data['Compound'].isin(sig_compounds)]

def from_dirs(paths:list[str]) -> list[CapstoneData]:
    """
    Imports `xlsx` files from a list of parent directories.
    The trial numbers are assigned based on the order of directories.
    """
    output = []
    for i, path in enumerate(paths):
        excel_files = glob.glob(os.path.join(path, '*.xlsx'))
        for file in excel_files:
            filename = os.path.basename(file).split(' ')
            envr = filename[0]
            medium = filename[1]
            species = filename[2]
            time = filename[3].rsplit('h')[0]
            data = pd.read_excel(file, index_col='Peak')
            
            output.append(
                CapstoneData(
                    envr,
                    medium,
                    species,
                    time,
                    trial=i+1,
                    data=data,
                )
            )
    
    return output

def from_json(path:str) -> list[CapstoneData]:
    """
    Imports a `json` file from directory.
    """
    output = []
    with open(path, 'r') as file:
        raw_records = json.load(file)
        
    for record in raw_records:
        output.append(
            CapstoneData(
                envr=record['envr'],
                medium=record['medium'],
                species=record['species'],
                time=record['time'],
                trial=record['trial'],
                data=pd.DataFrame.from_records(record['data'])
            )
        )
    
    return output

def to_json(records: list[CapstoneData], path:str) -> None:
    """
    Exports the current list of records into a `json` file.
    """
    data_list = []
    for record in records:
        data_dict = {
            'envr': record.envr,
            'medium': record.medium, 
            'species': record.species,
            'time': record.time,
            'trial': record.trial,
            'data': json.loads(record.data.to_json(orient='records'))
        }
        data_list.append(data_dict)
    with open(path, 'w') as file:
        json.dump(data_list, file, indent=4)

def to_df(records: list[CapstoneData]) -> pd.DataFrame:
    """
    Creates a dataframe where each column are different features (i.e., compounds). 
    
    A column of `Label` is appended in the last column for classification tasks.
    """
    output = pd.DataFrame()
    synonyms = {
        'Ethyl Alcohol': 'Ethanol',
        'Limonene': 'D-Limonene',
    }
    
    labels =[f'{record.envr} {record.medium} {record.species}' for record in records]
    for record in records:
        column_name = f'{record.envr} {record.medium} {record.species} {record.time}h {record.trial}'
        df = record.data.copy(deep=True)
        df.drop(['Retention Time', 'Relative Area', 'ID', 'CAS Number', 'Quality', 'Type', 'Width'], axis=1, inplace=True)
        df.rename({'Area': column_name}, axis=1, inplace=True)
        df.set_index('Compound', inplace=True)
        df.rename(index=synonyms, inplace=True)
        df = df.groupby(df.index).sum()
        output = pd.concat([output, df], axis=1)
        
    output.fillna(0.0, inplace=True)
    output = output.T
    output['Label'] = labels
    
    return output

# def train_test_split(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Randomly samples two records for each label to produce a training dataset.
    
#     Uses a deprecated feature, and would not likely work in the future. 
#     """
#     training_df = pd.DataFrame(columns=records.columns)
#     testing_df = pd.DataFrame(columns=records.columns)
#     grouped = records.groupby('Label')
#     for _, group in grouped:
#         sampled_rows = group.sample(n=2)
#         testing_df = pd.concat([testing_df, sampled_rows])
#         remaining_rows = group.drop(sampled_rows.index)
#         training_df = pd.concat([training_df, remaining_rows])
        
#     return (training_df, testing_df)

def standard_scaler(records: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalization on the dataframe. 
    """
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(records), index=records.index, columns=records.columns)
    return scaled_df

def bootstrap(records: pd.DataFrame, n_new:int=10) -> pd.DataFrame:
    output = []
    for group, group_data in records.groupby(level=0):
        group_size = len(group_data)
        bootstrap_samples = []
        for _ in range(n_new):
            bootstrap_indices = np.random.choice(group_size, size=group_size, replace=True)
            bootstrap_sample = group_data.iloc[bootstrap_indices]
            bootstrap_samples.append(bootstrap_sample)
            
        bootstrap_df = pd.concat(bootstrap_samples)
        output.append(bootstrap_df)
    
    return pd.concat(output)