import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class Basquehourly(BaseDataset):
    """Data set class for the URA data set by [#]_ and [#]_.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(Basquehourly, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # load file for basin
        filename = self.cfg.data_dir / 'hourly' / f'{basin}.txt'
        with open(filename, 'rt') as f:
            df = pd.read_csv(f)

        converted_date = [_convert_datetime(date) for date in df['date']]
        df['date'] = converted_date
        df = df.set_index("date")

        # replace invalid discharge values by NaNs
        df.loc[df['streamflowmean'] < 0, 'streamflowmean'] = np.nan
        df.loc[df['levelmean'] < 0, 'levelmean'] = np.nan    
        df.loc[df['streamflowinst'] < 0, 'streamflowinst'] = np.nan    
        df.loc[df['levelinst'] < 0, 'levelinst'] = np.nan    

        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_URA_attributes(self.cfg.data_dir, basins=self.basins)


def load_URA_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load URA attributes from the dataset provided by [#]_

    Parameters
    ----------
    data_dir : Path
        Path to the URA directory. This folder must contain a 'ura_attributes_v2.0' folder (the original 
        data set) containing the corresponding txt files for each attribute group.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pandas.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.

    References
    ----------
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The URA data set: catchment attributes and 
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    """
    attributes_path = Path(data_dir) / 'URA_attributes_v1.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('URA_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings
#    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
#    df = df.drop('huc_02', axis=1)

#    if basins:
#        if any(b not in df.index for b in basins):
 #           raise ValueError('Some basins are missing static attributes.')
  #      df = df.loc[basins]

    return df

def _convert_datetime(date: str) -> str:
    #datestr = date.split(' ')[0]
    #return pd.to_datetime(datetime.datetime.strptime(datestr, '%m/%d/%Y').strftime('%Y-%m-%d'))
    return pd.to_datetime(date)
