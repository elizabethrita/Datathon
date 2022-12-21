import os
import pandas as pd
import json
import numpy as np


if __name__ == '__main__':
    data_frames = []
    work_folder = os.path.join(os.path.dirname(__file__), './datasus-sim/Data/CID10/DORES/')
    for filename in os.listdir(work_folder):
        if filename.endswith('.csv'):
            current_df = pd.read_csv(os.path.join(work_folder, filename))
            data_frames.append(current_df)
    df = pd.concat(data_frames)

    coluna_target = 'CODMUNRES'
    df = df.fillna(-1.0).infer_objects()

    with open('cidades.json') as f:
        relevant_cities = list(json.load(f).keys())

    relevant_cities = np.array(relevant_cities).astype('int64')
    df_filtered = df[df[coluna_target].isin(relevant_cities)]

    df_filtered.to_csv('./datasus-sim/Data/filtered_DORES.csv',index=False)