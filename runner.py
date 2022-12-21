import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectKBest
from lightgbm import LGBMClassifier
import json

SEED = 42

def _baseline_evaluation(df, label_column):
    X_train, X_test, y_train, y_test = _train_test_split(df, label_column)
    cls = ExtraTreeClassifier(random_state=SEED)
    cls.fit(X_train, y_train)
    print(f"Shape: {df.shape} => Training Score: {cls.score(X_train, y_train)}")
    print(f"Shape: {df.shape} => Test Score    : {cls.score(X_test, y_test)}\n")

def _drop_less_relevant_columns(df, label_column, threshold=0):
    constant_filter = VarianceThreshold(threshold=threshold).fit(df.drop(columns=[label_column]))
    support_columns = df.drop(columns=[label_column]).columns[constant_filter.get_support()]
    non_constant_columns = [col for col in df.columns if col not in support_columns]
    print(f'\nNon (quasi-)constant columns: {non_constant_columns}')
    constant_columns = [col for col in df.columns if col in support_columns]
    print(f'\n(Quasi-)constant columns: {constant_columns}')
    return df.drop(columns=constant_columns)

def _plot_feature_importances(df, label_column, base_folder):
    # get importances from a tree-based classifier
    X, y = df.drop(columns=[label_column]), df[label_column]
    cls = ExtraTreeClassifier(random_state=SEED)
    cls.fit(X, y)
    importances = cls.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    # plot importances for each feature
    plt.figure(figsize=(32,24))
    plt.rcParams['font.size'] = '16'
    plt.clf()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(os.path.join(base_folder, f'feature_importances_{len(features)}.png'))

def _select_top_k_features(df, label_column, k=10):
    X, y = df.drop(columns=[label_column]).select_dtypes(include='number'), df[label_column]
    kbest = SelectKBest(k=k)
    X_new = kbest.fit_transform(X, y)
    feature_mask = X.columns[kbest.get_support()]
    relevant_columns = [col for col in X.columns if col in feature_mask]
    print(f'\nFeatures that will be kept: {relevant_columns}')
    irrelevant_cols = [col for col in X.columns if col not in feature_mask]
    print(f'\nFeatures that will be dropped: {irrelevant_cols}')
    return df.drop(columns=irrelevant_cols)

def _select_relevant_features(df, label_column, n_folds=5):
    print(f'\nPerforming {n_folds}-fold recursive feature elimination:')
    X, y = df.drop(columns=[label_column]).select_dtypes(include='number'), df[label_column]
    rfecv = RFECV(estimator=ExtraTreeClassifier(random_state=SEED), cv=n_folds, verbose=1)
    rfecv.fit(X, y)
    feature_mask = X.columns[rfecv.get_support()]
    relevant_columns = [col for col in X.columns if col not in feature_mask]
    print(f'\nFeatures that will be kept: {relevant_columns}')
    irrelevant_cols = [col for col in X.columns if col in feature_mask]
    print(f'\nFeatures that will be dropped: {irrelevant_cols}')
    return df.drop(columns=irrelevant_cols)

def _train_test_split(df, label_column, test_size=0.2):
    X, y = df.drop(columns=[label_column]), df[label_column]
    return train_test_split(X, y, test_size=test_size, random_state=SEED)

if __name__ == '__main__':

    # Leitura dos CSVs
    # data_frames = []
    work_folder = os.path.join(os.path.dirname(__file__), './datasus-sim/Data/CID10/DORES/')
    # for filename in os.listdir(work_folder):
    #     if filename.endswith('.csv'):
    #         current_df = pd.read_csv(os.path.join(work_folder, filename))
    #         data_frames.append(current_df)
    # df = pd.concat(data_frames)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), './datasus-sim/Data/filtered_DORES.csv'))
    # Coluna que indica o CID
    coluna_target = 'CAUSABAS'

    # Substituicao de valores no CID9
    # df['FILHVIVOS'] = df['FILHVIVOS'].replace({np.nan: -1.0, 'XX': -1.0})
    # df['FILHMORT'] = df['FILHMORT'].replace({np.nan: -1.0, 'XX': -1.0})
    # df = df.astype({'FILHVIVOS': 'int', 'FILHMORT': 'int', coluna_target: 'string'})
    df = df.fillna(-1.0).infer_objects()

    # Colunas antes da remocao manual das colunas irrelevantes
    print(df.columns.values)
    irrelevant_columns = ['CONTADOR', 'DTOBITO', 'HORAOBITO','LOCOCOR','CODESTAB', 'LINHAA','LINHAB','LINHAD','LINHAC','ATESTANTE']
    df = df.drop(columns=irrelevant_columns)
    # Colunas depois da remocao manual das colunas irrelevantes
    print("Colunas depois da remocao manual das colunas irrelevantes")
    print(df.columns.values)

    # Leitura do arquivo de CIDs relevantes
    print(df.shape)
    with open('cids.json') as f:
        relevant_cids = list(json.load(f).keys())

    # Contagens antes do mapeamento dos CIDS
    print(df[coluna_target].value_counts(normalize=True))
    # Mapeamento (cid relevante -> 'sim'; irrelevante -> 'nao')
    for cid in set(df[coluna_target].values):
        df[coluna_target] = df[coluna_target].replace(
            {cid: 'sim' if cid in relevant_cids else 'nao'})
    # Contagens antes do mapeamento dos CIDS
    print(df[coluna_target].value_counts(normalize=True))

    # Fatorizacao (transformacao de string para numeros fatorizados)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Series(pd.factorize(df[col])[0])

    # Remocao de colunas com pouca (ou nenhuma) variancia
    # df = _drop_less_relevant_columns(df, coluna_target, 1.00)

    # Importancia das caracteristicas antes da selecao
    print(f"Performing baseline evaluation and plotting feature importances before feature selection:")
    _baseline_evaluation(df, coluna_target)
    _plot_feature_importances(df, coluna_target, work_folder)

    # Selecao recursiva de caracteristicas
    # df = _select_relevant_features(df, coluna_target, 10)
    # Selecao das k melhores caracteristicas
    df = _select_top_k_features(df, coluna_target, 20)

    # Importancia das caracteristicas depois da selecao
    print(f"Performing baseline evaluation and plotting feature importances before feature selection:")
    _baseline_evaluation(df, coluna_target)
    _plot_feature_importances(df, coluna_target, work_folder)
