## %% load data and splits in training and testing sets
from pathlib import Path
import numpy as np

import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

import ephys_atlas.data
import ephys_atlas.encoding
from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions


regions = BrainRegions()
BENCHMARK = False

SCALE = False
label = '2023_W41'
RANDOM_SEED = 713705
LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
# FOLDER_GDRIVE = Path("/mnt/s0/ephys-atlas-decoding")
# FOLDER_GDRIVE = Path("/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")


df_voltage, _, _, _ = ephys_atlas.data.load_voltage_features(LOCAL_DATA_PATH.joinpath(label))
train_idx, test_idx = ephys_atlas.encoding.train_test_split_indices(df_voltage, include_benchmarks=True)
print(f"{df_voltage.shape[0]} channels", f'training set {np.sum(test_idx) / test_idx.size}')


FEATURE_SET = ['raw_ap', 'raw_lf', 'localisation', 'waveforms']
TRAIN_LABEL = 'cosmos_id'  # ['beryl_id', 'cosmos_id', 'atlas_id']


df_voltage_old = df_voltage.copy()
x_list = ephys_atlas.encoding.voltage_features_set(FEATURE_SET)

# import tqdm
# for pid in tqdm.tqdm(df_voltage.index.get_level_values(0).unique()):
#     df_voltage.loc[pid, x_list] = scipy.ndimage.median_filter(df_voltage.loc[pid, x_list], size=8, axes=0)

x_train = df_voltage.loc[train_idx, x_list].values
x_test = df_voltage.loc[test_idx, x_list].values
y_train = df_voltage.loc[train_idx, TRAIN_LABEL].values
y_test = df_voltage.loc[test_idx, TRAIN_LABEL].values

df_benchmarks = df_voltage.loc[ismember(df_voltage.index.get_level_values(0), ephys_atlas.data.BENCHMARK_PIDS)[0], :].copy()

classes = np.unique(df_voltage.loc[train_idx, TRAIN_LABEL])

match TRAIN_LABEL:
    case 'cosmos_id': # Gradient boost
        # this only works
        _, iy_train = ismember(y_train, classes)
        _, iy_test = ismember(y_test, classes)
        # create model instance
        classifier = XGBClassifier(device='gpu', verbosity=2)
        # fit model
        classifier.fit(x_train, iy_train)
        # make predictions
        y_pred = classes[classifier.predict(x_test)]
        df_benchmarks[f'cosmos_prediction'] = classes[classifier.predict(df_benchmarks.loc[:, x_list].values)]
        # # null model
        # accuracy = sklearn.metrics.accuracy_score(
        #     df_voltage.loc[test_idx, TRAIN_LABEL].values,
        #     regions.remap(df_voltage['atlas_id_target'], source_map='Allen', target_map='Cosmos')[test_idx]
        # )


    case 'beryl_id':
        # Random forest
        kwargs = {'n_estimators': 30, 'max_depth': 40, 'max_leaf_nodes': 10000,
                  'random_state': RANDOM_SEED, 'n_jobs': -1, 'criterion': 'entropy'}
        classifier = RandomForestClassifier(verbose=True, **kwargs)
        classifier.fit(x_train, df_voltage.loc[train_idx, TRAIN_LABEL].values)
        y_pred = classifier.predict(x_test)
        df_benchmarks['beryl_prediction'] = classifier.predict(df_benchmarks.loc[:, x_list].values)
        # null model
        # accuracy = sklearn.metrics.accuracy_score(
        #     df_voltage.loc[test_idx, TRAIN_LABEL].values,
        #     regions.remap(df_voltage['atlas_id_target'], source_map='Allen', target_map='Beryl')[test_idx]
        # )


accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
classes = np.unique(np.r_[y_test, y_pred])
confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='true')  # row: true, col: predicted
print(f"Accuracy: {accuracy}")

probas = classifier.predict_proba(df_benchmarks.loc[:, x_list].values)
for i, col in enumerate([label for label in regions.id2acronym(classes)]):
    df_benchmarks[col] = probas[:, i]

df_benchmarks.to_parquet(LOCAL_DATA_PATH.joinpath(f'{TRAIN_LABEL[:-3]}_predictions_benchmark.pqt'))
np.save(LOCAL_DATA_PATH.joinpath(f'{TRAIN_LABEL[:-3]}_predictions_classes.npy'), classes)
np.save(LOCAL_DATA_PATH.joinpath(f'{TRAIN_LABEL[:-3]}_confusion_matrix.npy'), confusion_matrix)









