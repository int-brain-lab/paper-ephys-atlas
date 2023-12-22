## %% load data and splits in training and testing sets
from pathlib import Path
import numpy as np

import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions
from iblutil.util import Bunch

import ephys_atlas.data
import ephys_atlas.encoding
import ephys_atlas.decoding


regions = BrainRegions()
meta = Bunch(dict(
    RANDOM_SEED=713705,
    VINTAGE='2023_W51',
    REGION_MAP='Cosmos',
    FEATURES=None,
    MODEL_CLASS=None,
))

LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")

# FOLDER_GDRIVE = Path("/mnt/s0/ephys-atlas-decoding")
# FOLDER_GDRIVE = Path("/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
df_voltage, _, _, _ = ephys_atlas.data.load_voltage_features(LOCAL_DATA_PATH.joinpath('features', meta.VINTAGE))

FEATURE_SET = ['raw_ap', 'raw_lf', 'raw_lf_csd', 'localisation', 'waveforms']
TRAIN_LABEL = f'{meta.REGION_MAP.lower()}_id'  # ['beryl_id', 'cosmos_id', 'atlas_id']
x_list = meta.FEATURES = sorted(ephys_atlas.encoding.voltage_features_set(FEATURE_SET))


# smooth features
# import tqdm
# for pid in tqdm.tqdm(df_voltage.index.get_level_values(0).unique()):
#     df_voltage.loc[pid, x_list] = scipy.ndimage.median_filter(df_voltage.loc[pid, x_list], size=8, axes=0)
# here we split the training and testing sets making sure the benchmark insertions are part of the testing set
train_idx, test_idx = ephys_atlas.encoding.train_test_split_indices(df_voltage, include_benchmarks=True)
print(f"{df_voltage.shape[0]} channels", f'training set {np.sum(test_idx) / test_idx.size}')

# here we split the training and testing sets making sure the benchmark insertions are part of the testing set
# we do it 4 times and get a list of test indices to loop over for each fold
test_idx_folds = ephys_atlas.encoding.train_test_split_folds(df_voltage, include_benchmarks=True)

path_models = []
for ifold, test_idx in enumerate(test_idx_folds):
    train_idx = ~test_idx
    x_train = df_voltage.loc[train_idx, x_list].values
    x_test = df_voltage.loc[test_idx, x_list].values
    y_train = df_voltage.loc[train_idx, TRAIN_LABEL].values
    y_test = df_voltage.loc[test_idx, TRAIN_LABEL].values

    df_benchmarks = df_voltage.loc[ismember(df_voltage.index.get_level_values(0), ephys_atlas.data.BENCHMARK_PIDS)[0], :].copy()
    df_test = df_voltage.loc[test_idx, :].copy()

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
            df_test[f'cosmos_prediction'] = classes[classifier.predict(df_test.loc[:, x_list].values)]
            # # null model
            # accuracy = sklearn.metrics.accuracy_score(
            #     df_voltage.loc[test_idx, TRAIN_LABEL].values,
            #     regions.remap(df_voltage['atlas_id_target'], source_map='Allen', target_map='Cosmos')[test_idx]
            # )
        case 'beryl_id':
            # Random forest
            kwargs = {'n_estimators': 30, 'max_depth': 40, 'max_leaf_nodes': 10000,
                      'random_state': meta.RANDOM_SEED, 'n_jobs': -1, 'criterion': 'entropy'}
            classifier = RandomForestClassifier(verbose=True, **kwargs)
            classifier.fit(x_train, df_voltage.loc[train_idx, TRAIN_LABEL].values)
            y_pred = classifier.predict(x_test)
            df_benchmarks['beryl_prediction'] = classifier.predict(df_benchmarks.loc[:, x_list].values)
            df_test['beryl_prediction'] = classifier.predict(df_test.loc[:, x_list].values)
            # null model
            # accuracy = sklearn.metrics.accuracy_score(
            #     df_voltage.loc[test_idx, TRAIN_LABEL].values,
            #     regions.remap(df_voltage['atlas_id_target'], source_map='Allen', target_map='Beryl')[test_idx]
            # )


    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='true')  # row: true, col: predicted
    print(f"Accuracy: {accuracy}")

    classes = np.unique(np.r_[y_test, y_pred])
    benchmark_probas = classifier.predict_proba(df_benchmarks.loc[:, x_list].values)
    test_probas = classifier.predict_proba(df_test.loc[:, x_list].values)
    for i, col in enumerate(classes):
        df_benchmarks[col] = benchmark_probas[:, i]
        df_test[col] = test_probas[:, i]

    # write model to disk
    path_model = ephys_atlas.decoding.save_model(
        LOCAL_DATA_PATH.joinpath('models'), classifier, meta, subfolder=f"FOLD{ifold:02d}")
    path_models.append(path_model)
    # add the testing sets (benchark and full testing)
    df_test.to_parquet(path_model.joinpath('predictions_test.pqt'))
    df_benchmarks.to_parquet(path_model.joinpath('predictions_benchmark.pqt'))
    np.save(path_model.joinpath(f'predictions_classes.npy'), classes)
    np.save(path_model.joinpath(f'confusion_matrix.npy'), confusion_matrix)


## output the complete set of predictions
import pandas as pd

df_predictions = []
for path_model in path_models:
    df_predictions.append(pd.read_parquet(path_model.joinpath('predictions_test.pqt')))

df_predictions = pd.concat(df_predictions)
columns = df_predictions.columns.difference(df_voltage.columns)
df_voltage = df_voltage.merge(df_predictions.loc[:, columns], left_index=True, right_index=True, how='left')

df_voltage.to_parquet(path_model.parent.joinpath('voltage_predictions.pqt'))