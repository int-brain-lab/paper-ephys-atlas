"""
Han wants to combine model outputs to boost performance.
Here we retain the same probes as her testing set for testing and train the model on the remaining probes
"""

from pathlib import Path
import numpy as np

import sklearn.metrics
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

from iblutil.numerical import ismember
from iblutil.util import Bunch

import ephys_atlas.data
import ephys_atlas.encoding
import ephys_atlas.decoding
import ephys_atlas.anatomy

# parede
LOCAL_DATA_PATH = Path('/mnt/s0/ephys-atlas-decoding')
NEMO_PATH = LOCAL_DATA_PATH

# this list contains the test probes that Han used originally in the "IBL_BWM_train_val_test_split.npz"
test_pids = ephys_atlas.data.BENCHMARK_PIDS

VOID_ROOT_HANDLING = 'inflate'  # 'keep'
REMOVE_VOID_ROOT = False

meta = Bunch(dict(
    RANDOM_SEED=42670,
    VINTAGE='2024_W04',
    REGION_MAP='Cosmos',
    FEATURES=None,
    MODEL_CLASS=None,
    CLASSES=None,
))


# %% load data and splits in training and testing sets
try:
    df_voltage, _, _, df_probes = ephys_atlas.data.load_voltage_features(LOCAL_DATA_PATH.joinpath('features', meta.VINTAGE))
except FileNotFoundError:
    from one.api import ONE
    one = ONE(mode='remote')
    df_voltage, _, _, df_probes = ephys_atlas.data.download_tables(LOCAL_DATA_PATH.joinpath('features'), meta.VINTAGE, one=one)

np.sum(~np.isin(test_pids, df_probes.index))  # hmm we have 3 probes from the BWM that are not in the voltage features

# Accuracy: 0.5509311541383488
brain_atlas = ephys_atlas.anatomy.EncodingAtlas()  # Accuracy: 0.5459850465017324 - this atlas splits void and void fluid
# brain_atlas = ephys_atlas.anatomy.AllenAtlas()  # Accuracy: 0.5536619920744102
aids = brain_atlas.get_labels(df_voltage.loc[:, ['x', 'y', 'z']].values, mode='clip')
df_voltage['Allen_id'] = aids
df_voltage['Cosmos_id'] = brain_atlas.regions.remap(aids, 'Allen', 'Cosmos')
regions = brain_atlas.regions

# here we are reassigning the channels found in void fluid and root to their nearest region
if VOID_ROOT_HANDLING == 'inflate':
    aids_replaced = np.array([2000, 997])
    _, rids_replaced = ismember(np.r_[aids_replaced, -aids_replaced ], regions.id)
    cosmos_volume = regions.mappings['Cosmos'][brain_atlas.label]
    i2replace = df_voltage['Cosmos_id'].isin(aids_replaced).values  # we are going to get the nearest region of those root and void inside brain
    i_volume_flat = brain_atlas._lookup(df_voltage.loc[i2replace, ['x', 'y', 'z']].values, mode='clip')
    i_volume = np.c_[np.unravel_index(i_volume_flat, cosmos_volume.shape)]
    around = np.c_[[x.flatten().astype(int) for x in np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])]]
    n, nearest_rid, isearch = (0, np.zeros(np.sum(i2replace)), np.ones(np.sum(i2replace), dtype=bool))
    while any(isearch):
        iclose = i_volume[isearch, :, np.newaxis] + around[np.newaxis, :, :] * n
        neighbours = np.take(cosmos_volume, np.ravel_multi_index([iclose[:, i, :] for i in np.arange(3)], dims=cosmos_volume.shape, mode='clip'))
        neighbours_mask = ~np.isin(neighbours, rids_replaced)
        ifound = np.any(neighbours_mask, axis=1)
        nearest_rid[np.where(isearch)[0][ifound]] = neighbours[ifound, np.argmax(neighbours_mask[ifound, :], axis=1)]
        isearch[np.where(isearch)[0][ifound]] = False
        print(n, np.sum(isearch))
        n += 1
    df_voltage['Cosmos_id_save'] = df_voltage['Cosmos_id'].copy()
    df_voltage.loc[i2replace, 'Cosmos_id'] = regions.id[nearest_rid.astype(int)]

# df_voltage = df_voltage.loc[~np.isin(df_voltage['Cosmos_id'], [0]), :]
FEATURE_SET = ['raw_ap', 'raw_lf', 'raw_lf_csd', 'localisation', 'waveforms']
TRAIN_LABEL = f'{meta.REGION_MAP}_id'  # ['beryl_id', 'cosmos_id', 'atlas_id']
x_list = meta.FEATURES = sorted(ephys_atlas.encoding.voltage_features_set(FEATURE_SET))
test_idx = np.isin(df_voltage.index.get_level_values(0), test_pids)
train_idx = ~test_idx

# look at the class balancing
# df_voltage.loc[train_idx, 'Cosmos_id_new'].value_counts()
# df_voltage.loc[test_idx, TRAIN_LABEL].value_counts()

print(f"{df_voltage.shape[0]} channels", f'training set {np.sum(test_idx) / test_idx.size}')

df_voltage.loc[train_idx, :].groupby(TRAIN_LABEL).count()
x_train = df_voltage.loc[train_idx, x_list].values
x_test = df_voltage.loc[test_idx, x_list].values
y_train = df_voltage.loc[train_idx, TRAIN_LABEL].values
y_test = df_voltage.loc[test_idx, TRAIN_LABEL].values
df_benchmarks = df_voltage.loc[ismember(df_voltage.index.get_level_values(0), ephys_atlas.data.BENCHMARK_PIDS)[0], :].copy()
df_test = df_voltage.loc[test_idx, :].copy()
classes = np.unique(df_voltage.loc[train_idx, TRAIN_LABEL])

_, iy_train = ismember(y_train, classes)
_, iy_test = ismember(y_test, classes)
# 0.5376271321378102
#  create model instance
classifier = XGBClassifier(device='gpu', verbosity=2)
# fit model
classifier.fit(x_train, iy_train)
# make predictions
y_pred = classes[classifier.predict(x_test)]
df_test[f'cosmos_prediction'] = classes[classifier.predict(df_test.loc[:, x_list].values)]
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='true')  # row: true, col: predicted
print(f"Accuracy: {accuracy}")
# write model to disk
meta.CLASSES = list(regions.id2acronym(classes))
meta.accuracy = float(accuracy)
path_model = ephys_atlas.decoding.save_model(NEMO_PATH, classifier, meta)
model_hash = path_model.name[-8:]
print(f"Model saved to {path_model}")

# add the predicted classes to the dataframe and write to disk
df_voltage['test_set'] = np.isin(df_voltage.index.get_level_values(0), test_pids)
classes = regions.acronym2id(meta['CLASSES'])
iclass_predicted = classifier.predict(df_voltage.loc[:, meta['FEATURES']].values)
df_voltage[f"{meta['REGION_MAP']}_prediction"] = classes[iclass_predicted]
y_probas = classifier.predict_proba(df_voltage.loc[:, meta['FEATURES']].values)

for i, col in enumerate(classes):
    df_voltage[col] = y_probas[:, i]

# add the testing sets (benchark and full testing)
df_voltage.to_parquet(NEMO_PATH.joinpath(
    f'voltage_features_with_predictions_{meta["VINTAGE"]}_{meta["REGION_MAP"]}_{model_hash}.pqt'))
