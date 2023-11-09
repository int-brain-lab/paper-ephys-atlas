"""
Here we perform post-hoc filtering of the predicted classes of the Cosmos classifier using a HMM.
The transition matrix is computed from the atlas in the previous script, while the emission matrix
is the confusion matrix of the classifier.
"""
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.sparse
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember
from brainbox.ephys_plots import plot_brain_regions
import ephys_atlas.encoding
import ephys_atlas.plots
from ephys_atlas.decoding import viterbi

FOLDER_GDRIVE = Path("/datadisk/team_drives/Task Force - Electrophysiology Atlas/Decoding/HMM_Decoding")

features_list = ephys_atlas.encoding.voltage_features_set()

regions = BrainRegions()


def load_decoding(label='cosmos'):
    df_benchmarks = pd.read_parquet(FOLDER_GDRIVE.joinpath(f'{label}_predictions_benchmark.pqt'))
    classes = np.load(FOLDER_GDRIVE.joinpath(f'{label}_predictions_classes.npy'))
    confusion_matrix = np.load(FOLDER_GDRIVE.joinpath(f'{label}_confusion_matrix.npy'))

    df_depths = df_benchmarks.reset_index().groupby(['pid', 'axial_um']).agg(
        **{f'{label}_id': pd.NamedAgg(column=f'{label   }_id', aggfunc=pd.Series.mode)},
        ml = pd.NamedAgg(column='x', aggfunc='mean'),
        ap = pd.NamedAgg(column='y', aggfunc='mean'),
        dv = pd.NamedAgg(column='z', aggfunc='mean'),
        **{r: pd.NamedAgg(column=r, aggfunc='mean') for r in regions.id2acronym(classes)},
        **{r: pd.NamedAgg(column=r, aggfunc='mean') for r in features_list},
    )
    df_depths[f'{label}_prediction'] = classes[np.argmax(df_depths.loc[:, regions.id2acronym(classes)].values, axis=1)]
    return df_depths, classes, confusion_matrix


label = 'cosmos'
df_depths, classes, confusion_matrix = load_decoding(label)
npz_transitions = np.load(FOLDER_GDRIVE.joinpath(f"region_transition_{label}.npz"))

# (nr, nr)
transition_down = npz_transitions['region_transitions']
transition_up = npz_transitions['region_transitions'].T

df_depths['hmm_prediction'] = 0
classes_priors = npz_transitions['region_counts'].squeeze() / np.sum(npz_transitions['region_counts'])

itr, _ = ismember(npz_transitions['region_aids'], classes)
transition_down = transition_down[itr, :][:, itr]
transition_up = transition_up[itr, :][:, itr]
transition_up = transition_up / np.sum(transition_up, axis=1)[:, np.newaxis]
transition_down = transition_down / np.sum(transition_down, axis=1)[:, np.newaxis]
emission = confusion_matrix
np.testing.assert_equal(np.sum(itr), classes.size)


if label == 'cosmos':
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(emission * 100, vmin=0, vmax=50, cmap='Blues', annot=True, ax=ax, fmt='.1f')
    ax.set(
        xticklabels=regions.id2acronym(classes),
        yticklabels=regions.id2acronym(classes),
        xlabel='True region',
        ylabel='Predicted region',
        title='Emission Probabilities (%)'
    )
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(transition_down * 100, vmin=0, vmax=1.2, annot=True, fmt='.1f', ax=ax, cmap='Blues')
    ax.set(
        xticklabels=regions.id2acronym(classes),
        yticklabels=regions.id2acronym(classes),
        xlabel='Lower region',
        ylabel='Upper region',
        title='Transition Probabilities (%)'
    )

## %%
"""
We are going to use a HMM to denoise the coding
"""
pids = df_depths.index.get_level_values(0).unique()
pids = ['1e104bf4-7a24-4624-a5b2-c2c8289c0de7']
pids = ['dc7e9403-19f7-409f-9240-05ee57cb7aea']
method = 'two_ways'  # 'two_ways', 'one_way'
for pid in pids:
    # pid = np.random.choice(pids)  # fixme
    _, predictions = ismember(df_depths.loc[pid, f'{label}_prediction'].values, classes)
    nd, nr = (predictions.shape[0], classes.size)
    probas = df_depths.loc[pid, regions.id2acronym(classes)].values  # this is the (ndepths, nregions) matrix of probabilities

    # we need to remove the root and void regions from the predictions

    _, root_void = ismember(regions.acronym2id(['root', 'void']), classes)
    irv, _ = ismember(predictions, root_void)
    predictions = scipy.interpolate.interp1d(np.arange(predictions.size)[~irv], predictions[~irv], kind='nearest', fill_value='extrapolate')(np.arange(predictions.size)).astype(int)
    probas[:, root_void] = 0
    probas = probas / probas.sum(axis=1)[:, np.newaxis]
    emission[:, root_void] = 0
    emission = emission / emission.sum(axis=1)[:, np.newaxis]
    transition_down[:, root_void] = 0
    transition_down = transition_down / transition_down.sum(axis=1)[:, np.newaxis]
    transition_up[:, root_void] = 0
    transition_up = transition_up / transition_up.sum(axis=1)[:, np.newaxis]
    # probas / classes_priors
    # probas / classes_priors

    n_runs = 1000
    all_obs = np.zeros((nd, n_runs), dtype=int)
    all_vit = np.zeros((nd, n_runs))
    vprobs = np.zeros(n_runs)
    vit, _ = viterbi(emission, transition_down, classes_priors, predictions)

    for i in range(n_runs):
        # generates a random set of observed states according to the classifier output probabilities
        all_obs[:, i] = np.flipud(np.mod(np.searchsorted(probas.flatten().cumsum(), np.random.rand(nd) + np.arange(nd)), nr))
        # run the viterbi algorithm once on the predicted labels for reference
        match method:
            case 'one_way':
                init_hidden_probs = probas[0, :]
                cl, p = viterbi(emission, transition_down, init_hidden_probs, all_obs[:, i])
                vprobs[i] = p
                all_vit[:, i] = cl
            case 'two_ways':
                # we take a random depth and start from this point, upwards, and downwards
                dstart = np.random.randint(1, nd - 1)
                init_hidden_probs = probas[dstart, :]
                # we take a random set of observed states according to the classifier output probabilities
                # this is the downward part
                all_vit[dstart:nd, i], pdown = viterbi(emission, transition_down, init_hidden_probs, all_obs[dstart:nd, i])
                # this is the upward part, note that we transpose the transition matrix to reflect the shift in direction
                _vit, pup = viterbi(
                    emission, transition_up, init_hidden_probs, np.flipud(all_obs[0:dstart, i]))
                all_vit[0:dstart, i] = np.flipud(_vit)
                vprobs[i] = np.sqrt(pup * pdown)

    vcprobs = scipy.sparse.coo_array(
        (all_vit.flatten() * 0 + 1, (np.tile(np.arange(nd), (n_runs,)), all_vit.flatten())), shape=(nd, nr)).todense()
    vcprobs = np.cumsum(vcprobs / vcprobs.sum(axis=1)[:, np.newaxis], axis=1)

    all_vit = np.flipud(all_vit.astype(int))
    vit_pred, _ = scipy.stats.mode(all_vit, axis=-1)
    df_depths.loc[pid, 'hmm_stochastic'] = classes[vit_pred]
    df_depths.loc[pid, 'hmm'] = classes[vit]
    vprobs = vprobs / vprobs.sum()
    # accuracies = np.mean(df_depths.loc[pid, 'cosmos_id'].values[:, np.newaxis] == classes[all_vit], axis=0)
    # plt.plot(np.log(vprobs), accuracies, '.')
    # print(np.corrcoef(np.log(vprobs), accuracies))
    # plt.plot(vprobs)
    #% Display the results and compute accuracy
    hacc = sklearn.metrics.accuracy_score(df_depths.loc[pid, f'{label}_id'].values, classes[vit_pred])
    pacc = sklearn.metrics.accuracy_score(df_depths.loc[pid, f'{label}_id'].values, df_depths.loc[pid, f'{label}_prediction'].values)
    vacc = sklearn.metrics.accuracy_score(df_depths.loc[pid, f'{label}_id'].values, classes[vit])

    fig, axs = plt.subplots(1, 7, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1, 7, 1, 1, 1, 7]})
    depths = df_depths.loc[pid].index.values
    plot_brain_regions(df_depths.loc[pid][f'{label}_id'].values, channel_depths=df_depths.loc[pid].index.values,
                       brain_regions=regions, display=True, ax=axs[0], title='Real')
    plot_brain_regions(df_depths.loc[pid][f'{label}_prediction'], channel_depths=df_depths.loc[pid].index.values,
                       brain_regions=regions, display=True, ax=axs[1], title='Prediction')

    ephys_atlas.plots.plot_cumulative_probas(probas, depths, aids=classes, regions=regions, ax=axs[2], legend=False)
    axs[2].set(title=f"{pacc:.2f} accuracy")
    axs[2].yaxis.tick_right()
    plot_brain_regions(df_depths.loc[pid][f'{label}_id'].values, channel_depths=df_depths.loc[pid].index.values,
                       brain_regions=regions, display=True, ax=axs[3], title='Real')
    plot_brain_regions(classes[vit], channel_depths=df_depths.loc[pid].index.values,
                       brain_regions=regions, display=True, ax=axs[4], title='Vit')
    plot_brain_regions(classes[vit_pred], channel_depths=df_depths.loc[pid].index.values,
                       brain_regions=regions, display=True, ax=axs[5], title='HMM')


    rgbs = regions.rgb[ismember(np.sort(classes[all_vit.astype(int)], axis=1), regions.id)[1]].reshape(nd, -1, 3)
    #ephys_atlas.plots.plot_cumulative_probas(vcprobs, depths, aids=classes, regions=regions, ax=axs[4], legend=False)
    axs[6].imshow(rgbs, aspect='auto', origin='lower')
    axs[6].set(title=f"Viterbi:{vacc:.2f}, HMM stochastic {hacc:.2f} accuracy")
    fig.suptitle(f"{pid}")
    fig.savefig(FOLDER_GDRIVE.joinpath("pics", f"viterbi_{pid}_{method}.png"), dpi=100)
    # plt.close(fig)
# TODO plot raw data features


acc_hmm = np.mean(df_depths.loc[pid, f'{label}_id'].values == df_depths.loc[pid, 'hmm_stochastic'].values)
acc_vit = np.mean(df_depths.loc[pid, f'{label}_id'].values == df_depths.loc[pid, 'hmm'].values)
acc = np.mean(df_depths.loc[pid, f'{label}_id'].values == df_depths.loc[pid, f'{label}_prediction'].values)

print(acc, acc_hmm, acc_vit)

## %%
df_acc = df_depths.loc[:, ['hmm', 'hmm_stochastic', f'{label}_prediction']].apply(lambda x: x == df_depths.loc[:, f'{label}_id'].values)

df_acc_s = df_acc.groupby('pid').mean()
# df_acc_s = df_acc_s.apply(lambda x: x - df_acc_s['cosmos_prediction'])
df_acc_s = df_acc_s.stack().reset_index().rename(columns={'level_1': 'method', 0: 'accuracy'})
df_acc_s['pid'] = df_acc_s['pid'].apply(lambda x: x[:6])

fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(df_acc_s, x='accuracy', y='method', ax=ax)
fig.tight_layout()

## %%
## %%


df_acc_s = df_acc.groupby('pid').mean()
df_acc_s = df_acc_s.apply(lambda x: x - df_acc_s['cosmos_prediction']).reset_index()
df_acc_s['pid'] = df_acc_s['pid'].apply(lambda x: x[:6])

fig, ax = plt.subplots()
sns.lineplot(df_acc_s, x='pid', y='hmm', ax=ax, palette='pastel')
sns.lineplot(df_acc_s, x='pid', y='hmm_stochastic', ax=ax, palette='pastel')
plt.xticks(rotation=45)

plt.legend(['hmm', 'hmm_stochastic'])
plt.title('Accuracy difference with Cosmos prediction')
plt.tight_layout()