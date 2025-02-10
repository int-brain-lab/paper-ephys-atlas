import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables, download_tables
from ephys_atlas.plots import plot_kde, plot_similarity_matrix
from ephys_atlas.feature_information import feature_overall_entropy
from ephys_atlas.encoding import FEATURES_LIST
from ephys_atlas.features import voltage_features_set
from ephys_atlas.plots import color_map_feature
import scipy
from one.api import ONE

one = ONE()
br = BrainRegions()

label = "2023_W51"  # label = '2023_W51_autism'
mapping = "Cosmos"
local_data_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Data")
local_fig_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Anaysis_Entropy2024")
if not local_fig_path.exists():
    local_fig_path.mkdir()

force_download = False

local_data_path_clusters = local_data_path.joinpath(label).joinpath("clusters.pqt")
if not local_data_path_clusters.exists() or force_download:
    print("Downloading table")
    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="local")
    df_voltage, df_clusters, df_channels, df_probes = download_tables(
        label=label, local_path=local_data_path, one=one
    )
else:
    df_voltage, df_clusters, df_channels, df_probes = load_tables(
        local_data_path.joinpath(label), verify=True
    )

df_voltage = pd.merge(
    df_voltage, df_channels, left_index=True, right_index=True
).dropna()
df_voltage["pids"] = df_voltage.index.get_level_values(0)

df_voltage = df_voltage.rename(
    columns={"atlas_id": "Allen_id", "acronym": "Allen_acronym"}
)

df_voltage[mapping + "_id"] = br.remap(
    df_voltage["Allen_id"], source_map="Allen", target_map=mapping
)
df_voltage[mapping + "_acronym"] = br.id2acronym(df_voltage[mapping + "_id"])
# Remove void / root
df_voltage.drop(
    df_voltage[df_voltage[mapping + "_acronym"].isin(["void", "root"])].index,
    inplace=True,
)

features = voltage_features_set()

##
# Compute information gain per feature (overall) by pair of brain regions

dict_feat = dict()  # Keys will be the features, containing the dataframes

for feature in features:
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, 600)[1:])
    quantiles = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(
        df_voltage,
        values=feature,
        index=mapping + "_id",
        columns=quantiles,
        aggfunc="count",
    )

    # Create a dataframe of Nregion x Nregion that will contain the entropy computed for a pair of region
    df_entropy = pd.DataFrame(index=counts.index, columns=counts.index)
    # df_entropy = pd.DataFrame(index=counts.index, columns=['reg1', 'reg2'])
    # Divide the counts into 2 regions
    for ireg1, reg1 in enumerate(counts.index):
        for reg2 in counts.index[ireg1 + 1 :]:
            counts_reg = counts.loc[[reg1, reg2], :]
            information_gain = feature_overall_entropy(counts_reg)

            # Save the result in both place in the DF
            df_entropy.at[reg1, reg2] = information_gain
            df_entropy.at[reg2, reg1] = information_gain
            # df_entropy['feature'] = feature

    dict_feat[feature] = df_entropy
##
"""
# Plot the information gain  / matrix per feature
"""
# Create multi-index dataframe (AXIS = 1)
df_multi = pd.concat(dict_feat.values(), axis=1, keys=dict_feat.keys())
# Replace Nans with zeros
df_multi.fillna(0, inplace=True)
##
# Display the matrix for a given feature
for feature in features:
    # Init figure
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches([13.51, 3.14])

    # KDE
    plot_kde(feature, df_voltage, ax=axs[0], brain_id=mapping + "_id")

    # Matrix
    plot_similarity_matrix(
        mat_plot=df_multi[feature].to_numpy(),
        regions=df_multi[feature].index,
        fig=fig,
        ax=axs[1],
    )

    # SUM of information
    df_info = df_multi[feature].sum().to_frame(name="Sum info")

    df_info[mapping + "_id"] = df_info.index.get_level_values(0)
    df_info[mapping + "_acronym"] = br.id2acronym(df_info[mapping + "_id"])

    sns.barplot(
        data=df_info, x=mapping + "_acronym", y="Sum info", color="b", ax=axs[2]
    )
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=90)
    axs[2].set_title(f'Overall sum: {df_info["Sum info"].sum()}')

    # Save
    plt.savefig(
        local_fig_path.joinpath(f"entropy_sim__{label}_{mapping}__{feature}.png")
    )
    plt.close()

##
"""
# Plot histogram of information gain per feature, color coded by feature type
"""
# Create multi-index dataframe (AXIS = 0)
df_multi = pd.concat(dict_feat.values(), axis=0, keys=dict_feat.keys())
# Replace Nans with zeros
df_multi.fillna(0, inplace=True)

information_gain = df_multi.groupby(level=[0]).sum()
##
fig, axs = plt.subplots(1, 2)
fig.set_size_inches([9.61, 4.81])

# Set tick labels as brain region acronyms
ax = axs[0]
sns.heatmap(information_gain, ax=ax)
regions = information_gain.index
regions_ac = br.id2acronym(regions)
ax.set_yticks(np.arange(regions.size))
ax.set_yticklabels(regions_ac)

##
info_feature = (
    information_gain.sum(axis=1)
    .to_frame(name="information_gain")
    .sort_values("information_gain", ascending=False)
    .reset_index()
)

# Add new column "color"
color_set = color_map_feature()
info_feature["color"] = 0  # init new column
for feature_i, color_i in zip(FEATURES_LIST, color_set):
    feat_list = voltage_features_set(feature_i)
    indx = info_feature["index"].isin(feat_list)
    info_feature["color"][indx] = color_i

# Plot
ax = axs[1]
sns.barplot(
    info_feature, y="information_gain", x="index", palette=info_feature["color"]
)
plt.xticks(rotation=90)
fig.tight_layout()

plt.savefig(local_fig_path.joinpath(f"entropy_sim__{label}_{mapping}__OVERALL.png"))
plt.close()

##
"""
# Plot correlation coefficient between matrix of pair of features
"""
# Create multi-index dataframe (AXIS = 1)
df_multi = pd.concat(dict_feat.values(), axis=1, keys=dict_feat.keys())
# Replace Nans with zeros
df_multi.fillna(0, inplace=True)
##
# Create a dataframe of Nfeature x Nfeature that will contain the correlation coefficient
df_corr = pd.DataFrame(index=features, columns=features)
df_corr_pass = pd.DataFrame(index=features, columns=features)
pval = 0.0000001
for ifet1, fet1 in enumerate(features):
    mat1 = df_multi[fet1].to_numpy()
    for fet2 in features[ifet1 + 1 :]:
        mat2 = df_multi[fet2].to_numpy()
        r, c = np.triu_indices(mat1.shape[0], 1)
        vec1 = mat1[r, c]
        vec2 = mat2[r, c]
        # Compute correlation coefficient only on top diagonal elements
        corr_coeff = scipy.stats.pearsonr(vec1, vec2)

        # Save the result in both place in the DF
        df_corr.at[fet1, fet2] = corr_coeff
        df_corr.at[fet2, fet1] = corr_coeff

        df_corr_pass.at[fet1, fet2] = corr_coeff.pvalue < pval
        df_corr_pass.at[fet2, fet1] = corr_coeff.pvalue < pval
##
# Fill nan val with 0
df_corr_pass.fillna(0, inplace=True)
df_corr_pass[df_corr_pass.columns] = df_corr_pass[df_corr_pass.columns].astype(int)
# Plot
fig, axs = plt.subplots(1, 2)
fig.set_size_inches([9.61, 4.81])

# Set tick labels as brain region acronyms
ax = axs[0]
sns.heatmap(df_corr_pass, ax=ax)
fig.tight_layout()

##
# Clustering -- does not work because of kernel
"""
data = df_corr_pass.copy().to_numpy()

from sklearn.cluster import SpectralCoclustering
n_clu = 2
model = SpectralCoclustering(n_clusters=n_clu, random_state=0)
model.fit(data)

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title(f"Clustered with {n_clu} clusters")

plt.show()

plt.imshow(fit_data)
plt.colorbar()
plt.show()
"""
