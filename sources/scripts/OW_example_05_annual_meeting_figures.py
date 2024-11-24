from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from iblatlas.atlas import AllenAtlas
from iblatlas.atlas.flatmaps import plot_swanson, annotate_swanson
import matplotlib.pyplot as plt

cmap = "Blues"
ba = AllenAtlas()

STAGING_PATH = Path("/datadisk/FlatIron/tables/bwm_sav")
FIGURES_PATH = Path("/datadisk/gdrive/2022/ephys-atlas")

STAGING_PATH = Path("/Volumes/GoogleDrive/My Drive/2022/05_tables/bwm_sav")
FIGURES_PATH = Path("/Volumes/GoogleDrive/My Drive/2022/ephys-atlas")


df_clusters = pd.read_parquet(STAGING_PATH.joinpath("clusters.pqt"))
df_probes = pd.read_parquet(STAGING_PATH.joinpath("probes.pqt"))
df_channels = pd.read_parquet(STAGING_PATH.joinpath("channels.pqt"))

df_voltage = pd.read_parquet(STAGING_PATH.joinpath("channels_voltage_features.pqt"))
df_voltage["rms_ap_db"] = 20 * np.log10(df_voltage["rms_ap"])
df_voltage["rms_lf_db"] = 20 * np.log10(df_voltage["rms_lf"])


if "atlas_id" not in df_clusters.keys():
    df_clusters = df_clusters.merge(
        df_channels[["atlas_id", "acronym"]],
        right_on=["pid", "raw_ind"],
        left_on=["pid", "channels"],
    )


# %% Group by
df_voltage["n_insertions"] = df_voltage.index.get_level_values(0)
dfv_regions = df_voltage.groupby("atlas_id").agg(
    {
        "psd_delta": "median",
        "psd_theta": "median",
        "psd_alpha": "median",
        "psd_beta": "median",
        "psd_gamma": "median",
        "rms_ap_db": "median",
        "rms_lf_db": "median",
        "spike_rate": "median",
        "x": "count",
        "n_insertions": "nunique",
    }
)
dfv_regions["channels_logcount"] = np.log10(np.maximum(1, dfv_regions["x"]))

dfc_regions = df_clusters.groupby("atlas_id").agg(
    {
        "amp_median": "median",
        "contamination": "median",
        "firing_rate": "median",
        "cluster_id": "count",
    }
)
dfc_regions["amp_median_mv"] = dfc_regions["amp_median"] * 1e6
dfc_regions[dfc_regions["contamination"] > 1] = 0
dfc_regions["clusters_logcount"] = np.log10(np.maximum(1, dfc_regions["cluster_id"]))


df_regions = pd.merge(dfc_regions, dfv_regions, left_index=True, right_index=True)


df_regions = df_regions.drop(np.where(dfc_regions.index == 0)[0])


df_regions["fr_ratio"] = df_regions["spike_rate"] / df_regions["firing_rate"]

df_regions.loc[~np.isfinite(df_regions["fr_ratio"]), "fr_ratio"] = 1
df_regions["fr_ratio_log"] = np.log10(df_regions["fr_ratio"])
df_regions.loc[~np.isfinite(df_regions["fr_ratio_log"]), "fr_ratio_log"] = 0
df_regions["fr_ratio_dos"] = (df_regions["spike_rate"] - df_regions["firing_rate"]) / (
    df_regions["spike_rate"] + df_regions["firing_rate"]
)
df_regions["fr_diff"] = df_regions["spike_rate"] - df_regions["firing_rate"]
df_regions["coverage"] = np.minimum(df_regions["n_insertions"], 2)
# %% Display for the last part
cmap_div = "PuOr"
feats = {
    "psd_delta": dict(cmap=cmap),
    "psd_theta": dict(cmap=cmap),
    "psd_alpha": dict(cmap=cmap),
    "psd_beta": dict(cmap=cmap),
    "psd_gamma": dict(cmap=cmap),
    "rms_ap_db": dict(cmap=cmap),
    "rms_lf_db": dict(cmap=cmap),
    "spike_rate": dict(cmap=cmap, vmin=0, vmax=20),
    "channels_logcount": dict(cmap=cmap, vmin=0, vmax=4),
    "amp_median_mv": dict(cmap=cmap),
    "contamination": dict(cmap=cmap, vmin=0, vmax=1),
    "firing_rate": dict(cmap=cmap, vmin=0, vmax=20),
    "clusters_logcount": dict(cmap=cmap, vmin=0, vmax=4),
    "fr_ratio": dict(cmap=cmap, vmin=0, vmax=10),
    "fr_ratio_log": dict(cmap=cmap_div, vmin=-1.5, vmax=1.5),
    "n_insertions": dict(cmap=cmap, vmin=0, vmax=20),
    "fr_ratio_dos": dict(cmap=cmap_div),
    "fr_diff": dict(cmap=cmap_div, vmin=-30, vmax=30),
    "coverage": dict(cmap="viridis"),
}


BRAIN_REGIONS_RE = ["PP", "CA1", "DG", "LP", "PO"]
re_regions = ba.regions.descendants(ba.regions.acronym2id(BRAIN_REGIONS_RE))["id"]
re_sel = np.isin(df_regions.index, re_regions)
# not_re_regions = np.setxor1d(ba.regions.id, re_regions)
# i2drop = np.isin(ba.regions.remap(df_regions.index, 'Swanson'), (ba.regions.acronym2id(['root', 'void'])))
# df_regions = df_regions.drop(df_regions.index[i2drop])

for i, k in enumerate(feats):
    kwargs = feats[k]
    if FIGURES_PATH.joinpath(f"{k}_.png").exists():
        continue
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.tight_layout()
    plot_swanson(
        acronyms=df_regions.index, values=df_regions[k], br=ba.regions, ax=ax, **kwargs
    )

    fig.colorbar(ax.get_images()[0], location="top")
    ax.set_axis_off()
    ax.set(title=k)
    fig.savefig(FIGURES_PATH.joinpath(f"{k}_.png"))
    # plt.close(fig)


for field_re in ["fr_diff", "fr_ratio_log"]:
    if not FIGURES_PATH.joinpath(f"{field_re}_re.png").exists():
        kwargs = feats[field_re]
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.tight_layout()
        plot_swanson(
            acronyms=df_regions.index[re_sel],
            values=df_regions[field_re][re_sel],
            br=ba.regions,
            ax=ax,
            **kwargs,
        )
        annotate_swanson(ax=ax, acronyms=BRAIN_REGIONS_RE)
        fig.colorbar(ax.get_images()[0], location="top")
        ax.set_axis_off()
        ax.set(title=field_re)

        fig.savefig(FIGURES_PATH.joinpath(f"{field_re}_re.png"))
        plt.close(fig)


if not FIGURES_PATH.joinpath(f"swanson.png").exists():
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.tight_layout()
    plot_swanson(ax=ax, annotate=BRAIN_REGIONS_RE)
    ax.set_axis_off()
    fig.colorbar(ax.get_images()[0], location="top")
    fig.savefig(FIGURES_PATH.joinpath(f"swanson.png"))
