from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ephys_atlas.data
import ephys_atlas.plots
from iblatlas.regions import BrainRegions

br = BrainRegions()
folder_file_save = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/Fig1')

label = 'latest'
local_path = Path("/Users/gaellechapuis/Documents/Work/EphysAtlas/Data/").joinpath(label)
df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_tables(
    local_path, verify=True)

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
df_voltage = df_voltage.rename(columns={"atlas_id": "Allen_id", "acronym": "Allen_acronym"})

df_voltage['Cosmos_id'] = br.remap(df_voltage['Allen_id'], source_map='Allen', target_map='Cosmos')
df_voltage['Beryl_id'] = br.remap(df_voltage['Allen_id'], source_map='Allen', target_map='Beryl')
df_voltage['pids'] = df_voltage.index.get_level_values(0)
df_voltage['Beryl_acronym'] = br.id2acronym(df_voltage['Beryl_id'], mapping='Beryl')
# Remove void / root
df_voltage.drop(df_voltage[df_voltage['Beryl_acronym'].isin(['void', 'root'])].index, inplace=True)

##
# aggregate ber brain region
df_regions = df_voltage.groupby('Beryl_id').agg(
    n_channels=pd.NamedAgg(column='Allen_id', aggfunc='count'),
    n_pids=pd.NamedAgg(column='pids', aggfunc='nunique')
)
df_regions['Beryl_id'] = df_regions.index.values


# Plot
ephys_atlas.plots.region_bars(atlas_id=df_regions.index.values,
                              feature=df_regions[f'n_pids'].values, regions=br)

# Save figure
plt.savefig(folder_file_save.joinpath(f"Beryl_Npids.svg"))
plt.show()

# Plot
ephys_atlas.plots.region_bars(atlas_id=df_regions.index.values,
                              feature=df_regions[f'n_channels'].values, regions=br)

# Save figure
plt.savefig(folder_file_save.joinpath(f"Beryl_Nchannels.svg"))
plt.show()
