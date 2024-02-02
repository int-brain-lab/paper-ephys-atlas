# %%
import matplotlib.pyplot as plt 
import pandas as pd
from iblatlas.genomics import agea 

from iblatlas.regions import BrainRegions
brain_regions = BrainRegions()

df_genes, gene_expression_volumes, atlas_agea = agea.load()

igenes = (0,)
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
atlas_agea.plot_cslice(0, ax=axs[0, 0])
atlas_agea.plot_cslice(0, ax=axs[1, 0], volume='annotation')
atlas_agea.plot_cslice(0, ax=axs[2, 0], volume=gene_expression_volumes[igenes[0]], cmap='viridis')
atlas_agea.plot_sslice(0, ax=axs[0, 1])
atlas_agea.plot_sslice(0, ax=axs[1, 1], volume='annotation')
atlas_agea.plot_sslice(0, ax=axs[2, 1], volume=gene_expression_volumes[igenes[0]], cmap='viridis')
fig.tight_layout()


# %% remap the the agea atlas at the cosmos level parcellation
import numpy as np
idx_beryl_root = atlas_agea.regions.mappings['Beryl'] == 1 #added

ne = gene_expression_volumes.shape[0]
sel = np.isin(atlas_agea.label,np.where(idx_beryl_root)[0]) #to select only root, lateralized
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)

atlas_agea.plot_cslice(0, ax=axs[2, 0], volume=sel, cmap='viridis')
atlas_agea.plot_sslice(0, ax=axs[2, 1], volume=sel, cmap='viridis')
fig.tight_layout()


sel = np.isin(atlas_agea.label.flatten(),np.where(idx_beryl_root)[0]) #to select only root, lateralized

# sel = atlas_agea.label.flatten() != 0  # remove void voxels
# reshape in a big array nexp x nvoxels this takes a little while
gexps = gene_expression_volumes.reshape((ne, -1))[:, sel].astype(np.float32).transpose()
aids = atlas_agea.regions.id[atlas_agea.label.flatten()[sel]]
# aids_cosmos = atlas_agea.regions.remap(aids, 'Allen-lr', 'Cosmos') 


# %% 
#KB 30Jan2024 
#================================= hdbscan ================================
import numpy as np
import hdbscan

# Generate a synthetic dataset
np.random.seed(0)
data = gexps

# Compute the clustering using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
clusterer.fit(data)

# # Plot the results
# from iblutil.numerical import ismember 
# _, rids = ismember(aids, brain_regions.id)
# colors_ids_rgb = brain_regions.rgb[rids, :]
# plt.scatter(data[:, 0], data[:, 1], c=clusterer.labels_, cmap=colors_ids_rgb)
# plt.title("HDBSCAN Clustering")
# plt.show() 

# Plot the results
from iblutil.numerical import ismember 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors_ids_rgb = brain_regions.rgb[rids, :]

# Create a custom colormap using LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_ids_rgb / 255.0, N=len(colors_ids_rgb))

# Plot the scatter plot with the custom colormap
plt.figure(figsize=(12, 8))  # Adjust the values (width, height) as needed
plt.scatter(data[:, 0], data[:, 1], c=clusterer.labels_, cmap=custom_cmap, alpha = 0.25, s = 10)
plt.title("HDBSCAN Clustering")
plt.show()




# %% 
#KB 30Jan2024 
#================================= PCA 1 ================================
df_aids = pd.DataFrame(aids, columns=['aids'])
from sklearn.preprocessing import StandardScaler

x = gexps
y = aids
# Standardizing the features
# x = StandardScaler().fit_transform(x)

#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=100)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = [f'pc{i}' for i in range(1, 101)])

finalDf = pd.concat([principalDf, df_aids[['aids']]], axis = 1)

finalDf_2 = finalDf

unique_aids = np.array(finalDf_2.aids.unique())


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('pc1', fontsize = 15)
ax.set_ylabel('pc2', fontsize = 15)
ax.set_title('100 component PCA', fontsize = 20)

targets = aids
# colors = ['r', 'g', 'b']
for target in targets:
    indicesToKeep = finalDf['aids'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
               , finalDf.loc[indicesToKeep, 'pc2']
               , s = 20)
# ax.legend(targets)
ax.grid() 

#===========================================================================
#%%
#================================= PCA 2 ================================
import plotly.express as px
from sklearn.decomposition import PCA

df = gexps
X = df

pca = PCA(n_components=100)
components = pca.fit_transform(X)

fig = px.scatter(components, x=0, y=1, color=aids) #use RGB code from A
fig.show()
#===========================================================================



# %%

a = np.array(principalDf['pc1'])
b = np.array(principalDf['pc2'])



gexps_pca = np.column_stack((a,b)) 





#%%
#================================= DIVISION ================================

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to plot PCA results
def plot_pca(ax, principal_df, title):
    unique_aids = np.array(principal_df['aids'].unique())
    targets = unique_aids
    
    for target in targets:
        indices_to_keep = principal_df['aids'] == target
        ax.scatter(principal_df.loc[indices_to_keep, 'pc1'],
                   principal_df.loc[indices_to_keep, 'pc2'],
                   s=20, label=f'AIDS {target}')
    
    ax.set_xlabel('pc1', fontsize=15)
    ax.set_ylabel('pc2', fontsize=15)
    ax.set_title(title, fontsize=20)
    ax.legend()
    ax.grid()

# Perform PCA with 3 components
pca_3 = PCA(n_components=3)
principal_components_3 = pca_3.fit_transform(x)
principal_df_3 = pd.DataFrame(data=principal_components_3,
                              columns=[f'pc{i}' for i in range(1, 4)])
final_df_3 = pd.concat([principal_df_3, df_aids[['aids']]], axis=1)

# Perform PCA with 100 components
pca_100 = PCA(n_components=100)
principal_components_100 = pca_100.fit_transform(x)
principal_df_100 = pd.DataFrame(data=principal_components_100,
                                columns=[f'pc{i}' for i in range(1, 101)])
final_df_100 = pd.concat([principal_df_100, df_aids[['aids']]], axis=1)

# Perform PCA with 300 components
pca_300 = PCA(n_components=300)
principal_components_300 = pca_300.fit_transform(x)
principal_df_300 = pd.DataFrame(data=principal_components_300,
                                columns=[f'pc{i}' for i in range(1, 301)])
final_df_300 = pd.concat([principal_df_300, df_aids[['aids']]], axis=1)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Plot for 3 components
plot_pca(axes[0], final_df_3, '3 component PCA')

# Plot for 100 components
plot_pca(axes[1], final_df_100, '100 component PCA')

# Plot for 300 components
plot_pca(axes[2], final_df_300, '300 component PCA')

plt.tight_layout()
plt.show()






# %%
