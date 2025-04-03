#%%
"""
    27March2025
    OW, KB
    Predicting a continuous variable - cortical depth
    Merging OW [get data and example model] and MF [atlas acronym / id and depth function] codes
"""

# %%
import matplotlib.pyplot as plt
from pathlib import Path
from one.api import ONE
import ephys_atlas.data 
from pathlib import Path
import pandas as pd
import numpy as np

import sklearn.metrics
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

from iblutil.numerical import ismember
import ephys_atlas.encoding
import ephys_atlas.decoding
import ephys_atlas.anatomy
import ephys_atlas.data
import ephys_atlas.features


import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor


from iblatlas.atlas import BrainRegions

# %%
LOCAL_DATA_PATH = Path.home().joinpath("Downloads")
LABEL = "2024_W50"  # or put "latest" # or '2024_W50'
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)

# %%
# Here we prepare the dataframe by selecting only the records in isocortex, and computing the cortical depth

df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)

df_voltage['cortical_depths']  = xyz_to_depth(df_voltage[['x', 'y', 'z']].to_numpy())
df_voltage = df_voltage.dropna().copy()

print(df_voltage.dropna().shape)

# %%
x_list = ephys_atlas.features.voltage_features_set(features_list= ['raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms'])
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)
stratify = df_voltage.index.get_level_values("pid").to_numpy()
Y = df_voltage['cortical_depths'].to_numpy()

train_idx, test_idx = ephys_atlas.encoding.train_test_split_indices(df_voltage)
X_test, y_test = (scaler.transform(X[test_idx, :]), Y[test_idx])
X_train, y_train = (scaler.transform(X[train_idx, :]), Y[train_idx])

# %%
# 3. Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# %%
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%% 
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%%
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%% 
from sklearn.svm import SVR

model = SVR(kernel='rbf')  # or 'linear', 'poly'
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%% 
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# 3. Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "BayesianRidge": BayesianRidge(),
    "HuberRegressor": HuberRegressor()
}

# 4. Fit & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}: R² = {r2:.4f}, MSE = {mse:.4f}")

#%% 
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)
print(f"MLPRegressor: R² = {r2_score(y_test, y_pred):.4f}, MSE = {mean_squared_error(y_test, y_pred):.4f}")

#%%
"""
PLOT
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel('Actual Depth')
plt.ylabel('Predicted Depth')
plt.title('Linear Regression: Predicted vs Actual Depth')
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
scatter = plt.scatter(y_test, y_pred, c=y_test, cmap='viridis', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Prediction')

plt.xlabel('Actual Depth')
plt.ylabel('Predicted Depth')
plt.title('Predicted vs Actual Depth (Color = True Depth)')
plt.colorbar(scatter, label='Actual Depth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
CODE TO PLOT BRAIN REGIONS AND A FEATURE SIDE BY SIDE - WE WANT THE DEPTH IN THE FEATURE PART
""" 
pids = df_voltage.index.get_level_values(0).unique().tolist()

for pid in pids:
    print(pid)
    pid_df = df_voltage.loc[pid]






for i, pid in enumerate(pids):
    # TODO label with the pids
    df_pid = df_voltage.loc[pid]
    depth = df_pid.groupby('axial_um').agg(atlas_id=pd.NamedAgg(column='atlas_id', aggfunc='first')).reset_index()
    brainbox.ephys_plots.plot_brain_regions(depth['atlas_id'].values, channel_depths=depth['axial_um'].values,
                                            brain_regions=regions, display=True, ax=ax[i * 3])
    ax[-2].plot(df_pid['x'].values * 1e6, df_pid['y'].values * 1e6, '.', color='r', markersize=2)
    ax[-2].plot(df_pid['x'].values[-1] * 1e6, df_pid['y'].values[-1] * 1e6, '*', color='k', markersize=2)
    ax[i * 3 + 2].axis('off')


    feature = df_pid["depths"].values
    if feature_name in transform:
        feature = transform["depths"](feature)
    data_bank, x_bank, y_bank = brainbox.plot_base.arrange_channels2banks(
        data=feature,
        chn_coords=df_pid[['lateral_um', 'axial_um']].values,
        pad=True,
    )
    data = brainbox.plot_base.ProbePlot(data_bank, x=x_bank, y=y_bank, cmap=colormap)
    data.clim = clim
    brainbox.ephys_plots.plot_probe(data.convert2dict(), ax=ax[i * 3 + 1], show_cbar=False)
    ax[i * 3 + 1].axis('off')
# %%
