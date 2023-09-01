
from pathlib import Path
from one.api import ONE
import ephys_atlas.data

config = ephys_atlas.data.get_config()
# LOCAL_DATA_PATH = Path(config['paths']['features']) 
LOCAL_DATA_PATH = Path("/home/kcenia/Documents/atlas") #KB 20230815
LABEL = config['decoding']['tag']  # LABEL = "2023_W14"
#new: (bef was 30)
LABEL = "2023_W34"

one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)
# df_raw_features, df_clusters, df_channels = load_tables(local_path=LOCAL_DATA_PATH)

df_depths = ephys_atlas.data.compute_depth_dataframe(df_raw_features, df_clusters, df_channels)
df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)


## %% 
#not saved; already in the code
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')

#%% 
import numpy as np
#Plotting df_voltage data
df = df_voltage.reset_index()
len(df.pid.unique()) 
"""
unique pid = 905

""" 
pid_unique = np.array(df.pid.unique()) 

df_0 = df[df.pid == "00a96dee-1e8b-44cc-9cc3-aca704d2b594"]

#saved
for columns in df_0: 
    plt.plot(df_0[columns])
    plt.title(columns,fontsize=20)
    # plt.savefig("/home/kcenia/Desktop/PEA/result_allplots1_"+columns+".png", bbox_inches='tight', pad_inches=0.0)
    plt.show() 

#saved
for columns in df_0: 
    plt.figure(figsize=(5,5))
    plt.plot(df_0[columns],df_0["channel"])
    plt.title(columns,fontsize=20)
    # plt.savefig("/home/kcenia/Desktop/PEA/result_allplots2_"+columns+".png", bbox_inches='tight', pad_inches=0.0)
    plt.show() 

#describe
#saved 
df_describe = df.describe()
print(df_describe)
for columns in df_describe: 
    sns.boxplot(data=df, x=columns, dodge=False) 
    plt.savefig("/home/kcenia/Desktop/PEA/result_boxplotallpid_"+columns+".png", bbox_inches='tight', pad_inches=0.0)
    plt.show()

df_0_describe = df_0.describe()
print(df_0_describe)
for columns in df_0_describe: 
    sns.boxplot(data=df_0, x=columns, dodge=False) 
    plt.savefig("/home/kcenia/Desktop/PEA/result_boxplotpid0_"+columns+".png", bbox_inches='tight', pad_inches=0.0)
    plt.show()


#%% 
#checking the most correlated ones, rms and psd 
#saved
test = ['rms_ap', 'rms_lf', 'psd_delta', 'psd_theta',
       'psd_alpha', 'psd_beta', 'psd_gamma', 'x', 'y', 'z']
data = df_0
for i in test: 
    for j in test: 
        x = i
        y = j
        hue = "acronym"
        plt.figure(figsize=(5,5))
        g = sns.scatterplot(data=data, x=x, y=y,hue = hue)
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1) 
        g.set_xlabel(x,fontsize=20) 
        g.set_ylabel(y,fontsize=20)
        plt.title("Plotting "+x+" vs "+y,fontsize=20)
        plt.savefig("/home/kcenia/Desktop/PEA/Plotting_"+x+"_vs_"+y+"_scatter.png", bbox_inches='tight', pad_inches=0.0)
        plt.show()

        plt.figure(figsize=(5,5))
        g = sns.jointplot(data=data,x=x, y=y, hue=hue,kind="kde") 
        g.set_axis_labels(x, y, fontsize=18)
        # g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1) 
        # g.set_xlabel(x,fontsize=20) 
        # g.set_ylabel("a",fontsize=20)
        # plt.title("Plotting "+x+" vs "+y,fontsize=20)
        plt.tight_layout()
        plt.savefig("/home/kcenia/Desktop/PEA/Plotting_"+x+"_vs_"+y+"_jointplot.png", bbox_inches='tight', pad_inches=0.0)
        plt.show() 

#%%

#%%
for columns in df_0: 
    fig,axs=plt.subplots(6,5)
    for i in range(0,6): 
        for j in range(0,5): 
            axs[i,j].plot(df_0[columns])
            axs[i,j].set_title(columns)
            plt.plot()
plt.show()

# %% 
"""CORRELATION"""
corr = df_0.corr()
corr.style.background_gradient(cmap='coolwarm')

corr.style.background_gradient(cmap='coolwarm').set_precision(2)

corr.style.background_gradient(cmap='coolwarm', axis=None)

#%%
#SAVED
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df.corr(), annot=True, fmt='.2f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
# plt.savefig("/home/kcenia/Desktop/PEA/result_peacorr1.png", bbox_inches='tight', pad_inches=0.0)
#%% 
""" benchmark pids """ 
benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724', 
                  '1e104bf4-7a24-4624-a5b2-c2c8289c0de7', 
                  '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e', 
                  '5f7766ce-8e2e-410c-9195-6bf089fea4fd', 
                  '6638cfb3-3831-4fc2-9327-194b76cf22e1', 
                  '749cb2b7-e57e-4453-a794-f6230e4d0226', 
                  'd7ec0892-0a6c-4f4f-9d8f-72083692af5c', 
                  'da8dfec1-d265-44e8-84ce-6ae9c109b8bd', 
                  'dab512bd-a02d-4c1f-8dbc-9155a163efc0', 
                  'dc7e9403-19f7-409f-9240-05ee57cb7aea', 
                  'e8f9fba4-d151-4b00-bee7-447f0f3e752c', 
                  'eebcaf65-7fa4-4118-869d-a084e84530e2', 
                  'fe380793-8035-414e-b000-09bfe5ece92a'] 
#saved
for i in benchmark_pids: 
    df = df_voltage.loc[i]
    df = df.reset_index()
    for columns in df: 
        plt.figure(figsize=(5,5))
        plt.plot(df[columns])
        plt.title(columns,fontsize=20,pad=1.05)
        plt.suptitle(i,fontsize=15)
        plt.savefig("/home/kcenia/Desktop/PEA/result_allplots3_benchmarkpids_"+i+'_'+columns+".png", bbox_inches='tight', pad_inches=0.0)
        plt.show() 
#saved 
for i in benchmark_pids: 
    df = df_voltage.loc[i]
    df = df.reset_index()
    for columns in df: 
        plt.figure(figsize=(5,5))
        plt.plot(df[columns],df["channel"])
        plt.title(columns,fontsize=20,pad=1.05)
        plt.suptitle(i,fontsize=15)
        plt.savefig("/home/kcenia/Desktop/PEA/result_allplots4_benchmarkpids_"+i+'_'+columns+".png", bbox_inches='tight', pad_inches=0.0)
        plt.show() 



# %% 
""" plot KDE for all features for the benchmark pids """ 
"""
KDE with median, Q1, Q3 
https://seaborn.pydata.org/generated/seaborn.kdeplot.html
""" 
benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724', 
                  '1e104bf4-7a24-4624-a5b2-c2c8289c0de7', 
                  '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e', 
                  '5f7766ce-8e2e-410c-9195-6bf089fea4fd', 
                  '6638cfb3-3831-4fc2-9327-194b76cf22e1', 
                  '749cb2b7-e57e-4453-a794-f6230e4d0226', 
                  'd7ec0892-0a6c-4f4f-9d8f-72083692af5c', 
                  'da8dfec1-d265-44e8-84ce-6ae9c109b8bd', 
                  'dab512bd-a02d-4c1f-8dbc-9155a163efc0', 
                  'dc7e9403-19f7-409f-9240-05ee57cb7aea', 
                  'e8f9fba4-d151-4b00-bee7-447f0f3e752c', 
                  'eebcaf65-7fa4-4118-869d-a084e84530e2', 
                  'fe380793-8035-414e-b000-09bfe5ece92a'] 

df_0 
for columns in df: 
sns.kdeplot(data=df_0, x=columns)


#"Estimate the cumulative distribution function(s), normalizing each subset:"
sns.kdeplot(
    data=df_0, x="alpha_mean", hue="acronym",
    cumulative=True, common_norm=False, common_grid=True,
)