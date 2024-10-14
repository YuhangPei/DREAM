import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
mathods=['CTW_con']
dataset='CBF'
file_dir='./loss_all'
seeds=[37,118,337,815,19]
# create a pd epoch,clean,noise

for method in mathods:
    df = pd.DataFrame()
    for seed in seeds:
        data_file_name=dataset+'_'+method+'_'+str(seed)+'.npy'
        label_file_name=dataset+'_'+method+'_'+str(seed)+'_label.npy'
        data=np.load(os.path.join(file_dir,data_file_name)) # (n,epoch)
        label=np.load(os.path.join(file_dir,label_file_name)) # (n,2)
        epoch=np.arange(data.shape[1]) # (epoch,)
        clean_mask=label[:,0]==label[:,1]
        clean_data=data[clean_mask]
        noise_data=data[~clean_mask]
        # lossdata shape 1,epoch
        loss_clean=np.mean(clean_data,axis=0) # (epoch,)
        loss_noise=np.mean(noise_data,axis=0) # (epoch,)
        # create a new df
        df_tmp=pd.DataFrame()
        df_tmp['epochs']=epoch
        df_tmp['Loss']=loss_clean
        df_tmp['type']='clean'
        df=pd.concat([df,df_tmp],ignore_index=True)
        df_tmp=pd.DataFrame()
        df_tmp['epochs']=epoch
        df_tmp['Loss']=loss_noise
        df_tmp['type']='noise'
        df=pd.concat([df,df_tmp],ignore_index=True)

    sns.set(style="darkgrid")
    plt.figure(figsize=(8,4))
    sns.lineplot(x='epochs', y='Loss', data=df, hue='type', palette={'clean': '#2878b5', 'noise': '#c82423'})
    # plt.title(dataset+'_'+method)
    # 去除图例名
    plt.legend(title=None,loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir,dataset+'_'+method+'.pdf'),bbox_inches='tight')