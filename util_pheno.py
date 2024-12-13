import re
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join

from MM_analysis.maldi_obj import load_data_maldi, createAdata_maldi,get_image_dict, coreg_merge_img_dict,restore_anndata
from MM_analysis.maldi_funcs import read_metaspace_annotation, read_metaspace_intensity, \
createAdata_maldi_from_metaspace, readXML_affine_matrix, visualize_coreg

from pyFlowSOM import map_data_to_nodes, som
from .consensusClustering import ConsensusCluster
from sklearn.cluster import AgglomerativeClustering

import anndata as ad
import os
join = os.path.join

import math
from functools import reduce
'''combine and then phenotyping main function'''

def combine_adata(adata_list, spheroid_names, spatial_key, mol_list=None, normalize=None):
    common_mol = None

    # Load AnnData objects and determine common molecules
    for adata_spheroid in adata_list:
        
        print(f'spheroid size: {adata_spheroid.obsm[spatial_key][:,0].max()}*{adata_spheroid.obsm[spatial_key][:,1].max()} pixel^2')
        # Get common molecule list
        if common_mol is None:
            common_mol = set(adata_spheroid.var_names)
        else:
            common_mol.intersection_update(set(adata_spheroid.var_names))
    if mol_list is None:
        mol_list = list(common_mol)
    else:
        mol_list = list(set(mol_list).intersection(common_mol))

    if normalize:
        adata_list = [
            adata.copy() for adata in adata_list  # Make copies to avoid altering original data
        ]
        for adata in adata_list:
            if normalize == 'max':
                if 'rangeMax' in adata.varm:
                    adata.X = adata.X / adata.varm['rangeMax']
                else:
                    print('No rangeMax found in AnnData.varm. Adopted zscore normalization instead.')
                    sc.pp.scale(adata)
            elif normalize == 'zscore':
                sc.pp.scale(adata)
            else:
                raise ValueError('Invalid normalization method. Please choose from "max" or "zscore"')
                
    # Subset the AnnData objects to the common molecules
    adata_list = [adata[:,mol_list] for adata in adata_list]

    # Concatenate AnnData objects while maintaining the correct metadata
    adata_all = ad.concat(adata_list, merge='same', uns_merge='unique', label='Spheroid', keys=spheroid_names)
    print(f'Number of cells: {adata_all.n_obs}, Number of molecules: {adata_all.n_vars} ')
    # Ensure that the Spheroid field is in the obs DataFrame
    adata_all.obs['Spheroid'] = adata_all.obs['Spheroid'].astype('category')

    if 'rangeMax' in adata_list[0].varm:
        # adata_all.varm['rangeMax'] = reduce(np.maximum, [adata_spheroid.varm['rangeMax'] for \
        #                                             adata_spheroid in adata_list])
        rangeMax_all = {spheroid: adata.varm['rangeMax'] for spheroid, adata in zip(spheroid_names, adata_list)}
        adata_all.uns['rangeMax_all'] = rangeMax_all  # Store as a dictionary in uns
    return adata_all
## Use 2 phase clustering to identify phenotypes
def phenoAdata(df,show=False):
    
    som_input_arr = df.to_numpy()
    # # train the SOM
    node_output = som(som_input_arr, xdim=10, ydim=10, rlen=10)

    # # use trained SOM to assign clusters to each observation in your data
    clusters, dists = map_data_to_nodes(node_output, som_input_arr)

    eno = pd.DataFrame(data=node_output, columns=df.columns)
    eco = pd.DataFrame(data=clusters, columns=["cluster"])


    # Append results to the input data
    df['cluster'] = clusters

    # Find mean of each cluster
    df_mean = df.groupby(['cluster']).mean()

    df_mean['cluster'] = df_mean.index
    df_mean['count'] = df['cluster'].value_counts().sort_index().values

    # Reset index to move 'cluster' from index to column
    df_mean = df_mean.reset_index(drop=True)
    
    # Make heatmap
    if show:
        sns_plot = sns.clustermap(df_mean.drop(columns=['cluster', 'count']), 
                                  z_score=1, cmap="vlag", center=0,xticklabels=True, yticklabels=True)
        
        #sns_plot.figure.savefig(f"example_cluster_heatmap.png")

    cc = ConsensusCluster(
                cluster=AgglomerativeClustering,
                L=5,
                K=20,
                H=10,
                resample_proportion=0.8
            )
    cc.fit(df_mean.drop(columns=['cluster', 'count']).values)
    print(f'consensus clustering best number of clusters: {cc.bestK}')
    df_mean['metacluster'] = cc.predict()

    cluster_to_metacluster = df_mean['metacluster'].to_dict()

    # Add metacluster column to the original df
    df['metacluster'] = df['cluster'].map(cluster_to_metacluster)

    df_mean_meta = df.drop(columns='cluster').groupby(['metacluster']).mean()
    
    if show:
        sns_plot = sns.clustermap(df_mean_meta, z_score=1, cmap="vlag", center=0,xticklabels=True, yticklabels=True)
        
        #sns_plot.figure.savefig(f"example_metacluster_heatmap.png")
    return df
def plot_phenotype(adata_all, spheroid_names,color_key, spatial_key, spot_size, num_cols,color_map=None):
    

    #plot spatial distribution of each metacluster
    if color_map is None:
        color_map = 'viridis'

    num_spheroids = len(spheroid_names)
    
    num_rows = math.ceil(num_spheroids / num_cols)

    # # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, sph_name in enumerate(sorted(spheroid_names)):
        ax = axs[i]
        sc.pl.spatial(adata_all[adata_all.obs['Spheroid'] == sph_name], 
                    basis=spatial_key,
                    color=color_key, 
                    spot_size=spot_size, 
                    palette=color_map,
                    frameon=False,
                    ax=ax,
                    show=False)
        ax.set_title(sph_name)  

    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

    # plot individually
    # for sph_name in sorted(spheroid_names):
    #     fig, ax = plt.subplots(figsize=(5, 5))
        
    #     sc.pl.spatial(
    #         adata_all[adata_all.obs['Spheroid'] == sph_name], 
    #         basis=spatial_key,
    #         color=color_key, 
    #         spot_size=spot_size, 
    #         palette=color_map,
    #         frameon=False,
    #         ax=ax,
    #         show=False
    #     )
        
    #     ax.set_title(sph_name)
    #     plt.show()
    

def combine_pheno_main(adata_list, spheroid_names,spatial_key,\
                       mol_list=None,Harmony=False,normalize=None,\
                        spot_size=1,num_cols = 5):
    '''
    normalize: str, optional (default: None). 
        When None, no normalization by slide is performed, absolute intensity is used for clustering
        When 'max', the data is divided by the maximum value of rangeMax in AnnData.varm. 
        When 'zscore', the data is z-score normalized.
    '''
    adata_all = combine_adata(adata_list, spheroid_names, spatial_key, mol_list=mol_list, normalize=normalize)

    # Features in Umap
    sc.tl.pca(adata_all, n_comps=min(100,adata_all.n_vars-1))
    sc.pp.neighbors(adata_all, use_rep='X_pca', n_neighbors=30, metric='cosine')
    #sc.pp.neighbors(adata_all, n_neighbors=30, metric='cosine')
    sc.tl.umap(adata_all,random_state=42)
    sc.pl.umap(adata_all,color='Spheroid',frameon=False,show=False)
    if Harmony:
        # Harmony
        sc.external.pp.harmony_integrate(adata_all, key='Spheroid',max_iter_harmony=20)
        sc.pp.neighbors(adata_all, use_rep='X_pca_harmony', n_neighbors=30, metric='cosine',key_added='neighbors_harmony')
        sc.tl.umap(adata_all,random_state=42,neighbors_key='neighbors_harmony')
        sc.pl.umap(adata_all,neighbors_key='neighbors_harmony',color='Spheroid',frameon=False,show=False)
        df1 = pd.DataFrame(data=adata_all.obsm['X_pca_harmony'],dtype='float64')
    else:
        # float64 to satisfy SOM requirement
        df1 = pd.DataFrame(data=adata_all.X.astype(np.float64), columns=adata_all.var_names)

    # Plot spatial distribution of each metacluster for each spheroid
    df = phenoAdata(df1,show=True)
    adata_all.obs['phenotype'] = df['metacluster'].astype('category').values
    color_key = 'phenotype'

    sc.pl.umap(adata_all, color=color_key, frameon=False, show=False)
    # get color map from sc.pl
    phenotype_categories = adata_all.obs[color_key].cat.categories
    colors = sc.pl.palettes.default_20[:len(phenotype_categories)]
    color_map = {category: colors[i] for i, category in enumerate(phenotype_categories)}

    plot_phenotype(adata_all, spheroid_names, color_key,spatial_key, spot_size, num_cols,color_map=color_map)
    return adata_all, df
def combine_PCA_main(adata_list, spheroid_names,spatial_key):
    adata_all = combine_adata(adata_list, spheroid_names, spatial_key)
    sc.tl.pca(adata_all)
    for i in range(5):
        adata_all.obs[f'PC{i+1}'] = adata_all.obsm['X_pca'][:, i]
        color_key = f'PC{i+1}'
        spot_size = 1
        num_cols = 10
        plot_phenotype(adata_all, spheroid_names,color_key, spatial_key, spot_size, num_cols)