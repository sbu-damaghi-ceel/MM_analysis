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
createAdata_maldi_from_metaspace, readXML_affine_matrix, visualize_coreg, phenoAdata



import anndata as ad
import os
join = os.path.join

import math
from functools import reduce
'''combine and then phenotyping main function'''

def combine_pheno_main(control_path_list, spheroid_names,spatial_key,Harmony=False):
    
    #spheroid_names = [re.search(r'[^/]+(?=\.h5ad)',adata_path).group() for adata_path in control_path_list]
    # Initialize lists to store AnnData objects and molecule names
    adata_list = []
    mol_list = None

    # Load AnnData objects and determine common molecules
    for adata_path in control_path_list:
        adata_spheroid = ad.read_h5ad(adata_path)
        print(f'spheroid size: {adata_spheroid.obsm[spatial_key][:,0].max()}*{adata_spheroid.obsm[spatial_key][:,1].max()} pixel^2')
        
        adata_list.append(adata_spheroid)
        # Get common molecule list
        if mol_list is None:
            mol_list = set(adata_spheroid.var_names)
        else:
            mol_list.intersection_update(adata_spheroid.var_names)

    # Convert mol_list back to a list
    mol_list = list(mol_list)

    # Subset the AnnData objects to the common molecules
    adata_list = [adata[:,mol_list] for adata in adata_list]

    # Concatenate AnnData objects while maintaining the correct metadata
    adata_all = ad.concat(adata_list, merge='same', uns_merge='unique', label='Spheroid', keys=spheroid_names)
    print(f'Number of cells: {adata_all.n_obs}, Number of molecules: {adata_all.n_vars} ')
    # Ensure that the Spheroid field is in the obs DataFrame
    adata_all.obs['Spheroid'] = adata_all.obs['Spheroid'].astype('category')

    adata_all.varm['rangeMax'] = reduce(np.maximum, [adata_spheroid.varm['rangeMax'] for \
                                                adata_spheroid in adata_list])

    # Features in Umap
    sc.tl.pca(adata_all, n_comps=min(100,adata_all.n_vars-1))
    
    if Harmony:
        # Harmony
        sc.external.pp.harmony_integrate(adata_all, key='Spheroid',max_iter_harmony=20)
        sc.pp.neighbors(adata_all, use_rep='X_pca_harmony', n_neighbors=30, metric='cosine',key_added='neighbors_harmony')
        sc.tl.umap(adata_all,random_state=42,neighbors_key='neighbors_harmony')
        sc.pl.umap(adata_all,neighbors_key='neighbors_harmony',color='Spheroid',frameon=False,show=False)
        df1 = pd.DataFrame(data=adata_all.obsm['X_pca_harmony'],dtype='float64')
    else:
        sc.pp.neighbors(adata_all, use_rep='X_pca', n_neighbors=30, metric='cosine')
        sc.pp.neighbors(adata_all, n_neighbors=30, metric='cosine')
        sc.tl.umap(adata_all,random_state=42)
        sc.pl.umap(adata_all,color='Spheroid',frameon=False,show=False)
        df1 = pd.DataFrame(data=adata_all.X, columns=adata_all.var_names)

    # Plot spatial distribution of each metacluster for each spheroid
    df = phenoAdata(df1,show=True)
    adata_all.obs['phenotype'] = df['metacluster'].astype('category').values

    sc.pl.umap(adata_all,color='phenotype',frameon=False,show=False)

    #plot spatial distribution of each metacluster
    phenotype_categories = adata_all.obs['phenotype'].cat.categories
    colors = sc.pl.palettes.default_20[:len(phenotype_categories)]
    color_map = {category: colors[i] for i, category in enumerate(phenotype_categories)}

    num_spheroids = len(spheroid_names)
    num_cols = 5
    num_rows = math.ceil(num_spheroids / num_cols)

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, sph_name in enumerate(spheroid_names):
        ax = axs[i]
        sc.pl.spatial(adata_all[adata_all.obs['Spheroid'] == sph_name], 
                      basis=spatial_key,
                      color='phenotype', 
                      spot_size=1, 
                      palette=color_map,
                      frameon=False,
                      ax=ax,
                      show=False)
        ax.set_title(sph_name)  # Optionally set a title for each subplot

    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
    return adata_all, df


def merge_anndata_on_spatial(adata1, col1,adata2,col2):
    spatial1 = adata1.obsm[col1].astype(int)
    spatial2 = adata2.obsm[col2]
    
    df_spatial1 = pd.DataFrame(spatial1, columns=['x', 'y'])
    df_spatial2 = pd.DataFrame(spatial2, columns=['x', 'y'])
    

    df_spatial1['index1'] = df_spatial1.index
    df_spatial2['index2'] = df_spatial2.index
    
    # Merge the dataframes on spatial coordinates
    merged_df = pd.merge(df_spatial1, df_spatial2, on=['x', 'y'], how='left')
    
    new_X = []
    pheno_list = []
    for idx1, row in merged_df.iterrows():
        if pd.notna(row['index2']):
            idx2 = int(row['index2'])
            new_row = np.concatenate((adata1.X[idx1], adata2.X[idx2]))
            new_pheno = adata2.obs['phenotype'][idx2]
        else:
            new_row = np.concatenate((adata1.X[idx1], np.full(adata2.X.shape[1], np.nan)))
            new_pheno = np.nan
        new_X.append(new_row)
        pheno_list.append(new_pheno)
    
    new_X = np.array(new_X)
    pheno = np.array(pheno_list)
    
    
    
    new_adata = ad.AnnData(X=new_X)
    
    new_adata.obs_names = adata1.obs_names.copy()
    new_var_names = np.concatenate((adata1.var_names, adata2.var_names))
    new_adata.var_names = new_var_names
    new_adata.var['source'] = ['macsima']*len(adata1.var_names) + ['maldi']*len(adata2.var_names)

    new_adata.obsm = adata1.obsm.copy()
    new_adata.obs['phenotype']  = pheno
    
    return new_adata
