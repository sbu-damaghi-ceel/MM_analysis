import pdb

import re
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns

import xml.etree.ElementTree as ET
from .maldi_obj import create_intensity_image





from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


########################  
# Untargeted MALDI data loading
'''
Use mz as the identifier as it is unique for each molecule(even for same formula with different adducts)
'''

def read_metaspace_intensity(intensity_file,perc_thres=95,verbose = False):
    intensities = pd.read_csv(intensity_file,skiprows=2)

    regex_pattern = r'x(\d+)_y(\d+)'
    xy_columns = [col for col in intensities.columns if re.match(regex_pattern, col)]

    #drop all background pixels
    for col in xy_columns:
        if intensities[col].sum() == 0:
            intensities = intensities.drop(columns=col)
            xy_columns.remove(col)
    ####remove molecules with 95% or more 0 intensities
    row_percentiles = intensities[xy_columns].apply(lambda row: np.percentile(row, perc_thres), axis=1)
    non_zero_rows = row_percentiles > 0
    if verbose:
        print(f'Molecules with {perc_thres}% or more 0 intensity: {intensities.loc[~non_zero_rows,"mol_formula"].values}')
    intensities = intensities[non_zero_rows]
    
    intensities = intensities[xy_columns+['mz']]
    
    return intensities

def read_metaspace_annotation(annotation_file, fdr_thres=None):
    annotations = pd.read_csv(annotation_file,skiprows=2)
    if fdr_thres is not None:
        annotations = annotations[annotations['fdr'] <= fdr_thres]
    return annotations
def createAdata_maldi_from_metaspace(intensities,annotations):
    '''
    centroided imzML output from SCiLab with TIC normalization, metaspace did hotspot removal
    input:
    intensities: pd.DataFrame, each row is a molecule, each column is a pixel. 
                contain xy_columns('x{}_y{}') and 'mz'
    annotations: pd.DataFrame, each row is a molecule, 
                contain 'formula','mz','adduct', 'moleculeNames', 'moleculeIds',etc
    '''
    combined = intensities.merge(annotations, on=['mz'],how='inner')
    
    regex_pattern = r'x(\d+)_y(\d+)'
    xy_columns = [col for col in combined.columns if re.match(regex_pattern, col)]
    xy_tuples = [re.match(regex_pattern, col).groups() for col in xy_columns]
    xy_array = np.array(xy_tuples, dtype=int)
    X_df = combined[xy_columns]

    adata = ad.AnnData(X=X_df.T.values)
    adata.obs_names = X_df.columns #the xy_columns
    adata.obsm['spatial'] = xy_array

    #concatenate formula and adduct to get identifier
    combined['identifier'] =  combined['formula'] +'_'+ combined['adduct']
    adata.var_names = combined['identifier'].values
    adata.var = combined.drop(columns=xy_columns).set_index('identifier')
    
    

    ####For visualization purpose, when need to compare hearmaps of the same molecule across different samples, should unify rangeMax first
    adata.varm['rangeMax'] = np.percentile(adata.X,99,axis=0)#GET RID OF OUTLIERS#np.max(adata.X,axis=0)
    
    print(f'adata shape: {adata.X.shape}')
    return adata


def plot_spatial_subplots(adata, spot_size=1, ncols=5):
    mols = adata.var_names
    n_plots = len(mols)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    
    for idx, mol in enumerate(mols):
        ax = axs[idx // ncols, idx % ncols]
        sc.pl.spatial(adata, color=mol, spot_size=spot_size, ax=ax, show=False,frameon=False)
        ax.set_title(mol)
    
    # Hide any unused subplots
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(axs.flatten()[i])
    
    plt.tight_layout()
    plt.show()

######################## Co-registration
# from .coreg_util import *




def create_channel_image(image, channel):
    # Normalize the image to [0, 1] range
    if np.max(image) > 1:
        print('Normalizing image to [0, 1] range')
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Create an empty RGB image
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3))

    # Assign the image to the specified channel
    rgb_image[:, :, channel] = image

    return rgb_image

######################## Phenotypying
#from MM_analysis.util_pheno import phenoAdata
    


######################## Differential Analysis

def plot_volcano(differential_results, df_mean_meta, metacluster, dominant_variables=None, id_to_name=None):
    
    # Calculate log2 fold change
    offset = abs(df_mean_meta.min().min()) + 1e-10  # Smallest value + small constant
    log2_fold_change = np.log2(df_mean_meta.loc[metacluster] + offset) - np.log2(df_mean_meta.drop(index=metacluster).mean() + offset)

    # Safely calculate -log10 of p-values
    p_values = differential_results[metacluster]
    minus_log10_p_values = -np.log10(p_values)
    max_finite_p = np.nanmax(minus_log10_p_values[np.isfinite(minus_log10_p_values)])  # Max finite p-value
    minus_log10_p_values[minus_log10_p_values == np.inf] = 1.1 * max_finite_p  # Cap inf values in p-values
    
    
    # Plot volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_change, minus_log10_p_values, alpha=0.5)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-value')
    plt.title(f'Volcano Plot for Metacluster {metacluster}')

    # Highlight dominant variables
    # if dominant_variables is not None:
    #     for gene in dominant_variables[metacluster]:
    #         plt.scatter(log2_fold_change[gene], minus_log10_p_values[gene], color='red')
    #         label = id_to_name[gene] if id_to_name is not None else gene
    #         plt.text(log2_fold_change[gene], minus_log10_p_values[gene], label, fontsize=9)

    # Hightlight variables with mapping that contain 'Cer' or 'SM'
    if id_to_name is not None:
        for gene in df_mean_meta.columns:
            if gene in id_to_name.keys():
                for isomer in id_to_name[gene]:
                    if len(isomer) > 20:
                        isomer = isomer.split('-')[-1]
                    if 'Cer' in isomer or 'SM' in isomer:
                        plt.scatter(log2_fold_change[gene], minus_log10_p_values[gene], color='red')
                        plt.text(log2_fold_change[gene], minus_log10_p_values[gene], isomer, fontsize=9)
                        break
            else:
                print(f'No annotation mapping found for {gene}')
    plt.show()

    return log2_fold_change, minus_log10_p_values

def diffAnalysis(df, top_n, id_to_name=None,heatmap=False):
    #drop NA rows of 'metacluster' column
    df.dropna(subset=['metacluster'], inplace=True)
    
    df_mean_meta = df.groupby(['metacluster']).mean()
    df_mean_meta_t = df_mean_meta.transpose()

    # Initialize a DataFrame to store the differential analysis results
    differential_results = pd.DataFrame(index=df_mean_meta_t.index)

    # Perform t-test for each variable across metaclusters
    for col in df_mean_meta_t.columns:
        p_values = []
        for var in df_mean_meta_t.index:
            group1 = df[df['metacluster'] == col][var]
            group2 = df[df['metacluster'] != col][var]
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Use Welch's t-test for unequal variance
            p_values.append(p_val)
        differential_results[col] = p_values

    # Adjust the p-values for multiple testing (e.g., using Benjamini-Hochberg correction)
    adjusted_p_values = multipletests(differential_results.values.flatten(), method='fdr_bh')[1]
    differential_results = pd.DataFrame(adjusted_p_values.reshape(differential_results.shape),
                                        index=differential_results.index, columns=differential_results.columns)

    # Identify the dominant variables for each metacluster
    dominant_variables = {}
    for col in differential_results.columns:
        p_values_col = differential_results[col]
        top_vars = p_values_col.nsmallest(top_n).index.tolist()
        inf_vars = p_values_col[p_values_col == 0].index.tolist()
        dominant_variables[col] = list(set(top_vars) | set(inf_vars))  # Combine top_n and inf variables

    # Create a list to store the dominant variables for visualization
    dominant_list = []
    for metacluster, variables in dominant_variables.items():
        for var in variables:
            dominant_list.append({
                'metacluster': metacluster,
                'variable': var,
                'p_value': differential_results.loc[var, metacluster]
            })

    # Convert the list to a DataFrame
    # dominant_df = pd.DataFrame(dominant_list)
    # dominant_df.to_csv('dominant_variables_per_metacluster.csv', index=False)

    unique_metaclusters = df['metacluster'].unique()
    for i in unique_metaclusters:
        plot_volcano(differential_results, df_mean_meta, i, dominant_variables=dominant_variables,id_to_name=id_to_name)
    
    return differential_results, dominant_variables

############################Formula Analysis

def extract_formulas(variables):
    return [var.split('_')[0] for var in variables]

def kendrick_analysis(formulas):
    '''Kendrick Mass Defect Analysis'''
    # Define the base mass for CH2 (methylene group)
    base_mass = 14.01565
    
    # Initialize dictionary to store Kendrick mass and defect
    kendrick_dict = {}

    for formula in formulas:
        # Calculate the exact mass of the formula (this requires a function to compute the exact mass from a formula)
        exact_mass = compute_exact_mass(formula)
        
        # Calculate Kendrick mass and Kendrick mass defect
        kendrick_mass = exact_mass * (base_mass / 14.0)
        kendrick_mass_defect = kendrick_mass - round(kendrick_mass)
        
        kendrick_dict[formula] = {
            'exact_mass': exact_mass,
            'kendrick_mass': kendrick_mass,
            'kendrick_mass_defect': kendrick_mass_defect
        }
    
    # Group formulas into families based on Kendrick mass defect
    families = {}
    for formula, values in kendrick_dict.items():
        defect = values['kendrick_mass_defect']
        # Use some threshold to determine if formulas belong to the same family (e.g., 0.001)
        threshold = 0.001
        found_family = False
        for family_defect in families:
            if abs(defect - family_defect) < threshold:
                families[family_defect].append(formula)
                found_family = True
                break
        if not found_family:
            families[defect] = [formula]
    
    return families

def compute_exact_mass(formula):
    # Function to compute the exact mass from a chemical formula
    # This requires a periodic table with atomic masses
    atomic_masses = {
        'H': 1.00784, 'C': 12.00000, 'O': 15.99491, 'N': 14.00307, 'S': 31.97207,
        # Add more elements as needed
    }
    mass = 0.0
    # Parse the formula to compute the exact mass
    import re
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    for (element, count) in elements:
        count = int(count) if count else 1
        mass += atomic_masses[element] * count
    return mass

def enrichment_analysis(families):
    # Perform enrichment analysis on the grouped formulas
    # This is a placeholder for actual enrichment analysis
    enriched_families = {}
    for family, formulas in families.items():
        # Example enrichment criteria: number of formulas in family
        enriched_families[family] = {
            'formulas': formulas,
            'enrichment_score': len(formulas)  # Replace with actual enrichment calculation
        }
    return enriched_families

