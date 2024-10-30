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


from pyFlowSOM import map_data_to_nodes, som
from .consensusClustering import ConsensusCluster
from sklearn.cluster import AgglomerativeClustering


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
        print(f'Molecules with {perc_thres}% or more 0 intensity: {intensities.loc[~non_zero_rows,'mol_formula'].values}')
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
    adata.var['rangeMax'] = np.percentile(adata.X,99,axis=0)#GET RID OF OUTLIERS#np.max(adata.X,axis=0)
    
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

######################## 
# Co-registration
'''Co-registration related functions'''

import cv2
import numpy as np
import matplotlib.pyplot as plt



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
def readXML_affine_matrix(xml_file_path):
    identity_transform = np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 1., 0.]])
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # Assume only one image is transformed, the other remains the same(identity_matrix)
    for transform in root.findall('.//ManualSourceTransforms/SourceTransform'):
        affine_text = transform.find('affine').text
        affine_values= [float(val) for val in affine_text.split()]
        affine_matrix = np.array(affine_values).reshape(3,4)
        
        if not np.array_equal(affine_matrix, identity_transform):
            return affine_matrix
    #If no affine matrix found, return identity matrix
    return identity_transform

def transform_image_single(base_img, transform_img, affine_matrix):
    affine_matrix_2x3 = affine_matrix[:2, [0, 1, 3]]
    rows, cols, _ = base_img.shape
    transformed_img = cv2.warpAffine(transform_img, affine_matrix_2x3, (cols, rows))
    return transformed_img

def overlay_images_with_affine(base_img, base_name, *args,bgr=False,plot=False):
    """
    Overlay images with affine transformations.

    Parameters:
        base_img (numpy.ndarray): The base image.
        base_name (str): The name of the base image.
        bgr (bool): Whether the images are in BGR format.
        *args: Arbitrary number of tuples each containing (transform_img, transform_name, affine_matrix).

    """
    # Convert BGR (OpenCV format, from cv2.imread) to RGB (matplotlib format)
    if bgr:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        args = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), name, affine) for img, name, affine in args]

    # Ensure the affine matrix is 2x3 for cv2.warpAffine by taking the first two rows and 1,2,4-th columns
    transformed_images = []
    for transform_img, transform_name, affine_matrix in args:
        transformed_img = transform_image_cv2(base_img, transform_img, affine_matrix)
        transformed_images.append((transformed_img, transform_name))

    if plot:
        # Plot the images separately and overlayed together
        fig, ax = plt.subplots(1, len(transformed_images) + 2, figsize=(4 * (len(transformed_images) + 2), 4))
        ax[0].imshow(base_img)
        ax[0].set_title(base_name)
        ax[0].axis('off')

        for i, (transformed_img, transform_name) in enumerate(transformed_images):
            ax[i + 1].imshow(transformed_img)
            ax[i + 1].set_title(f'Transformed {transform_name}')
            ax[i + 1].axis('off')

        ax[-1].imshow(base_img, alpha=0.5)
        for transformed_img, _ in transformed_images:
            ax[-1].imshow(transformed_img, alpha=0.3)
        ax[-1].set_title(f'Overlay of {base_name} and Transformed Images')
        ax[-1].axis('off')

        plt.show()

    return transformed_images


def visualize_coreg(parent_dir,adata_cer,adata_met,adata_sm,affine_matrix_met,affine_matrix_sm,show='generated'):
    if show == 'generated':
        #IMAGES generated from intensity data(the first molecule)
        cer_gray,_ = create_intensity_image(adata_cer,adata_cer.var_names[0],spatial_key='spatial')
        cer_img = create_channel_image(cer_gray, 0)
        met_gray,_ = create_intensity_image(adata_met,adata_met.var_names[0],spatial_key='spatial')
        met_img = create_channel_image(met_gray, 1)
        sm_gray,_ = create_intensity_image(adata_sm,adata_sm.var_names[0],spatial_key='spatial')
        sm_img = create_channel_image(sm_gray, 2)

        overlay_images_with_affine(cer_img, 'cer',(met_img,'met',affine_matrix_met),\
                                   (sm_img,'sm',affine_matrix_sm))
        
    elif show == 'metaspace':
        #images downloaded from METASPACE
        cer_img = cv2.imread(join(parent_dir,'coreg','Cer.png'))
        met_img = cv2.imread(join(parent_dir,'coreg','Met.png'))
        sm_img = cv2.imread(join(parent_dir,'coreg','SM.png'))
        
        
        overlay_images_with_affine(cer_img, 'cer',(met_img,'met',affine_matrix_met),\
                                   (sm_img,'sm',affine_matrix_sm),bgr=True)
    else:
        print('Invalid show option, should be either "generated" or "metaspace"')
        return

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
    obs_data = []
    for idx1, row in merged_df.iterrows():
        if pd.notna(row['index2']):
            idx2 = int(row['index2'])
            new_row = np.concatenate((adata1.X[idx1], adata2.X[idx2]))
            obs_row = adata2.obs.iloc[idx2].to_dict()
        else:
            new_row = np.concatenate((adata1.X[idx1], np.full(adata2.X.shape[1], np.nan)))
            obs_row = {col: np.nan for col in adata2.obs.columns}
        new_X.append(new_row)
        obs_data.append(obs_row)
    
    new_X = np.array(new_X)
    obs_df = pd.DataFrame(obs_data, index=adata1.obs_names)
    
    new_adata = ad.AnnData(X=new_X)
    new_adata.obs_names = adata1.obs_names.copy()
    new_var_names = np.concatenate((adata1.var_names, adata2.var_names))
    new_adata.var_names = new_var_names
    new_adata.var['source'] = ['macsima']*len(adata1.var_names) + ['maldi']*len(adata2.var_names)

    
    new_adata.obsm = adata1.obsm.copy()
    new_adata.obs = pd.concat([adata1.obs.add_prefix('macsima_'), obs_df.add_prefix('maldi_')], axis=1)
    
    return new_adata
######################## Phenotypying
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
    


######################## Differential Analysis

def plot_volcano(differential_results, df, metacluster, dominant_variables=None):
    log2_fold_change = np.log2(df.groupby('metacluster').mean().loc[metacluster]) - np.log2(df.groupby('metacluster').mean().drop(index=metacluster).mean())
    
    p_values = differential_results[metacluster]
    minus_log10_p_values = -np.log10(p_values)
    # Trim the infinity to 2 * max number
    minus_log10_p_values[minus_log10_p_values == np.inf] = 1.1 * minus_log10_p_values[minus_log10_p_values != np.inf].max()

    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_change, minus_log10_p_values, alpha=0.5)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-value')
    plt.title(f'Volcano Plot for Metacluster {metacluster}')

    # Highlight dominant variables
    if dominant_variables is not None:
        for i,gene in enumerate(dominant_variables[metacluster]):
            plt.scatter(log2_fold_change[gene], minus_log10_p_values[gene], color='red')
            plt.text(log2_fold_change[gene], minus_log10_p_values[gene], gene, fontsize=9)
        
    plt.show()

    return log2_fold_change, minus_log10_p_values

def diffAnalysis(df, top_n, heatmap=False):
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
        plot_volcano(differential_results, df_mean_meta, i, dominant_variables=dominant_variables)
    
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

