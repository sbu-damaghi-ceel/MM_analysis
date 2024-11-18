import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk

from MM_analysis.maldi_obj import get_image_dict
'''functions for calculating corr and plotting heatmap/barplot'''
def calculate_gene_distance_correlations(adata):
    expression_matrix = adata.X
    distances = adata.obs['distances'].values
    correlation_results = []

    for gene_idx in range(expression_matrix.shape[1]):
        gene_expression = expression_matrix[:, gene_idx].flatten()  # Flatten if it's a sparse matrix
        correlation = np.corrcoef(gene_expression, distances)[0, 1]
        correlation_results.append(correlation)

    gene_names = adata.var.index
    return pd.Series(correlation_results, index=gene_names)

def plot_heatmap(correlation_df):
    avg_correlations = correlation_df.mean(axis=1)
    sorted_genes = avg_correlations.sort_values(ascending=False).index
    sorted_correlation_df = correlation_df.loc[sorted_genes]

    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_correlation_df, cmap='coolwarm', center=0)
    plt.title('Gene Expression Correlation with Distances for Different Spheroids')
    plt.xlabel('Spheroids')
    plt.ylabel('Genes')
    plt.show()
def plot_combined_bar_boxplot(correlation_df):
    avg_correlations = correlation_df.mean(axis=1)
    sorted_avg_correlations = avg_correlations.sort_values(ascending=False)

    plt.figure(figsize=(15, 10))

    # Bar plot
    sns.barplot(x=sorted_avg_correlations.index, y=sorted_avg_correlations.values, color='skyblue')

    # Box plot with individual points
    for i, gene in enumerate(sorted_avg_correlations.index):
        sns.boxplot(x=[i] * len(correlation_df.columns), y=correlation_df.loc[gene], 
                    color='lightgray', width=0.5, fliersize=0)
        sns.stripplot(x=[i] * len(correlation_df.columns), y=correlation_df.loc[gene], 
                      color='black', size=4, jitter=True)


    plt.xticks(rotation=90)
    plt.xlabel('Genes')
    plt.ylabel('Correlation with Distances')
    plt.title('Average Gene Expression Correlation with Distances Across Spheroids')
    plt.tight_layout()
    plt.show()

    
def plot_corr_all(adata):
    data = pd.DataFrame(adata.X, columns=adata.var_names)  # If adata.X is sparse, use .toarray()
    correlation_matrix = data.corr()

    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation_matrix, 
                annot=False,  # Set to True if you want to see the correlation values
                cmap='coolwarm', 
                square=True, 
                linewidths=.5, 
                center=0, 
                cbar_kws={"shrink": 0.75},
                xticklabels=correlation_matrix.columns, 
                yticklabels=correlation_matrix.index)
    plt.title(f'Correlation Matrix', fontsize=14)
    plt.show()
def plot_corr_1vsall(adata,target_variable,selected_variables=None):
    
    data = pd.DataFrame(adata.X, columns=adata.var_names)  # If sparse, use .toarray()
    if selected_variables is None:
        selected_variables = data.columns
    correlations = data[selected_variables].corrwith(data[target_variable])

    # Sort the correlations from highest to lowest
    correlations_sorted = correlations.sort_values(ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(x=correlations_sorted.values, y=correlations_sorted.index, palette='coolwarm')

    plt.title(f'Correlation between {target_variable} and Selected Variables', fontsize=14)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variables')
    plt.grid(True)
    plt.show()

def get_boundary_mask(predictions,dilate_size=0):
    if len(predictions.shape) != 4:
        raise ValueError(f'Predictions must be 4D, got {predictions.shape}')
    boundaries = np.zeros_like(predictions)
    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, ..., 0], connectivity=1, mode='inner')
        # Increase the boundary thickness by dilating it
        boundary_dilated = dilation(boundary, disk(dilate_size))  
        boundaries[img, boundary_dilated > 0, :] = 1
    # squeeze the last dimension
    boundaries = boundaries[...,0]
    return boundaries

# adapted from deepcell.utils.plot_utils.make_outline_overlay
def make_outline_overlay(rgb_data, predictions,dilate_size=0):
    """Overlay a segmentation mask with image data for easy visualization

    Args:
        rgb_data: 3 channel array of images, output of ``create_rgb_data`` (n,h,w,c)
        predictions: segmentation predictions to be visualized (n,h,w,1)

    Returns:
        numpy.array: overlay image of input data and predictions

    Raises:
        ValueError: If predictions are not 4D
        ValueError: If there is not matching RGB data for each prediction
    """
    if len(predictions.shape) != 4:
        raise ValueError(f'Predictions must be 4D, got {predictions.shape}')
    # Handle 3D rgb_data(grayscale) by expanding to 4D
    if len(rgb_data.shape) == 3:
        rgb_data = np.expand_dims(rgb_data, axis=-1)  # Add channel dimension

    if predictions.shape[0] > rgb_data.shape[0]:
        raise ValueError('Must supply an rgb image for each prediction')

    boundaries = np.zeros_like(rgb_data)
    overlay_data = np.copy(rgb_data)
    max_v = 1 if np.max(rgb_data) <= 1 else 255

    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, ..., 0], connectivity=1, mode='inner')
        # Increase the boundary thickness by dilating it
        boundary_dilated = dilation(boundary, disk(dilate_size))  

        boundaries[img, boundary_dilated > 0, :] = 1
        #boundaries[img, boundary > 0, :] = 1

    overlay_data[boundaries > 0] = max_v # max_v
    # If the input was originally 3D, squeeze the last dimension
    if rgb_data.shape[-1] == 1:
        overlay_data = overlay_data[..., 0]

    return overlay_data
def plot_boxplot_stripplot_with_images_singleRow(ax, title,names, distributions, images):
    """
    Plot a boxplot with stripplot and overlay images for given names and distributions.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - names: List of names corresponding to columns/groups.
    - distributions: List of arrays representing distributions. Some elements can be None.
    - images: List of images corresponding to names. Some elements can be None.
    The function ensures that all names are represented on the x-axis, even if their
    corresponding distribution or image is None.
    """
    # Prepare a DataFrame with empty entries for None distributions
    data = []
    labels = []
    for i, dist in enumerate(distributions):
        if dist is not None:
            data.extend(dist)
            labels.extend([names[i]] * len(dist))
        else:
            # Add a placeholder for empty distribution to keep alignment
            labels.append(names[i])
            data.append(None)

    df = pd.DataFrame({'Value': data, 'Group': labels})

    # Plot the boxplot and stripplot
    sns.boxplot(
        x='Group', y='Value', data=df, ax=ax,
        color='lightblue', showfliers=False
    )
    sns.stripplot(
        x='Group', y='Value', data=df, ax=ax,
        jitter=True, color='black', size=4, alpha=0.8
    )

    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Value')
    ax.set_xlabel('Group')

    num_columns = len(names)
    inset_width = 1 / num_columns  # Adjust the width of each inset
    inset_height = inset_width

    for i, name in enumerate(names):
        image = images[i] if i < len(images) else None
        if image is not None:
            # Calculate position of the inset
            inset_x_position = i / num_columns 
            inset_ax = ax.inset_axes(
                [inset_x_position, 1.05, inset_width, inset_height],
                transform=ax.transAxes
            )
            inset_ax.imshow(image)
            inset_ax.axis('off')  # Hide axes for the inset

def plot_boxplots_stripplots(ad_list, ad_names, common_var_names=None, image_dict_list=None):
    """
    For each common molecule, plot a box plot with dots for all AnnData objects in one subplot.
    Additionally, display an image on top of each boxplot.
    
    Parameters:
    - ad_list: List of AnnData objects
    - ad_names: List of names corresponding to each AnnData object
    - common_var_names: List of common molecule names
    - image_dict_list: List of dictionaries containing molecules as keys and corresponding images as values
    """
    if common_var_names is None:
        common_var_names = set(ad_list[0].var_names)
        for adata in ad_list[1:]:
            common_var_names = common_var_names.intersection(set(adata.var_names))
        common_var_names = sorted(common_var_names)
    
    num_molecules = len(common_var_names)
    fig, axes = plt.subplots(num_molecules, 1, figsize=(20, 5 * num_molecules))

    for i, molecule in enumerate(common_var_names):
        # Collect data for the current molecule across all AnnData objects
        data_list = []
        labels = []
        
        for j, adata in enumerate(ad_list):
            expression_values = adata[:, molecule].X.flatten()  # Get expression values for this molecule
            data_list.append(expression_values)
            labels += [ad_names[j]] * len(expression_values)
        
        # Create a DataFrame for seaborn plotting
        df = pd.DataFrame({
            f'{molecule}_Expression': np.concatenate(data_list),
            'Experiment': labels
        })
        
        # Create the boxplot with dots for the current molecule
        sns.boxplot(x='Experiment', y=f'{molecule}_Expression', data=df, ax=axes[i], color='lightblue', showfliers=False)
        sns.stripplot(x='Experiment', y=f'{molecule}_Expression', data=df, ax=axes[i], jitter=True, color='black', size=4)

        # Set title for the molecule
        axes[i].spines['top'].set_visible(False)
        axes[i].set_title(f'{molecule}', fontsize=20)
        axes[i].set_ylabel('intensity')

        # Set the y-limit to 99th percentile to avoid outliers affecting the plot
        y_limit = np.percentile(df[f'{molecule}_Expression'], 99)
        axes[i].set_ylim([0, y_limit])

        # Add image on top of the boxplot
        if image_dict_list is not None:
            num_anndata = len(ad_list)
            inset_width = 1 / (num_anndata)  # Calculate width of each inset based on number of AnnDatas
            inset_height = inset_width  # Keep height the same as width to maintain aspect ratio

            for j in range(num_anndata):
                image = image_dict_list[j].get(molecule)
                if image is not None:
                    # Calculate dynamic position for each inset
                    inset_x_position = j * inset_width 
                    inset_ax = axes[i].inset_axes([inset_x_position-inset_width/2, 1.02, inset_width*2, inset_height*2], transform=axes[i].transAxes)
                    inset_ax.imshow(image) 
                    inset_ax.axis('off')  
    
    plt.tight_layout()
    return fig, axes

def unify_ad_max(ad_list,common_var_names=None):
    if common_var_names is None:
        common_var_names = set(ad_list[0].var_names)
        for adata in ad_list[1:]:
            common_var_names = common_var_names.intersection(set(adata.var_names))
        common_var_names = sorted(common_var_names)

    # not only subset but also reorder the anndata by the order of common_var_names
    reordered_adata_list = []
    for adata in ad_list:
        # Subset and reorder the AnnData by common_vars
        adata_reordered = adata[:, common_var_names].copy()
        common_var_indices = [adata.var_names.get_loc(var) for var in common_var_names]
            
        for key in adata_reordered.varm.keys():
            # Ensure that varm gets reordered according to common_vars using their indices
            adata_reordered.varm[key] = adata.varm[key][common_var_indices]
        reordered_adata_list.append(adata_reordered)

    rangeMax_max = np.max(np.vstack([adata.varm['rangeMax'] for adata in reordered_adata_list]),axis=0)
    for adata in reordered_adata_list:
        adata.varm['rangeMax'] = rangeMax_max
    return reordered_adata_list
def plot_unified_image_dict(ad_list,ad_names,common_var_names=None):
    reordered_adata_list = unify_ad_max(ad_list,common_var_names=common_var_names)
    num_ad = len(ad_list)
    num_molecules = len(ad_list[0].var_names)

    fig,axs = plt.subplots(num_molecules,num_ad,figsize=(20,5*num_molecules))
    for j, (name, adata) in enumerate(zip(ad_names, reordered_adata_list)):
        axs[0, j].set_title(name,fontsize=20)
        image_dict = get_image_dict(adata,spatial_key='spatial_convert')
        for i, molecule in enumerate(common_var_names):
            image, rangeMax = image_dict.get(molecule)
            if image is not None:
                im = axs[i,j].imshow(image,aspect='auto',cmap='jet',vmin=0,vmax=1) ##SPECIFYING VMIN VMAX!!
                #axs[i,j].axis('off')
                # Turn off ticks, but keep the axis frame so labels can show
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

                if j == 0:
                    axs[i, j].set_ylabel(molecule, rotation=90,labelpad=20 ,va='center',fontsize=15)
                if j == num_ad - 1:
                    cbar = fig.colorbar(im, ax=axs[i, :], orientation='vertical', fraction=0.02, pad=0.04)

                    # Multiply colorbar ticks by rangeMax
                    cbar_ticks = cbar.get_ticks()  # Get current ticks
                    cbar.set_ticks(cbar_ticks)  # Set the same ticks
                    cbar.set_ticklabels([f'{tick * rangeMax:.2f}' for tick in cbar_ticks])  # Scale by rangeMax
