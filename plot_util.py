import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar


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


def plot_heatmap_adata(adata):
    if hasattr(adata, "X") and isinstance(adata.X, (pd.DataFrame, np.ndarray)):
        data = pd.DataFrame(adata.X, columns=adata.var_names)
    else:
        raise ValueError("AnnData object must have a valid data matrix in 'X'.")

    # Compute the correlation matrix
    correlation_df = data.corr()
    plot_heatmap(correlation_df)

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

    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Value')
    ax.set_xlabel('Group')

    bbox = ax.get_window_extent()
    axis_aspect_ratio = bbox.width / bbox.height # need to also consider the aspect ratio of the axis because they can represent different pixel sizes

    num_columns = len(names)
    group_width = 1 / num_columns  
    inset_width = group_width * 0.8  
    inset_y_position = 0.85

    for i, name in enumerate(names):
        image = images[i] if i < len(images) else None
        if image is not None:
            inset_x_position = i / num_columns + group_width * 0.1  # Center inset in the group
            aspect_ratio = image.shape[1] / image.shape[0]
            inset_height = inset_width / (aspect_ratio / axis_aspect_ratio)
            # need to scale the height up if ax_width>ax_height because then 0.01 height is less pixels than 0.01 width 
            
            inset_ax = ax.inset_axes(
                [inset_x_position, inset_y_position, inset_width, inset_height],
                transform=ax.transAxes,
                clip_on=False #ensure that the inset is not clipped even though it's outside the axis
            )
            inset_ax.imshow(image,origin='lower',vmin=0,vmax=1) #aspect='auto' #vmin=0,vmax=1 to ensure the uniform color scale
            inset_ax.axis('off')  # Hide axes for the inset
    #ensure title is above the image

    ax.set_title(title, fontsize=16,pad = 50)
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


def unify_ad_max(ad_list, mode='intersection', common_var_names=None):
    """
    Unify multiple AnnData objects based on either the intersection or union of variable names.
    
    Parameters:
    - ad_list: list of AnnData objects
    - mode: 'intersection' (default) or 'union'
    - common_var_names: Optional pre-defined list of common variable names (only used for 'intersection')

    Returns:
    - A list of AnnData objects with harmonized varm['rangeMax'] values.
    """
    if mode == 'intersection':
        var_names = set.intersection(*[set(adata.var_names) for adata in ad_list])
    elif mode == 'union':
        var_names = set.union(*[set(adata.var_names) for adata in ad_list])
    else:
        raise ValueError("mode must be either 'intersection' or 'union'")

    if common_var_names:
        common_var_names = set(common_var_names).intersection(var_names)
    else:
        common_var_names = var_names
    common_var_names = sorted(common_var_names)

    # Prepare reordered AnnData objects
    reordered_adata_list = []
    for adata in ad_list:
        if mode == 'intersection':
            # Subset and reorder AnnData by common_var_names
            adata_reordered = adata[:, common_var_names].copy()
        elif mode == 'union':
            # Create a new AnnData object that includes all common_var_names
            new_X = np.full((adata.shape[0], len(common_var_names)), np.nan)  # Fill missing values with NaN
            existing_indices = [i for i, var in enumerate(common_var_names) if var in adata.var_names]
            adata_indices = [adata.var_names.get_loc(var) for var in common_var_names if var in adata.var_names]
            new_X[:, existing_indices] = adata.X[:, adata_indices]

            # Create new AnnData with the union of variable names
            adata_reordered = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=pd.DataFrame(index=common_var_names))
        
        # Reorder varm according to common_var_names
        common_var_indices = [adata.var_names.get_loc(var) for var in common_var_names if var in adata.var_names]
        for key in adata_reordered.varm.keys():
            if mode == 'intersection':
                adata_reordered.varm[key] = adata.varm[key][common_var_indices]
            elif mode == 'union':
                new_varm = np.full((len(common_var_names), adata.varm[key].shape[1]), np.nan)  # Fill missing with NaN
                if common_var_indices:
                    new_varm[existing_indices] = adata.varm[key][common_var_indices]
                adata_reordered.varm[key] = new_varm
        
        reordered_adata_list.append(adata_reordered)

    # Compute the maximum rangeMax across all datasets
    rangeMax_max = np.nanmax(np.vstack([adata.varm['rangeMax'] for adata in reordered_adata_list]), axis=0)
    for adata in reordered_adata_list:
        adata.varm['rangeMax'] = rangeMax_max

    return reordered_adata_list

def unify_dicts(dict_list, mode='intersection',common_var_names=None):
    """
    Unify a list of dictionaries where each dictionary contains molecules as keys 
    and values as (image, rangeMax).

    Parameters:
    - dict_list: List of dictionaries {molecule: (image, rangeMax)}
    - mode: 'intersection' (only common molecules) or 'union' (all molecules)

    Returns:
    - List of unified dictionaries with updated rangeMax and re-normalized images.
    """
    # Get molecule sets based on mode
    if mode == 'intersection':
        var_names = set.intersection(*[set(d.keys()) for d in dict_list])
    elif mode == 'union':
        var_names = set.union(*[set(d.keys()) for d in dict_list])
    else:
        raise ValueError("mode must be either 'intersection' or 'union")

    if common_var_names:
        common_var_names = set(common_var_names).intersection(var_names)
    else:
        common_var_names = var_names
    common_var_names = sorted(common_var_names)

    # Compute max rangeMax for each molecule
    rangeMax_max = {}
    for molecule in common_var_names:
        rangeMax_max[molecule] = max(
            (d[molecule][1] for d in dict_list if molecule in d), default=np.nan
        )

    # Normalize images using the new max rangeMax
    unified_dicts = []
    for d in dict_list:
        new_dict = {}
        for molecule in common_var_names:
            if molecule in d:
                image, old_rangeMax = d[molecule]
                new_rangeMax = rangeMax_max[molecule]
                # Avoid division by zero
                if new_rangeMax > 0 and old_rangeMax > 0:
                    image = image * (old_rangeMax / new_rangeMax)
                new_dict[molecule] = (image, new_rangeMax)
            else:
                new_dict[molecule] = (None, np.nan)  # Fill missing molecules with None/NaN
        unified_dicts.append(new_dict)

    return unified_dicts



def plot_unified_image_dict(ad_list,ad_names,spatial_key='spatial_convert',\
                            mode='intersection',common_var_names=None\
                            ,plot_size=5, title_size=20, ylabel_size=20):
    image_dict_list = [get_image_dict(adata,spatial_key=spatial_key) \
                       for adata in ad_list]
    unified_image_dict_list = unify_dicts(image_dict_list,mode=mode,common_var_names=common_var_names)
    common_var_names = sorted(unified_image_dict_list[0].keys())
    num_ad = len(ad_list)
    num_molecules = len(common_var_names)
    if num_molecules == 0: 
        raise ValueError('No common features')

    fig,axs = plt.subplots(num_molecules,num_ad,figsize=(plot_size*num_ad,plot_size*num_molecules))
    axs = np.atleast_2d(axs)
    # Determine consistent image shape per dataset
    image_shapes_rangeMax = {}
    for j, image_dict in enumerate(unified_image_dict_list):
        for molecule in common_var_names:
            image, rangeMax = image_dict.get(molecule, (None, None))
            if image is not None:
                image_shapes_rangeMax[j] = image.shape,rangeMax  # Store the first valid shape for each column
                break
    
    for j, (name, image_dict) in enumerate(zip(ad_names, unified_image_dict_list)):
        axs[0, j].set_title(name,fontsize=title_size)
        for i, molecule in enumerate(common_var_names):
            image, rangeMax = image_dict.get(molecule)
            if image is None:
                img_shape,rangeMax = image_shapes_rangeMax.get(j, (100, 100))  # Default to 100x100 if no valid image is found
                image = np.ones(img_shape)
            
            im = axs[i,j].imshow(image,cmap='jet',\
                                    vmin=0,vmax=1)#aspect='auto' ##SPECIFYING VMIN VMAX!!
            #axs[i,j].axis('off')
            # Turn off ticks, but keep the axis frame so labels can show
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            pixel_length=20
            scalebar = ScaleBar(pixel_length, 'um', location='lower right',box_alpha=0,color='white')#μm
            axs[i, j].add_artist(scalebar)
            if j == 0: #1st column
                axs[i, j].set_ylabel(molecule, rotation=90, labelpad=20, va='center', fontsize=ylabel_size)
            if j == num_ad - 1: #last column
                cbar = fig.colorbar(im, ax=axs[i, :], orientation='vertical', fraction=0.02, pad=0.04)

                # Multiply colorbar ticks by rangeMax
                cbar_ticks = cbar.get_ticks()  # Get current ticks
                cbar.set_ticks(cbar_ticks)  # Set the same ticks
                cbar.set_ticklabels([f'{tick * rangeMax:.2f}' for tick in cbar_ticks])  # Scale by rangeMax


    #remove unused subplots
    for i in range(num_molecules):
        for j in range(num_ad):
            if j >= num_ad:
                axs[i,j].set_visible(False)
'''
Whenever trying to unify the color bar for multiple images, besides normalizing with the common rangeMax
always specify vmin and vmax in the imshow function
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import kruskal, f_oneway
from collections import defaultdict

def plot_boxplot_with_dots_compare(
    adata_list, id_list, group_by=None, features=None, num_cols=5, stat_test="anova"
):
    """
    Plots boxplots with overlaid dots for selected features in adata objects, with optional grouping,
    and includes p-values (Kruskal-Wallis or ANOVA) indicating significant differences across groups.

    Parameters:
    - adata_list: List of AnnData objects.
    - id_list: List of dataset IDs corresponding to the adata objects.
    - group_by: Optional dictionary mapping group names to lists of keywords (all keywords must match).
                Example: {'1 Week Group': ['1 week', 'week1'], '24 Hours Group': ['24 hrs', '1 day']}
    - features: Optional list of features to plot. If None, all features will be plotted.
                Example: ['Feature1', 'Feature2']
    - num_cols: Number of columns for the subplot grid. Default is 5.
    - stat_test: Statistical test to perform. Options are 'kruskal' (Kruskal-Wallis) or 'anova' (ANOVA).
    """
    if stat_test not in {"kruskal", "anova"}:
        raise ValueError("stat_test must be 'kruskal' or 'anova'")

    # Step 1: Get the union of features
    all_features = set()
    for adata in adata_list:
        all_features.update(adata.var_names)

    all_features = sorted(all_features)  # Sort for consistency

    # Step 2: Filter provided features and print missing features
    if features is not None:
        missing_features = [feature for feature in features if feature not in all_features]
        if missing_features:
            print(f"The following features are not present in the data: {missing_features}")
        features_to_plot = [feature for feature in features if feature in all_features]
    else:
        features_to_plot = all_features

    # Step 3: Group IDs if group_by is provided
    if group_by:
        grouped_data = defaultdict(list)
        for adata, exp_id in zip(adata_list, id_list):
            matched = False
            for group_name, keywords in group_by.items():
                # Match only if all keywords are present in the ID
                if all(keyword in exp_id for keyword in keywords):
                    grouped_data[group_name].append((adata, exp_id))
                    matched = True
                    break
            if not matched:  # If no group matches, treat it as its own group
                grouped_data[exp_id].append((adata, exp_id))
    else:
        grouped_data = {exp_id: [(adata, exp_id)] for adata, exp_id in zip(adata_list, id_list)}

    # Step 4: Prepare the subplot grid
    num_features = len(features_to_plot)
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten() if num_features > 1 else [axes]

    # Step 5: Plot for each feature in the specified list
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        data_for_plot = []

        # Collect data for the current feature
        for group_name, adata_id_pairs in grouped_data.items():
            for adata, exp_id in adata_id_pairs:
                if feature in adata.var_names:
                    feature_values = adata[:, feature].X.flatten()
                    data_for_plot.append(pd.DataFrame({
                        'Value': feature_values,
                        'Dataset': group_name
                    }))

        # Concatenate data from all datasets for this feature
        if data_for_plot:
            plot_data = pd.concat(data_for_plot, ignore_index=True)

            # Perform the selected statistical test
            groups = [plot_data[plot_data['Dataset'] == group]['Value'] for group in plot_data['Dataset'].unique()]
            if all(len(group) > 1 for group in groups):  # Ensure sufficient data in each group
                try:
                    if stat_test == "kruskal":
                        stat, p_value = kruskal(*groups)
                    elif stat_test == "anova":
                        stat, p_value = f_oneway(*groups)
                except Exception as e:
                    p_value = float('nan')  # If test fails
                    print(f"{stat_test.capitalize()} test failed for feature {feature}: {e}")
            else:
                p_value = float('nan')  # Not enough data for statistical testing

            # Plot the boxplot with overlaid dots
            sns.boxplot(
                data=plot_data,
                x='Dataset',
                y='Value',
                showcaps=True,
                boxprops={'facecolor': 'None'},
                showmeans=True,
                meanline=True,
                meanprops={"color": "red", "ls": "-", "lw": 2},
                showfliers=False,
                ax=ax
            )
            sns.stripplot(
                data=plot_data,
                x='Dataset',
                y='Value',
                color='black',
                alpha=0.6,
                jitter=True,
                size=1,
                ax=ax
            )
            ax.set_title(f'{feature}\nP-Value ({stat_test.capitalize()}): {p_value:.3e}')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Feature Value')
            ax.tick_params(axis='x', rotation=90)
        else:
            ax.set_visible(False)

    # Hide unused subplots
    for ax in axes[len(features_to_plot):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

'''Given a list of adata, show how many featurs are shared between each pair of adata'''
def plot_intersection_matrix(spheroid_names,adata_list):
    var_names_sets = {spheroid_names[i]: set(ad.var_names) for i, ad in enumerate(adata_list)}
    #all_var_names = set.union(*var_names_sets.values())
    # Create a set-to-set overlap matrix
    overlap_matrix = np.zeros((len(var_names_sets), len(var_names_sets)))
    for i, set_i in enumerate(var_names_sets.values()):
        for j, set_j in enumerate(var_names_sets.values()):
            overlap_matrix[i, j] = len(set_i & set_j)  # Intersection size

    plt.figure(figsize=(10, 8))
    plt.imshow(overlap_matrix, cmap="Blues", interpolation="none")
    plt.colorbar(label="Intersection Size")
    plt.xticks(range(len(var_names_sets)), list(var_names_sets.keys()), rotation=90)
    plt.yticks(range(len(var_names_sets)), list(var_names_sets.keys()))
    plt.title("Set Intersection Matrix")
    plt.show()
