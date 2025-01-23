import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from matplotlib_scalebar.scalebar import ScaleBar

import anndata as ad
from scipy.stats import linregress
from scipy.stats import ttest_ind

from scipy.spatial import distance

import cv2

import alphashape

from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from scipy.interpolate import make_interp_spline

import h5py

import pdb

def get_mz_mol_mapping(df_feature_list, mz_list):
    df = df_feature_list.copy()
    mapped_molecules = []
    
    for mz in mz_list: #could have duplicates in mz_list
        mask = abs(df['m/z'] - float(mz)) <= df['Interval Width (+/- Da)']
        candidates = df[mask]

        if not candidates.empty:
            # Select the first candidate
            molecule_name = candidates.iloc[0]['Name']
            mapped_molecules.append(molecule_name)
            # Drop the matched row to prevent it from being used again
            df = df.drop(candidates.index[0])
        else:
            mapped_molecules.append(None)

    return mapped_molecules
    
    

def load_data_maldi(int_path,coor_path,feat_path, skip_rows_intensity, skip_rows_coordinates, skip_rows_featureList):
    # Data loading
    ##REQUIRES df_intensity in COL when exporting
    df_intensity = pd.read_csv(int_path, skiprows=skip_rows_intensity, header=0, delimiter=';')
    df_coordinates = pd.read_csv(coor_path, skiprows=skip_rows_coordinates, delimiter=';')
    df_feature_list = pd.read_csv(feat_path,skiprows=skip_rows_featureList, delimiter=';')

    df_coordinates.columns = ['Spot', 'x', 'y']
    # Correcting spot numbering mismatch
    df_coordinates['Spot'] = 'Spot '+ (df_coordinates['Spot'] + 1).astype(str)
    return df_intensity, df_coordinates,df_feature_list

def convert_coordinates(df_coordinates):
    min_x_micro, min_y_micro, max_x_micro, max_y_micro = df_coordinates['x'].min(), df_coordinates['y'].min(), df_coordinates['x'].max(), df_coordinates['y'].max()
    pixel_length_x = df_coordinates['x'].drop_duplicates().sort_values().diff().value_counts().idxmax()
    pixel_length_y = df_coordinates['y'].drop_duplicates().sort_values().diff().value_counts().idxmax()
    df_converted = df_coordinates.copy()
    df_converted['x'] = ((df_converted['x'] - min_x_micro) / pixel_length_x).round().astype(int)
    df_converted['y'] = ((df_converted['y'] - min_y_micro) / pixel_length_y).round().astype(int)
    print(f'y min: {min_y_micro}',f'x min: {min_x_micro}',f'pixel length_y: {pixel_length_y}, pixel_length_x: {pixel_length_x}')
    return df_converted, (min_y_micro, min_x_micro, max_y_micro, max_x_micro), (pixel_length_y, pixel_length_x)
def average_columns(df, group_size=4):
    '''Average every group_size columns in the dataframe for df_intensity when output in 'row' mode from SciLab'''
    num_columns = df.shape[1]
    
    # Create a list to hold the averaged columns
    averaged_columns = []

    # Iterate over the dataframe in steps of group_size
    for i in range(0, num_columns, group_size):
        # Select the group of columns
        group = df.iloc[:, i:i + group_size]
        
        # Compute the mean of the group
        mean_series = group.mean(axis=1)
        
        # Use the name of the first column in the group as the new column name
        new_col_name = df.columns[i]
        
        # Append the resulting series to the list of averaged columns
        averaged_columns.append((new_col_name, mean_series))

    # Create a new dataframe from the averaged columns
    averaged_df = pd.DataFrame({name: data for name, data in averaged_columns})

    return averaged_df
def createAdata_maldi(df_intensity, df_coordinates,df_feature_list,intensity_format='col',thres=95,verbose=False):
    '''
    thres: the percentile threshold to remove molecules with {100*thres}% or more 0 intensities
    OR the number of non-zero pixels < thres 
    '''
    if intensity_format == 'row':
        #convert 1st column of df_intensity to index
        df_intensity.set_index(df_intensity.columns[0], inplace=True,drop=True)
        
        counts = average_columns(df_intensity)
        #handle repeated columns
        counts.columns = [col[:-2] if col.endswith('.1') else col for col in counts.columns]
    else:
        #df_intensity 1st row is 'Spot{i}', 1st column is 'm/z' values
        counts = df_intensity.T 
        counts.columns = counts.iloc[0] #use the 1st row(mz values) as column names
        counts = counts.iloc[1:]

    mapped_molecule = get_mz_mol_mapping(df_feature_list, counts.columns)
    counts.columns=mapped_molecule#convert from mz to molecule names
    counts.columns.name = 'molecule name'
    counts = counts.loc[:, counts.columns.notnull()] # if mz does not have a corresponding name in feature list, it will be None
    
    ####remove molecules with 95% or more 0 intensities OR the number of non-zero pixels < thres
    if thres < 1:
        percentiles = counts.apply(lambda x: np.percentile(x, thres*100), axis=0)
        low_intensity_columns = percentiles[percentiles == 0].index
    else:
        low_intensity_columns = counts.columns[(counts != 0).sum(axis=0) <= thres]
    if verbose:
        for molecule_name in low_intensity_columns:
            print(f'{molecule_name} does not have enough positive pixels')
    counts_filtered = counts.drop(columns=low_intensity_columns)
    
    if counts_filtered.shape[1] == 0:
        raise ValueError('All molecules do not have enough positive pixels')
    adata = ad.AnnData(counts_filtered)
    adata.obs_names = counts_filtered.index #index is 'Spot {i}'
    adata.var_names = counts_filtered.columns 

    #merge by obs_name
    merged_coor = pd.merge(pd.DataFrame({'Spot':list(adata.obs_names)}),
                                        df_coordinates,on='Spot', how='inner')

    adata.obsm['spatial'] = merged_coor[['x','y']].values
    df_converted,_,_ = convert_coordinates(merged_coor)
    adata.obsm['spatial_convert'] = df_converted[['x','y']].values

    #merge by var_names
    merged_featureList = pd.merge(pd.DataFrame({'Name':list(adata.var_names)}),
                                                df_feature_list,on = 'Name', how='inner')
    
    for col in merged_featureList:
        col_1 = col.replace('/','Over')#/ in varm causes loading h5ad problem
        adata.varm[col_1] = np.array(merged_featureList[col].values)

    adata.var_names = [name.replace(":", "-").replace(' ','_').replace('\xa0', '') for name in adata.var_names]
    
    ####For visualization purpose, when need to compare hearmaps of the same molecule across different samples, should unify rangeMax first
    adata.varm['rangeMax'] = np.percentile(adata.X,99,axis=0)#GET RID OF OUTLIERS#np.max(adata.X,axis=0)
    return adata

def create_intensity_image(adata, molecule, spatial_key,norm=True, denoise=False, smooth=True, smooth_method='gaussian', kernel_size=3):
    if molecule in adata.var_names:
        intensities = adata[:,molecule].X.toarray().flatten()
    elif molecule in adata.obs.columns:
        intensities = adata.obs[molecule].to_numpy()
    else:
        raise ValueError(f"Molecule {molecule} not found in AnnData var_names or obs columns")
    # assert intensities is numeric
    assert np.issubdtype(intensities.dtype, np.number)
    max_x, max_y = adata.obsm[spatial_key][:,0].max(), adata.obsm[spatial_key][:,1].max()
    image = np.zeros((int(max_y)+1,int(max_x)+1))

    if 'rangeMax' in adata.varm:
        max = adata.varm['rangeMax'][adata.var_names == molecule][0]
    else:
        max = np.max(intensities)
    
    for i, intensity in enumerate(intensities):
        
        x, y = adata.obsm[spatial_key][i,:]
        image[y, x] = intensity
    
    if norm:
        if max!=0:
            image = np.clip(image / max,0,1)
    if denoise:
        image = cv2.fastNlMeansDenoising(np.array(image, dtype=np.float32), None)
    if smooth:
        if smooth_method == 'gaussian':
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif smooth_method == 'median':
            image = cv2.medianBlur(image, kernel_size)
    return image,max

def get_image_dict(adata,spatial_key,molecule_list=None,verbose=True):
    image_dict = {}
    if molecule_list is None:
        molecule_list = adata.var_names
    for molecule in molecule_list:
        image,max = create_intensity_image(adata,molecule,spatial_key)
        image_dict[molecule] = image,max
        if not verbose:
            print(f'Molecule {molecule} done')
    return image_dict


    
def coreg_merge_img_dict(base_img_dict, *additional_img_dicts_coreg_tuple):
    """
    Merges additional image dictionaries into a base image dictionary. Images in additional
    dictionaries are resized to match the first image in the base dictionary.
    
    Parameters:
        base_img_dict (dict): The base image dictionary to merge into.
        additional_img_dicts_coreg_tuple (tuple of (dict,coreg)): One or more image dictionaries to merge with the base
                            and corresponding coregistration.
            - dict: The image dictionary to merge into the base.
            - coreg: The affine matrix of shape 3*4

    Returns:
        dict: The merged image dictionary with resized images from additional dictionaries.
    """
    
    if not base_img_dict:
        raise ValueError("Base image dictionary is empty. Cannot determine target dimensions for resizing.")
    
    target_image = next(iter(base_img_dict.values()))[0]
    target_height, target_width = target_image.shape[:2]

    merged_img_dict = base_img_dict.copy()
    for img_dict,affine_matrix in additional_img_dicts_coreg_tuple:
        for mz_name, (image,max) in img_dict.items():
            # Ensure the affine matrix is 2x3 for cv2.warpAffine by taking the first two rows and 1,2,4-th colujmns
            affine_matrix_2x3 = affine_matrix[:2, [0, 1, 3]]
            transformeed_image = cv2.warpAffine(image, affine_matrix_2x3, (target_width, target_height))
            merged_img_dict[mz_name] = (transformeed_image,max)
    
    return merged_img_dict

def restore_anndata(image_dict, spatial_key,molecule_names=None):
    if molecule_names is None:
        molecule_names = list(image_dict.keys())
    
    mask,_,_ = getForegroundMask(image_dict)
    all_coords = np.argwhere(mask > 0)[:, [1, 0]]  # Convert to (x, y) format
    
    range_max_list = []
    X = np.zeros((len(all_coords), len(molecule_names)))
    # Populate the intensities and spatial coordinates
    for i,molecule in enumerate(molecule_names):
        image, max_intensity = image_dict[molecule]
      
        range_max_list.append(max_intensity)
        
        # Only consider pixels inside the mask
        intensities = image[mask].flatten()
        X[:,i] = intensities * max_intensity
        

    new_adata = ad.AnnData(X=X)
    # Set the spatial coordinates
    new_adata.obsm[spatial_key] = all_coords

    # Set the varm['rangeMax']
    new_adata.varm['rangeMax'] = np.array(range_max_list)

    # Set the variable names
    new_adata.var_names = molecule_names
    
    print(f'adata shape: {new_adata.X.shape}')
    return new_adata

def plot_images_single(ax,image,rangeMax,key,pixel_length=20):
    #vmin=0 vmax=1 ensure color bar covers between 0 and 1 regardless of the actual image range
    cax = ax.imshow(np.flipud(image), cmap='jet',vmin = 0,vmax=1)#flip upside down
    ax.set_title(key)
    ax.axis('off')

    ############Color bar, show the absolute value(image value*rangeMax)
    cbar = plt.colorbar(cax,ax=ax,orientation='vertical',shrink=.78)
    num_ticks = 5
    ticks = np.linspace(0, 1, num_ticks) 
    cbar.set_ticks(ticks)
    tick_labels = np.round(ticks * rangeMax,decimals=1)#label with the absolute intensit value
    cbar.set_ticklabels(tick_labels)
    
    ###############Scale bar

    scalebar = ScaleBar(20, 'um', location='lower right',box_alpha=0,color='white')#'μm'
    ax.add_artist(scalebar)
    
def plot_images_main(image_dict, cols, keys=None,show=False):
    #PLOT INDIVIDUAL IMAGES FOR EVERY KEY
    if keys is None:
        keys = list(image_dict.keys())
    keys = sorted(keys)
    # Plot individual images
    n = len(keys)
    cols = min(cols, n)
    rows = np.ceil(n / cols).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if n == 1:
        axs = np.array([axs])
    elif axs.ndim == 1: 
        axs = np.expand_dims(axs, axis=0) # Ensure axs is 2D for uniform handling
    for i, key in enumerate(keys):
        ax = axs.flat[i]
        plot_images_single(ax,image_dict[key][0],image_dict[key][1],key)
    for j in range(n, rows * cols):
        axs.flat[j].axis('off')
    plt.tight_layout()
    if not show:
        return fig,axs
    plt.show()
def get_pair_colocalization_score(image_dict,key1,key2):
    # Calculate colocalization_score using Pearson correlation coefficient
    image1_flat = image_dict[key1][0].flatten()
    image2_flat = image_dict[key2][0].flatten()
    
    correlation_matrix = np.corrcoef(image1_flat, image2_flat)
    colocalization_score = correlation_matrix[0, 1]
    
    return colocalization_score

def get_colocal_mat(image_dict):
    mz_names = list(image_dict.keys())
    colocalization_matrix = np.zeros((len(mz_names), len(mz_names)))

    for i, mz_name1 in enumerate(mz_names):
        for j, mz_name2 in enumerate(mz_names):
            if i <= j:  # Correlation matrix is symmetric, no need to calculate twice
                score = get_pair_colocalization_score(image_dict,mz_name1,mz_name2)
                colocalization_matrix[i, j] = score
                colocalization_matrix[j, i] = score  # Mirror the score across the diagonal
    # Convert to DataFrame for easier handling and visualization
    colocalization_df = pd.DataFrame(colocalization_matrix, index=mz_names, columns=mz_names)
    
    return colocalization_df


def pair_plot_single(image_dict,key1,key2, ax1, ax2):
    image1, image2 = image_dict[key1][0], image_dict[key2][0]
    # Create overlaid image
    overlay_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.float32)
    image1_normalized = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2_normalized = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    overlay_image[:, :, 0] = image1_normalized
    overlay_image[:, :, 1] = image2_normalized
    
    
    ax1.imshow(overlay_image)
    ax1.set_title(f'Overlay: {key1} and {key2}')
    ax1.axis('off')
    
    # Create scatter plot
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    plt.scatter(image1_flat, image2_flat, alpha=0.5, color='purple')
    
    # Fit a linear regression line to the data points
    slope, intercept, r_value, _, _ = linregress(image1_flat, image2_flat)
    
    ax2.plot(image1_flat, intercept + slope * image1_flat, 'r', label=f'Fit: r={r_value:.2f}')
    ax2.set_title(f'Colocalization Scatter Plot\n{key1} vs {key2}')
    ax2.set_xlabel(f'Intensity of {key1}')
    ax2.set_ylabel(f'Intensity of {key2}')
    ax2.legend()
    ax2.grid(True)

def pair_plot_main(image_dict, threshold,feat_subset=None):
    colocalization_df = get_colocal_mat(image_dict)
    
    high_pairs = np.where(colocalization_df > threshold)
    # Avoid plotting duplicates or self-comparisons
    if feat_subset is None:
        valid_pairs = [(i, j) for i, j in zip(*high_pairs) if i < j]
    else:
        valid_pairs = [(i, j) for i, j in zip(*high_pairs) if i < j and colocalization_df.columns[j] in feat_subset]
    
    num_pairs = len(valid_pairs)
    row_offset = 2

    height_ratios = [20, 20] + [3] * num_pairs


    fig = plt.figure(figsize=(20, 40 + 3 * num_pairs))
    gs = GridSpec(row_offset + num_pairs, 2, figure=fig, height_ratios=height_ratios)
    axs = []
    ################################ Create heatmap subplot
    
    ax_heatmap = fig.add_subplot(gs[0, :])
    sns.heatmap(colocalization_df, cmap='coolwarm', annot=False, fmt=".2f", ax=ax_heatmap) # 'viridis'/'coolwarm'
    ax_heatmap.set_title('Colocalization Score Heatmap')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=45)
    axs.append([ax_heatmap])

    ax_heatmap_thres = fig.add_subplot(gs[1, :])
    mask = colocalization_df <= threshold
    cmap = sns.color_palette(["gray", "red"])
    sns.heatmap(colocalization_df, mask=mask, cmap=cmap, annot=True, fmt=".2f", ax=ax_heatmap_thres)
    ax_heatmap_thres.set_title('Thresholded Colocalization Score Heatmap')
    ax_heatmap_thres.set_xticklabels(ax_heatmap_thres.get_xticklabels(), rotation=45, ha='right')
    ax_heatmap_thres.set_yticklabels(ax_heatmap_thres.get_yticklabels(), rotation=45)
    axs.append([ax_heatmap_thres])
    
    for idx, (i, j) in enumerate(valid_pairs):
        key1, key2 = colocalization_df.index[i], colocalization_df.columns[j]
        print(f'{key1} and {key2} have a colocalization score of {colocalization_df.iloc[i,j]}')
        ax1 = fig.add_subplot(gs[row_offset + idx, 0])
        ax2 = fig.add_subplot(gs[row_offset + idx, 1])
        pair_plot_single(image_dict, key1, key2, ax1, ax2)
        axs.append([ax1,ax2])
    plt.tight_layout()
    return fig,axs

def getForegroundMask(image_dict,plot_flag=False):
    
    images = [image for image,_ in image_dict.values()]
    # Ensure all images have the same dimensions
    if not all(image.shape == images[0].shape for image in images):
        raise ValueError("All images must have the same dimensions")
    
    mask = np.zeros(images[0].shape, dtype=bool)
    for image in images:
        mask |= (image != 0) 
    points = np.argwhere(mask > 0)  

    # Use alphashape to find the concave hull
    points = points[:, [1, 0]]# Convert to (x, y) format!!
    common_boundary = alphashape.alphashape(points,alpha=0)
    if plot_flag:
        ####visualize the boundary with the 1st image
        fig, ax = plt.subplots()
        ax.imshow(images[0], cmap='gray')
        polygon_points = np.array(common_boundary.exterior.coords)
        ax.plot(polygon_points[:, 0], polygon_points[:, 1], 'r-', linewidth=2) 
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], 'r', alpha=0.3)  
        plt.title('Alpha Shape Polygon Overlay on Image')
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    distances_array = np.full(images[0].shape, np.nan)

    # Only calculate distances for pixels within the mask
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:  # If the pixel is within the combined mask
                point = Point(x, y)
                nearest_geom = nearest_points(common_boundary.boundary, point)[0]
                distance = point.distance(nearest_geom)
                distances_array[y, x] = distance

    distances_flat = distances_array[mask].flatten()

    return mask, common_boundary,distances_flat
def add_distance_obs(adata,spatial_key='spatial'):
    image_dict = get_image_dict(adata,spatial_key=spatial_key)
    mask, common_boundary,distances_flat = getForegroundMask(image_dict)
    distances_array = np.zeros_like(mask, dtype=distances_flat.dtype)
    distances_array[mask] = distances_flat

    #add a new obs column to store the distances
    adata.obs['distances'] = np.zeros(adata.shape[0])
    for i,idx in enumerate(adata.obs.index):
        x,y = adata.obsm[spatial_key][i]
        if 0 <= y < distances_array.shape[0] and 0 <= x < distances_array.shape[1]:
            adata.obs.loc[idx, 'distances'] = distances_array[y, x]#images are x,y flipped
        else:
            print(f"Warning: Coordinates ({x},{y}) are out of bounds for distances_array")
    return
def dist2Bdry_plot_single(image, molecule_name, distances_flat, intensities_flat, distance_threshold, common_boundary, pixel_len,axs):
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    
    #scatter plot
    # axs[0].scatter(distances_flat, intensities_flat, alpha=0.5)
    # axs[0].set_xlabel('Distance to boundary')
    # axs[0].set_ylabel('Pixel Intensity')
    # axs[0].set_title(f'Distance vs. Intensity for {molecule_name}')
    # axs[0].grid(True)


    ####histogram and KDE fitted curve
    n_bins = 30  
    _, bin_edges = np.histogram(distances_flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # mean intensities for each bin
    mean_intensities = np.zeros(n_bins)
    for i in range(n_bins):
        bin_mask = (distances_flat >= bin_edges[i]) & (distances_flat < bin_edges[i+1])
        if np.any(bin_mask):  
            mean_intensities[i] = np.mean(intensities_flat[bin_mask])
        else:
            mean_intensities[i] = np.nan  
    valid_bins = ~np.isnan(mean_intensities)
    x_smooth = bin_centers[valid_bins]
    y_smooth = mean_intensities[valid_bins]

    # Creating a spline of x and y
    spl = make_interp_spline(x_smooth, y_smooth, k=3)  # k is the degree of the spline
    xnew = np.linspace(x_smooth.min(), x_smooth.max(), 300)  # 300 represents number of points to make between x.min and x.max
    ynew = spl(xnew)

    # Plotting the smoothed curve
    axs[0].bar(bin_centers, mean_intensities, width=bin_edges[1] - bin_edges[0], alpha=0.6, label='Mean Intensity per Bin')
    axs[0].plot(xnew, ynew, 'r-', label='Smoothed Curve')
    axs[0].set_xlabel('Distance to Boundary')
    axs[0].set_ylabel('Mean Intensity')
    xticks = axs[0].get_xticks()
    xticks_scaled = xticks * pixel_len  # Scale x-ticks by pixel_len
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(['']+[f"{tick:.0f}" for tick in xticks_scaled[1:-1]]+[''])
    #axs[0].set_title(f'Mean Intensity per Distance Bin for {molecule_name}')
    axs[0].legend()

    distances_less_than_threshold = intensities_flat[distances_flat < distance_threshold]
    distances_greater_than_threshold = intensities_flat[distances_flat >= distance_threshold]
    
    data_to_plot = [distances_less_than_threshold, distances_greater_than_threshold]
    axs[1].boxplot(data_to_plot, showfliers = False,patch_artist=True)
    axs[1].set_xticklabels([f'< {distance_threshold*pixel_len} µms', f'>= {distance_threshold*pixel_len} µms'])
    #axs[1].set_title('Intensity Distribution by Distance Threshold')
    axs[1].set_ylabel('Intensity')
    t_stat, p_value = ttest_ind(distances_less_than_threshold, distances_greater_than_threshold, equal_var=False)
    #t test
    p_val_color = 'red' if p_value < 1e-4 else 'black'
    axs[1].text(1.5, axs[1].get_ylim()[1] * 0.9,
    f'p = {p_value:.2e}', color=p_val_color, horizontalalignment='center', verticalalignment='top')


    axs[2].imshow(image, cmap='jet',vmin=0,vmax=255)##Ensure the color range is uniform for different images of the same molecule
    polygon_points = np.array(common_boundary.exterior.coords)
    axs[2].plot(polygon_points[:, 0], polygon_points[:, 1], 'r-', linewidth=2) 
    #axs[2].fill(polygon_points[:, 0], polygon_points[:, 1], 'r', alpha=0.3) 
    inner_points = np.array(common_boundary.buffer(-distance_threshold).exterior.coords)
    axs[2].plot(inner_points[:, 0], inner_points[:, 1], 'y-', linewidth=2) 
    #axs[2].fill(inner_points[:, 0], inner_points[:, 1], 'r', alpha=0.2) 
    axs[2].set_title(f'{molecule_name}')

        

def dist2Bdry_plot_main(image_dict,distance_threshold,pixel_len=20,keys=None):
    if keys is None:
        keys = list(image_dict.keys())
    keys = sorted(keys)
    
    image_dict = {key:image_dict[key] for key in keys}
    mask,common_boundary,distances_flat = getForegroundMask(image_dict)
    common_boundary_points = np.array(common_boundary.exterior.coords)

    fig, axs = plt.subplots(len(image_dict), 3, figsize=(12, 3 * len(image_dict)))
    if len(image_dict) == 1:
        axs = np.array([axs])
    elif len(axs) == 1:
        axs = np.expand_dims(axs, axis=0)  # Ensure axs is 2D for uniform handling
    for idx,key in enumerate(keys):
        molecule_name = key
        (image,max) = image_dict[key]
    
        axs_row = axs[idx,:]
        intensities_flat = image[mask].flatten()
        dist2Bdry_plot_single(image, molecule_name, distances_flat, intensities_flat, distance_threshold, common_boundary, pixel_len, axs_row)
    plt.tight_layout()
    return fig,axs



##########################################Tuple format to store image data
def get_image_tuple(adata, spatial_key, h5_filename=None, molecule_list=None, verbose=False):
    if molecule_list is None:
        molecule_list = list(adata.var_names)
    images = []
    max_values = []

    for molecule in molecule_list:
        # norm=False to store the raw intensities
        image, max_value = create_intensity_image(adata, molecule, spatial_key,norm=False)
        images.append(image)
        max_values.append(max_value)
        if verbose:
            print(f'Molecule {molecule} done')

    image_array = np.stack(images)
    image_tuple = (image_array, molecule_list, max_values)
    if h5_filename is not None:
        save_image_tuple(image_tuple, h5_filename)
    return image_tuple
def save_image_tuple(image_tuple, h5_filename):
    image_array, molecule_list, max_values = image_tuple
    with h5py.File(h5_filename, 'w') as hf:
        
        hf.create_dataset('images', data=image_array)
        hf.create_dataset('molecule_names', data=np.array(molecule_list, dtype='S'))  # Save molecule names as bytes
        hf.create_dataset('max_values', data=np.array(max_values))

    print(f"Data successfully saved in {h5_filename}")
    return
def load_image_tuple(h5_filename,load='all'):
    with h5py.File(h5_filename, 'r') as hf:
        if load == 'images':
            return hf['images'][:]
        elif load == 'molecule_names':
            return [m.decode('utf-8') for m in hf['molecule_names'][:]]  # Decode bytes to strings
        elif load == 'max_values':
            return hf['max_values'][:]
        elif load == 'all':
            image_array = hf['images'][:]
            molecule_list = [m.decode('utf-8') for m in hf['molecule_names'][:]]
            max_values = hf['max_values'][:]
            return (image_array, molecule_list, max_values)
        else:
            raise ValueError(f"Invalid value {load} for 'load'. Must be 'all', 'images', 'molecule_names', or 'max_values'.")

def lazy_load_image_by_molecule(h5_filename, molecule_name):
    """
    Lazy load a specific image array from the HDF5 file based on the molecule name.
    """
    with h5py.File(h5_filename, 'r') as hf:
        molecule_names = [m.decode('utf-8') for m in hf['molecule_names'][:]]
        if molecule_name not in molecule_names:
            raise ValueError(f"Molecule '{molecule_name}' not found in the file.")
        index = molecule_names.index(molecule_name)
        # Lazy load the image array using the index
        image = hf['images'][index, ...]
        return image

def getForegroundMask_tuple(image_tuple,plot_flag=False):
    image_array, molecule_list, max_values = image_tuple
    
    mask = np.zeros(image_array[0].shape, dtype=bool)
    for i in range(image_array.shape[0]):
        image = image_array[i]/max_values[i]
        mask |= (image != 0) 
    points = np.argwhere(mask > 0)  

    # Use alphashape to find the concave hull
    points = points[:, [1, 0]]# Convert to (x, y) format!!
    common_boundary = alphashape.alphashape(points,alpha=0)
    if plot_flag:
        ####visualize the boundary with the 1st image
        fig, ax = plt.subplots()
        ax.imshow(image_array[0], cmap='gray')
        polygon_points = np.array(common_boundary.exterior.coords)
        ax.plot(polygon_points[:, 0], polygon_points[:, 1], 'r-', linewidth=2) 
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], 'r', alpha=0.3)  
        plt.title('Alpha Shape Polygon Overlay on Image')
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    distances_array = np.full(image_array[0].shape, np.nan)

    # Only calculate distances for pixels within the mask
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:  # If the pixel is within the combined mask
                point = Point(x, y)
                nearest_geom = nearest_points(common_boundary.boundary, point)[0]
                distance = point.distance(nearest_geom)
                distances_array[y, x] = distance

    distances_flat = distances_array[mask].flatten()

    return mask, common_boundary,distances_flat





def restore_anndata_from_tuple(image_tuple,spatial_key,molecule_names=None):
    image_array, molecule_list, max_values = image_tuple
    if molecule_names is None:
        molecule_names = molecule_list
    
    mask,_,_ = getForegroundMask_tuple(image_tuple)
    all_coords = np.argwhere(mask > 0)[:, [1, 0]]  # Convert to (x, y) format
   
    X = np.zeros((len(all_coords), len(molecule_names)))
    # Populate the intensities and spatial coordinates
    for i in range(image_array.shape[0]):
        image = image_array[i]
        
        # Only consider pixels inside the mask
        intensities = image[mask].flatten()*max_values[i]
        X[:,i] = intensities
        

    new_adata = ad.AnnData(X=X)
    # Set the spatial coordinates
    new_adata.obsm[spatial_key] = all_coords
    # Set the varm['rangeMax']
    new_adata.varm['rangeMax'] = np.array(max_values)
    # Set the variable names
    new_adata.var_names = molecule_names
    
    print(f'Restored adata shape: {new_adata.X.shape}')
    return new_adata