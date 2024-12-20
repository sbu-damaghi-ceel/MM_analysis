import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

################ affine matrix operations ################
def readXML_affine_matrix(xml_file_path):
    '''read a 3*4 affine matrix from xml file'''
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
def reverse_affine_matrix(affine_matrix):
    # Reverse the affine transformation matrix
    matrix_square = np.vstack([affine_matrix, [0, 0, 0, 1]])
    return np.linalg.inv(matrix_square)
def apply_multiple_affine(matrix_list):
    '''
    Applies a series of transformation matrices(3x4 or 4x4) in the specified order.
    
    Parameters:
    - matrix_list: list of np.array, each of shape (3, 4), 
    representing transformation matrices in the order they should be applied.
    (The first matrix in the list is applied first.)
    '''
    matrix_list_4x4 = [np.vstack([matrix, [0, 0, 0, 1]]) if
                       matrix.shape == (3, 4)else matrix
                        for matrix in matrix_list ]

    # Start with the last matrix in the list and apply matrices in reverse order since the matrix on the rightmost is multiplied first with the (x,y) array
    result_4x4 = matrix_list_4x4[-1]
    for matrix in reversed(matrix_list_4x4[:-1]):
        result_4x4 = result_4x4 @ matrix

    result_matrix = result_4x4[:3, :4]
    return result_matrix

##########operate on point or image ##########
# point simply transform a point from one coordinate system to another, suitable for translation/rotation
#(image will do interpolation to have enlarged/shrinked pixel blocks, suitabe for when resizing is involved)

def transform_points(points, affine_matrix):
    '''
        points: np.array of shape (n, 2)
        affine_matrix: np.array of shape (3, 4) or (4,4)
        eg. adata.obsm['spatial'] = transform_points(adata.obsm['spatial'], affine_matrix)
    '''
    points_3d = np.hstack([points, np.ones((points.shape[0], 1))])
    affine_matrix_2x3 = affine_matrix[:2, [0, 1, 3]]
    transformed_points = (affine_matrix_2x3 @ points_3d.T).T
    return transformed_points[:, :2]


def transform_image_single(target_shape, transform_img, affine_matrix,interpolation='linear'):
    affine_matrix_2x3 = affine_matrix[:2, [0, 1, 3]]
    # rows, cols = target_img.shape[:2]
    rows, cols = target_shape
    if interpolation == 'linear':
        transformed_img = cv2.warpAffine(transform_img, affine_matrix_2x3, (cols, rows),\
                                     flags=cv2.INTER_LINEAR)
    elif interpolation == 'nearest':
        transformed_img = cv2.warpAffine(transform_img, affine_matrix_2x3, (cols, rows),\
                                     flags=cv2.INTER_NEAREST)
    else:
        raise ValueError('Invalid interpolation method. Must be either "linear" or "nearest".')
    return transformed_img

def transform_image_tuple(target_shape, image_tuple, affine_matrix,interpolation='linear'): 
    image_array, molecule_list, max_values = image_tuple
    transformed_images = []
    for i in range(image_array.shape[0]):
        normalized_image = image_array[i] / max_values[i] # Normalize for the cv2 format
        transformed_image = transform_image_single(target_shape, normalized_image, affine_matrix,\
                                                    interpolation=interpolation)
        transformed_images.append(transformed_image*max_values[i])  # Denormalize after transformation
    return np.stack(transformed_images), molecule_list, max_values

def coreg_merge_img_tuple(base_tuple,coreg_tuple,coreg_affine_matrix,replace=False,plot=False):
                        
    base_image_array, base_molecule_list, base_max_values = base_tuple
    coreg_image_array, coreg_molecule_list, coreg_max_values = coreg_tuple

    
    target_height, target_width = base_image_array[0].shape[:2] 
    for i in range(coreg_image_array.shape[0]):
        if not replace and coreg_molecule_list[i] in base_molecule_list:
            print(f'Duplicate molecule {coreg_molecule_list[i]} found in coregistered image. Skipping...')
            continue
            
        # Resize the coregistered image to match the base image
        affine_matrix_2x3 = coreg_affine_matrix[:2, [0, 1, 3]]
        coreg_image = coreg_max_values[i]*cv2.warpAffine(coreg_image_array[i]/coreg_max_values[i], \
                                     affine_matrix_2x3, (target_width, target_height))
        if replace and coreg_molecule_list[i] in base_molecule_list:
            idx = base_molecule_list.index(coreg_molecule_list[i])
            base_image_array[idx] = coreg_image
            print(f'Overwriting molecule {coreg_molecule_list[i]}')
        else:
            base_image_array = np.append(base_image_array, coreg_image[np.newaxis, ...], axis=0)
            base_molecule_list.append(coreg_molecule_list[i])
            base_max_values.append(coreg_max_values[i])
    if plot :
        example_idx = 0
        base_image = base_image_array[0]  # For demonstration, using first base image as example
        original_image = coreg_image_array[example_idx]
        normalized_image = original_image / coreg_max_values[example_idx]
        converted_image = cv2.warpAffine(normalized_image, affine_matrix_2x3, (target_width, target_height))
        
        # Plot the images
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(base_image, cmap='Blues')
        axes[0].set_title("Base Image")
        axes[0].axis('off')

        axes[1].imshow(original_image, cmap='Reds')
        axes[1].set_title("Original Coregistered Image")
        axes[1].axis('off')

        axes[2].imshow(converted_image, cmap='Reds')
        axes[2].set_title("Converted Image (Warped)")
        axes[2].axis('off')

        axes[3].imshow(base_image, cmap='Blues', alpha=0.5)
        axes[3].imshow(converted_image, cmap='Reds', alpha=0.5)
        axes[3].set_title("Overlay of Base and Converted Image")
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()
    return base_image_array, base_molecule_list, base_max_values

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

############## visualization utilities ################
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
        transformed_img = transform_image_single(base_img, transform_img, affine_matrix)
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

