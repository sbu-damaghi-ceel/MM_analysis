import numpy as np
import pandas as pd
import shapely
import shapely.plotting
from shapely import Polygon,MultiPolygon,Point

import matplotlib.pyplot as plt
import matplotlib

import tps
#tps code is from https://github.com/tzing/tps-deformation

def apply_affine_transformation(points, affine_matrix):
    #affine matrix can be either 3*3 or 3*4
    homogeneous_points = np.column_stack((points, np.ones((len(points), affine_matrix.shape[1] - 2))))
    transformed_points = np.dot(homogeneous_points, affine_matrix.T)
    return transformed_points[:, :2]
def trans_multiPolygon(multiPoly,trans):
    '''
        recursively handle the shapely tranform error
        TO DO: transform interior too
    '''

    try:
        transformed_multiPoly = shapely.transform(multiPoly,trans)
    except Exception as e:
        print(f"MultiPoly transformation failed : {e}")
        transformed_geoms = []
        for j,poly in enumerate(multiPoly.geoms):
            try:
                transformed_poly = shapely.transform(poly,trans)
                transformed_geoms.append(transformed_poly)
            except Exception as e:
                print(f"Polygon Transformation failed for geoms{j}: {e}")
                xy_tuple = poly.exterior.xy
                xy = np.column_stack(xy_tuple)
                transformed_geoms.append(Polygon(trans(xy)))
        transformed_multiPoly = MultiPolygon(transformed_geoms)
    return transformed_multiPoly

def plot_dict(dict_obj,cols,plt_func):
    n = len(dict_obj.items())
    cols = min(cols, n)
    rows = np.ceil(n / cols).astype(int)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    for i,(key,value) in enumerate(dict_obj.items()):
        
        ax = axes.flat[i]
        plt_func(ax,value)
        ax.set_title(f'{key}')
    for j in range(n,rows*cols):
        axes.flat[j].axis('off')
    plt.tight_layout()
    return fig,axes

def plot_multiPoly(ax,multiPoly):
    patch = shapely.plotting.patch_from_polygon(multiPoly)
    ax.add_patch(patch)
    ax.autoscale()
    ax.set_aspect('equal', 'box')



def plot_poly_overlay(ax,adata,multiPoly,key=None,options=None):
    if options is None:
        options = {}
    patch_color = options.get('patch_color', 'blue')
    patch_alpha = options.get('patch_alpha', 0.5)
    scatter_size = options.get('scatter_size', 10)
    title = options.get('title','')
    title_size = options.get('title_size', 10)
    
    if 'color_map' not in options:
        unique_labels = np.sort(np.unique(adata.obs[key]))
        cmap = matplotlib.colormaps.get_cmap('viridis')  # viridis/plasma/coolwarm
        n = len(unique_labels)
        color_map = {label: cmap(i*cmap.N//n) for i, label in enumerate(unique_labels)}
    else:
        color_map = options['color_map']

    
    if key is None:
        ax.scatter(adata.obsm['spatial'][:,0],adata.obsm['spatial'][:,1],color='blue')
    else:
        for class_label in adata.obs[key].unique():
            x_values = adata.obsm['spatial'][adata.obs[key] == class_label, 0]
            y_values = adata.obsm['spatial'][adata.obs[key] == class_label, 1]
            ax.scatter(x_values, y_values, label=class_label,s=scatter_size,color=color_map.get(class_label, 'gray'))
    patch = shapely.plotting.patch_from_polygon(multiPoly)
    patch.set(fill=True,alpha=patch_alpha,color=patch_color,zorder=10)

    ax.add_patch(patch)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    ax.set_title(title,fontsize=title_size)

    ax.invert_yaxis()

def classify_cells_byMALDI(adata,multiPoly_dict,molecule,ordered_keys,labels):
    xy = adata.obsm['spatial']
    adata.obs[molecule] = pd.Series([None] * adata.shape[0])
    for i, (x, y) in enumerate(xy):
        point = Point(x, y)
        classified = False
        for key, label in zip(ordered_keys, labels):
            if point.within(multiPoly_dict[key].buffer(0)):  #to make it valid
                adata.obs.loc[adata.obs.index[i],molecule] = label
                classified = True
                break
        if not classified:
            adata.obs.loc[adata.obs.index[i],molecule] = 'Low'