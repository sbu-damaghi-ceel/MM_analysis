import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_poly
from matplotlib.collections import PatchCollection

from shapely.geometry import Polygon, MultiPolygon
import shapely

def extractContour(image,percentile=50,title=None):
    #cv requires image to be 0-255 scale
    if np.max(image) <= 1:
        image = (image * 255).astype(np.uint8)
    threshold_value = np.percentile(image[image>0], percentile)#percentile of non-zero pxiels
    threshold_value = int(threshold_value)

    ret, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.RETR.EXTERNAL
    
    return contours,hierarchy

def Contour2MultiPoly(contours,hierarchy,area_threshold=100):
    hierarchy = hierarchy[0]  # Get the first hierarchy list(should only be one)
    polygons = []
    for i, contour in enumerate(contours):
        if hierarchy[i][3] == -1:  # No parent contour, it's an outer boundary
            polygon = construct_polygon(contour, contours, hierarchy, i,area_threshold)
            if polygon and not polygon.is_empty:
                polygons.append(polygon)
    
    # Create a MultiPolygon from the list of polygons
    multi_polygon = MultiPolygon(polygons)
    return multi_polygon

def construct_polygon(contour, contours, hierarchy, current_index,area_threshold):
    #print(current_index)
    if contour.shape[0] < 4: 
        return None #contour too small to make a polygon
    exterior = np.squeeze(contour)
    if Polygon(exterior).area < area_threshold:
        #print(f'polygon area {Polygon(exterior).area}, too small')
        return None
    interiors = []
    child_index = hierarchy[current_index][2]  # Index of the first child
    while child_index != -1:
        child_contour = contours[child_index]
        # Recursively construct polygon for the child, which might have its own children
        child_polygon = construct_polygon(child_contour, contours, hierarchy, child_index,area_threshold)
        # Subtract child's polygon (which includes its own nested structures)
        if child_polygon and not child_polygon.is_empty:
            interiors.append(child_polygon.exterior) #interiors should be LINEAR RINGS
        child_index = hierarchy[child_index][0]  # Sibling of the child
    if interiors:
        return Polygon(exterior,interiors)
    else:
        return Polygon(exterior)

def plot_single_overlay(ax,image,contours=None,multiPoly=None):
    ax.imshow(image,cmap='jet')
    if contours is not None:
        for contour in contours:
            poly = plt_poly(contour.reshape(-1, 2),edgecolor='red',fill=False)
            ax.add_patch(poly)
    elif multiPoly is not None:
        patch = shapely.plotting.patch_from_polygon(multiPoly)
        patch.set(fill=True,alpha=.5,color='red',edgecolor = 'red',zorder=10)
        ax.add_patch(patch)
        ax.autoscale()
        ax.set_aspect('equal', 'box')
    return

