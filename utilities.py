import numpy as np
import numpy.ma as ma
import time

import tifffile
from aicsimageio import AICSImage
import matplotlib.pyplot as plt
import scipy
import skimage

from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity, histogram
from skimage.morphology import skeletonize
from skimage import filters

# skeleton analysis
from scipy.ndimage import convolve, label, distance_transform_edt, generate_binary_structure, binary_dilation
import networkx as nx
from networkx.algorithms.components import connected_components

# geodesic distance
import skfmm

import h5py

# dataframe
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import itertools as it

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import math

from collections import Counter

import requests
from pathlib import Path

###########################################################
'''input utilities'''
###########################################################
# read image and get the physical resolution of the pixels
def get_picture(path):
    '''
    reading picture and metadata

    output[0] picture
    output[1] spatial relotuion in x, y and z
    '''
    obj = AICSImage(path)
    #print(f'size (y, x, z): {img.GetSize()}')

    # axis order before t - c - z - y - x
    img = obj.data
    # axis order after x - y - z
    return (np.transpose(img[0, 0, :, :, :], axes=(1, 2, 0)), (obj.physical_pixel_sizes.X, obj.physical_pixel_sizes.Y, obj.physical_pixel_sizes.Z))

def read_h5(filepath):
    with h5py.File(filepath, 'r') as h5_file:
        a_group_key = list(h5_file.keys())[0]

        dataset = h5_file[a_group_key][()]
        dataset = np.squeeze(dataset)

    return np.transpose(dataset, (1, 2, 0))

def get_zenodo(filename_prefix, output_dir = Path.cwd() / 'example_data'):
    doi = '10.5281/zenodo.16037032'


    output_dir.mkdir(parents=True, exist_ok=True)


    record_id = doi.split('.')[-1]
    api_url = f'https://zenodo.org/api/records/{record_id}'


    response = requests.get(api_url)

    data = response.json()
    files = data.get('files', [])

    matching_files = [f for f in files if f['key'].startswith(filename_prefix)]


    print(f'{len(matching_files)} file - downloading to {output_dir}')

    for file in matching_files:
        file_name = file['key']
        file_url = file['links']['self']
        file_path = output_dir / file_name

        print(f'downloading: {file_name}')
        r = requests.get(file_url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print('download complete')

    return str(output_dir)

###########################################################
'''plotting utilities'''
###########################################################
# maximum projection
def max_proj(img):
    return np.amax(img, axis=2)

def coord_shower(coord_list, pic, radius = 5):
    mip = np.max(pic, axis = 2)

    fig, ax = plt.subplots()

    # Display the MIP image
    ax.imshow(mip)

    # Step 3: Overlay circles on the projection
    # The circles are projected onto the y-x plane (ignoring the z-coordinate)
    for coord in coord_list:
        y, x = coord[0], coord[1]  # Extract the y, x from the 3D coordinates
        circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
    
    plt.show()

def create_overlay(img, color):
    max_img = max_proj(img).astype(bool)
    overlay = np.zeros((max_img.shape[0], max_img.shape[1], 4))
    overlay[max_img] = [*color, 1]

    return overlay

def show_pic(pic, title = '', palette = 'viridis', show_legend = False):
    img = plt.imshow(max_proj(pic), cmap = palette)
    if show_legend:
        plt.colorbar(img)
    plt.title(title)
    plt.show()

# plot the network with edge weights
def plot_graph_from_skeleton(G, title="Skeleton Graph with Edge Weights", node_size=2):
    # Create a figure for the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions of nodes
    pos = {node: node for node in G.nodes()}  # 3D positions are the node coordinates themselves
    
    # Draw nodes (at their 3D coordinates)
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs, c='r', s=node_size, label="Nodes")
    
    # Draw edges with thickness proportional to the edge weight (length)
    for edge in G.edges(data=True):
        n1, n2, edge_data = edge
        weight = edge_data['weight']
        
        # Get node positions for the edge
        x_coords = [pos[n1][0], pos[n2][0]]
        y_coords = [pos[n1][1], pos[n2][1]]
        z_coords = [pos[n1][2], pos[n2][2]]
        
        # Use weight to set line width (scale appropriately for visualization)
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=weight * 2, alpha=0.6, label=f'Edge weight: {weight:.2f}')
    
    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    
    plt.show()

# plot the network with edge weights
def plot_graph_from_skeleton_2D(G):
    fig = plt.figure(figsize=(10, 10))
    
    # Extract positions of nodes
    pos = {node: node for node in G.nodes()}
    
    # Draw nodes (at their 3D coordinates)
    ys = [pos[node][0] for node in G.nodes()]
    xs = [pos[node][1] for node in G.nodes()]
    plt.scatter(xs, ys, s= 1)

    center_coord = np.array(from_graph_to_element_centers(G))
    plt.scatter(center_coord[:,1], center_coord[:,0], c='r')

    
    # Draw edges with thickness proportional to the edge weight (length)
    for edge in G.edges(data=True):
        n1, n2, edge_data = edge
        
        # Get node positions for the edge
        y_coords = [pos[n1][0], pos[n2][0]]
        x_coords = [pos[n1][1], pos[n2][1]]
        
        # Use weight to set line width (scale appropriately for visualization)

        
        plt.plot(x_coords, y_coords, c='k', linewidth=1)
    
    plt.axis('equal')
    plt.gca().invert_yaxis()
    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()

# plot a labeled graph
def plot_lab_graph(G, l):
    weights = nx.get_node_attributes(G, l)
        
    norm = plt.Normalize(vmin=min(weights.values()), vmax=max(weights.values()))
    colors = [plt.cm.viridis(norm(w)) for w in weights.values()]


    pos = {node: (node[1], -node[0]) for node in G.nodes}
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=10)

    plt.show()


###########################################################
'''images utilities'''
###########################################################
# setting up the proper structure for the geodesic distance
def zero_contour(mask, marker):
    '''
    mask is boolean
    marker is whatever value, with zero outside
    '''

    mask = ~mask.astype(bool)
    marker = ~marker.astype(bool)
    
    m = ma.masked_array(marker, mask)
    return m

# proper geodesic distance
def geodesic_dist(mask, marker, res):
    '''https://stackoverflow.com/questions/28187867/geodesic-distance-transform-in-python'''
    return skfmm.distance(zero_contour(mask, marker), dx= res)

# from list of pixels to mask
def pixels_to_mask(img, pixels):
    mask = np.zeros_like(img, dtype=np.uint8)

    for coord in pixels:
        mask[coord] = 1

    return mask.astype(bool)

# get the bounding box containing the cell
def bounding_box(cell_mask):
    true_indices = np.where(cell_mask)

    min_x, max_x = true_indices[0].min(), true_indices[0].max()
    min_y, max_y = true_indices[1].min(), true_indices[1].max()
    min_z, max_z = true_indices[2].min(), true_indices[2].max()


    bounding_box = (
        slice(min_x, max_x + 1),
        slice(min_y, max_y + 1),
        slice(min_z, max_z + 1)
    )

    return bounding_box

# remove random non connected components and keeps only the biggest one
def pixels_cleaning(pic):
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    labeled_array, n_components = label(pic, structure=structure)

    actual_obj_pixels = 0
    actual_obj = 0
    for comp in range(1, n_components+1):
        current_comp = labeled_array == comp
        current_tot_pix = np.sum(current_comp)

        if current_tot_pix > actual_obj_pixels:
            actual_obj_pixels = current_tot_pix
            actual_obj = comp
    
    return labeled_array == actual_obj

# finding the labels associated with a cell (pixels of that cell will have that specific value)
def get_masks(img):
    hist = histogram(img)[0]

    labels = np.where(hist > 0)[0][1:]

    # voxels contains information about how many voxels for every cell
    voxels = list()
    masks = list()

    # masks containing True for every pixel belonging to the cell
    for i in labels:
        voxels.append(int(hist[i]))
        masks.append(img == i)

    # labels, how many voxels per label, array of masks
    return (labels, np.array(voxels), np.array(masks))

# setting the pixels outside the mask to 0
def apply_mask(img, mask):
    result = np.copy(img)
    return result * mask

def connected_annotation(vol, seed):
    # PROCESSES AS CONNECTED COMPONENTS
    # cell mask outside soma
    processes = ma.masked_array(vol, seed)

    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    # masked array needs to be transformed into regular array for the connected components to be properly recognized
    labeled_array, num_components = label(processes.filled(fill_value=0), structure=structure)
    #plt.imshow(max_proj(labeled_array))
    #plt.title('labelled components')
    #plt.show()

    return labeled_array, num_components


###########################################################
'''compartment annotation'''
###########################################################

def cell_annotation(cell_mask, cell_soma_mask):
    '''
    returns:
        annotate image
            background is 0
            soma is 1
            every process is annotated with a different label
        number of subcellular components
    '''
    
    labeled_processes, num_components = connected_annotation(cell_mask.astype(bool), cell_soma_mask.astype(bool))
    labeled_cell = np.zeros_like(labeled_processes)
    labeled_cell[cell_mask] = labeled_processes[cell_mask] + 1
    labeled_cell[cell_soma_mask] = 1

    return labeled_cell, num_components + 1

def apical_progenitor_compartment_annotation(cell_mask, cell_soma_mask, soma_dist, vent_dist, threshold = 2):
    '''
    returns
    labelled compartments
        1 - apical soma
        2 - basal soma
        3 - apical process
        4 - basal process
    distance of soma from the ventricle
    '''
    
    vent_dist_on_soma = ma.masked_array(vent_dist, ~cell_soma_mask)
    cell_dist = (np.max(vent_dist_on_soma) + np.min(vent_dist_on_soma)) / 2

    # PROCESSES AS CONNECTED COMPONENTS
    labeled_array, num_processes = connected_annotation(cell_mask, cell_soma_mask)

    AB_masks = np.zeros_like(labeled_array)
    apical_mask = np.full_like(labeled_array, fill_value=False, dtype=bool)
    basal_mask = np.full_like(labeled_array, fill_value=False, dtype=bool)
    for process in range(1, num_processes+1):
        process_mask = labeled_array == process

        #process_on_soma_dist = ma.masked_array(soma_dist, ~process_mask)
        process_on_soma_dist = apply_mask(soma_dist, process_mask)
        process_length = np.max(process_on_soma_dist)

        process_min = np.min(process_on_soma_dist)

        # filter out processes that are too little
        # and merge them into the soma
        if process_length < threshold:
            soma_dist[process_mask] = 0
        else:
            # keeping the portion of the process attached to the soma
            start_process = (process_on_soma_dist > 0) & (process_on_soma_dist < threshold)
            # checking where this portion is in relation to the ventricle
            start_process_on_vent_dist = ma.masked_array(vent_dist, ~start_process)

            if np.min(start_process_on_vent_dist) < cell_dist:  # apical
                AB_masks = ma.masked_array(AB_masks, process_mask).filled(fill_value=3)
            else:
                AB_masks = ma.masked_array(AB_masks, process_mask).filled(fill_value=4)
        

    cell_soma_mask = soma_dist == 0
    cell_soma_mask = cell_soma_mask.filled(fill_value=0)
    #show_pic(cell_soma_mask, 'cell_soma_mask')


    # 1 - apical soma
    apical_soma_mask = apply_mask(vent_dist < cell_dist, cell_soma_mask)
    AB_masks[apical_soma_mask] = 1

    # 2 - basal soma
    basal_soma_mask = apply_mask(vent_dist >= cell_dist, cell_soma_mask)
    AB_masks[basal_soma_mask] = 2

    #show_pic(AB_masks)

    return AB_masks, cell_dist



###########################################################
'''core analysis'''
###########################################################
def picture_analysis(folder_name, picture_name, mito_suffix, mito_bin_suffix, cell_suffix, soma_suffix, vent_suffix, info, h5_flag = False):
    '''
    function for mitochondrial apical radial glia analysis in picture

    the compartments are based on the ventricular distance and apico-basal position
    '''

    # read mitochondrial channel
    img_mito, res = get_picture(folder_name + '/' + picture_name + mito_suffix)
    voxel_vol = np.prod(res)

    # cell segmentation
    img_cell = get_picture(folder_name + '/' + picture_name + cell_suffix)[0]
    # soma
    img_soma = get_picture(folder_name + '/' + picture_name + soma_suffix)[0]
    # ventricle
    img_vent = get_picture(folder_name + '/' + picture_name + vent_suffix)[0]

    img_mito_bin = read_h5(folder_name + '/' + picture_name + mito_bin_suffix)

    info_cell = get_masks(img_cell)
    cell_labels = info_cell[0]
    cell_masks = info_cell[2]

    n_cells = len(cell_labels)
    print(' - ', end='')
    print(cell_labels, end='')

    # euclidean distance used for the ventricle
    vent_dist = distance_transform_edt(~img_vent, sampling= res)
    print(' - ventricular distance DONE')

    
    ###################################################

    # creating result dataframes
    results_pic = pd.DataFrame()
    results_mito_pic = pd.DataFrame()
    a_process_df = pd.DataFrame()
    b_process_df = pd.DataFrame()

    for c in range(n_cells):
        
        # mask of current cell
        cell_mask = cell_masks[c]

        cell_box = bounding_box(cell_mask)

        # cropping pic to size of the cell we are considering
        cell_mask = cell_mask[cell_box]
        #

        # removing non connected random pixels
        cell_mask = pixels_cleaning(cell_mask)

        # value of current cell
        cell_label = cell_labels[c]
        print(f'--- {cell_label}', end='')

        #
        vent_dist_subset = vent_dist[cell_box]

        # soma within the cell
        img_soma_subset = img_soma[cell_box]
        cell_soma_mask = ma.masked_array(img_soma_subset == cell_label, ~cell_mask.astype(bool)).filled(fill_value=False)

        # removing non connected random pixels
        cell_soma_mask = pixels_cleaning(cell_soma_mask)

        # binarized mitochondria
        '''
        binarized mitochondria could be obtained in program from mito_raw
        '''



        mito_bin = img_mito_bin[cell_box]

        '''fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(max_proj(cell_mask))
        axes[0].axis('off')
        axes[1].imshow(max_proj(mito_bin))
        axes[1].axis('off')

        plt.show()'''


        mito_bin = apply_mask(mito_bin, cell_mask).astype(bool)
        
        tot_mito_pixels = np.sum(mito_bin.astype(bool))

        # computing distance from soma
        soma_dist = geodesic_dist(cell_mask, cell_soma_mask, res)
    
        print(' - distance', end='')

        # cell_dist is the average between max and min of soma dist from the ventricle
        comp_labels, cell_dist = apical_progenitor_compartment_annotation(cell_mask, cell_soma_mask, soma_dist, vent_dist_subset)


        cell_tables = cell_analysis(mito_bin, comp_labels, soma_dist, res)

        results_cell = cell_tables[0]
        results_mito_cell = cell_tables[1]

        # these are values common for all the rows of the same cell, so let's add them at the end
        #################################

        results_mito_cell['cell_label'] = cell_label
        results_mito_cell['pict_code'] = picture_name
        results_mito_cell['cell_code'] = picture_name + '_' + str(cell_label)


        results_cell.loc[:,'pict_code'] = picture_name
        results_cell.loc[:,'cell_label'] = cell_label
        results_cell.loc[:,'cell_code'] = picture_name + '_' + str(cell_label)

        for x, y in info.items():
            results_cell.loc[:, x] = y
        
        results_cell.loc[:,'cell_dist'] = cell_dist
        results_cell.loc[:,'vent_perc'] = cell_dist / info['vent_thick']


        
        # I leave this comment because it's funny
        ### TRY TO UNDERSTAND WHY THIS IS WORKING

        ################################

        
        a_process_cell_df = pd.DataFrame()
        b_process_cell_df = pd.DataFrame()

        a_process_cell_df['range'] = np.nan
        a_process_cell_df['vol'] = np.nan
        a_process_cell_df['mito_vol'] = np.nan

        b_process_cell_df['range'] = np.nan
        b_process_cell_df['vol'] = np.nan
        b_process_cell_df['mito_vol'] = np.nan

        if np.sum(comp_labels == 3) != 0:

            
            a_range, a_data, a_mito_data = process_analysis(comp_labels == 3, soma_dist, mito_bin, res)
            a_process_cell_df = pd.DataFrame({'range': a_range, 'vol': a_data, 'mito_vol': a_mito_data})
            #a_process_df.to_csv('results/processes/' + results_cell.loc[0, 'cell_code'] +'_apical.csv')
        
        if np.sum(comp_labels == 4) != 0:
            b_range, b_data, b_mito_data = process_analysis(comp_labels == 4, soma_dist, mito_bin, res)
            b_process_cell_df = pd.DataFrame({'range': b_range, 'vol': b_data, 'mito_vol': b_mito_data})
            #b_process_df.to_csv('results/processes/' + results_cell.loc[0, 'cell_code'] +'_basal.csv')
        
        a_process_cell_df.loc[:,'pict_code'] = picture_name
        a_process_cell_df.loc[:,'cell_label'] = cell_label
        a_process_cell_df.loc[:,'cell_code'] = picture_name + '_' + str(cell_label)

        b_process_cell_df.loc[:,'pict_code'] = picture_name
        b_process_cell_df.loc[:,'cell_label'] = cell_label
        b_process_cell_df.loc[:,'cell_code'] = picture_name + '_' + str(cell_label)


        results_mito_pic = pd.concat([results_mito_pic, results_mito_cell])
        results_pic = pd.concat([results_pic, results_cell])

        a_process_df = pd.concat([a_process_df, a_process_cell_df])
        b_process_df = pd.concat([b_process_df, b_process_cell_df])

        print(' - DONE')

    return results_pic, results_mito_pic, a_process_df, b_process_df

def picture_analysis_general(folder_name, picture_name, mito_suffix, mito_bin_suffix, cell_suffix, nucl_suffix, info, h5_flag = False):
    '''
    function for mitochondrial cell analysis in picture

    compartments are not calculated
    '''


    # read mitochondrial channel
    img_mito, res = get_picture(folder_name + '/' + picture_name + mito_suffix)
    voxel_vol = np.prod(res)

    # cell segmentation
    img_cell = get_picture(folder_name + '/' + picture_name + cell_suffix)[0]
    # soma
    img_nucl = get_picture(folder_name + '/' + picture_name + nucl_suffix)[0]

    info_cell = get_masks(img_cell)
    cell_labels = info_cell[0]
    cell_masks = info_cell[2]

    n_cells = len(cell_labels)
    print(' - ', end='')
    print(cell_labels)

    
    ###################################################

    # creating result dataframes
    results_pic = pd.DataFrame()
    results_mito_pic = pd.DataFrame()

    for c in range(n_cells):
        
        # mask of current cell
        cell_mask = cell_masks[c]

        cell_box = bounding_box(cell_mask)

        # cropping pic to size of the cell we are considering
        cell_mask = cell_mask[cell_box]
        #

        # removing non connected random pixels
        cell_mask = pixels_cleaning(cell_mask)

        # value of current cell
        cell_label = cell_labels[c]
        print(f'--- {cell_label}', end='')

        # nucl within the cell
        img_nucl_subset = img_nucl[cell_box]
        cell_nucl_mask = ma.masked_array(img_nucl_subset == cell_label, ~cell_mask.astype(bool)).filled(fill_value=False)

        # removing non connected random pixels
        cell_nucl_mask = pixels_cleaning(cell_nucl_mask)

        # binarized mitochondria
        '''
        binarized mitochondria could be obtained in program from mito_raw
        '''

    
        mito_bin = read_h5(folder_name + '/' + picture_name + mito_bin_suffix)


        mito_bin = mito_bin[cell_box]


        mito_bin = apply_mask(mito_bin, cell_mask).astype(bool)
        

        # computing distance from nucl
        nucl_dist = geodesic_dist(cell_mask, cell_nucl_mask, res)
    
        print(' - distance', end='')


        comp_labels = cell_mask


        cell_tables = cell_analysis(mito_bin, comp_labels, nucl_dist, res)

        results_cell = cell_tables[0]
        results_mito_cell = cell_tables[1]

        # these are values common for all the rows of the same cell, so let's add them at the end
        #################################

        results_mito_cell['cell_label'] = cell_label
        results_mito_cell['pict_code'] = picture_name
        results_mito_cell['cell_code'] = picture_name + '_' + str(cell_label)

        results_cell.loc[:,'pict_code'] = picture_name
        results_cell.loc[:,'cell_label'] = cell_label
        results_cell.loc[:,'cell_code'] = picture_name + '_' + str(cell_label)


        for x, y in info.items():
            results_cell.loc[:, x] = y
        

        
        # I leave this comment because it's funny
        ### TRY TO UNDERSTAND WHY THIS IS WORKING

        ################################


        results_mito_pic = pd.concat([results_mito_pic, results_mito_cell])
        results_pic = pd.concat([results_pic, results_cell])

        print(' - DONE')

    return results_pic, results_mito_pic

def cell_analysis(mito_bin, label_masks, soma_dist, res):
    '''
    analysis of mito inside a cell
    label_masks: subcellular components to consider to organize mito info
    soma_dist: distance map from the soma
    '''

    labels = np.unique(label_masks)[np.unique(label_masks) != 0]
    #print(labels)
    voxel_vol = np.prod(res)

    results_cell = pd.DataFrame()
    results_mito_cell = pd.DataFrame()

    # mitochondria results
    element_n, element_info = mito_analysis(mito_bin, res)


    if element_n == 0:
        results_mito_cell.loc[0, 'mito_label'] = np.nan
        results_mito_cell.loc[0, 'n_branches'] = np.nan
        results_mito_cell.loc[0, 'n_junctions'] = np.nan
        results_mito_cell.loc[0, 'length'] = np.nan
        results_mito_cell.loc[0, 'soma_dist'] = np.nan
        results_mito_cell.loc[0, 'compartment'] = np.nan
        results_mito_cell.loc[0, 'element_type'] = np.nan
    else:
        for m in range(element_n):
            results_mito_cell.loc[m, 'mito_label'] = m
            results_mito_cell.loc[m, 'n_branches'] = element_info['branches'][m]
            results_mito_cell.loc[m, 'n_junctions'] = element_info['junctions'][m]
            results_mito_cell.loc[m, 'length'] = element_info['length'][m]
            results_mito_cell.loc[m, 'soma_dist'] = soma_dist[element_info['coords'][m]]

            results_mito_cell.loc[m, 'compartment'] = label_masks[element_info['coords'][m]]

            if element_info['branches'][m] == 0:
                results_mito_cell.loc[m, 'element_type'] = 'p'
            # I could be classifying donuts as rods
            elif element_info['branches'][m] == 1:
                results_mito_cell.loc[m, 'element_type'] = 'r'
            else:
                results_mito_cell.loc[m, 'element_type'] = 'n'


    ##################


    for comp in labels:
        current_compartment = label_masks == comp
        current_compartment_mito = results_mito_cell[results_mito_cell['compartment'] == comp]

        results_cell.loc[comp-1, 'compartment'] = comp

        results_cell.loc[comp-1, 'volume'] = np.sum(current_compartment)*voxel_vol
        results_cell.loc[comp-1, 'mito_volume'] = np.sum(apply_mask(mito_bin, current_compartment))*voxel_vol
        soma_dist_on_current = ma.masked_array(soma_dist, ~current_compartment)

        # the length is not super accurate for cycling objects, in that case I need to use the length from the skeleton
        results_cell.loc[comp-1, 'length'] = np.max(soma_dist_on_current) - np.min(soma_dist_on_current)

        # mitochondria
        value_counts = current_compartment_mito['element_type'].value_counts()
        try:
            num_p = value_counts['p']
        except KeyError:
            num_p = 0

        try:
            num_r = value_counts['r']
        except KeyError:
            num_r = 0

        try:
            num_n = value_counts['n']
        except KeyError:
            num_n = 0
        
        results_cell.loc[comp-1, 'num_mito'] = num_p + num_r + num_n
        results_cell.loc[comp-1, 'num_p'] = num_p
        results_cell.loc[comp-1, 'num_r'] = num_r
        results_cell.loc[comp-1, 'num_n'] = num_n

        results_cell.loc[comp-1, 'mito_length'] = current_compartment_mito['length'].sum()
    
    return results_cell, results_mito_cell

def mito_analysis(mito_bin, res):
    # just to make sure everything is boolean
    mito_bin = mito_bin.astype(bool)

    # skeletonized mitochondria
    mito_ske = skeletonize(mito_bin)

    #show_pic(mito_ske, 'skeleton')

    mito_G = graph_from_skeleton(mito_ske, res)

    #plot_graph_2d(mito_G)

    mito_G = graph_branching_cleaner(mito_G, res)

    #plot_graph_2d(mito_G)

    element_coords = from_graph_to_element_centers(mito_G)

    simplified_G = graph_simplifier(mito_G)




    element_info = calculate_branches_and_junctions(simplified_G)
    element_info_array = np.array(element_info)

    element_lengths = calculate_real_length(mito_G)

    element_n = len(element_coords)

    mito_output = {
        'coords': element_coords,
        'length': element_lengths}
    
    try:
        mito_output['branches'] = element_info_array[:,0]
        mito_output['junctions'] = element_info_array[:,1]

    except IndexError:
        mito_output['branches'] = []
        mito_output['junctions'] = []

    return element_n, mito_output

def process_analysis(process_mask, soma_dist, mito_bin, res, step = 0.5):
    dist_in_process = apply_mask(soma_dist, process_mask)

    min_dist = np.min(dist_in_process)
    max_dist = np.max(dist_in_process)

    # I am losing the last half bin or so
    span = np.arange(min_dist, max_dist, step)

    bins = len(span)-1

    vol_bins = np.zeros(bins)
    mito_bins = np.zeros(bins)


    for i in range(bins):
        slice = np.logical_and(dist_in_process > span[i], dist_in_process <= span[i+1])
        vol_bins[i] = np.sum(slice)*np.prod(res)

        # only mitochondria in the considered slice
        mito_in_slice = apply_mask(mito_bin, slice)
        mito_bins[i] = np.sum(mito_in_slice)*np.prod(res)

    #plt.axhline(0, color='black', linewidth=1)
    #plt.plot(span[:-1], vol_bins, label='volume', color='green')
    #plt.plot(span[:-1], mito_bins, label='mito', color='red')
    #plt.show()

    # range of distances, process volume, mito volume
    return span[:-1], vol_bins, mito_bins


###########################################################
'''mitochondrial network managing'''
###########################################################
def get_neighbors(x, y, z, shape):
    '''
    get_neighbors produces neighbor coordinates in 3d

    x, y, z are the coordinates of the point
    shape is the 3d shape of the picture the point belongs to
    (to avoid that neighbors are out of the range when their value is checked)
    '''
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    neighbors.append((nx, ny, nz))
    return neighbors

# identify Endpoints and Junction Points
def identify_points(skeleton):
    # Create a convolution kernel to count neighbors
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0  # Exclude the center point itself
    
    # Count the number of neighbors for each skeleton voxel
    neighbors_count = convolve(skeleton.astype(int), kernel, mode='constant')
    
    # Find endpoints (1 neighbor) and junction points (3 or more neighbors)
    endpoints = np.argwhere((skeleton == 1) & (neighbors_count == 1))
    junctions = np.argwhere((skeleton == 1) & (neighbors_count >= 3))
    
    return endpoints, junctions

# calculate number of branches and junctions # IT DOESN'T WORK IN THE ORIGINAL NETWORK
def calculate_branches_and_junctions(G):
    element_info = []
    
    # iterate through each connected component in the graph
    for element in nx.connected_components(G):
        subgraph = G.subgraph(element)  # get the subgraph corresponding to this element
        
        # count the number of edges (branches)
        num_branches = subgraph.number_of_edges()
        
        # count the number of junction points (degree >= 3)
        num_junctions = sum(1 for node in subgraph if subgraph.degree(node) >= 3)
        
        element_info.append([num_branches, num_junctions])
    
    return element_info

# recursive function to explore the linear tracts of my mitochondrial graph
def network_linear_explorer(G, node, dist, neighbor):
    new_neighbors = list(G.neighbors(neighbor))

    next_neighbor = [n for n in new_neighbors if n != node][0]
    
    next_dist = G.get_edge_data(neighbor, next_neighbor)['weight']

    if G.degree(next_neighbor) != 2:
        return [next_neighbor, dist + next_dist]
    else:
        return network_linear_explorer(G, neighbor, dist + next_dist, next_neighbor)

# Compacts the linear tracts into single edges
def graph_simplifier(G):
    H = nx.MultiDiGraph()
    #H = nx.Graph()
    visited_edges = set()  # To track visited edges and avoid duplication

    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        
        # Check if all nodes in the component have degree 2 -> DONUT
        all_rank_2 = all(G.degree(n) == 2 for n in subgraph.nodes())

        if all_rank_2:
            # Handle cycle case: treat the whole component as a self-loop
            node = next(iter(subgraph.nodes()))  # Pick any node from the cycle
            total_weight = sum(G.get_edge_data(u, v)['weight'] for u, v in subgraph.edges())
            H.add_node(node)  # Add the node to the simplified graph
            H.add_edge(node, node, weight=total_weight)  # Add self-loop with total weight
        else:
            # Handle normal components (with some degree != 2 nodes)
            for node in subgraph.nodes():
                if G.degree(node) != 2:
                    # Add the node to the new simplified graph
                    H.add_node(node)
                    
                    # Traverse through each neighbor of the current node
                    for neighbor in G.neighbors(node):
                        # Avoid revisiting edges
                        if (node, neighbor) in visited_edges or (neighbor, node) in visited_edges:
                            continue

                        if G.degree(neighbor) != 2:
                            # Copy edges between non-degree-2 nodes directly
                            weight = G.get_edge_data(node, neighbor)['weight']
                            H.add_edge(node, neighbor, weight=weight)
                        else:
                            # Explore the linear tract and collapse it into a single edge
                            linear_edge = G.get_edge_data(node, neighbor)['weight']
                            node_to_connect, new_weight = network_linear_explorer(G, node, linear_edge, neighbor)
                            H.add_edge(node, node_to_connect, weight=new_weight)
                            
                            # Mark the entire linear tract as visited to avoid duplication
                            visited_edges.add((node, node_to_connect))
    
    '''for u, v, key, data in H.edges(keys=True, data=True):
        print(f"Edge {key}: {u} -> {v}, Attributes: {data}")'''
    
    try:
        max_x = max(node[0] for node in G.nodes) +1
        max_y = max(node[1] for node in G.nodes) +1
        max_z = max(node[2] for node in G.nodes) +1
    except ValueError:
        max_x, max_y, max_z = (1,1,1)

    # edges are doubled, there are couples with opposite direction
    # let's arbitrary remove one of the direction
    edges_to_remove = []
    for u, v, key, data in H.edges(keys=True, data=True):
        x1, y1, z1 = u  # Extract x-coordinate of the source node
        x2, y2, z2 = v  # Extract x-coordinate of the target node
        
        if (z1 + y1*max_z + x1*max_z*max_y) > (z2 + y2*max_z + x2*max_z*max_y):
            edges_to_remove.append((u, v, key))
        
    H.remove_edges_from(edges_to_remove)

    print()

    H_undirected = nx.MultiGraph(H)

    '''for u, v, key, data in H_undirected.edges(keys=True, data=True):
        print(f"Edge {key}: {u} -> {v}, Attributes: {data}")'''

    return H_undirected

def calculate_real_length(G):
    '''
    calculating real length based on voxel resolution on the edges
    every edge has a length depending on the position of the two voxels
    info about pixels distance already in the graph edges

    return
    array with length of each mito element
    '''
    element_lengths_real = []

    for element in nx.connected_components(G):
        length = 0.0
        subgraph = G.subgraph(element)  # Get the subgraph corresponding to this element
        # Sum the weights of all edges in the subgraph
        for (u, v, d) in subgraph.edges(data=True):
            length += d['weight']
        element_lengths_real.append(length)

    return element_lengths_real

def graph_barycenter_node(G):
    total_distances = {}

    for node in G.nodes():
        # Get the shortest path lengths from the current node to all other nodes
        shortest_paths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        # Sum the distances
        total_distances[node] = sum(shortest_paths.values())

    # Step 3: Find the node with the minimum total distance (the barycenter)
    barycenter_node = min(total_distances, key=total_distances.get)

    return barycenter_node

def graph_from_skeleton(skeleton, res):
    endpoints, junctions = identify_points(skeleton)



    # Create a Graph Representation of the Skeleton
    G = nx.Graph()


    # Add nodes for all points in the skeleton
    skeleton_points = np.argwhere(skeleton != 0)
    for point in skeleton_points:
        G.add_node(tuple(point))

    # Add edges for neighboring points in the skeleton
    for point in skeleton_points:
        neighbors = get_neighbors(*point, skeleton.shape)
        for neighbor in neighbors:
            if skeleton[neighbor] != 0:
                p1 = np.array(point)
                p2 = np.array(neighbor)
                # Calculate Euclidean distance considering voxel spacing
                distance = np.sqrt(((p1[0] - p2[0]) * res[0])**2 + 
                                ((p1[1] - p2[1]) * res[1])**2 + 
                                ((p1[2] - p2[2]) * res[2])**2)
                G.add_edge(tuple(point), tuple(neighbor), weight=distance)

    return G

def euclidean_distance(coord1, coord2, res=(1,1,1)):
    """
    Compute the Euclidean distance between two 3D coordinates considering resolution.
    
    Parameters:
        coord1 (tuple): First coordinate (x, y, z).
        coord2 (tuple): Second coordinate (x, y, z).
        res (tuple): Resolution of the pixels in (x_res, y_res, z_res).
    
    Returns:
        float: Euclidean distance between coord1 and coord2.
    """

    diff = np.array(coord1) - np.array(coord2)
    scaled_diff = diff * np.array(res)

    return np.sqrt(np.sum(scaled_diff ** 2))

# from the mitochondrial graph to the centers of all the elements
def from_graph_to_element_centers(G):
    elements = list(nx.connected_components(G))

    coords = []

    for m in elements:
        subgraph = G.subgraph(m)
        element_center = graph_barycenter_node(subgraph)
        coords.append(element_center)
    
    return coords

def rank_nodes_from_graph(G, r):
    return [node for node in G.nodes if G.degree(node) == r]


def graph_branching_cleaner(graph, res=(1, 1, 1)):
    """
    Collapse clusters of high-degree nodes into a single node using a custom center function
    and assign weights based on Euclidean distance.
    
    Parameters:
        graph (networkx.Graph): The input graph.
        center_function (callable): Function to compute the center of a cluster of nodes.
        res (tuple): Resolution of the pixels in (x_res, y_res, z_res).
        degree_threshold (int): The degree threshold for identifying high-degree nodes.
    
    Returns:
        networkx.Graph: The modified graph with clusters collapsed.
    """
    # Identify nodes with degree > 2, which are branching nodes
    high_degree_nodes = [node for node, degree in graph.degree() if degree > 2]
    
    # Extract connected components of high-degree nodes
    subgraph = graph.subgraph(high_degree_nodes)
    clusters = list(connected_components(subgraph))
    
    # Create a copy of the graph to modify
    modified_graph = graph.copy()
    
    for cluster in clusters:
        cluster = list(cluster)

        # Remove all nodes in the cluster
        for node in cluster:
            if node in modified_graph:
                modified_graph.remove_node(node)
        
        # Compute the center of the cluster using the provided function
        cluster_center = from_graph_to_element_centers(graph.subgraph(cluster))[0]

        
        # Find all neighbors of the cluster
        neighbors = set()
        for node in cluster:
            neighbors.update(graph.neighbors(node))

        
        # Remove the cluster nodes from neighbors set
        neighbors -= set(cluster)

        
        # Add the cluster center as a new node (if not already in the graph)
        if cluster_center not in modified_graph:
            modified_graph.add_node(cluster_center)
        
        # Add edges between the cluster center and all neighbors of the cluster
        for neighbor in neighbors:
            distance = euclidean_distance(cluster_center, neighbor, res)
            modified_graph.add_edge(cluster_center, neighbor, weight=distance)
        

    
    return modified_graph

def plot_graph_2d(graph):
    """
    Plot a 3D graph in 2D using nx.draw, projecting nodes onto a specified 2D plane.
    
    Parameters:
        graph (networkx.Graph): The graph to plot.
            Node names must be 3D coordinates (x, y, z).
        axis_projection (tuple): Indices of the axes to project onto (default: x-y plane).
    """
    # Create a position dictionary for nodes using the specified projection
    pos = {node: (node[0], node[1]) for node in graph.nodes}
    
    # Draw the graph using nx.draw with the custom position
    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos, with_labels=False, node_size=2, font_size=10)
    plt.xlabel(f"Axis {0}")
    plt.ylabel(f"Axis {1}")
    plt.title("Graph Projection")
    plt.show()

###########################################################
'''branching identification and mapping'''
###########################################################
def graph_linear_explorer(graph, current_node, component_id, visited):
    '''
    graph_explorer explores a linear componet of a graph from a starting point
    used within graph_linear_components

    starting from a node, it goes in both directions until it finds rank-2 nodes

    graph: graph to explore
    current_node: node where to start the exploration
    component_id: label of the current linear component
    visited: array with visited nodes, it's globally updated
    '''
    if graph.degree[current_node] <= 2:
        visited.add(current_node)

        graph.nodes[current_node]['component_label'] = component_id


        # Find the next unvisited neighbor
        neighbors = list(graph.neighbors(current_node))
        
        if neighbors:
            for next_node in neighbors:
                if next_node not in visited:
                    graph_linear_explorer(graph, next_node, component_id, visited)
        else:
            return

def graph_linear_components(G, th_length = 0):
    '''
    graph_linear_components produces an annotated graph with the linear components

    linear components are considered as neighborhoods of nodes with rank 2
    the nodes delimiting such components (branching points and end points)
        receive in the end an annotation from one of the neighboring linear components
        if they don't have annotated neighbors, they are removed

    G: graph, the length of the connections is stored as weight of the edges
    th_length: the minimum length for a connected component to be considered

    output:
        annotated graph
        the label of the linear component is stored in 'component_label' node attribute
        if a linear component is shorter than the threshold, the nodes belonging to it are removed
    '''

    H = G.copy()

    #finding linear components
    visited = set()
    component_id = 0
    for node in H.nodes:
        if node in visited or H.degree[node] > 2:
            if H.degree[node] > 2:
                H.nodes[node]['component_label'] = None
            continue
        
        # recursively explores the whole linear component and asign the component_id to all the nodes
        # updating visited also
        graph_linear_explorer(H, node, component_id, visited)
        # now we need to move to the next node and so next linear component
        component_id = component_id + 1
    

    # checking their length
    labels = range(component_id)
    label_lengths = np.zeros_like(labels, dtype=float)


    for l in labels:
        filtered_nodes = [n for n, data in H.nodes(data=True) if data['component_label'] == l]

        # subgraph keeping only nodes with specified lable
        subgraph = H.subgraph(filtered_nodes)

        # total sum of edge weight (length)
        current_label_length = sum(data['weight'] for u, v, data in subgraph.edges(data=True))

        label_lengths[l] = current_label_length

        # if component length is less than the threshold, I am removing the nodes
        if current_label_length < th_length:
            H.remove_nodes_from(filtered_nodes)
    

    # I need to repeat the linear components finding, this time without the little pieces
    visited = set()
    component_id = 0
    for node in H.nodes:
        if node in visited or H.degree[node] > 2:
            if H.degree[node] > 2:
                H.nodes[node]['component_label'] = None
            continue
        
        graph_linear_explorer(H, node, component_id, visited)
        component_id = component_id + 1

    # asigning component_label to branching and end nodes
    for node in H.nodes:
        if H.nodes[node]['component_label'] == None:
            # find the first neighbor with a label and assign it
            neighbors = H.neighbors(node)
            for neighbor in neighbors:
                if H.nodes[neighbor]['component_label'] != None:
                    H.nodes[node]['component_label'] = H.nodes[neighbor]['component_label']
                    break
    
    # if the node has still no label, it means it has no valid neighbors, so just remove it
    nodes_to_remove = [node for node in H.nodes if H.nodes[node].get('component_label') is None]
    H.remove_nodes_from(nodes_to_remove)

    output = {
        'graph': H,
        'labels': list(labels),
        'lengths': label_lengths
    }
    return output



###########################################################
'''functions to perform strahler analysis'''
###########################################################

def masked_voronoi_from_points(mask_pic, pixel_graph, attribute, show=False):
    '''
    masked_voronoi_from_points produces the voronoi diagram, where subset of points are aggregate under the same attribute value

    mask_pic: binary mask
    pixel_graph: graph where
        every node name is its coordinates tuple
        the atribute value is the one used to cluster the areas together
    attribute: which attribute of the graph nodes is used to cluster the areas
    show: flag to plot the voronoi results to debug

    output:
        masked array where every pixel value is the attribute values of the closest graph node
    '''
    mask_pic = mask_pic.astype(bool)
    points = list(pixel_graph.nodes)

    dims = mask_pic.shape


    # initialize the Voronoi regions array
    voronoi_array = np.full(dims, -1, dtype=int)  # Fill with -1 (unassigned)

    coords = np.argwhere(mask_pic)

    # assign each voxel to the nearest Voronoi seed
    for coord in coords:
        distances = np.linalg.norm(points - coord, axis=1)
        nearest_seed = np.argmin(distances)
        voronoi_array[coord[0], coord[1], coord[2]] = nearest_seed
    #######

    voronoi_output = voronoi_array.copy()

    unique_labels = set(data.get(attribute) for _, data in pixel_graph.nodes(data=True))
    '''for n, data in pixel_graph.nodes(data=True):
        print(f'{n}, - {data}')'''

    # cycling through all the labels on the points
    for c in unique_labels:
        nodes_with_label = [node for node, attr in pixel_graph.nodes(data=True) if attr.get(attribute) == c]
        nodes_coords = list(nodes_with_label)
        values = voronoi_output[tuple(zip(*nodes_coords))]
        voronoi_output[np.isin(voronoi_output, values)] = c

    if show:
        fig, (pixels_ax, compon_ax) = plt.subplots(1, 2, figsize=(12, 6))

        pixels_ax.imshow(max_proj(voronoi_array), cmap='plasma')
        pixels_ax.set_xticks([])
        pixels_ax.set_yticks([])

        compon_ax.imshow(max_proj(voronoi_output), cmap='plasma')
        compon_ax.set_xticks([])
        compon_ax.set_yticks([])


    voronoi_output = ma.masked_array(voronoi_output, ~mask_pic)

    return voronoi_output

def strahler_analysis(mask, soma, res, compartment_annotation=None, show = False):
    # we need the same network structure, for this reason we use the G_strahler that doesn't have loops

    mask = connection_corrector(mask)
    dist = geodesic_dist(mask, soma, res)

    G, _ = strahler_network(mask, soma, res)

    if show:
        plot_lab_graph(G, 'strahler_label')

    results = []

    # cycle through all the strahler numbers
    strahler_numbers = set(nx.get_node_attributes(G, 'strahler_label').values())

    current_strahler_id = 1

    for strahler_number in strahler_numbers:
        nodes_with_strahler = [n for n, d in G.nodes(data=True) if d.get('strahler_label') == strahler_number]

        subG = G.subgraph(nodes_with_strahler)

        num_components = nx.number_connected_components(subG)

        print(f'{strahler_number}: {num_components}')

        for component_nodes in nx.connected_components(subG):
            component_subG = subG.subgraph(component_nodes)
            
            # getting the distance values of the considered section to calculate the distance as max - min
            coords = np.array(list(component_subG.nodes))
            dist_values = dist[coords[:, 0], coords[:, 1], coords[:, 2]]

            if compartment_annotation is not None:
                comp_values = compartment_annotation[coords[:, 0], coords[:, 1], coords[:, 2]]

                counter = Counter(comp_values)
                compartment_label, _ = counter.most_common(1)[0]


            for node in component_nodes:
                G.nodes[node]['strahler_id'] = current_strahler_id

            current_result = {
                'strahler_id': current_strahler_id,
                'strahler_number': strahler_number,
                'length': np.max(dist_values) - np.min(dist_values)
            }

            if compartment_label is not None:
                current_result['compartment'] = compartment_label
            

            results.append(current_result)

            current_strahler_id += 1
    
    branching_df = pd.DataFrame(results)


    return branching_df, G

def strahler_network(mask, soma, res, show = False):
    mask = mask.astype(bool)

    ske = skeletonize(mask)

    mask = connection_corrector(mask)
    dist = geodesic_dist(mask, soma, res)


    G = graph_from_skeleton(ske, res)


    # edge weights depend on the distance from the soma, we prioritize cutting the loops away from the soma
    for u, v in G.edges:
        if not dist.mask[u]:
            edge_value = dist[u]
        elif not dist.mask[v]:
            edge_value = dist[v]
        else:
            edge_value = 0

        G.edges[u, v]['weight'] = edge_value

        # it can happen if one of the two values is masked
        if math.isnan(G.edges[u, v]['weight']):
            G.edges[u, v]['weight'] = 0

    T = nx.minimum_spanning_tree(G, weight='weight')

    nx.set_node_attributes(T, 0, 'strahler_label')

    ####
    temp_T = T.copy()
    current_l = 1

    old_unlabel_count = 0

    while any(degree >= 1 for _, degree in temp_T.degree()):
        # obtaining the nodes with that specific label
        current_nodes = leaf_labeler(temp_T, dist, current_l)

        for node in current_nodes:
            T.nodes[node]['strahler_label'] = current_l

        temp_T.remove_nodes_from(current_nodes)
        current_l += 1

        #plot_lab_graph(T, 'label')

        unlabel_count = sum(1 for _, data in T.nodes(data=True) if data.get("strahler_label") == 0)
        # it stops if it's not annotating any new node
        if unlabel_count == old_unlabel_count:
            break
        else:
            old_unlabel_count = unlabel_count
    ####

    if show:
        plot_lab_graph(T, 'strahler_label')

    T_0 = T.copy()
    # let's remove the nodes not annotated, in theory they are just the roots

    nodes_to_remove = [node for node, data in T.nodes(data=True) if data.get('strahler_label') == 0]

    # coords = np.array(list(T.nodes))
    # root_values = soma[coords[:,0], coords[:,1], coords[:,2]]
    # nodes_to_remove = [tuple(coord) for coord in coords[root_values]]
    T.remove_nodes_from(nodes_to_remove)

    if show:
        plot_lab_graph(T, 'strahler_label')

    return T, T_0

def annotate_from_network(mask, soma, T, l = 'label', show = False):
    '''
    annotating a cell following the strahler analysis of it's branching structure
    input:
        cell or process binary mask
        soma mask
        strahler annotated network

    returns:
        annotated mask
    '''

    mask = mask.astype(bool)


    cell_mask = mask  & (~soma)


    annotate_mask = masked_voronoi_from_points(cell_mask, T, l)

    return annotate_mask

def leaf_labeler(T, dist, l):
    graph = T.copy()

    # finding end nodes
    end_nodes = [node for node, degree in graph.degree() if degree == 1]


    for end_node in end_nodes:
        connected_component = nx.node_connected_component(graph, end_node)
        coordinates = list(connected_component)

        min_value = min(dist[coord] for coord in coordinates)
        #print(f'min value : {min_value},    dist[end_node] : {dist[end_node]}')
        
        # taking all the leaf nodes with the exception of the one closer to the soma
        if dist[end_node] != min_value:
            graph.nodes[end_node]['strahler_label'] = l

            # there should be only a neighbor
            neighbor_node = next(graph.neighbors(end_node))

            while graph.degree(neighbor_node) == 2:
                graph.nodes[neighbor_node]['strahler_label'] = l

                # take the neighbor that has not been asigned a lable yet
                next_node = [node for node in graph.neighbors(neighbor_node) if graph.nodes[node]['strahler_label'] == 0]

                if len(next_node) != 1:
                    break

                neighbor_node = next_node[0]
    
    # returning the list of nodes with the new label    
    return [node for node, data in graph.nodes(data=True) if data.get("strahler_label") == l]

def connection_corrector(image):
    '''
    binary image
    the calculation of the geodesic distance requires a 26-connected volume
    since cells containing thin processes might be 6-connected, but not 26,
    pixels that are touching this problematic junctions are expanded to achieve 26-connectivity

    returns:
        26-connected picture
    '''
    # connectivity structure
    connectivity_6 = generate_binary_structure(3, 1)
    connectivity_26 = generate_binary_structure(3, 3)

    # label connected components
    labels_6, num_labels_6 = label(image, structure=connectivity_6)
    labels_26, num_labels_26 = label(image, structure=connectivity_26)

    # pixels responsible for bridging
    bridging_pixels = np.zeros_like(image, dtype=bool)

    for label_idx in range(1, num_labels_26 + 1):
        # current 26-connected coponent
        component_26 = (labels_26 == label_idx)
        
        # unique labels in the 6-connected components within this 26-component
        unique_labels_6 = np.unique(labels_6[component_26])
        unique_labels_6 = unique_labels_6[unique_labels_6 > 0]  # remove background
        
        # If more than one unique 6-connected component exists, find bridging pixels
        if len(unique_labels_6) > 1:
            # Iterate through each pixel in the 26-connected component
            for x, y, z in zip(*np.where(component_26)):
                # Check the labels of the 6-connected neighbors
                neighbors_6 = labels_6[max(0, x-1):x+2, max(0, y-1):y+2, max(0, z-1):z+2]
                unique_neighbors = np.unique(neighbors_6)
                unique_neighbors = unique_neighbors[unique_neighbors > 0]  # Exclude background
                
                # If the pixel connects multiple 6-connected components, mark it
                if len(unique_neighbors) > 1:
                    bridging_pixels[x, y, z] = True
        

    expanded_bridging_pixels = binary_dilation(bridging_pixels, structure=connectivity_26)

    # Update the original image to include the expanded bridging pixels
    image_fixed = np.logical_or(image, expanded_bridging_pixels)

    # Validate the result
    labels_fixed, num_labels_fixed = label(image_fixed, structure=connectivity_6)

    #print(f"Original components (6-connectivity): {num_labels_6}")
    #print(f"Fixed components (6-connectivity): {num_labels_fixed}")

    return image_fixed