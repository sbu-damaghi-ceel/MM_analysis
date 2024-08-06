import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt

import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import matrix_power
from collections import Counter





import seaborn as sns
import phenograph


def load_raw(df_raw,id_col,loc_cols):
    
    #print(df_raw.head())
    spatial_locs = df_raw[loc_cols].values
    sample_names = df_raw[id_col].values
    gene_names = [col for col in df_raw.columns if col not in loc_cols+[id_col]]
    count_matrix = df_raw[gene_names].values
    return count_matrix,spatial_locs,gene_names,sample_names


def plot_raw_histo(count_matrix,gene_names):

    num_cols = count_matrix.shape[1]
    num_rows = (num_cols + 3) // 4 

    fig, axes = plt.subplots(num_rows, 4, figsize=(16, num_rows * 4))

    if num_rows == 1:
        axes = [axes]

    for i in range(num_rows):
        for j in range(4):
            col_index = i * 4 + j
            if col_index < num_cols:
                axes[i][j].hist(count_matrix[:, col_index], bins=20, color='skyblue', edgecolor='black')
                axes[i][j].set_xlabel('Values')
                axes[i][j].set_ylabel('Frequency')
                axes[i][j].set_title(f'Histogram of {gene_names[col_index]}')

    plt.tight_layout()
    plt.show()
def preprocess_count(count_matrix,pth=95,normal=None):
    #trim normalization method(replace extreme values with 99th percentile)
    log_count_matrix = np.log(count_matrix+1)
    count_matrix_clipped = log_count_matrix.copy()
    percentiles_99 = np.percentile(log_count_matrix, pth, axis=0)
    for j in range(count_matrix.shape[1]):
        count_matrix_clipped[log_count_matrix[:,j] > percentiles_99[j],j] = percentiles_99[j]
        count_matrix_clipped[log_count_matrix[:,j] < -percentiles_99[j],j] = -percentiles_99[j]
        #print(f'column {j} replace {np.sum(log_count_matrix[:,j] > percentiles_99[j])+\
        #    np.sum(log_count_matrix[:,j] < -percentiles_99[j])}')
    if normal is None:
        count_matrix_normalized = count_matrix_clipped
    elif normal == 'maxmin':
        max = np.max(count_matrix_clipped, axis=0)
        min = np.min(count_matrix_clipped, axis=0)
        count_matrix_normalized = (count_matrix_clipped - min) / (max-min)
    elif normal == 'zscore':
        mean = np.mean(count_matrix_clipped, axis=0)
        std = np.std(count_matrix_clipped, axis=0)
        count_matrix_normalized = (count_matrix_clipped - mean) / std
    else:
        raise ValueError("Invalid value for 'normal'. Supported options are: None, 'maxmin', 'zscore'")
    return count_matrix_normalized

def createAdata_macsima(count_matrix_normalized,gene_names,sample_names,spatial_locs=None):
    adata = ad.AnnData(count_matrix_normalized.astype(float))
    if spatial_locs is not None:
        adata.obsm['spatial'] = spatial_locs.astype(float)
    adata.var_names = np.array(gene_names).astype(str)
    adata.obs_names = np.array(sample_names).astype(str)
    #sc.pp.log1p(adata)
    #sc.pp.scale(adata)
    #sc.pp.recipe_zheng17(adata,n_top_genes=len(adata.var_names))

    print(f'max of adata:{np.max(adata.X)}')
    return adata



def construct_graph(adata,use_rep='X', method='kNN', key_added='neighbors', n_neighbors=15, threshold=None):
    """
    Constructs a graph based on the specified method and updates the adata object.

    Parameters:
        adata (AnnData): The AnnData object to compute the graph on.
        method (str): The method to use for graph construction ('kNN' or 'distance').
        key_added (str): Base key to add to adata.obsp and adata.obs.
        n_neighbors (int): Number of neighbors to consider for the 'kNN' method.
        threshold (float): Distance threshold for creating edges in the 'distance' method.
    """
    
    if method == 'kNN':
        sc.pp.neighbors(adata, n_neighbors=n_neighbors,use_rep=use_rep, key_added=key_added)
    elif method == 'distance':
        
        if use_rep is None or use_rep == 'X':
            use_rep = 'X'
            values = adata.X
        elif use_rep in adata.obsm.keys():
            values = adata.obsm[use_rep]
        else:
            raise ValueError('use_rep needs to be X or a key of adata.obsm')
        dist_matrix = squareform(pdist(values, metric='euclidean'))
        dist_matrix[dist_matrix>threshold] = 0
        dist_sparse = csr_matrix(dist_matrix)  # Convert to sparse format for efficiency
        
        adata.obsp[key_added + '_distances'] = dist_sparse
        
        if threshold is not None:
            connectivities = csr_matrix((dist_matrix > 0).astype(int))
            adata.obsp[key_added + '_connectivities'] = connectivities
        else:
            raise ValueError('Requires threshold when method is distance')
        # Mimic the structure of sc.pp.neighbors output in adata.obs
        adata.uns[key_added] = {'connectivities_key': key_added + '_connectivities',
                                'distances_key': key_added + '_distances',
                                'params':{
                                    'method': method,
                                    'threshhold':threshold,
                                    'use_rep': use_rep,

                                }}
    else:
        raise ValueError("Method not supported. Use 'kNN' or 'distance'.")


def getNeighborComposition(adata,connectivities_key,pheno_key,neighbor_radius):
   # Convert adjacency matrix to binary if it's not already
    binary_adjacency = adata.obsp[connectivities_key].astype(bool)

    #dist_matrix = csgraph.shortest_path(binary_adjacency, directed=False)
    path_matrix = matrix_power(binary_adjacency,neighbor_radius)

    pheno_values = adata.obs[pheno_key]
    unique_pheno = pheno_values.unique()
    neighbor_composition = np.zeros((len(adata), len(unique_pheno)))
    for i in range(len(adata)):
        # Get neighbors within 'neighbor_radius' layers from the source node
        neighbors_indices = path_matrix[i].indices
        neighbors_phenos = pheno_values.iloc[neighbors_indices]
        neighbor_count = len(neighbors_indices)
        if neighbor_count > 0:
            composition = Counter(neighbors_phenos)
            for j, pheno in enumerate(unique_pheno):
                neighbor_composition[i, j] = composition.get(pheno, 0) / neighbor_count
    var_names = [f'radius{neighbor_radius}_{pheno}_prop' for _,pheno in enumerate(unique_pheno)]

    new_adata = ad.AnnData(neighbor_composition)
    new_adata.var_names = var_names
    new_adata.obs_names = adata.obs_names.copy()
    new_adata.obsm['spatial'] = adata.obsm['spatial'].copy()
    return new_adata




def plot_correlation_matrix(adata,varlist=None):
    if varlist is None:
        varlist = adata.var_names
    var_indices = [adata.var_names.get_loc(var) for var in varlist]
    
    submatrix = adata.X[:, var_indices]
    correlation_matrix = submatrix.T.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=varlist, yticklabels=varlist)
    plt.title("Correlation Matrix of Variables in adata.X")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()



###########################################################
# def create_community_adata(adata,comm_key='community',pheno_key='phenotype'):
#     unique_communities = np.unique(adata.obs[comm_key])
#     unique_phenotypes = np.unique(adata.obs[pheno_key])
    
#     prop_matrix = np.zeros((len(unique_communities), len(unique_phenotypes)))
    
#     for comm_label in unique_communities:
#         obs_subset = adata[adata.obs[comm_key] == comm_label]
#         # Calculate proportions of each leiden_exp label
#         proportions = obs_subset.obs[pheno_key].value_counts(normalize=True)
#         for exp_label in unique_phenotypes:
#             if exp_label in proportions:
#                 prop_matrix[int(comm_label), int(exp_label)] = proportions[exp_label]
#     new_adata = ad.AnnData(prop_matrix.astype(float))

#     # Set observation and feature names
#     obs_names = [f'community{label}' for label in unique_communities]
#     feature_names = [f'prop_phenotype{label}' for label in unique_phenotypes]
    
#     new_adata.obs_names = obs_names
#     new_adata.var_names = feature_names
    
#     return new_adata
# def plot_stacked_histogram(community_adata,show_legend=False):
#     prop_matrix = community_adata.X
#     unique_communities = community_adata.obs_names
#     unique_phenotypes = community_adata.var_names
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ind = np.arange(len(unique_communities))
#     width = 0.35
    
#     bottom = np.zeros(len(unique_communities))
#     for j, exp_label in enumerate(unique_phenotypes):
#         ax.bar(ind, prop_matrix[:, j], width, bottom=bottom, label=f'Phenotype {exp_label[-1]}')
#         bottom += prop_matrix[:, j]
    
#     ax.set_ylabel('Proportion')
#     ax.set_title('Stacked Histogram of Phenotype Proportions by Community')
#     ax.set_xticks(ind)
#     ax.set_xticklabels(unique_communities)
#     if show_legend:
#         ax.legend()
    
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

# def manual_gating(raw_data,feat_tree):
#     sample_labels = []
#     manual_gating_recursive(raw_data,feat_tree.root_node,sample_labels)
#     n = raw_data.shape[0]
#     reorganized_labels = [[label[i] for label in sample_labels] for i in range(n)]
#     concatenated_labels = []
#     for labels in reorganized_labels:
#         concatenated_label = '_'.join(labels)
#         concatenated_labels.append(concatenated_label)

#     return concatenated_labels
# def manual_gating_recursive(raw_data,node,sample_labels):
    
#         threshold = node['threshold']
#         if threshold is not None:
#             gene_expression = raw_data[:, node['name']]
#             sample_labels.append(np.where(gene_expression >= threshold, 
#                                           node['pos_label'], node['neg_label']))
#             for child_node in node['pos_children']+node['neg_children']:
#                 manual_gating_recursive[child_node]
#         #when threshold is None, the node is a leave node
#         return sample_labels

            
        

# class feat_tree:
#     def __init__(self, gene_names,root_name):
#         self.gene_nodes = {}
#         for gene_name in gene_names:
#             self.gene_nodes[gene_name] = {
#                 'name': gene_name,
#                 'threshold': None,
#                 'pos_label': f'{gene_name}_+',
#                 'neg_label': f'{gene_name}_-',
#                 'pos_children': [],
#                 'neg_children': []
#             }
#         self.root_node = self.gene_nodes[root_name]
    
#     def set_node(self, gene_name, threshold=None, pos_label=None, neg_label=None, pos_children=None, neg_children=None):
#         if gene_name in self.gene_nodes:
#             if threshold is not None:
#                 self.gene_nodes[gene_name]['threshold'] = threshold
#             if pos_label is not None:
#                 self.gene_nodes[gene_name]['pos_label'] = pos_label
#             if neg_label is not None:
#                 self.gene_nodes[gene_name]['neg_label'] = neg_label
#             if pos_children is not None:
#                 self.gene_nodes[gene_name]['pos_children'] = [self.gene_nodes[gene] if isinstance(gene, str) else gene for gene in pos_children]
#             if neg_children is not None:
#                 self.gene_nodes[gene_name]['neg_children'] = [self.gene_nodes[gene] if isinstance(gene, str) else gene for gene in neg_children]
#         else:
#             print(f"Gene '{gene_name}' not found in the feature tree.")