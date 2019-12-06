| pickle filename | description |
| :-------- : | :----- |
| ind.dataset_str.x | the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object |
| ind.dataset_str.tx | the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object |
| ind.dataset_str.allx | the feature vectors of both labeled and unlabeled training instances (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object |
| ind.dataset_str.y | the one-hot labels of the labeled training instances as numpy.ndarray object |
| ind.dataset_str.ty | the one-hot labels of the test instances as numpy.ndarray object |
| ind.dataset_str.ally | the labels for instances in ind.dataset_str.allx as numpy.ndarray object |
| ind.dataset_str.graph | a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object |
| ind.dataset_str.test.index | the indices of test instances in graph, for the inductive setting as list object. 

All objects above are saved using python pickle module.

```python
def load_data(dataset_str):
    '''
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).

    '''
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data2/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("../data2/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[:, 0] = 1  # set labels 1 for those isolated nodes, they do not appear in the test set
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    # convert to torch tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

```