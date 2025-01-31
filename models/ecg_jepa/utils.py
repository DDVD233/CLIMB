import numpy as np

def return_unique(waves, use_quarter = True):
    '''
        waves.shape = (n_samples, n_channels, timestep)
        returns unique indices
    '''
    timestep = len(waves[0,0,:])
    if use_quarter:
        timestep = int(timestep/4)
    _, uniq = np.unique(waves[:,0,:timestep], return_index=True, axis=0)
    uniq.sort()
    unique_indices = np.zeros(len(waves)).astype(bool)
    for i in uniq:
        unique_indices[i] = True

    return unique_indices


def return_normal(waves):
    '''
        waves.shape = (n_samples, n_channels, timestep)
        returns nonzero wave indices
    '''
    
    n_samples = len(waves)
    n_channels = len(waves[0])

    dead_indicies = np.zeros(n_samples).astype(bool)

    for i in range(n_samples):
        for j in range(n_channels):
            if not np.any(waves[i][j][:500]):   # True if the first 500 moments is the  zero wave 
                dead_indicies[i] = True
            if not np.any(waves[i][j][-500:]):  # True if the last 500 moments is the zero wave 
                dead_indicies[i] = True
    normal_indices1 = ~dead_indicies

    max_values = np.max(waves, axis=2)
    min_values = np.min(waves, axis=2)
    differences = max_values - min_values
    normal_indices2 = np.all(differences <= 5., axis=1)

    normal_indices = normal_indices1 & normal_indices2

    return normal_indices

def return_unique_normal(waves):
    unique_indices = return_unique(waves)
    normal_indices = return_normal(waves)

    unique_normal_indices = unique_indices & normal_indices
    return unique_normal_indices

def return_purified(waves, labels):
    '''
        use this
    '''
    if len(waves) != len(labels):
        print('len(waves) does not agree with len(labels). Make sure they are of same length.')

    labels_indices = ~np.isnan(labels)
    waves, labels = waves[labels_indices], labels[labels_indices]

    unique_indices = return_unique(waves)
    waves, labels = waves[unique_indices], labels[unique_indices]

    normal_indices = return_normal(waves)
    waves, labels = waves[normal_indices], labels[normal_indices]

    return waves, labels


def return_purified_feature(waves, labels, feature):
    '''
        use this
    '''
    if len(waves) != len(labels):
        print('len(waves) does not agree with len(labels). Make sure they are of same length.')

    labels_indices = ~np.isnan(labels)
    waves, labels = waves[labels_indices], labels[labels_indices]
    feature = feature[labels_indices]

    unique_indices = return_unique(waves)
    waves, labels = waves[unique_indices], labels[unique_indices]
    feature = feature[unique_indices]

    normal_indices = return_normal(waves)
    waves, labels = waves[normal_indices], labels[normal_indices]
    feature = feature[normal_indices]

    return waves, labels, feature
