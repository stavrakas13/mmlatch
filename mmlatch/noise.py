import numpy as np
import copy

np.random.seed(42)

def add_gaussian_noise(features, sigma=0.01):
    """
    Adds gasussian noise to the feature vectors of a modality.
    sigma = noise_level
    """
    noise = np.random.normal(0, sigma, features.shape)
    return features + noise

def add_dropout_noise(features, rate=0.5):
    """
    Adds dropout noise to the feature vectors of a modality.
    rate = noise_level
    """
    mask = np.random.binomial(1, 1-rate, features.shape)
    return features * mask

def pad_or_truncate(datum, goal_length):
    """
    Pads or truncates the input datum to match the goal_length.
    
    If the datum is shorter, it will be padded with zeros.
    If it is longer, it will be truncated to match goal_length.

    Args:
        datum (numpy array): The input feature embedding.
        goal_length (int): The desired length.

    Returns:
        numpy array: Padded or truncated array.
    """
    current_length = len(datum)
    
    if current_length < goal_length:
        # Pad with zeros
        padding = np.zeros((goal_length - current_length, datum.shape[1])) if datum.ndim == 2 else np.zeros(goal_length - current_length)
        return np.concatenate((datum, padding), axis=0)
    elif current_length > goal_length:
        # Truncate to match the goal length
        return datum[:goal_length]
    return datum  # Return as-is if already the correct length

def length_preserving_shuffler(datum1, datum2):
    """
    Ensures that datum2 matches the length of datum1 before replacement.
    
    datum1: current embedding (to be replaced)
    datum2: replacing embedding
    """
    goal_length = len(datum1)
    datum2_resized = pad_or_truncate(datum2, goal_length)
    return datum2_resized  # Return the resized datum2

def shuffle_modalities(dataset, modality_to_shuffle="all", shuffle_prob=0.1):
    """
    Shuffles modalities between samples to create out-of-context combinations.

    Args:
        dataset (list): List of samples, where each sample is a dict with keys "text", "audio", and "visual".
        modality_to_shuffle (str or list): Modality to shuffle. Options: "all", "text", "audio", "visual", or a list of modalities.
        shuffle_prob (float): Probability of shuffling a modality for a given sample.

    Returns:
        list: Dataset with shuffled modalities.
    """            
    # Determine which modalities to shuffle
    if modality_to_shuffle == "all":
        modalities = ["text", "audio", "visual"]
    else:
        modalities = [modality_to_shuffle]

    # Shuffle each modality
    for modality in modalities:
        # Collect all features for this modality across the dataset
        print(f"Expected modality keys: {dataset[0].keys() if dataset else 'Empty Dataset'}") # DEBUGGING
        print(f"Modality requested: {modality}") # DEBUGGING

        modality_features = [sample[modality] for sample in dataset]

        # Shuffle the features
        np.random.shuffle(modality_features)

        # Apply shuffled features to samples based on shuffle_prob
        for i, sample in enumerate(dataset):
            if np.random.rand() < shuffle_prob:
                sample[modality] = length_preserving_shuffler(sample[modality], modality_features[i])


    return dataset

def add_noise(dataset, noise_type='none', noise_modality='all', noise_level=0, augment=False):
    """
    Adds noise to the selected dataset (test/train)
    noise_type = 'none', 'gaussian', 'dropout', 'shuffle', 'all'
    noise_modality = 'all', 'text', 'audio', 'visual' (note that 'text'=='glove')
    modifies dataset by reference
    """
    if noise_type == 'none':
        return dataset

    ds = copy.deepcopy(dataset)
    modalities = ['text', 'audio', 'visual']
    if noise_type == 'gaussian':
        for i in range(len(ds)):
            if noise_modality == 'all':
                for modality in modalities:
                    ds[i][modality] = add_gaussian_noise(ds[i][modality], noise_level)
            else:
                ds[i][noise_modality] = add_gaussian_noise(ds[i][noise_modality], noise_level)

    elif noise_type == 'dropout':
        for i in range(len(ds)):
            if noise_modality == 'all':
                for modality in modalities:
                    ds[i][modality] = add_dropout_noise(ds[i][modality], noise_level)
            else:
                ds[i][noise_modality] = add_dropout_noise(ds[i][noise_modality], noise_level)

    elif noise_type == 'shuffle':
        ds = shuffle_modalities(ds, modality_to_shuffle=noise_modality, shuffle_prob=noise_level)
    
    elif noise_type == 'all':
        noise_level_gaussian  = noise_level[0]
        noise_level_dropout = noise_level[1]
        noise_level_shuffle = noise_level[2]
        for i in range(len(ds)):
            if noise_modality == 'all':
                for modality in modalities:
                    ds[i][modality] = add_dropout_noise(add_gaussian_noise(ds[i][modality], noise_level_gaussian), noise_level_dropout)
            else:
                ds[i][noise_modality] = add_dropout_noise(add_gaussian_noise(ds[i][noise_modality], noise_level_gaussian), noise_level_dropout)
        ds = shuffle_modalities(ds, modality_to_shuffle=noise_modality, shuffle_prob=noise_level_shuffle)

    if augment:
        return dataset + ds
    else:
        return ds
