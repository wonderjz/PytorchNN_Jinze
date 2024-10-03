import numpy as np


def generate_cross_powered_features(arr0):
    """
    Generates new features by multiplying existing features and then raising the result to the power of 1, 2, 3, and 4.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.

    Returns:
    pd.DataFrame: A new DataFrame with the original features and the newly generated features.
    """
    # Ensure the input array is numeric
    arr0 = arr0.astype(float)
    arr_first = arr0[:, :40]
    arr_last = arr0[:,-1:] # for target, not transfer but add after transfer
    arr = arr0[:, 40:-1]
    # Get the number of original features
    num_features = arr.shape[1]
    
    # Create a list to store the new features
    new_features = []
    
    # Generate new features by multiplication and power operations
    for i in range(num_features):
        for j in range(i + 1, num_features):
            multiplied_feature = arr[:, i] * arr[:, j]
            for power in range(1, 3):
                new_features.append(multiplied_feature ** power)
    
    # Stack the new features horizontally
    new_features_array = np.column_stack(new_features)
    # Concatenate the original array with the new features array
    new_arr = np.hstack((arr, new_features_array))
    new_arr = np.hstack((arr_first, new_arr))
    new_arr = np.hstack((new_arr, arr_last))