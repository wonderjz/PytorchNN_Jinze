
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def compare_distributions(train_data, test_data):
    """
    Compare each column of the training data and test data using the Kolmogorov-Smirnov test.
    
    Parameters:
    train_data (numpy.ndarray): Training data with shape (num_samples, num_features).
    test_data (numpy.ndarray): Test data with shape (num_samples, num_features).
    
    Returns:
    list: A list of tuples where each tuple contains the K-S statistic and p-value for each feature.
    """
    num_features = train_data.shape[1]
    results = []
    
    for i in range(num_features):
        train_column = train_data[:, i]
        test_column = test_data[:, i]
        
        # Perform the K-S test
        ks_statistic, p_value = ks_2samp(train_column, test_column)
        
        # Store the result
        results.append((ks_statistic, p_value))
    
    return results



'''
df_train = pd.read_csv('/Users/lijinze/Downloads/HW1.train.csv')
df_test = pd.read_csv('/Users/lijinze/Downloads/HW1.test.csv')
df_train = df_train.iloc[:, :-1]
df_test = df_test.iloc[:, 1:]
columns_to_check = ['AR', 'IL', 'IA', 'KS', 'KY', 'NE', 'OK']
filtered_df_train = df_train[df_train[columns_to_check].any(axis=1)]
filtered_df_test = df_test[df_test[['MO']].any(axis=1)]

# Compare distributions
results = compare_distributions(np.array(filtered_df_train), np.array(pd.DataFrame(filtered_df_train)) )

# Print the results
for i, (ks_statistic, p_value) in enumerate(results):
    print(f"Feature {i+1}: K-S Statistic = {ks_statistic:.4f}, p-value = {p_value:.4f}")

'''
