import pandas as pd

def select_features_via_correlation(data, threshold = 0.3):
    """ Select features based on correlation with the target, keeping the state features """
    df = pd.DataFrame(data)  # Convert numpy array to pandas DataFrame for easier processing
    
    features = df.iloc[:, 41:-1]  # All columns except the last one (which is the target), without all states
    target = df.iloc[:, -1]       # The last column is the target
    
    # Calculate correlation between each feature and the target
    correlations = features.corrwith(target)

    # Select feature names with correlation above the threshold
    selected_features = correlations[correlations.abs() < threshold].index.tolist()
       
    # Get the first 40 state features (assuming columns 1 to 40 are state features)
    state_features = df.columns[1:41].tolist()  # Make sure this is a list of feature names
    
    # Combine the two lists (state features and selected features)
    selected_features = state_features + selected_features  # Combine the two lists of feature names

    return selected_features