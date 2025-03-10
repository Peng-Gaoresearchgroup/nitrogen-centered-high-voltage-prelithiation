import numpy as np
import pandas as pd
class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA model
        :param n_components: Target dimension after dimensionality reduction
        """
        self.n_components = n_components  #  Target dimension
        self.mean = None  # Mean of the data for centering
        self.components = None  # Principal components (eigenvectors)
        self.explained_variance = None  # Eigenvalues corresponding to principal components
        self.column_names = None  # Column names of the input data (if DataFrame is used)
    def fit(self, X):
        """
        Fit the PCA model
        :param X: Input data matrix (n_samples, n_features)
        """
        if isinstance(X, pd.DataFrame):
            self.column_names = X.columns
            X = X.values
        else:
            self.column_names = [f"Feature{i+1}" for i in range(X.shape[1])]
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2:  Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Step 3: Perform eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Step 5:  Select the top n_components principal components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
    
    def transform(self, X):
        """
        Transform the data into lower-dimensional space
        :param X:  Input data matrix (n_samples, n_features)
        :return: Transformed data matrix (n_samples, n_components)
        """
        if isinstance(X, pd.DataFrame):
            self.column_names = X.columns
            X = X.values

        # Ensure the PCA model is already fitted
        if self.components is None:
            raise ValueError("PCA model has not been fitted yet.")
        
        # Center the data
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        X_reduced = np.dot(X_centered, self.components)
        
        # Return a DataFrame with appropriate column names
        reduced_column_names = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(X_reduced, columns=reduced_column_names)
    
    def fit_transform(self, X):     

        """
        Fit the PCA model and transform the data
        :param X: Input data matrix (n_samples, n_features)
        :return: Transformed data matrix (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
