import torch

def default_target_func(X, centroids):
    return torch.cdist(X, centroids, p=2)**2

class KMeansPlusPlus:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, device='cuda', logging=False):
        """
        Initialize the KMeans++ class.

        Parameters:
            n_clusters (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
            device (str): Device type, 'cuda' or 'cpu'.
            logging (bool): Whether to log convergence information.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.centroids = None
        self.logging = logging
        self.loss = None
        self.empty_cluster_warning = 0

    def initialize_centroids(self, X, target_func, sample_weight=None):
        """
        Initialize centroids using the k-means++ method with sample weights.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).
            sample_weight (Tensor, optional): Sample weights, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        centroids = torch.empty((self.n_clusters, n_features), device=self.device)
        
        # Randomly select the first centroid
        indices = torch.randint(0, n_samples, (1,), device=self.device)
        centroids[0] = X[indices]
        
        # Initialize distances
        distances = torch.full((n_samples,), float('inf'), device=self.device)
        
        for i in range(1, self.n_clusters):
            # Compute distances from each point to each centroid
            dist = target_func(X, centroids[None, i - 1]).squeeze(-1)
            
            distances = torch.min(distances, dist)
            
            if sample_weight is not None:
                # Incorporate sample weights into probabilities
                weighted_distances = distances * sample_weight
            else:
                weighted_distances = distances
            
            # Select the next centroid with probability proportional to the weighted distance
            probabilities = weighted_distances / torch.sum(weighted_distances)
            categorical = torch.distributions.Categorical(probs=probabilities)
            index = categorical.sample().item()
            centroids[i] = X[index]
        
        self.centroids = centroids

    def fit(self, X, target_func=default_target_func, sample_weight=None):
        """
        Train the KMeans model with optional sample weights.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).
            sample_weight (Tensor, optional): Sample weights, shape (n_samples,).
        """
        assert len(X.shape) == 2

        input_device = X.device
        X = X.to(self.device).to(torch.float32)
        
        if sample_weight is not None:
            sample_weight = sample_weight.to(self.device).to(torch.float32)
            # Normalize sample weights to sum to 1
            sample_weight = sample_weight / torch.sum(sample_weight)
        
        self.initialize_centroids(X, target_func, sample_weight)
        
        for iteration in range(self.max_iter):
            # Compute distances from each point to each centroid
            distances = target_func(X, self.centroids)

            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)

            self.loss = distances.min(dim=-1).values.sum()
            if self.logging:
                print(f"[INFO] clustering loss {self.loss}")
            
            # Compute new centroids with sample weights
            new_centroids = torch.zeros_like(self.centroids)
                
            if sample_weight is not None:
                # Multiply X by sample weights
                weighted_X = X * sample_weight.unsqueeze(1)
                new_centroids.index_add_(0, labels, weighted_X)
                counts = torch.zeros(self.n_clusters, device=self.device)
                counts.scatter_add_(0, labels, sample_weight)
            else:
                new_centroids.index_add_(0, labels, X)
                counts = torch.zeros(self.n_clusters, device=self.device)
                counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float, device=self.device))
                
            if not torch.all(counts > 0):
                self.empty_cluster_warning = (counts == 0).sum().item()
                counts = torch.where(counts == 0, 1, counts)
            
            new_centroids /= counts.unsqueeze(1)
            
            # Compute the shift in centroids
            centroid_shift = torch.sum((self.centroids - new_centroids) ** 2).sqrt().item()
            self.centroids = new_centroids
            
            # Check for convergence
            if centroid_shift < self.tol:
                if self.logging:
                    print(f"[INFO] converged at iteration {iteration}")
                break
        else:
            print("[WARNING] kmeans clustering did not converge")

        self.centroids = self.centroids.to(input_device)
        if self.empty_cluster_warning:
            print(f"[WARNING] {self.empty_cluster_warning} empty clusters")
            self.empty_cluster_warning = 0


    def predict(self, X, target_func=default_target_func):
        """
        Predict cluster labels for new data points.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).

        Returns:
            labels (Tensor): Cluster labels for each point.
        """
        input_device = X.device
        X = X.to(self.device).to(torch.float32)
        distances = target_func(X, self.centroids.to(self.device))
        labels = torch.argmin(distances, dim=1)
        return labels.to(input_device)

    def fit_predict(self, X, target_func=default_target_func, sample_weight=None):
        """
        Train the model and predict cluster labels with optional sample weights.

        Parameters:
            X (Tensor): Input data, shape (n_samples, n_features).
            sample_weight (Tensor, optional): Sample weights, shape (n_samples,).

        Returns:
            labels (Tensor): Cluster labels for each point.
        """
        self.fit(X, target_func, sample_weight)
        return self.predict(X, target_func)