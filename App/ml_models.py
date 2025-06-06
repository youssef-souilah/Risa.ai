import numpy as np

class LinearRegression:
    
    def __init__(self, alpha=0.01, iterations=1000):
        
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
        self.training_history = []
        
    def _add_bias_term(self, X):
        # Check if bias term is already added (last column is all ones)
        if X.shape[1] > 0 and np.all(X[:, -1] == 1):
            return X
        return np.c_[X, np.ones(len(X))]
    
    def _compute_cost(self, X, y, theta):
        m = len(X)
        predictions = X.dot(theta)
        error = predictions - y.reshape(-1, 1)
        cost = (1/(2*m)) * np.sum(error ** 2)
        return cost
    
    def _compute_gradient(self, X, y, theta):
        m = len(X)
        predictions = X.dot(theta)
        error = predictions - y.reshape(-1, 1)
        gradient = (1/m) * X.T.dot(error)
        return gradient
    
    def fit(self, X, y):
        # Print initial shapes
        print(f"LinearRegression.fit - Initial X shape: {X.shape}")
        print(f"LinearRegression.fit - Initial y shape: {y.shape}")
        
        # Add bias term to features
        X = self._add_bias_term(X)
        print(f"LinearRegression.fit - X shape after adding bias: {X.shape}")
        
        # Ensure y is a column vector
        y = y.reshape(-1, 1)
        print(f"LinearRegression.fit - y shape after reshape: {y.shape}")
        
        # Initialize parameters - ensure theta has same number of parameters as features
        n_features = X.shape[1]  # This includes the bias term
        print(f"LinearRegression.fit - Number of features (including bias): {n_features}")
        self.theta = np.zeros((n_features, 1))
        print(f"LinearRegression.fit - Theta shape: {self.theta.shape}")
        
        # Initialize training history
        self.training_history = []
        
        # Gradient descent
        for i in range(self.iterations):
            # Compute gradient
            gradient = self._compute_gradient(X, y, self.theta)
            
            # Update parameters
            self.theta -= self.alpha * gradient
            
            # Record training history every 10 iterations
            if i % 10 == 0:
                cost = self._compute_cost(X, y, self.theta)
                self.training_history.append({
                    'iteration': i,
                    'cost': float(cost),  # Convert to float for JSON serialization
                    'theta': self.theta.tolist()  # Store theta for visualization
                })
        
        # Record final state
        final_cost = self._compute_cost(X, y, self.theta)
        self.training_history.append({
            'iteration': self.iterations,
            'cost': float(final_cost),
            'theta': self.theta.tolist()
        })
        print(y[0])
        print(X[0])
        return self
    
    def predict(self, X):
        print(f"LinearRegression.predict - Input X shape: {X.shape}")
        X = self._add_bias_term(X)
        print(f"LinearRegression.predict - X shape after adding bias: {X.shape}")
        return X.dot(self.theta)
    
    def score(self, X, y):
        X = self._add_bias_term(X)
        y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
    def get_params(self):
        return {
            'theta': self.theta.tolist(),
            'alpha': self.alpha,
            'iterations': self.iterations,
            'training_history': self.training_history
        }
    
    def set_params(self, params):
        self.theta = np.array(params['theta'])
        self.alpha = params['alpha']
        self.iterations = params['iterations']
        self.training_history = params.get('training_history', []) 