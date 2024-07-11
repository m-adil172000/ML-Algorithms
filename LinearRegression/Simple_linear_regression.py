# The LinearRegression method of sklearn uses the Ordinary Least Square method

class simpleLR:

    def __init__(self):
        self.m = None
        self.b = None
    
    def fit(self, X_train, y_train):

        num = 0
        den = 0
        n_iter = X_train.shape[0]

        for i in range(n_iter):

            num += (X_train[i] -X_train.mean())*(y_train[i] - y_train.mean())
            den += (X_train[i] - X_train.mean())**2
        
        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(f"coefficient: {self.m}, intercept: {self.b}")
    
    def predict(self, X_test):
        return (self.m)*(X_test) + self.b
