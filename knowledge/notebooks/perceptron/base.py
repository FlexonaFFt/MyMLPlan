import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None 

    '''Ступенчатая функция активации'''
    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Инициализация весов и смещения bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias 
                y_pred = self.activation(linear_output)

                # Правило обновления весов
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error 

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias 
        return self.activation(linear_output)



'''Как можно использовать написанный перцептрон'''
def simple_and_function():
    X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])

    # Запускаем процесс обучения 
    model = Perceptron(learning_rate=0.1, n_epochs=20)
    model.fit(X, y)

    predictions = model.predict(X)
    print("Predictions:", predictions)


if __name__ == '__main__':
    simple_and_function()