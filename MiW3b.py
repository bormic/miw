from typing import List

import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_sizes: List[int], output_size: int, input_size: int) -> None:

        self.weights = []
        self.layers = []
        self.output_size = output_size
        self.input_size = input_size

        self.weights.append(np.random.random((input_size, hidden_sizes[0])))

        for i in range(len(hidden_sizes) - 1):
            weight = np.random.random((hidden_sizes[i], self.output_size))
            self.weights.append(weight)

    def set_layers(self, data):
        """ Ustawianie wag poszczegolnych warstw. """
        self.layers.clear()
        for i in range(len(data)):        
            self.layer0 = data[i:i+1]
            self.layers.append(self.layer0)
            self.layer1 = self.layer0.dot(self.weights[0])
            self.layers.append(self.layer1)
            self.layer2 = self.layer1.dot(self.weights[1])
            self.layers.append(self.layer2)

    def predict(self, input_: np.ndarray) -> np.ndarray:
        # zwracamy wartosci znajdujace sie w ostatniej warstwie sieci
        return self.layers[len(self.layers) - 1]

    def fit(self, data: np.ndarray, labels: np.ndarray, iterations: int, alpha: float) -> float:
        # zwracamy wartosc bledu treningu
        for _ in range(iterations):
            error = 0
            #correct_count = 0
            for i in range(len(data)):
                self.set_layers(data)
                
                #layer = self.predict(index)

                error += np.sum((labels[i] - self.layers[len(self.layers) - 1]) ** 2)
                #correct_count += int(self.layers[len(self.layers) - 1] == labels[i])
                
                delta2 = labels[i] - self.layers[len(self.layers) - 1]

                lenght = len(self.weights) - 1

                while lenght >= 0:
                    delta = delta2.dot(self.weights[lenght-1].T)
                    self.weights[lenght] += alpha * (self.layers[lenght].T.dot(delta))
                    lenght -= 1
        return print(error)


input_size = 2
hidden_size = [10,4,16,16]
output_size = 1
network = NeuralNetwork(hidden_size, output_size, input_size)
#network.__init__(hidden_size, output_size, input_size)

train_inputs = np.array([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
])

train_outputs = np.array([0, 1, 1, 0])

train_iterations = 500  

train_alpha = 0.01

network = NeuralNetwork(hidden_size, output_size, input_size)
network.fit(train_inputs, train_outputs, train_iterations, train_alpha)
