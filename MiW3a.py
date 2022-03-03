from typing import List

import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_sizes: List[int], output_sizes: int, input_sizes: int) -> None:
        self.hidden_size = hidden_sizes
        self.output_size = output_sizes
        self.input_size = input_sizes
        self.weights_0_1 = np.random.random((self.input_size, self.hidden_size))  # inicjujemy wagi sieci dla warstw ukrytych
        self.weights_1_2 = np.random.random((self.hidden_size, self.output_size))
        self.layers = []  # umieszczamy wartosci znajdujace sie w poszczegolnych ukrytych warstwach sieci


    def set_layers(self, data):
        """ Ustawianie wag poszczegolnych warstw. """
        for i in range(len(data)):
            self.layer0 = data[i:i+1]
            self.layer1 = self.layer0.dot(self.weights_0_1)
            self.layer2 = self.layer1.dot(self.weights_1_2)
        
        # umieszczamy wartosci znajdujace sie w poszczegolnych ukrytych warstwach sieci
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)


    def predict(self, input_: np.ndarray) -> np.ndarray:
        # zwracamy wartosci znajdujace sie w ostatniej warstwie sieci
        inputs = input_.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output


    def fit(self, data: np.ndarray, labels: np.ndarray, iterations: int, alpha: float) -> float:
        # zwracamy wartosc bledu treningu
        self.set_layers(data)
        for _ in range(iterations):
            error = 0.0
            correct_count = 0
           #delta = []
            
            for i in range(len(data)):
                error += np.sum((labels[i] - self.layer2) ** 2)
                correct_count += int(self.layer2 == labels[i])
                              
                delta2 = labels[i] - self.layer2
                delta1 = delta2.dot(self.weights_1_2.T)
                self.weights_1_2 += alpha * self.layer1.T.dot(delta2)
                self.weights_0_1 += alpha * self.layer0.T.dot(delta1)



input_size = 2
hidden_size = 10
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

train_iterations = 5   

train_alpha = 0.01

network.fit(train_inputs, train_outputs, train_iterations, train_alpha)
network.predict(train_inputs)
print(network.weights_0_1)
print(network.weights_1_2)