from typing import List

import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_sizes: List[int], output_sizes: int, input_sizes: int) -> None:
        for i in hidden_sizes:
            self.hidden_size = i
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
        #print(self.layers)


    def predict(self, input_: np.ndarray) -> np.ndarray:
        # zwracamy wartosci znajdujace sie w ostatniej warstwie sieci
        output = self.layers[len(self.layers) - 1]
        return output


    def fit(self, data: np.ndarray, labels: np.ndarray, iterations: int, alpha: float) -> float:
        # zwracamy wartosc bledu treningu
        for _ in range(iterations):
            error = 0.0
            correct_count = 0 
            for i in range(len(data)):
                self.set_layers(data)
                error += np.sum((labels[i] - self.layer2) ** 2)
                print("iteracja:", i, "błąd: ", error)
                correct_count += int(self.layer2 == labels[i])
                              
                delta2 = labels[i] - self.layer2
                delta1 = delta2.dot(self.weights_1_2.T)
                self.weights_1_2 += alpha * self.layer1.T.dot(delta2)
                self.weights_0_1 += alpha * self.layer0.T.dot(delta1)
        return error



input_size = 2
hidden_size = [10 , 4 , 16 , 16]
output_size = 1

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
print("error: " , network.fit(train_inputs, train_outputs, train_iterations, train_alpha))