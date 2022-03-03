from typing import Any, Callable

import numpy as np
import pandas as pd
import math as mt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


def euclidean_distance(row1, row2):
    distance = 0.0
    print("row1: ", row1, "len row1: ", len(row1))
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return mt.sqrt(distance)

def get_cosine_sim(row1, row2):
    return mt.acos(
        sum([x1*x2 for x1,x2 in zip(row1,row2)])/(sum([i**2 for i in row1]) * sum([i**2 for i in row2]))
    )

def norm_data(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)



class DecisionSystem:
    __objects: np.ndarray
    __decisions: np.ndarray

    def __init__(self, filename: str) -> None:
        self.file_columns = ['sepal_len','sepal_width','petal_len','petal_width','class']
        self.__objects = pd.read_csv(filename, header=None, names=self.file_columns, delimiter='\t')
        self.__decisions = pd.read_csv(filename, header=None, names=self.file_columns, delimiter='\t')
        pass  # wczytuje dane z pliku do zbioru obiektow i decyzji

    def get_object(self, i: int) -> np.ndarray:
        return self.__objects.loc[i]
        pass  # zwraca obiekt o i-tym indeksie

    def get_decision(self, i: int) -> np.ndarray:
        return self.__objects.loc[i].at['class']
        pass  # zwraca decyzje dla obiektu o i-tym indeksie

    def __len__(self) -> int:
        return self.__objects.size
        pass  # zwraca liczbe obiektow w systemie decyzyjnym

# Calculate the Euclidean distance between two vectors

class Knn:
    __decision_system: DecisionSystem

    def __init__(self, decision_system: DecisionSystem) -> None:
        self.__decision_system = decision_system
        pass  # przekazuje do instancji system decyzyjny

    def predict(self, obj: np.ndarray, k: int, metric: Callable[[np.ndarray], float]) -> Any:
        
        
        X = self.__decision_system.get_object(k)
        y = self.__decision_system.get_decision(k)        
        
        #distances = np.linalg.norm(X - y, axis=1) # wektor odległości np.linalg.norm
        #distances = get_cosine_sim(X,y) # wektor odległości get_cosine_sim
        distances = euclidean_distance(X,y) # wektor odległości euclidean_distance
        
        nearest_neighbor_ids = distances.argsort()[:k] # którzy k sąsiedzi są najbliżej
        nearest_neighbor_rings = y[nearest_neighbor_ids] # wartości dla tych k sąsiadów
        
        prediction = nearest_neighbor_rings.mean() # wybiera średnią wartość k sąsiadów lub
        #prediction = norm_data(nearest_neighbor_rings) # w/w przy użyciu funkcji
        print("średnia wartość k sąsiadów: ",prediction)
        
        # Dzielenie danych na zbiory szkoleniowe i testowe w celu oceny modelu
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2) # from sklearn.model_selection import train_test_split
        
        knn_model = obj
        knn_model.fit(X_train, y_train) # dopasowanie modelu do zestawu danych szkoleniowych. Używając .fit(), pozwalasz modelowi uczyć się na podstawie danych.
        
        # ocena błędu przewidywania w danych uczących
        train_preds = knn_model.predict(X_train)
        mse = mean_squared_error(y_train, train_preds)
        rmse = mt.sqrt(mse)
        print("błąd przewidywania: ",rmse)
        
        # ocena predykcyji działania zestawu testowego
        test_preds = knn_model.predict(X_test)
        mse1 = mean_squared_error(y_test, test_preds)
        rmse1 = mt.sqrt(mse1)
        print("średni błąd: ",rmse1)
        
        # użycie Seaborn do utworzenia wykresu punktowego na obiekcie uczącym się
        cmap = sns.cubehelix_palette(as_cmap=True)
        f, ax = plt.subplots()
        points = ax.scatter(X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap)
        f.colorbar(points)
        plt.show()

        # wykres punktowy danych wczytanych powinien być podobny
        cmap = sns.cubehelix_palette(as_cmap=True)
        f, ax = plt.subplots()
        points = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cmap)
        f.colorbar(points)
        plt.show()
        
        
        
        self.__decision_system.predict(metric)
        pass  # zwraca przydzielona decyzje

    def __find_closest_objects(self, obj: np.ndarray, k: int,
                               metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
        
        
        

        self.__decision_system.__find_closest_objects(metric)
        pass  # zwraca k najblizszych obiektow






url = (
       "http://wmii.uwm.edu.pl/~bnowak/userfiles/downloads/dydaktyka/Metody_Inzynierii_Wiedzy/iris.txt"
       )
file = ("C:/Users/nonew/Desktop/III-lato/MiW/cw5/iris.txt")




# zmienne
q = 149 # wybiera obiekty od q min 0
z = 150 # wybiera obiekty do z-1 max 150
ka = 5 # ka najbliższych sąsiadów

# wczytanie danych do wybrania/wygenerowania obiektu
columns = ['sepal_len','sepal_width','petal_len','petal_width','class']
dataset = pd.read_csv(url, header=None, names=columns, delimiter='\t')
obje = np.array([])
df = dataset.iloc[q:z, 0:4]
obje = df.values
#print("typ obiektu: ",type(obje))
print("obiekt testowany:\n", obje)
# koniec w/w
data1 = DecisionSystem(url)
data = Knn(DecisionSystem(url))
print("dane obiektu testowanego z pliku:\n", data1.get_object(q), "\n")
print("decyzja obiektu testowanego z pliku: ", data1.get_decision(q), "\n")
print("data len: ",data1.__len__())

data.predict(obje, ka, "cos2")