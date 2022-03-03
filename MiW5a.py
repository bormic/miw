from typing import Any, Callable

import numpy as np
import pandas as pd
import numpy_ml 

def euclidian_dist(e1, e2):
    return np.linalg.norm(e1-e2)

def Manhattan_distances(point1, point2):
    dimension = len(point2)
    result = 0.0
    for i in range(dimension):
        result += abs(point1[i] - point2[i]) * 0.1
        #result = sum(result)
    return result

def Minkowski_distances(A, B, p):
    A = A[:5]
    return numpy_ml.utils.distance_metrics.minkowski(A, B, p)

def chebyshev_dist(x, y):
    x = x[:5]
    return numpy_ml.utils.distance_metrics.chebyshev(x, y)

def hamming_dist(A, B):
    return np.count_nonzero(A != B, axis=-1)

def norm_data(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)



class DecisionSystem:
    __objects: np.ndarray
    __decisions: np.ndarray

    def __init__(self, filename: str) -> None:
        self.file_columns = ['sepal_len','sepal_width','petal_len','petal_width', 'class']
        self.dataset = pd.read_csv(filename, header=None, names=self.file_columns, delimiter='\t')
        self.__objects = self.dataset.values
        self.__decisions = self.dataset.values
        pass  # wczytuje dane z pliku do zbioru obiektow i decyzji

    def get_object(self, i: int) -> np.ndarray:
        return self.dataset.loc[i]
        pass  # zwraca obiekt o i-tym indeksie

    def get_decision(self, i: int) -> np.ndarray:
        return self.dataset.loc[i].at['class']
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
        #obj = np.delete(obj, int(obj[4]))
        #print("TEST: ", obj)
        train_set = train_set = self.find_closest_objects(obj,k,metric)
        
        dists, train_size = {}, len(train_set)
            
        for i in range(train_size):
            #d = Manhattan_distances(train_set[i], obj)
            #d = chebyshev_dist(train_set[i], obj)
            #d = Minkowski_distances(train_set[i], obj, p=2) # p=1 lub p=2
            d = euclidian_dist(train_set[i,4], obj)
            #d = hamming_dist(train_set[i,4], obj)
            dists[i] = d
        
        k_neighbors = sorted(dists, key=dists.get)[:k]

        qty_label1, qty_label2, qty_label3 = 0, 0, 0
        for index in k_neighbors:
            if train_set[index][-1] == 1.0:
                qty_label1 += 1
            elif train_set[index][-1] == 2.0:
                qty_label2 += 1
            else:
                qty_label3 += 1
        
        print("zliczone drogi: |1:", qty_label1, "|2:", qty_label2, "|3:", qty_label3)
        
        if qty_label1 >= qty_label2 and qty_label1 >= qty_label3:
            return print("Wyuczona decyzja to: 1.0")
        elif qty_label2 >= qty_label1 and qty_label2 >= qty_label3:
            return print("Wyuczona decyzja to: 2.0")
        else:
            return print("Wyuczona decyzja to: 3.0")
        
        self.__decision_system.predict(metric)
        pass  # zwraca przydzielona decyzje

    def find_closest_objects(self, obj: np.ndarray, k: int,
                               metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
        
        idx = int(obj[-1])
        #print("TEST idx: ", idx)
        dafr = self.__decision_system.dataset.iloc[0:, 0:5]
        dafr = dafr.sort_values("sepal_width", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
        pe1 = k/2
        pe2 = idx + pe1
        pe3 = idx - pe1
        if pe3<0:
            pe2 += abs(pe3)
            pe3 = 0
        if pe2>150:
            pe2=150
            pe3 += pe2-150
        dafr = dafr.iloc[int(pe3):int(pe2), 0:5]
        #dafr = dafr.sample(k)
        dafr = dafr.values
        return dafr

        #self.__decision_system.find_closest_objects(metric)
        pass  # zwraca k najblizszych obiektow


url = (
       "http://wmii.uwm.edu.pl/~bnowak/userfiles/downloads/dydaktyka/Metody_Inzynierii_Wiedzy/iris.txt"
       )
file = ("C:/Users/nonew/Desktop/III-lato/MiW/cw5/iris.txt")


# zmienne
q = 55 # wybiera obiekty od q min 0
z = 56 # wybiera obiekty do z-1 max 150
ka = 49 # ka najbliższych sąsiadów

# wczytanie danych do wybrania/wygenerowania obiektu
columns = ['sepal_len','sepal_width','petal_len','petal_width','class']
datasets = pd.read_csv(url, header=None, names=columns, delimiter='\t')
obje = np.array([])
df = datasets.iloc[q:z, 0:4]
idi = df.index
obje = df.values
obje = np.append(obje, idi)
#print("typ obiektu: ",type(obje))
print("obiekt testowany (indeks na koncu):\n", obje)
# koniec w/w

data1 = DecisionSystem(url)
data = Knn(DecisionSystem(url))

print("SPRAWDZENIE:\n")
print("dane obiektu testowanego z pliku:\n", data1.get_object(q), "\n")
print("decyzja obiektu testowanego z pliku: ", data1.get_decision(q), "\n")
print("len to obiektow: ",int((data1.__len__())/5), "i danych: ", data1.__len__())

data.predict(obje, ka, "cos")
data.find_closest_objects(obje, ka, "cos")