from typing import Any, Callable

import numpy as np
import pandas as pd

class DecisionSystem:
    __objects: np.ndarray
    __decisions: np.ndarray

    def __init__(self, filename: str) -> None:
        columny = ['sepal_len','sepal_width','petal_len','petal_width', 'class']
        dane = pd.read_csv(filename, header=None, names=columny, delimiter='\t')
        self.__objects = dane.values
        self.__decisions = dane.values
        pass  # wczytuje dane z pliku do zbioru obiektow i decyzji

    def get_object(self, i: int) -> np.ndarray:
        return self.__objects[i]
        pass  # zwraca obiekt o i-tym indeksie

    def get_decision(self, i: int) -> np.ndarray:
        return self.__decisions[i,5]
        pass  # zwraca decyzje dla obiektu o i-tym indeksie

    def __len__(self) -> int:
        return self.__objects.size
        pass  # zwraca liczbe obiektow w systemie decyzyjnym


class Knn:
    __decision_system: DecisionSystem

    def __init__(self, decision_system: DecisionSystem) -> None:
        self.__decision_system = decision_system
        pass  # przekazuje do instancji system decyzyjny

    def euklides(self, e1, e2):
        return np.linalg.norm(e1-e2)

    def Manhattan(self, e1, e2):
        dimension = len(e1)
        result = 0.0
        for i in range(dimension):
            result += abs(e1[i] - e2[i]) * 0.1
        return result


    def predict(self, obj: np.ndarray, k: int, metric: Callable[[np.ndarray], float]) -> Any:
        dists = {}
        obj = self.__decision_system.get_object(k)
        najblizsi_sasiedzi = self.find_closest_objects(obj,k,metric)
        print("k najbliższych sąsiadów: ", najblizsi_sasiedzi)
        rozmiar = len(najblizsi_sasiedzi)
        for i in range(rozmiar):
            d = self.euklides(najblizsi_sasiedzi[i], obj)
            dists[i] = d
            
        sasiedzi = sorted(dists, key=dists.get)[:k]
        print(sasiedzi)
        
        #self.__decision_system.predict(metric)
        pass  # zwraca przydzielona decyzje        

    def find_closest_objects(self, obj: np.ndarray, k: int,
                               metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
        tab_obje = {}
        self.obj = self.__decision_system.get_object(k)
        print("objekt testowy: ", self.obj)
        for i in range(k):
            obje = self.__decision_system.get_object(i)
            tab_obje[i] = obje
            #print(i, tab_obje[i])
        return tab_obje

        #self.__decision_system.find_closest_objects(metric)
        pass  # zwraca k najblizszych obiektow



plik = (
       "http://wmii.uwm.edu.pl/~bnowak/userfiles/downloads/dydaktyka/Metody_Inzynierii_Wiedzy/iris.txt"
       )

irys_a = DecisionSystem(plik)
irys_b = Knn(DecisionSystem(plik))
o = np.ndarray
irys_b.predict(o, 10, "met")
irys_b.find_closest_objects(o, 10, "met")