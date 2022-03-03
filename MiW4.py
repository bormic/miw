from typing import Any, List, Callable, Optional

class TreeNode:
    """
    klasa reprezntujaca wezel grafu gry
    value: wartosc odpowiadajaca biezacemu graczowi (antagonista/protagonista)
    state: wartosc odpowiadajaca dokonanemu ruchowi
    score: wartosc odpowiadajaca wynikowi w biezacym stanie
    children: lista obiektow wszystkich wezlow potomnych
    """

    value: Any
    state: Optional[Any]
    score: Optional[int]
    children: List['TreeNode']

    def __init__(self, value: Any, state: Optional[Any] = None, score: Optional[int] = 0) -> None:
        self.value = value
        self.state = state
        self.score = score
        self.children = []
        
        if (self.value == "Protagonista"):
            if (score < 21):
                self.add(TreeNode("Antagonista", 4, score + 4))
                self.add(TreeNode("Antagonista", 5, score + 5))
                self.add(TreeNode("Antagonista", 6, score + 6))

        if (self.value == "Antagonista"):
            if (score < 21):
                self.add(TreeNode("Protagonista", 4, score + 4))
                self.add(TreeNode("Protagonista", 5, score + 5))
                self.add(TreeNode("Protagonista", 6, score + 6))

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
  
    def add(self, child: 'TreeNode') -> None:
        self.children.append(child)
    
    def traverse_deep_first(self, visit: Callable[['TreeNode'], None]) -> None:
        visit(self)
        for child in self.children:
            child.traverse_deep_first(visit)
      
      
class Tree:
    root: TreeNode
    
    def __init__(self, node: TreeNode) -> None:
        self.root = node
      
    def traverse_deep_first(self, visit: Callable[[TreeNode], None]) -> None:
        for child in self.children:
            child.traverse_deep_first(visit)
        visit(self)
      
    def min_max(self) -> None:
        
        for child in self.root.children:
            Tree(child).min_max()

        if self.root.is_leaf is not True:
            if (self.root.value == "Protagonista"):
                self.root.podpowiedz = max(self.root.children[0].state, self.root.children[1].state,
                                           self.root.children[2].state)
            if (self.root.value == "Antagonista"):
                self.root.podpowiedz = min(self.root.children[0].state, self.root.children[1].state,
                                           self.root.children[2].state)
        pass  # nalezy zmodyfikowac wyniki w kazdym wezle tego grafu
            # wykonujemy rekurencyjnie metode min-max dla wszystkich wezlow potomnych

    def show(tree):
        pass  # wizualizujemy wszystkie wezly grafu gry wraz z krawedziami




start = Tree(TreeNode("Protagonista"))
start.min_max()
start.root.traverse_deep_first(Tree)
#start.drawtree(start.root)
