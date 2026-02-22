
from pathlib import Path
import sys

sys.path.append(str(Path().cwd().parent))
import networkx as nx
import numpy as np
from matrix import are_similar
from itertools import combinations
from numpy import trace
from networkx import is_bipartite



tolerance = 1e-10