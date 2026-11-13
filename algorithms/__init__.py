"""Meta-sezgisel algoritmalar: GA, ACO, Binary PSO, SA ve GA-SA hibriti."""

from algorithms.ant_colony import AntColonyOptimization
from algorithms.base import BaseOptimizer, OptimizationResult
from algorithms.binary_pso import BinaryPSO
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.hybrid_ga_sa import HybridGASA
from algorithms.simulated_annealing import SimulatedAnnealing

#: Algoritma adı -> sınıf (deney betikleri bu tablo üzerinden örnek üretir).
ALGORITHM_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "GA": GeneticAlgorithm,
    "ACO": AntColonyOptimization,
    "BPSO": BinaryPSO,
    "SA": SimulatedAnnealing,
    "GA-SA": HybridGASA,
}

__all__ = [
    "ALGORITHM_REGISTRY",
    "AntColonyOptimization",
    "BaseOptimizer",
    "BinaryPSO",
    "GeneticAlgorithm",
    "HybridGASA",
    "OptimizationResult",
    "SimulatedAnnealing",
]
