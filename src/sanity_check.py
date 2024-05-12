import algorithms
import random
from src import similarities

non_hom = [{str(i):0 for i in range(30)} for j in range(30)]
hom = [{str(i):288 for i in range(30)} for j in range(30)]
semi_hom = [{str(i):random.randrange(0, 5) for i in range(30)} for j in range(30)]

coverage = 0.95

res_non_hom = algorithms.nndescent_reverse_neighbors(non_hom,
                                                     coverage=coverage,
                                                     sample_rate=0.5,
                                                     similarity=freq_similarity,
                                                     K = 7,
                                                     threshold=0.8)
print(len(res_non_hom))
res_hom = algorithms.nndescent_reverse_neighbors(hom,
                                                 coverage=coverage,
                                                 sample_rate=0.5,
                                                 similarity=freq_similarity,
                                                 K = 7,
                                                 threshold=0.8)
print(len(res_hom))
res_semi_hom = algorithms.nndescent_reverse_neighbors(semi_hom,
                                                      coverage=coverage,
                                                      sample_rate=0.5,
                                                      similarity=freq_similarity,
                                                      K = 7,
                                                      threshold=0.8)
print(len(res_semi_hom))
