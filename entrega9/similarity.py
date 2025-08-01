import numpy as np

def compute_similarity(vector1: list, vector2: list):
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.
    
    # Cálculo de la similitud de Pearson entre dos usuarios

    vec1 = np.array(vector1)
    vec2 = np.array(vector2)

    mask = ~np.isnan(vec1) & ~np.isnan(vec2)    # Cojo solo aquellas películas que ambos usuarios han calificado
    vec1_common = vec1[mask]    # Aplico la máscara para obtener las calificaciones
    vec2_common = vec2[mask]

    if len(vec1_common) == 0:   # Si no comparten películas en común su similitud es 0.0
        return 0.0
    
    mean_vec1 = np.mean(vec1[np.isfinite(vec1)])    # Calculo la media de puntuación de cada usuario aplicando una máscara para eliminar NaN o valores infinitos
    mean_vec2 = np.mean(vec2[np.isfinite(vec2)])

    numerator = np.sum((vec1_common - mean_vec1) * (vec2_common - mean_vec2))
    denominator = np.sqrt(np.sum((vec1_common - mean_vec1) ** 2)) * np.sqrt(np.sum((vec2_common - mean_vec2) ** 2))

    if denominator == 0:
        return 0.0
    
    return numerator / denominator

    

if __name__ == "__main__":
    
    vector_a, vector_b = [4, 4, 4, 4], [1, 1, 1, 1]
    print(compute_similarity(vector_a, vector_b))
    