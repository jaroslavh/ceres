from numba import njit


@njit
def get_triangle_inequality_misses_and_hits(distances):
    misses = 0  # how many times did the triangle inequality not hold
    triangle_holds = 0

    l = distances.shape[0]
    for row in range(0, l - 1):
        for col in range(row + 1, l - 1):
            for p in range(col + 1, l):
                x = distances[row][col]
                y = distances[row][p]
                z = distances[col][p]
                if (x + y >= z) and (y + z >= x) and (z + x >= y):
                    triangle_holds += 1
                else:
                    misses += 1
    return misses, triangle_holds
