import numpy as np

maximum_k = 10000
maximum_norm = 10 ** 8
epsilon = 10 ** -8

identity_matrix = None
SIGNALS = {'continue': 0, 'converge': 1, 'diverge': 2, 'iteration': 3}


def iteration_schulz(matrix, vector):
    func = lambda a, b: b.dot(2 * identity_matrix - a.dot(b))
    count = 0
    while True:
        count += 1
        next_vector = func(matrix, vector)
        yield count, vector, next_vector
        vector = next_vector


def iteration_lili_1(matrix, vector):
    func = lambda a, b: b.dot(3 * identity_matrix - a.dot(b.dot(3 * identity_matrix - a.dot(b))))
    count = 0
    while True:
        count += 1
        next_vector = func(matrix, vector)
        yield count, vector, next_vector
        vector = next_vector


def iteration_lili_2(matrix, vector):
    func = lambda a, b: (identity_matrix + (1.0 / 4) * (identity_matrix - b.dot(a))).dot((3 * identity_matrix - b.dot(a))).dot(b)
    count = 0
    while True:
        count += 1
        next_vector = func(matrix, vector)
        yield count, vector, next_vector
        vector = next_vector


def norm_1(matrix):
    dimension = len(matrix)
    sums = []
    for idx in range(dimension):
        sums.append(sum(matrix[:, idx]))
    return max(sums)


def norm_infinity(matrix):
    dimension = len(matrix)
    sums = []
    for idx in range(dimension):
        sums.append(sum(matrix[idx, :]))
    return max(sums)


def get_vectors(matrix):
    dimension = len(matrix)
    vector_1 = matrix.T / (norm_1(matrix) * norm_infinity(matrix))
    vector_2 = np.diag([1.0 / nr for nr in matrix.diagonal()])
    total = 0
    for idx in range(dimension):
        for jdx in range(dimension):
            total += matrix[idx, jdx] ** 2
    total = total ** (1.0 / 2)
    vector_3 = (1 / total) * identity_matrix
    vector_4 = identity_matrix
    return vector_1, vector_2, vector_3, vector_4


def choose_vmat(amat):
    for idx, vector in enumerate(get_vectors(amat)):
        if norm_1(amat.dot(vector) - identity_matrix) < 1:
            return vector


def check(count, vmat, next_vmat):
    if count > maximum_k:
        return SIGNALS['iteration']
    norm = norm_1(next_vmat - vmat)
    if norm < epsilon:
        return SIGNALS['converge']
    if norm > maximum_norm:
        return SIGNALS['diverge']
    return SIGNALS['continue']


if __name__ == "__main__":
    dimension = int(input("Dimensiunea matricii\n"))
    algorithm = int(input("Algoritm folosit:\n1.Schultz\n2.Lili 1\n3.Lili 2\n"))
    if algorithm == 1:
        function_name = "Schulz"
        iteration_function = iteration_schulz
    elif algorithm == 2:
        function_name = "Lili 1"
        iteration_function = iteration_lili_1
    else:
        function_name = "Lili 2"
        iteration_function = iteration_lili_2
    matrix, identity_matrix = map(np.identity, [dimension] * 2)
    for idx in range(dimension - 1):
        matrix[idx, idx + 1] = 4
    print("Matrice:\n", matrix)
    print("Algoritm folosit: ", function_name)
    vector = choose_vmat(matrix)
    iteration_generator = iteration_function(matrix, vector)
    status = None
    next_vector = None
    for count, vector, next_vector in iteration_generator:
        status = check(count, vector, next_vector)
        if status != SIGNALS['continue']:
            break
    if status == SIGNALS['converge']:
        print("Converge:\n", next_vector)
        print("Norm: ", norm_1(matrix.dot(next_vector) - identity_matrix))
    else:
        if status == SIGNALS['diverge']:
            print("Diverge prin valoare")
        else:
            print("Diverge prin valoare numar de iteratii")
