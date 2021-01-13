# distutils: language=c++
import numpy as np
import pandas as pd
cimport numpy as cnp
import cython
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray target_mean_v6(data, str y_name, str x_name):
    cdef int n = data.shape[0]
    cdef cnp.ndarray[double] result = np.zeros(n,dtype=float)

    cdef cnp.ndarray[long] x_data = data[x_name].values
    cdef cnp.ndarray[long] y_data = data[y_name].values
    cdef cnp.ndarray[double] sum_data = np.zeros(10)
    cdef cnp.ndarray[double] count_x = np.zeros(10)
    cdef int i = 0

    for i in prange(n, nogil=True):
        sum_data[x_data[i]] += y_data[i]
        count_x[x_data[i]] += 1

    for i in prange(n,nogil=True):
        result[i] = (sum_data[x_data[i]] - y_data[i]) / (count_x[x_data[i]] - 1)
    return result



cpdef main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])



if __name__ == '__main__':
    main()