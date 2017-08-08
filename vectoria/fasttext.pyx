from libcpp.string cimport string
from builtins import bytes
import numpy as np
cimport numpy as cnp

cdef extern from "./fasttext/fasttext.h" namespace "fasttext":
    cdef cppclass FastText:
        void loadModel(const string&)
        void getVector(Vector&, const string&)
        int getDimension() const

cdef extern from "./fasttext/vector.h" namespace "fasttext":
    cdef cppclass Vector:
        int m_
        float *data_
        Vector(int)
        float norm() const

cdef class VectorFinalizer:
    """
    Memory management wrapper to allow a zero copy projection
    of fasttext::Vector as a numpy array.
    """
    cdef Vector *_vec

    def __dealloc__(self):
        del self._vec

cdef cnp.ndarray numpy_wrap_vector(Vector *vec):
    cdef float[:] memory = <float[:vec.m_]>vec.data_
    cdef cnp.ndarray array = np.asarray(memory)
    fin = VectorFinalizer()
    fin._vec = vec
    cnp.set_array_base(array, fin)
    return array

cdef class FastTextModelWrapper:
    """
    Wrapper for a pretrained FastText model. This will provide
    [word] -> numpy array dense vector encoding for a single string.

    A zero copy share of the underlying C++ memory is utilized in the
    interest of performance.
    """
    cdef FastText *fm

    def __cinit__(self):
        self.fm = new FastText()

    def __dealloc(self):
        del self.fm

    def __getitem__(self, word):
        cdef int dims = int(self.fm.getDimension())
        word_bytes = bytes(word, 'utf-8')
        vec = new Vector(dims)
        self.fm.getVector(vec[0], word_bytes)
        return numpy_wrap_vector(vec)
    

def load_model(filename, encoding='utf-8'):
    filename_bytes = bytes(filename, encoding)
    ft = FastTextModelWrapper()
    ft.fm.loadModel(filename_bytes)
    return ft