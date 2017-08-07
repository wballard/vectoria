from libcpp.string cimport string
from builtins import bytes

cdef extern from "../fasttext/fasttext/cpp/src/fasttext.h" namespace "fasttext":
    cdef cppclass FastText:
        void loadModel(const string&)

cdef class FastTextWrapper:
    cdef FastText *fm

    def __cinit__(self):
        self.fm = new FastText()

    def __dealloc(self):
        del self.fm

def loadFastText(filename, encoding='utf-8'):
    filename_bytes = bytes(filename, encoding)
    ft = FastTextWrapper()
    ft.loadModel(filename_bytes)
    return ft