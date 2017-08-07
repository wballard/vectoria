from libcpp.string cimport string
from builtins import bytes

cdef extern from "./fasttext/fasttext.h" namespace "fasttext":
    cdef cppclass FastText:
        void loadModel(const string&)
        int getDimension() const

cdef class FastTextWrapper:
    cdef FastText *fm

    def __cinit__(self):
        self.fm = new FastText()

    def __dealloc(self):
        del self.fm

    def __getitem__(self, word):
        dims = self.fm.getDimension()
        return dims
    

def loadFastText(filename, encoding='utf-8'):
    filename_bytes = bytes(filename, encoding)
    ft = FastTextWrapper()
    ft.fm.loadModel(filename_bytes)
    return ft