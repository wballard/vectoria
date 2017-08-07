from libcpp.string cimport string
from builtins import bytes

cdef extern from "./fasttext/fasttext.h" namespace "fasttext":
    cdef cppclass FastText:
        void loadModel(const string&)
        void getVector(Vector&, const string&)
        int getDimension() const

cdef extern from "./fasttext/vector.h" namespace "fasttext":
    cdef cppclass Vector:
        Vector(int)
        float norm() const

cdef class FastTextWrapper:
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
        norm = vec.norm()
        del vec    
        return dims, norm
    

def loadFastText(filename, encoding='utf-8'):
    filename_bytes = bytes(filename, encoding)
    ft = FastTextWrapper()
    ft.fm.loadModel(filename_bytes)
    return ft