from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
setup(name='TPM',
      ext_modules = cythonize([
      Extension("Variable", ["Variable.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]), 
      Extension("Util", ["Util.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]), 
      Extension("Function", ["Function.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]),
      Extension("BN", ["BN.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]),
      Extension("MT", ["MT.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]),
      Extension("CNode", ["CNode.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]),
      Extension("CN", ["CN.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"]),
      Extension("MCN", ["MCN.pyx"], include_dirs=[numpy.get_include()], language="c++", extra_compile_args=["-std=c++14"])
      ])
)