cmake_minimum_required(VERSION 3.14)
project(cplusplus-code)

set(CMAKE_CXX_STANDARD 14)
set(COMMON_SOURCES include/Variable.h include/Function.h src/Function.cpp include/MyTypes.h src/Utils.cpp include/Utils.h include/GM.h src/Data.cpp include/Data.h src/CLT.cpp include/CLT.h src/MT.cpp include/MT.h include/CN.h src/CN.cpp include/MCN.h src/MCN.cpp  src/HyperParameters.cpp include/HyperParameters.h src/myRandom.cpp include/myRandom.h include/InferenceEngine.h src/BTP.cpp include/BTP.h src/MT_BTP.cpp include/MT_BTP.h include/CN_BTP.h src/CN_BTP.cpp include/MCN_BTP.h src/MCN_BTP.cpp include/BN_Sampler.h src/BN_Sampler.cpp include/SamplingFunction.h src/MT_Sampler.cpp include/MT_Sampler.h include/CN_Sampler.h src/CN_Sampler.cpp include/MCN_Sampler.h src/MCN_Sampler.cpp)

#set(Python3_LIBRARIES "/usr/local/Cellar/python@3.8/3.8.11/Frameworks/Python.framework/Versions/3.8/lib/libpython3.8.dylib")
#set(Python3_EXECUTABLE "/usr/local/Cellar/python@3.8/3.8.11/Frameworks/Python.framework/Versions/3.8/bin/python3.8")
#set(Python3_INCLUDE_DIRS "/usr/local/Cellar/python@3.8/3.8.11/Frameworks/Python.framework/Versions/3.8/include/python3.8")
#set(Boost_INCLUDE_DIRS "/usr/local/opt/")



FIND_PACKAGE(Python3 COMPONENTS Interpreter Development REQUIRED)

INCLUDE(FindPythonLibs)
INCLUDE(FindPythonInterp)

IF(FEATURE_BOOST_BUNDLED_CONFIGURED)
  MESSAGE(FATAL_ERROR
    "DEAL_II_COMPONENT_PYTHON_BINDINGS has unmet configuration requirements: "
    "Python bindings require an external boost library, but deal.II was "
    "configured with bundled boost."
    )
ENDIF()

#
# Unset Boost_Found and run the low level FindBOOST.cmake module again to
# pick up libboost_python.so
FIND_PACKAGE(Boost 1.76 COMPONENTS python3 numpy3)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules/)

IF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR
    "DEAL_II_COMPONENT_PYTHON_BINDINGS has unmet configuration requirements: "
    "The external boost library does not provide Boost.Python"
    )
ENDIF()

#
# FIXME: Once finalized, reconsider moving this definitions into
# cmake/setup_dealii.cmake


#FIND_PACKAGE(Boost 1.76 COMPONENTS python numpy REQUIRED)

message(STATUS "Python_Interpreter = ${Python3_INTERPRETER_ID}")
message(STATUS "PYTHON_LIBRARIES = ${Python3_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "PYTHON_Version = ${Python3_VERSION}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(include ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 REQUIRED)
message(STATUS "PYBIND11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
pybind11_add_module(pygmlib ${COMMON_SOURCES} src/PBPythonInterface.cpp)

#include(FetchContent)
#FetchContent_Declare(
#        pybind11
#        GIT_REPOSITORY https://github.com/pybind/pybind11
#        GIT_TAG        v2.2.6
#)

#FetchContent_GetProperties(pybind11)
#if(NOT pybind11_POPULATED)
#    FetchContent_Populate(pybind11)
#    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
#    #add_subdirectory(pybind11)
#    pybind11_add_module(pygmlib ${COMMON_SOURCES} src/PBPythonInterface.cpp)
#endif()

message(${CMAKE_CXX_FLAGS_RELEASE})

#add_executable(ol ${COMMON_SOURCES} target_src/main_online_learning.cpp)
#add_executable(gen-samples ${COMMON_SOURCES} target_src/main_generate_samples.cpp)
#add_executable(marg_kl ${COMMON_SOURCES} target_src/main_marginal_kl.cpp)
add_executable(tw ${COMMON_SOURCES} target_src/modifyTreewidth.cpp)
#add_executable(learn-CN ${COMMON_SOURCES} target_src/main_cn.cpp)
add_executable(learn-oracle-MCN ${COMMON_SOURCES} target_src/main_mcn.cpp)
add_executable(learn-MCN ${COMMON_SOURCES} target_src/main_mcn.cpp)
add_executable(merge_mcns ${COMMON_SOURCES} target_src/main_merge_mcns.cpp)
add_executable(tree_uai ${COMMON_SOURCES} target_src/main_tree_uai.cpp)
add_executable(inf-bn ${COMMON_SOURCES} target_src/main_inf.cpp)
add_executable(inf-mcn ${COMMON_SOURCES} target_src/main_mcn_samples.cpp)

