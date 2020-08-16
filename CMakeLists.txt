cmake_minimum_required(VERSION 3.14)
project(TMaP)

set(CMAKE_CXX_STANDARD 11)
set(COMMON_SOURCES include/Variable.h include/Function.h src/Function.cpp include/MyTypes.h src/Utils.cpp include/Utils.h src/GM.cpp include/GM.h src/Data.cpp include/Data.h src/CLT.cpp include/CLT.h src/MT.cpp include/MT.h  src/HyperParameters.cpp include/HyperParameters.h src/myRandom.cpp include/myRandom.h include/InferenceEngine.h src/BTP.cpp include/BTP.h src/MT_BTP.cpp include/MT_BTP.h include/BN_Sampler.h src/BN_Sampler.cpp include/SamplingFunction.h src/MT_Sampler.cpp include/MT_Sampler.h include/LogDouble.h src/LogDouble.cpp)

#add_library(Proposals include/Variable.h include/Function.h src/Function.cpp include/MyTypes.h src/Utils.cpp include/Utils.h src/GM.cpp include/GM.h src/Data.cpp include/Data.h target_src/main_learn.cpp)

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development REQUIRED)
FIND_PACKAGE(Boost 1.71.0 COMPONENTS python36 numpy36)
FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

message(STATUS "PYTHON_LIBRARIES = ${Python3_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message(STATUS "Python_Interpreter = ${PYTHON_VERSION_STRING}")

INCLUDE_DIRECTORIES(include ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.2.3
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    #add_subdirectory(pybind11)
    pybind11_add_module(pygmlib ${COMMON_SOURCES} src/PBPythonInterface.cpp)
endif()

message(${CMAKE_CXX_FLAGS_RELEASE})