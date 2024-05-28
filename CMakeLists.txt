cmake_minimum_required(VERSION 3.5)
project(
  ipu_multi_models
  VERSION 1.0.0
  LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")

# unreferenced formal parameter
string(APPEND CMAKE_CXX_FLAGS " /wd4100")
string(APPEND CMAKE_CXX_FLAGS " /EHsc")

string(APPEND CMAKE_CXX_FLAGS " /wd4996")
string(APPEND CMAKE_CXX_FLAGS " /WX")
string(APPEND CMAKE_CXX_FLAGS " /W0")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zi")


find_package(onnxruntime)
find_package(OpenCV COMPONENTS core highgui imgproc)
add_subdirectory(src)
