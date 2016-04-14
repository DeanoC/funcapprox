
set(MODULE_NAME machinelearning)

set(SOURCE_FILES machinelearning.cpp machinelearning.h machinelearning.cpp machinelearning.h layer.cpp layer.h ActivationFunction.cpp ActivationFunction.h ANNetwork.cpp ANNetwork.h)

add_library(${MODULE_NAME} ${SOURCE_FILES})