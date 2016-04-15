
set(MODULE_NAME machinelearning)

set(SOURCE_FILES machinelearning.cpp machinelearning.h machinelearning.cpp machinelearning.h layer.cpp layer.h ActivationFunction.cpp ActivationFunction.h ANNetwork.cpp ANNetwork.h connections.cpp connections.h inputlayer.cpp inputlayer.h hiddenlayer.cpp hiddenlayer.h outputlayer.cpp outputlayer.h)

add_library(${MODULE_NAME} ${SOURCE_FILES})