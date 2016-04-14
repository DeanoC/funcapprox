
set(MODULE_NAME core)

set(SOURCE_FILES core.h core.cpp vectoralu.h vectoralu.cpp basiccppvectoralu.h basiccppvectoralu.cpp)

add_library(${MODULE_NAME} ${SOURCE_FILES})