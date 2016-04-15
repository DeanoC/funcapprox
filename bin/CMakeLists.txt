
set(SOURCE_FILES binshared.h realfunc.cpp realfunc.h)
add_executable(funcapprox ${SOURCE_FILES} funcapprox.cpp)
target_link_libraries(funcapprox ${Boost_LIBRARIES} core machinelearning)

add_library(bintest_lib ${SOURCE_FILES})