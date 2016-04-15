
set(SOURCE_FILES bin_check.cpp core_check.cpp main.cpp machinelearning_check.cpp)
add_executable(bintests ${SOURCE_FILES})
target_link_libraries(bintests ${Boost_LIBRARIES} core machinelearning gtest bintest_lib)