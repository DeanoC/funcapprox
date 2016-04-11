
set(SOURCE_FILES binshared.h)
add_executable(funcapprox ${SOURCE_FILES} funcapprox.cpp)
target_link_libraries(funcapprox ${Boost_LIBRARIES} core)