//
// Created by Dean Calver on 11/04/2016.
//
#include <iostream>
#include <boost/log/trivial.hpp>

int main() {
    BOOST_LOG_TRIVIAL(trace) << "funcapprox main starting...";
    std::cout << "Hello, World!" << std::endl;
    BOOST_LOG_TRIVIAL(trace) << "funcapprox main ending";
    return 0;
}