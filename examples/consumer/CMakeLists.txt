# A small program that uses an installed namturbo:
#   cmake -S examples/consumer -B build-consumer -DCMAKE_PREFIX_PATH=<install prefix>
cmake_minimum_required(VERSION 3.16)
project(namturbo_consumer CXX)

find_package(namturbo CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE namturbo::namturbo)
