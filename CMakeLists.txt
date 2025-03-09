cmake_minimum_required(VERSION 3.10)
project(MeuProjetoOpenMP LANGUAGES CXX)


find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(meu_executavel src/main.cpp)

target_link_libraries(meu_executavel PUBLIC OpenMP::OpenMP_CXX)