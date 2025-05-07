cmake_minimum_required(VERSION 3.28)
project(SFMLProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.x
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

add_executable(main 
    src/main.cpp
    src/env/Environment.cpp
    src/env/Rewarder.cpp
    src/env/RolloutBuffer.cpp
    src/env/RungeKutta.cpp
    src/rl/Networks.cpp
    src/window/Window.cpp
)

target_link_libraries(main PRIVATE SFML::Graphics)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)


if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Configuring for AppleClang with OpenMP and Apple M2 optimization")

    # Specifiy the include and library directives for libomp
    target_include_directories(main PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(main PRIVATE /opt/homebrew/opt/libomp/lib)

    # Link against the OpenMP library
    target_link_libraries(main PRIVATE omp
        -fsanitize=address
    )

    # Add compiler options for OpenMP and Apple M2 optimization
    target_compile_options(main PRIVATE
        -Xpreprocessor -fopenmp
        -mcpu=apple-m2
        -fsanitize=address -g
    )

endif()

