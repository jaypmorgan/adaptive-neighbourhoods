add_requires("pybind11")

target("main")
    set_kind("binary")
    add_headerfiles("adaptive_neighbourhoods/src/*.h")
    add_files("adaptive_neighbourhoods/src/*.cpp")
    set_languages("c++11")
    add_packages("pybind11")

