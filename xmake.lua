add_requires("pybind11")

target("main")
    set_kind("binary")
    add_headerfiles("src/*.h")
    add_files("src/*.cpp")
    set_languages("c++11")
    add_packages("pybind11")

