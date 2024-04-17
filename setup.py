import os
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


## download EIGEN if it doesn't exist in build.
DIR = os.path.abspath(os.path.dirname(__file__))
EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
EIGEN3_DIRNAME = "eigen-3.3.7"
if not os.path.exists(EIGEN3_DIRNAME):
    import zipfile
    import urllib.request
    download_target = "eigen.zip"
    response = urllib.request.urlretrieve(EIGEN3_URL, download_target)
    with zipfile.ZipFile(download_target) as f:
        f.extractall()    

ext_modules = [
    Pybind11Extension(
        "adaptive_neighbourhoods",
        sorted(glob("src/*.cpp")),
        include_dirs=[EIGEN3_DIRNAME],
        extra_compile_args=["-fopenmp", "-O2"],
        extra_link_args=['-lgomp'],
    ),
]

setup(
    ext_modules=ext_modules,
)
