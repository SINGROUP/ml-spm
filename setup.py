import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build


class Build(build):
    """Custom build for setuptools to compile C++ shared libraries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        self.build()
        super().run()

    def build(self):
        current_dir = Path.cwd()
        source_dir = Path(__file__).resolve().parent / "mlspm" / "_c"
        os.chdir(source_dir)
        os.system("g++ -fPIC -O3 -c matching.cpp peaks.cpp")
        os.system("g++ -shared matching.o peaks.o -o mlspm_lib.so")
        os.chdir(current_dir)


setup(cmdclass={"build": Build}, has_ext_modules=lambda: True)
