# setup script for pyimcom_croutines


import numpy as np
from setuptools import Extension, setup

# C routines
ec = ["-fopenmp", "-O2"]
el = ["-fopenmp"]

setup(
    name="furry_parakeet",
    ext_modules=[
        Extension(
            "furry_parakeet.pyimcom_croutines",
            sources=["src/furry_parakeet/pyimcom_croutines.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=ec,
            extra_link_args=el,
        )
    ],
)
