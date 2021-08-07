from distutils.core import setup
from setuptools import find_packages
from glob import glob
from os import environ as env, path
from Cython.Distutils import build_ext
import os
import sys
import numpy as np
import petsc4py
import versioneer

from firedrake_configuration import get_config

try:
    from Cython.Distutils.extension import Extension
    config = get_config()
    complex_mode = config['options'].get('complex', False)
except ImportError:
    # No Cython Extension means no complex mode!
    from distutils.extension import Extension
    complex_mode = False


def get_petsc_dir():
    try:
        petsc_dir = os.environ["PETSC_DIR"]
        petsc_arch = os.environ.get("PETSC_ARCH", "")
    except KeyError:
        try:
            petsc_dir = os.path.join(os.environ["VIRTUAL_ENV"], "src", "petsc")
            petsc_arch = "default"
        except KeyError:
            sys.exit("""Error: Firedrake venv not active.""")

    return (petsc_dir, path.join(petsc_dir, petsc_arch))


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext


cython_compile_time_env = {'COMPLEX': complex_mode}

cythonfiles = [("evalpatch", ["petsc"]), ]

petsc_dirs = get_petsc_dir()
if os.environ.get("HDF5_DIR"):
    petsc_dirs = petsc_dirs + (os.environ.get("HDF5_DIR"), )
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]
dirs = (sys.prefix, *petsc_dirs)
link_args = ["-L%s/lib" % d for d in dirs] + ["-Wl,-rpath,%s/lib" % d for d in dirs]

extensions = [Extension("fdutils.{}".format(ext),
                        sources=[os.path.join('fdutils', "{}.pyx").format(ext)],
                        include_dirs=include_dirs,
                        libraries=libs,
                        extra_link_args=link_args,
                        cython_compile_time_env=cython_compile_time_env) for (ext, libs) in cythonfiles]
if 'CC' not in env:
    env['CC'] = "mpicc"

setup(name='fdutils',
      version="0.0.1",
      cmdclass=cmdclass,
      description="""An patch for function evaluation""",
      # author="Imperial College London and others",
      # author_email="firedrake@imperial.ac.uk",
      # url="http://firedrakeproject.org",
      packages=find_packages(),
      package_data={"fdutils": ["locate.c",
                              "mpi-compat.h"]},
      # scripts=glob('scripts/*'),
      ext_modules=extensions)


