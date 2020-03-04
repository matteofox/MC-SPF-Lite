#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob
import subprocess

try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.build import build

except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.build import build

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

        for root, dirs, files in list(os.walk('pyspark')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build', 'dist', ):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("MC-SPF requires cython to install")


class build_ext(_build_ext):
    def build_extension(self, ext):
        _build_ext.build_extension(self, ext)
 

if __name__ == "__main__":

    include_dirs = ["include", numpy.get_include(),]

    cmodules = []
    cmodules += [Extension("mcspflite.utils.magtools", 
                           ["mcspflite/utils/magtools.pyx"], 
                           include_dirs=include_dirs)]
    ext_modules = cythonize(cmodules)


    scripts = ['scripts/'+file for file in os.listdir('scripts/')]  

    cmdclass = {'clean': CleanCommand,
                'build_ext': build_ext}
    
    from ctypes import cdll
    
    libname = 'libmultinest'
    libname += {
	'darwin' : '.dylib',
	'win32'  : '.dll',
	'cygwin' : '.dll',
    }.get(sys.platform, '.so')

    try:
      lib = cdll.LoadLibrary(libname)
    except:
      print('ERROR:   Could not load MultiNest library')
      print('ERROR:   You have to build it first and link it to your LD_LIBRARY_PATH')
      print('ERROR:   Instructions at: http://johannesbuchner.github.com/PyMultiNest/install.html')
      sys.exit(1)
    
    with open('mcspflite/_version.py') as f:
      exec(f.read())
      
    setup(
        name = "mcspflite",
        url="NO_URL",
        version= __version__,
        author="Matteo Fossati",
        author_email="matteo.fossati@durham.ac.uk",
        ext_modules = ext_modules,
	cmdclass = cmdclass,
        scripts = scripts, 
        packages=["mcspflite", 
	          "mcspflite.routines", 
		  "mcspflite.utils"],
        license="LICENSE",
        description="Monte-Carlo Stellar Population Fitter, (LITE version optimized for photometry)",
        install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mpi4py',
          'astropy',
	  'pymultinest',
	  'corner'],
        package_data={"": ["README.md", "LICENSE"],
	              "mcspflite": ["models/Dust_Emi_models/alpha_DH02.dat",
		      "models/Dust_Emi_models/spectra_DH02.dat",
		      "models/Dust_Emi_models/nebular_Byler.lines",
		      "models/Filters/FILTER_LIST",
		      "models/Filters/allfilters.dat",
		      "models/Filters/allindices.dat",
		      "models/Filters/filter_lambda_eff.dat",
		      "models/SPS/Models_exp_bc03hr.fits",
		      "models/SPS/Models_del_bc03hr.fits",
		      ]},
        include_package_data=True,
        zip_safe=False,
    )

