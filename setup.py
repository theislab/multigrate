from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.md').read_text('utf-8')

try:
    from multigrate import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(name='multigrate',
      version='1.0.0',
      description='Multi-omic data integration and transformation for single-cell genomics',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theislab/multigrate',
      author=__author__,
      author_email=__email__,
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
      ],
      )
