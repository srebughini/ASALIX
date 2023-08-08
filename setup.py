from setuptools import setup
from pathlib import Path

import codecs
import os.path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='asalix',
    version=get_version("asalix/__init__.py"),
    description='A comprehensive collection of mathematical tools and utilities designed to support Lean Six Sigma practitioners in their process improvement journey.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/srebughini/ASALIX',
    author='Stefano Rebughini',
    author_email='ste.rebu@outlook.it',
    license='GNU General Public License v3.0',
    packages=['asalix'],
    install_requires=['et-xmlfile==1.1.0',
                      'numpy==1.25.1',
                      'openpyxl==3.1.2',
                      'pandas==2.0.3',
                      'python-dateutil==2.8.2',
                      'pytz==2023.3',
                      'scipy==1.11.1',
                      'six==1.16.0',
                      'termcolor==2.3.0',
                      'tzdata==2023.3'],

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries'
    ]
)
