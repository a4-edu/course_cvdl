import os
from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


setup(
    name='a4_course_cvdl_t3',
    version='0.1.1',
    python_requires='>=3.7.0',
    packages=[''],
    description='Task3 for A4 CV DL course: Text Detection Challenge',
    url='',
    author='Boris Zimka',
    author_email='zimka@phystech.edu',
    install_requires=['numpy', 'matplotlib', 'editdistance', 'requests'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering'
    ],
)