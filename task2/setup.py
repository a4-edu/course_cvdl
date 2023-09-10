import os
from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


setup(
    name='a4_course_cvdl_t2',
    version='0.2.0',
    python_requires='>=3.7.0',
    packages=[''],
    description='Task2 for A4 CV DL course: CenterNet',
    url='',
    author='Boris Zimka',
    author_email='zimka@phystech.edu',
    install_requires=['numpy', 'torch', 'segmentation-models-pytorch', 'torchvision'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering'
    ],
)