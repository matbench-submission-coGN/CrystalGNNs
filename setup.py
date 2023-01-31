from setuptools import setup
from setuptools import find_packages  # or find_namespace_packages

setup(
    name='crystalgnns',
    version='0.0.1',
    package_dir={'':'src'},
    install_requires=[
        'networkx',
        'tensorflow',
        'pymatgen',
        'pyxtal',
        'kgcnn@git+https://github.com/aimat-lab/gcnn_keras'
    ],
    packages=find_packages(
        where='src', 
        include='crystalgnns*'
    ),
)
