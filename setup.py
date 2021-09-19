from setuptools import setup

setup(
    name='PySALESetup',
    version='1.0.1',
    packages=['PySALESetup'],
    url='https://github.com/jgd10/PySALESetup',
    license='MIT',
    author='jgd10',
    author_email='jgd10.github@gmail.com',
    description='iSALE setup package',
    install_requires=[
        'matplotlib~=3.4.3',
        'numpy~=1.21.2',
        'Shapely~=1.7.0',
        'scipy~=1.2.1'
    ]
)
