from setuptools import setup

version = {}
with open("./PySALESetup/_version.py") as fp:
    exec(fp.read(), version)

setup(
    name='PySALESetup',
    python_requires='>3.6, <3.10',
    version=version['__version__'],
    packages=['PySALESetup'],
    url='https://github.com/jgd10/PySALESetup',
    license='MIT',
    author='jgd10',
    author_email='jgd10.github@gmail.com',
    description='iSALE setup package',
    include_package_data=True,
    package_data={'PySALESetup.grain_library_directory': ['grain_library_directory/*'],
                  'PySALESetup.input_templates': ['input_templates/*']},
    install_requires=[
        'matplotlib~=3.4.3',
        'numpy~=1.21.2',
        'Shapely~=1.7.0',
        'scipy~=1.2.1'
    ]
)
