from setuptools import setup

version = {}
with open("./PySALESetup/_version.py") as fp:
    exec(fp.read(), version)

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='PySALESetup',
    python_requires='>3.6, <3.10',
    version=version['__version__'],
    packages=['PySALESetup'],
    url='https://github.com/jgd10/PySALESetup',
    license='MIT',
    author='James Derrick',
    author_email='jgd10.github@gmail.com',
    description='iSALE setup package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        'PySALESetup.grain_library_directory':
            ['grain_library_directory/*'],
        'PySALESetup.input_templates':
            ['input_templates/*']
    },
    install_requires=[
        'matplotlib~=3.4.3',
        'numpy~=1.21.2',
        'Shapely~=1.7.0',
        'scipy~=1.2.1'
    ]
)
