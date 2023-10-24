from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='EDPS_neural_network',
    version='0.9.0',
    packages=find_packages(),
    url='',
    license='',
    author='Hudson Ferreira',
    author_email='hudsonfferreira@gmail.com',
    description='',
    install_requires=requirements
)
