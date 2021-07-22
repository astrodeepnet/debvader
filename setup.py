from setuptools import setup

setup(
    name = 'debvader',
    url = 'https://github.com/BastienArcelin/debvader',
    author = 'Bastien Arcelin, Cyrille Doux, Biswajit Biswas',
    author_email = 'arcelin@apc.in2p3.fr',
    packages=['debvader'],
    install_requires=['numpy'],
    version='0.1',
    description="Galaxy deblender from variational autoencoders",
    # long_description=open('README.txt').read(),
    )