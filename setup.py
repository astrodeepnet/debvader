from setuptools import setup, find_packages

# setup(
#     name = 'debvader',
#     url = 'https://github.com/BastienArcelin/debvader',
#     author = 'Bastien Arcelin, Cyrille Doux, Biswajit Biswas',
#     author_email = 'arcelin@apc.in2p3.fr',
#     packages=['debvader'],
#     install_requires=['numpy'],
#     version='0.1',
#     description="Galaxy deblender from variational autoencoders",
#     # long_description=open('README.txt').read(),
#     )


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy==1.17.4","tensorflow==2.1.0","tensorflow-probability==0.9.0"]

setup(
    name="debvader",
    version="0.0.111",
    author="Bastien Arcelin, Cyrille Doux, Biswajit Biswas",
    author_email="arcelin@apc.in2p3.fr",
    description="Galaxy deblender from variational autoencoders",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/BastienArcelin/debvader",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
)