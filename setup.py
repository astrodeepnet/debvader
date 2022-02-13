from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy==1.17.4",
    "tensorflow==2.1.0",
    "tensorflow-probability==0.9.0",
    "matplotlib",
    "jupyter",
    "sep",
    "scikit-image",
]

setup(
    name="debvader",
    version="0.0.81191",
    author="Bastien Arcelin, Cyrille Doux, Thomas Sainrat, Biswajit Biswas, Alexandre Boucaud",
    author_email="arcelin@apc.in2p3.fr",
    description="Galaxy deblender from variational autoencoders",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/BastienArcelin/debvader",
    include_package_data=True,
    packages=["debvader"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
    package_data={"debvader": ["data/*"]},
)
