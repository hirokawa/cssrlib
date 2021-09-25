import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cssrlib",
    version="0.0.1",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "cbitstruct",
        "ephem",
    ],
    author='Rui Hirokawa',
    author_email="rui.hirokawa@gmail.com",
    description="Compact SSR Library for PPP/PPP-RTK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hirokawa/cssrlib",
    packages=setuptools.find_packages(),
    package_data={
        "cssrlib": ["data/*.*", "tests/*.py"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
