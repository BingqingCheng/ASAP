import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asaplib",
    version="0.0.1",
    author="Bingqing Cheng",
    author_email="tonicbq@gmail.com",
    description=
    "Automatic Selection And Prediction tools for materials and molecules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BingqingCheng/ASAP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'dscribe==1.2.2', 'click>=7.0', 'numpy', 'scipy', 'scikit-learn',
        'ase', 'umap-learn', 'PyYAML', 'tqdm', 'pandas'
    ],
    extras_require={'testing': ['pytest>=5.0']},
    python_requires='>=3.7',
    entry_points="""
    [console_scripts]
    asap=asaplib.cli.cmd_asap:asap
    """)
