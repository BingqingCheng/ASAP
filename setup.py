import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asaplib",
    version="0.0.1",
    author="Bingqing Cheng",
    author_email="tonicbq@gmail.com",
    description="Automatic Selection And Prediction tools for materials and molecules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BingqingCheng/ASAP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=['scripts/clustering.py',
             'scripts/frame_select.py',
             'scripts/gen_soap_descriptors.py',
             'scripts/kernel_density_estimation.py',
             'scripts/kpca.py',
             'scripts/kpca_sparse.py',
             'scripts/krr.py',
             'scripts/pca.py',
             'scripts/kpca_for_projection_viewer.py',
             'scripts/ridge_regression.py',
             'scripts/tsne.py',
             'scripts/umap_reducer.py'])
