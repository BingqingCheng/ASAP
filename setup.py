import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ASAP2", # Replace with your own username
    version="0.0.12",
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
    scripts=['asap/gen_soap_descriptors.py', 'asap/frame_select.py', 'asap/clustering.py',
             'asap/gen_soap_kmat.py', 'asap/kernel_density_estimation.py', 'asap/kpca.py',
             'asap/krr.py', 'asap/pca.py', 'asap/ridge_regression.py'
             ],
)
