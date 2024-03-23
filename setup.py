from setuptools import setup, find_packages

setup(
    name="computer_vision_datasets",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "opencv-python",
        "numpy",
    ],  # Add your package dependencies here
    python_requires='>=3.6',
    author="Franco Marchesoni",
    author_email="marchesoniacland@gmail.com",
    description="Easy Computer Vision Datasets",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
