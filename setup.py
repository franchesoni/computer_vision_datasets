from setuptools import setup, find_packages

setup(
    name="computer_vision_datasets",
    version="0.2.2",
    packages=find_packages(include=['computer_vision_datasets', 'computer_vision_datasets.*']),
    install_requires=[
        "tqdm",
        "opencv-python",
        "numpy",
        "requests",
        "fire",
    ],  # Add your package dependencies here
    entry_points={
        'console_scripts': [
            'list-ninja:computer_vision_datasets.module:print_datasets',
        ]
    }
    python_requires='>=3.6',
    author="Franco Marchesoni",
    author_email="marchesoniacland@gmail.com",
    description="Easy Computer Vision Datasets",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
