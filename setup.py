from setuptools import setup, find_packages
import os

# Create a tracklib namespace package
packages = []
for root, dirs, files in os.walk('.'):
    if '__pycache__' in root or 'track_lib' in root or '.git' in root:
        continue
    if '__init__.py' in files or any(f.endswith('.py') for f in files):
        package = root.replace('./', '').replace('/', '.')
        if package and package != '.':
            packages.append(f'tracklib.{package}')

setup(
    name="tracklib",
    version="0.1.0",
    packages=['tracklib'] + packages,
    package_dir={'tracklib': '.'},
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "matplotlib",
        "pillow",
        "tqdm",
        "PyYAML",
        "scipy",
        "filterpy",
        "lap",
        "ultralytics",
        "wandb",
        "loguru",
        "scikit-learn",
    ],
    python_requires=">=3.8",
    author="TrackLib",
    description="A comprehensive tracking library",
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.txt', '*.md', '*.pth', '*.pt'],
    },
)