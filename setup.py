from setuptools import setup, find_packages
setup(
    name='ais',
    version='0.0.1',
    packages=find_packages(include=[
        'lmi_utils.*',
        'object_detectors.*',
        'anomoly_detectors.*',
        'classifiers.*']),
    include_package_data=True,
    description='LMI AIS Python Package',
    author='LMI AIS Team',
    install_requires=[
        'numpy >= 1.23.0',
        'opencv-python',
        'torch',
        'torchvision',
        'tqdm',
    ],  # Add any dependencies here
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)