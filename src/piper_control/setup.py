from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'piper_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('piper_control/configs/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('piper_control/urdf/*.urdf')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'PyYAML',
        "python-can",
        "typing_extensions",
        "piper_sdk"
        ],
    zip_safe=True,
    maintainer='Philip Han',
    maintainer_email='philiphan0109@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "piper_node = piper_control.piper_node:main"
        ],
    },
)
