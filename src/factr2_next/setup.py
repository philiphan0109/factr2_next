import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'factr2_next'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('factr2_next/config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='philip',
    maintainer_email='philipha@andrew.cmu.edu',
    description='NEXT external joint torque estimation for FACTR2.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'next_record = factr2_next.data_collection.recorder_node:main',
            'next_check_h5 = factr2_next.data_collection.check_h5:main',
            'next_train = factr2_next.training.train:main',
            'next_infer = factr2_next.inference.inference_node:main',
            'next_visualize = factr2_next.visualization.web_node:main',
            'next_eval_record = factr2_next.evaluation.recorder_node:main',
            'next_eval_plot = factr2_next.evaluation.plot_episode:main',
        ],
    },
)
