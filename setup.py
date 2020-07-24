from __future__ import absolute_import
from setuptools import setup, find_packages
from io import open

# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mmdnn',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.3.1',

    description='Deep learning model converter, visualization and editor.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/Microsoft/MMdnn',

    # Author details
    author = 'System Research Group, Microsoft Research Asia',
    author_email='mmdnn_feedback@microsoft.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
    ],

    # What does your project relate to?
    keywords='deep learning model converter visualization',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    package_data={
        'mmdnn':['visualization/public/*',
                'visualization/*.json',
                'visualization/*.js',
                'visualization/*.html',
                'visualization/*.css']
    },

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy >= 1.15.0',
        'protobuf >= 3.6.0',
        'six >= 1.10.0',
        'pillow >= 6.2.1',
    ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'mmconvert  = mmdnn.conversion._script.convert:_main',
            'mmdownload = mmdnn.conversion._script.extractModel:_main',
            'mmvismeta  = mmdnn.conversion.examples.tensorflow.vis_meta:_main',
            'mmtoir     = mmdnn.conversion._script.convertToIR:_main',
            'mmtocode   = mmdnn.conversion._script.IRToCode:_main',
            'mmtomodel  = mmdnn.conversion._script.dump_code:_main',
        ],
    },
)
