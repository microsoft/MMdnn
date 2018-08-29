#!/usr/bin/python
"""
To have a single pip command that uses the specific requirements file use this
in a shell script for posix OS::

  pip install -r $(select_requirements.py)

On windows, create a bat of cmd file that loads the windows-specific
requirements directly::

  for /f %%i in ('python select_requirements.py') do (set req_file="%%i")
  pip install -r %req_file%
"""

from __future__ import print_function

import os
import platform
import struct
import sys

# major python major_python_versions as python2 and python3
major_python_versions = tuple(map(str, platform.python_version_tuple()))
python2 = major_python_versions[0] == '2'
python3 = major_python_versions[0] == '3'


# operating system
sys_platform = str(sys.platform).lower()
linux = 'linux' in sys_platform
windows = 'win32' in sys_platform
cygwin = 'cygwin' in sys_platform
solaris = 'sunos' in sys_platform
macosx = 'darwin' in sys_platform
posix = 'posix' in os.name.lower()

def select_requirements_file():
    """
    Print the path to a requirements file based on some os/arch condition.
    """
    if windows:
        print('requirements/win.txt')
    elif macosx:
        print('requirements/mac.txt')
    elif linux:
        if python2:
            print('requirements/linux-py2.txt')
        elif python3:
            print('requirements/linux-py3.txt')
    elif cygwin:
        print('requirements/cygwin.txt')
    else:
        raise Exception('Unsupported OS/platform')

if __name__ == '__main__':
    select_requirements_file()