#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2016
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
"""
Main module where application-specific tasks are defined. The main procedure
is dependent on the fabfileTemplate module.
"""
import os, sys
from fabric.state import env
from fabric.colors import red
from fabric.operations import local
from fabric.decorators import task
from fabric.context_managers import settings, cd

from fabfileTemplate.utils import home

if sys.version_info.major == 3:
    import urllib.request as urllib2
else:
    import urllib2

# The following variable will define the Application name as well as directory
# structure and a number of other application specific names.
APP = 'ASTROVIS'

# The username to use by default on remote hosts where APP is being installed
# This user might be different from the initial username used to connect to the
# remote host, in which case it will be created first
APP_USER = APP.lower()

# Name of the directory where APP sources will be expanded on the target host
# This is relative to the APP_USER home directory
APP_SRC_DIR_NAME = APP.lower() + '_src'

# Name of the directory where APP root directory will be created
# This is relative to the APP_USER home directory
APP_ROOT_DIR_NAME = APP.upper()

# Name of the directory where a virtualenv will be created to host the APP
# software installation, plus the installation of all its related software
# This is relative to the APP_USER home directory
APP_INSTALL_DIR_NAME = APP.lower() + '_rt'

# Sticking with Python3.6 because that is available on AWS instances.
APP_PYTHON_VERSION = '3.6' # really requires 3.6 else await list comprehension does not work

# URL to download the correct Python version
APP_PYTHON_URL = 'https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz'

# NOTE: Make sure to modify the following lists to meet the requirements for
# the application.
APP_DATAFILES = []

# AWS specific settings
env.AWS_PROFILE = 'NGAS'
env.AWS_REGION = 'us-east-1'
env.AWS_AMI_NAME = 'Debian'
env.AWS_INSTANCES = 1
env.AWS_INSTANCE_TYPE = 't2.micro'
env.AWS_KEY_NAME = 'icrar_{0}'.format(APP_USER)
env.AWS_SEC_GROUP = 'NGAS' # Security group allows SSH and other ports
env.AWS_SUDO_USER = 'ec2-user' # required to install init scripts.

env.APP_NAME = APP
env.APP_USER = APP_USER
env.APP_INSTALL_DIR_NAME = APP_INSTALL_DIR_NAME
env.APP_ROOT_DIR_NAME = APP_ROOT_DIR_NAME
env.APP_SRC_DIR_NAME = APP_SRC_DIR_NAME
env.APP_PYTHON_VERSION = APP_PYTHON_VERSION
env.APP_PYTHON_URL = APP_PYTHON_URL
env.APP_DATAFILES = APP_DATAFILES
env.APP_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Alpha-sorted packages per package manager
env.pkgs = {
            'YUM_PACKAGES': [
                     'python36-devel',
                     'python-devel',
                     'readline-devel',
                     'openssl-devel',
                     'gcc',
                     'git',
                     'yasm',  
                     ],
            'APT_PACKAGES': [
                    'python3-venv',
                    'python3-dev',
                    'python-setuptools',
                    'gcc',
                    'make',
                    'zlib1g-dev',
                    'libssl-dev',
                    'libsqlite3-dev',
                    'libbz2-dev',
                    'libreadline-dev',
                    'pkg-config',
                    'ffmpeg',
                    'libavformat-dev',
                    'libavcodecdev',
                    'libavdevice-dev',
                    'libavutil-dev',
                    'libavfilter-dev',
                    'libswscale-dev',
                    'libswresample-dev',
                    'libopus-dev',
                    'libvpx-dev',
                    'git',
                    ],
            'SLES_PACKAGES': [
                    'python-devel',
                    'wget',
                    'zlib',
                    'zlib-devel',
                    'gcc',
                    ],
            'BREW_PACKAGES': [
                    'wget',
                    ],
            'PORT_PACKAGES': [
                    'wget',
                    ],
            'APP_EXTRA_PYTHON_PACKAGES': [
                    'jupyter',
                    'pycrypto',
                    'sphinx',
                    'numpy',
                    'uvloop',
                    'asyncio',
                    'bqplot',
                    'opencv-python',
                    ],
        }

# This dictionary defines the visible exported tasks.
__all__ = [
    'sysinitstart_EAGLE_and_check_status',
    'ffmpeg_install',
    'start_jupyter',
    'install_jupyterHub',
]

# >>> The following lines need to be after the definitions above!!!

from fabfileTemplate.utils import sudo, info, success, default_if_empty, run
from fabfileTemplate.system import check_command
from fabfileTemplate.APPcommon import virtualenv, APP_doc_dependencies, APP_source_dir, APP_root_dir
from fabfileTemplate.APPcommon import extra_python_packages, APP_user, build, APP_install_dir

def APP_build_cmd():

    # The installation of the bsddb package (needed by ngamsCore) is in
    # particular difficult because it requires some flags to be passed on
    # (particularly if using MacOSX's port
    # >>>> NOTE: This function potentially needs heavy customisation <<<<<<
    build_cmd = ['cd {0}; pip install -e .'.format(APP_SRC_DIR_NAME)]
    # linux_flavor = get_linux_flavor()

    env.APP_INSTALL_DIR = os.path.abspath(os.path.join(home(), APP_INSTALL_DIR_NAME))
    env.APP_ROOT_DIR = os.path.abspath(os.path.join(home(), APP_ROOT_DIR_NAME))
    env.APP_SRC_DIR = os.path.abspath(os.path.join(home(), APP_SRC_DIR_NAME))
 
    return ' '.join(build_cmd)


def install_sysv_init_script(nsd, nuser, cfgfile):
    """
    Install the uwsgi init script for an operational deployment of EAGLE.
    The init script is an old System V init system.
    In the presence of a systemd-enabled system we use the update-rc.d tool
    to enable the script as part of systemd (instead of the System V chkconfig
    tool which we use instead). The script is prepared to deal with both tools.
    """
    with settings(user=env.AWS_SUDO_USER):
        pass
    success("Init scripts installed")


@task
def sysinitstart_astrovis_and_check_status():
    """
    Starts the APP daemon process and checks that the server is up and running
    then it shuts down the server
    """
    # We sleep 2 here as it was found on Mac deployment to docker container
    # that the shell would exit before the APPDaemon could detach, thus
    # resulting in no startup self.
    #
    # Please replace following line with something meaningful
    # 
    pass


def dummy():
    pass

@task
def ffmpeg_install():
    """
    Compiles and installs the ffmpeg library required by pyastrovis
    Only required on AWS Linux instances. Debian has packages.
    """
    with cd('/tmp'):
        sudo('wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2')
        sudo('tar -vxjf ./ffmpeg-snapshot.tar.bz2')
    with cd ('/tmp/ffmpeg-snapshot'):
        sudo('./configure')
        sudo('make')
        sudo('make install')

@task
def start_jupyter():
    """
    Starts the jupyter server in the correct directory
    """
    with cd('{0}'.format(APP_SRC_DIR_NAME)):
        run('jupyter notebook --ip=0.0.0.0')

@task
def install_jupyterHub():
    """
    Install the tiniest JupyterHub
    """
    cmd = """curl https://raw.githubusercontent.com/jupyterhub/the-littlest-jupyterhub/master/bootstrap/bootstrap.py \
  | python3 - --admin jupyter"""
    sudo(cmd)


def npm_install():
    """
    Install Node.js to get access to the npm package manager required by pyastrovis
    NOTE: The code below works only for Debian style Linux
    """
    sudo('curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -')
    sudo('sudo apt-get install -y nodejs')


env.build_cmd = APP_build_cmd
env.APP_init_install_function = install_sysv_init_script
env.sysinitAPP_start_check_function = sysinitstart_astrovis_and_check_status
env.APP_extra_sudo_function = npm_install
