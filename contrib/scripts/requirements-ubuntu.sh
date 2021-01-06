#! /bin/bash

#  __HEAD__
#  Copyright (c) 2020, Centre de Morphologie Mathematique
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#     #  Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     #  Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     #  Nor the name of "Centre de Morphologie Mathematique" nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#  THE POSSIBILITY OF SUCH DAMAGE.
#
#  Description :
#    Install requirements prior to install or build Smil
#
#  History :
#    - 20/11/2020 - by Jose-Marcio Martins da Cruz
#      Just created this file
#
#  __HEAD__

Python3=yes
Qt5=yes
Doc=yes

doInstall()
{
  for arg in $*
  do
    Has=$(apt-cache search $arg 2>/dev/null)
    [ "$Has" = "" ] && continue
    apt-get -y install $arg
  done
}

# Required
doInstall git \
          gcc gcc++ \
          cmake cmake-data \
          cmake-extras extra-cmake-modules \
          cmake-curses-gui \
          swig \
          pkg-config


# Optional
#  to create doxygen documentation
if [ "$Doc" = "yes" ]
then
  doInstall doxygen doxygen-latex doxygen-doc \
            inkscape graphviz \
            topmenu-gtk2
fi

#
# libraries
doInstall libfreetype6 libfreetype6-dev
# image manipulation
doInstall libjpeg-turbo8-dev \
          libtiff5-dev \
          libpng++-dev
# to get an image from an URL
doInstall libcurl4-openssl-dev
#

#
# Recommended
#  CLI tools to manipulate and get information about image files
doInstall pngtools \
          libtiff-tools
#
# Option - pour les AddOns
# * AddOn FFT
doInstall libfftw3-dev libfftw3-double3
# * AddOn VTK
doInstall libvtk6-dev

if [ "Qt5" = "no" ]
then
  # Qt 4
  # With ccmake, set the value USE_QT_VERSION => 4
  doInstall qt4-qmake \
            qt4-dev-tools
            qt4-default
else
  # Qt 5
  # With ccmake, set the value USE_QT_VERSION => 5
  doInstall qt5-qmake \
            qtbase5-dev \
            libqwt-qt5-dev
fi


if [ "Python3" = "no" ]
then
  # Python 2.x
  # Required
  doInstall python python-dev
  # Recommended
  doInstall ipython
  doInstall python-numpy python-numpy-doc
  # Recommended to be able to manipulate 3d images
  doInstall python-libtiff
  # Option, if AddOn VTK is enabled
  doInstall python-vtk6
else
  # Python 3.x
  # Required
  doInstall python3 python3-dev
  # Recommended
  doInstall ipython3
  doInstall python3-numpy python3-numpy-dev python-numpy-doc
  # Recommended to be able to manipulate 3d images
  doInstall python3-pip
  pip3 install libtiff
  # Option, if AddOn VTK is enabled
  pip3 install vtk
fi
