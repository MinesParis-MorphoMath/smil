HOWTO: Build and use Smil on Windows
====================================

This guide is an attempt to describe the building process of the
[Smil](https://smil.cmm.mines-paristech.fr/doc/index.html) image
processing library on the Microsoft Windows platform, from scratch,
using the native Visual C++ compiler.

Dependencies
------------

- The [Visual Studio Build
  Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools) is
  a set of compilers and libraries to build native software on MS Windows.
- [CMake](https://cmake.org/download/) is the build system used by Smil.
- [Swig](http://www.swig.org/download.html) is used to generate Python
  bindings from the Smil C++ API. Select the *swigwin* version, which
  contains the Windows pre-built executable.
- [Anaconda](https://www.anaconda.com/download/) is a Python
  distribution loaded with scientific libraries, such as Numpy or
  Scipy. It also provides some useful libraries (libcurl, libfreetype,
  libpng, libjpeg, libtiff, zlib, Qt). Choose the Python3 variant, since
  the Python2 one is known to be not fully compatible with Smil during
  the MSVC build.
- [Git](https://git-scm.com/downloads) may also be needed to clone the
  Smil repository.

Installing the dependencies
---------------------------

0. Download and install a sane keyboard layout, such as
   [Bépo](http://bepo.fr/wiki/Accueil)
1. Install the VS Build Tools, CMake and Git (in whichever
   order). Make sure that CMake is added to the PATH environment
   variable for all users. Reboot Windows.
2. Install Anaconda for all users, and add Anaconda to the PATH, in
   order for CMake to automatically find the dependencies.
4. Download and decompress the *swigwin* archive. Edit the PATH
   environment variable and add the Swig executable folder.

Build process
-------------

1. Get the Smil source (using Git?).
2. Create an empty build directory near or inside the Smil sources
   folder.
3. Open a *native* Visual Studio console, corresponding to your
   Windows version (x86 for a 32 bits Windows vs. x64 for 64 bits)
4. In this console, navigate using `cd` to the build directory you
   just created.
5. Launch CMake with `cmake-gui path/to/the/Smil/sources/folder`.
6. Select the *Nmake* back-end.
7. Check the *WRAP_PYTHON* option and click on *Configure*. Some
   errors may be popping in the output console.
8. [Numpy] Set the *PYTHON_NUMPY_INCLUDE_DIR* to
   `path/to/anaconda/Libs/site-packages/numpy/core/include`
9. Click on *Configure* then, if there is no errors, on *Generate*.
10. Go back to the Visual Studio console, and type `nmake`. Wait for
    it…
11. Set the PYTHONPATH environment variable: `set
    PYTHONPATH=path/to/build/folder/lib`
12. Launch Python and try to import Smil: `import smilPython`

Caveats
-------

* With Anaconda2 (Python 2), some embedded libraries cause link
  errors. One solution is to install a separate
  [Qt](https://www.qt.io/download) run-time, and prepend it to the
  PATH environment variable (before Anaconda paths), to be detected by
  CMake.
* OpenMP is not (yet) supported by Smil with MSVC
* Cannot build a debug version of Smil with the Python wrapper,
  because Anaconda does not provides any Python library with debug
  symbols (python**_d.lib). That's why CMake sets automatically the
  build type to Release when asking for the Python wrapper on Windows.

Alternatives
------------

### Using the Cygwin environment

[Cygwin](https://cygwin.com/install.html) tries to replicate an Unix
environment inside Windows. It provides a set of compatibles
open-source packages, such as GCC and Python, which can only be used
in this environment.

#### Pros

- An Unix-like environment.
- A package manager with quite a few open-source packages.

#### Cons

- Still missing some packages, such as python-scipy, which has to be
  manually installed with `pip` after its dependencies.
- We did not find any way to install pyopencv or call OpenCV from
  Python inside Cygwin.

### Using Clang/LLVM with the MSVC back-end

[Clang/LLVM](http://releases.llvm.org/download.html) is a
multi-platform, open-source, C/C++ compiler software. It can link to
Visual C++ libraries, thus providing a native alternative to the
Microsoft C++ compiler. Google has managed to use it to [build its web
browsers Chrome and
Chromium](http://blog.llvm.org/2018/03/clang-is-now-used-to-build-chrome-for.html).

The Visual Studio Build Tools are still needed, though.

To use it, change the *CLANG_CXX_COMPILER* option to point to
`LLVM/bin/clang-cl.exe`, a binary that accepts MSVC's cl.exe CLI
flags.
