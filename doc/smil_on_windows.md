HOWTO: Build and use Smil on Windows
====================================

This guide will teach you the building process of the
[Smil](https://smil.cmm.mines-paristech.fr/doc/index.html) image
processing library on the Microsoft Windows platform, using the native
Visual C++ compiler.

Prerequisites
-------------

- The [Visual Studio Build
  Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools) is
  a set of compilers and libraries to build native software on MS Windows.
- [CMake](https://cmake.org/download/) is the build system used by Smil.
- [Swig](http://www.swig.org/download.html) is used to generate Python
  bindings from the Smil C++ API. You should download the *swigwin*
  version.
- [Anaconda](https://www.anaconda.com/download/) is a Python
  distribution loaded with scientific libraries, such as Numpy or
  Scipy. It also provides image processing libraries (libpng, libjpeg,
  tiff, zlib). To be installed globally, and added to PATH.
- [Qt](https://www.qt.io/download), although packaged by Anaconda,
  should be used in its standalone version to avoid a (still to be
  debugged) link error. You should prepend the Qt bin folder to the
  %PATH% environment variable (after installing Anaconda).
- [Git](https://git-scm.com/downloads) may also be needed to clone the
  Smil repository.

Build process
-------------

0. Install a sane keyboard layout, such as
   [Bépo](http://bepo.fr/wiki/Accueil)
1. Install the Visual Studio Build Tools, CMake, Anaconda and
   Qt. Uncompress the swigwin archive.
2. Get the Smil source (using Git?)
3. Create an empty build directory near/inside the Smil sources
   folder.
4. Open a *native* Visual Studio console, corresponding to your
   Windows version (x86 for 32 bits vs. x64 for 64 bits)
5. In this console, navigate using `cd` to the build directory you
   just created.
6. Launch CMake with the command `cmake-gui
   path/to/the/Smil/sources/folder`
7. Select the *Nmake* back-end.
8. Uncheck the *USE_OPEN_MP* and *USE_OPTIMIZATIONS* options (till
   they work...)
9. Check the *WRAP_PYTHON*, *USE_JPEG*, *USE_PNG*, *USE_TIFF*,
   *USE_QT*, *BUILD_TESTS* options and click on *Configure*. Some
   errors should be popping in the output console.
10. [Image encoding/decoding libraries] Check the
    *Advanced* button to show all the available CMake variables, and
    replace {PNG,JPEG,TIFF,ZLIB}_INCLUDE_DIR-NOTFOUND by
    `path/to/anaconda/Libraries/include`. Also replace
    {PNG,JPEG,TIFF,ZLIB}_LIBRARY-NOTFOUND by the corresponding
    libraries in the `path/to/anaconda/Libraries/lib` folder:
    - libjpeg.lib (this variant solves an unresolved symbol link error)
    - libpng.lib
    - libtiff_i.lib (this variant solves an unresolved symbol link error)
    - zlib.lib
11. [Python bindings] Points the *SWIG_EXECUTABLE* variable to
    `path/to/swig/swig.exe`. Switch the *CMAKE_BUILD_TYPE* to
    Release. Set the *PYTHON_NUMPY_INCLUDE_DIR* to
    `path/to/anaconda/Libs/site-packages/numpy/core/include`
12. [Qt GUI] Replace in *USE_QT_VERSION* 4 by 5, and point the
    *Qt5Widgets_DIR* to `path/to/Qt/lib/cmake/Qt5Widgets`.
13. [Building] Go back to the Visual Studio console, and type
    `nmake`. Wait for it... (You may have to add /bigobj to the
    *CMAKE_CXX_FLAGS* option)
14. Launch `nmake test` to launch Smil's test suite.
15. Set the PYTHONPATH environnement variable: `set
    PYTHONPATH=path/to/build/folder/lib`
16. Launch Python and try to import Smil: `import smilPython`

Alternatives
------------

### Using the Cygwin environment

[Cygwin](https://cygwin.com/install.html) tries to replicate an Unix
environment inside Windows. It provides a set of compatibles
open-source packages, such as GCC and Python, which can only be used
in this environment.

#### Pros

- an Unix-like environment
- a lot of open-source packages with dependency management

#### Cons

- still missing some packages, like python-scipy: we had to manually
  install the dependencies before `pip install`ing it
- we did not find any way to call OpenCV from Python inside Cygwin

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

#### Cons

- Uses MSVC link.exe. As a consequence, does not solve the link
  problem with Anaconda Qt.
