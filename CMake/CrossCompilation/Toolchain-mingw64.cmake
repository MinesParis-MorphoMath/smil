# Toolchain configuration for SUSE/Fedora like MinGW32 setup

# the name of the target operating system
set(CMAKE_SYSTEM_NAME Windows)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

# here is the target environment located
set(CMAKE_FIND_ROOT_PATH
    /usr/x86_64-w64-mingw32/
    /usr/lib/gcc/x86_64-w64-mingw32/
    /usr/lib/gcc/x86_64-w64-mingw32/4.6/
    ~/src/w64libs/
    ~/src/w64libs/include/
    ~/src/w64libs/bin/
    ~/src/w64libs/Qt/4.8.1/
    ~/src/w64libs/Qt/4.8.1/bin/
    ~/src/w64libs/Qt/4.8.1/include/
    ~/src/w64libs/Python/
    ~/src/w32libs/java_jdk1.6.0_25/
    ~/src/w32libs/java_jdk1.6.0_25/include/
    ~/src/w32libs/java_jdk1.6.0_25/lib/
    ~/src/w32libs/java_jdk1.6.0_25/lib/)

set(USE_64BIT_IDS ON)

# adjust the default behaviour of the FIND_XXX() commands: search headers and
# libraries in the target environment, search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Windows libraries names
set(WIN_LIB_ICONV) # builtin
set(WIN_LIB_INTL libintl-8.dll)
set(WIN_LIB_CURL
    libcurl-4.dll
    libidn-11.dll
    libnspr4.dll
    nss3.dll
    libssh2-1.dll
    ssl3.dll
    zlib1.dll
    nssutil3.dll
    libplc4.dll
    libplds4.dll
    libgcrypt-11.dll
    libgpg-error-0.dll)
set(WIN_LIB_MYSQL libmysql.dll)
set(WIN_LIB_PGSQL libpq.dll)
set(WIN_LIB_GLIB libglib-2.0-0.dll libgobject-2.0-0.dll libiconv-2.dll
                 libgthread-2.0-0.dll)

# Disable pkg-config lookups
set(PKG_CONFIG_EXECUTABLE /bin/false)
