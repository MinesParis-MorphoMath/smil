Use:
- Cross-compile Smil fro android. You can use the android.toolchain.cmake from Google: http://code.google.com/p/android-cmake/source/browse/toolchain/android.toolchain.cmake
- Turn cmake option WRAP_JAVA to ON. Give the path for android java sources (not the oracle sources!).
- Compile Smil.
- Copy the smilJava directory in the SmilDemo/src directory and the smilJava libraries in the SmilDemo/libs/armeabi folder.
- Compile the SmilDemo project using the android SDK.

