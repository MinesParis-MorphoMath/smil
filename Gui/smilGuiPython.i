%module smilPython

%{
/* Includes the header in the wrapper code */
#include "DImage.hpp"
#include "DGui.h"
#include "DImage.hxx"

#ifdef USE_QT
#include "Qt/QtApp.h"
#endif // USE_QT
%}


%extend Image 
{
}

%include "DGui.h"
#ifdef USE_QT
%include "Qt/QtApp.h"
#endif // USE_QT


