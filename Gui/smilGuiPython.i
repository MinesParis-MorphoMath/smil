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
	void _show(const char* name=NULL)
	{
	    self->show(name);
	}
}

%include "DGui.h"
#ifdef USE_QT
%include "Qt/QtApp.h"
#endif // USE_QT


