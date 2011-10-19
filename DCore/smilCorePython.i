%module smilCorePython

%feature("autodoc", "1");

%include <windows.i>

%{
/* Includes the header in the wrapper code */
#include "DCommon.h"
#include "DImage.h"
#include "DImage.hpp"
#include "DImage.hxx"
#include "D_Types.h"
#include "D_BaseObject.h"
#include "DBaseImageOperations.hpp"
#include "DBaseLineOperations.hpp"
#include "DImageArith.hpp"
#include "DImageMorph.hpp"
/*#include "D_BaseOperations.h"*/
#include "DImageIO_PNG.h"
#include "memory"

#ifdef USE_QT
#include "gui/Qt/QtApp.h"
#endif // USE_QT
%}
 
%include <std_string.i>
%include <typemaps.i>


/*%rename(__int__) *::operator int; */
/* %rename(__float__) *::operator float; */
/* %rename(__irshift__) *::operator>>=; */
%rename(__lshift__)  operator<<; 
/* %rename(__rshift__)  *::operator>>; */
%rename(__add__) *::operator+; 
/*%ignore *::operator=;*/
%rename(__assign__) *::operator=;

#define __attribute__(x)

%include "DCommon.h"
%include "DImage.hpp"
%include "DImage.hxx"
%include "DImage.h"
%include "DBaseImage.h"
%include "D_Types.h"
%include "DImageIO_PNG.h"
/*%include "D_BaseOperations.h" */
%include "DBaseImageOperations.hpp"
%include "DBaseLineOperations.hpp"
%include "DLineArith.hpp"
%include "DImageArith.hpp"
%include "DStructuringElement.h"
%include "DImageMorph.hpp"
#ifdef USE_QT
%include "gui/Qt/QtApp.h"
#endif // USE_QT

%extend Image 
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
	void show()
	{
	    
	}
}

%extend StrElt
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}

%define TEMPLATE_WRAP_CLASS_TYPE(_class) 
  %template(_class ## _UINT8) _class<UINT8>;
  %template(_class ## _UINT16) _class<UINT16>;
/*  %template(_class ## _UINT32) _class<UINT32>;*/
%enddef

%define TEMPLATE_WRAP_FUNC_TYPE(func)
  %template(func) func<UINT8>;
/*  %template(func) func<UINT16>; */
/*  %template(func) func<UINT32>; */
%enddef

%define TEMPLATE_WRAP_FUNC_TYPE2(func)
  %template(func) func<UINT8,UINT8>;
  %template(func) func<UINT8,UINT16>;
  %template(func) func<UINT16,UINT8>;
  %template(func) func<UINT32>;
%enddef

%define TEMPLATE_WRAP_FUNC_IMG_TYPE(func) 
  %template(func) func<Image_UINT8>;
  %template(func) func<Image_UINT16>;
  %template(func) func<Image_UINT32>;
%enddef

TEMPLATE_WRAP_CLASS_TYPE(Image);


TEMPLATE_WRAP_FUNC_TYPE(createImage);

TEMPLATE_WRAP_FUNC_TYPE(copy);
TEMPLATE_WRAP_FUNC_TYPE(inv);
TEMPLATE_WRAP_FUNC_TYPE(fill);
TEMPLATE_WRAP_FUNC_TYPE(add);
TEMPLATE_WRAP_FUNC_TYPE(addNoSat);
TEMPLATE_WRAP_FUNC_TYPE(sub);
TEMPLATE_WRAP_FUNC_TYPE(subNoSat);

TEMPLATE_WRAP_FUNC_TYPE(sup);
TEMPLATE_WRAP_FUNC_TYPE(inf);

//TEMPLATE_WRAP_FUNC_TYPE(label);

TEMPLATE_WRAP_FUNC_TYPE(dilate);
TEMPLATE_WRAP_FUNC_TYPE(erode);
TEMPLATE_WRAP_FUNC_TYPE(close);
TEMPLATE_WRAP_FUNC_TYPE(open);
TEMPLATE_WRAP_FUNC_TYPE(gradient);
TEMPLATE_WRAP_FUNC_TYPE(geoDil);
TEMPLATE_WRAP_FUNC_TYPE(geoEro);
TEMPLATE_WRAP_FUNC_TYPE(build);
TEMPLATE_WRAP_FUNC_TYPE(dualBuild);

TEMPLATE_WRAP_FUNC_TYPE(vol);





/* %template(smartImage) boost::shared_ptr< D_Image<UINT8> >; */

/*%extend UINT8

{
  void operator << (UINT8 val)
  {
      cout << "ok" << endl;
  }
}*/



/* %rename(__eq__) setVal; */

/* %template(Create) createImage<UINT8>; */

/*%template(__lshift__) D_Image::operator << <UINT16>;
%template(__lshift__) D_Image::operator << <UINT8>;*/

/* %template(__rshift__) D_Image::operator >> <UINT8>(string filename); */

/* %template(Int) Int<UINT8>; */

