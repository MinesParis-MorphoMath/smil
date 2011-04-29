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
%include "DImageMorph.hpp"
#ifdef USE_QT
%include "gui/Qt/QtApp.h"
#endif // USE_QT

%extend Image 
{
//	std::string  __str__() {
//	    self->printSelf();
//	}
}

%define TEMPLATE_WRAP_CLASS_TYPE(_class) 
  %template(_class ## _UINT8) _class<UINT8>;
  %template(_class ## _UINT16) _class<UINT16>;
/*  %template(_class ## _UINT32) _class<UINT32>;*/
%enddef

%define TEMPLATE_WRAP_FUNC_TYPE(func)
  %template(func) func<UINT8>;
  %template(func) func<UINT16>;
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

TEMPLATE_WRAP_FUNC_TYPE(copyIm);
TEMPLATE_WRAP_FUNC_TYPE(fillIm);
TEMPLATE_WRAP_FUNC_TYPE(addIm);
TEMPLATE_WRAP_FUNC_TYPE(addNoSatIm);
TEMPLATE_WRAP_FUNC_TYPE(subIm);
TEMPLATE_WRAP_FUNC_TYPE(subNoSatIm);

TEMPLATE_WRAP_FUNC_TYPE(supIm);

//TEMPLATE_WRAP_FUNC_TYPE(labelIm);

TEMPLATE_WRAP_FUNC_TYPE(dilateIm);
TEMPLATE_WRAP_FUNC_TYPE(erodeIm);
TEMPLATE_WRAP_FUNC_TYPE(closeIm);
TEMPLATE_WRAP_FUNC_TYPE(openIm);
TEMPLATE_WRAP_FUNC_TYPE(gradientIm);

TEMPLATE_WRAP_FUNC_TYPE(volIm);





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

