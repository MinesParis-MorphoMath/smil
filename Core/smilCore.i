



%{
/* Includes the header in the wrapper code */
#include "DCommon.h"
#include "DImage.h"
#include "DImage.hpp"
#include "DImage.hxx"
#include "DTypes.hpp"
#include "DBaseObject.h"
#include "DBaseImageOperations.hpp"
#include "DBaseLineOperations.hpp"
#include "DImageArith.hpp"
#include "DImageDraw.hpp"
#include "DLineHistogram.hpp"
#include "DImageHistogram.hpp"
/*#include "D_BaseOperations.h"*/
#include "memory"

%}
 
%extend Image 
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}

%define PTR_ARG_OUT_APPLY(name)
  %apply unsigned char *OUTPUT{ unsigned char *name };
  %apply unsigned short *OUTPUT{ unsigned short *name };
  %apply unsigned int *OUTPUT{ unsigned int *name };
  %apply char *OUTPUT{ char *name };
  %apply short *OUTPUT{ short *name };
  %apply int *OUTPUT{ int *name };
%enddef

PTR_ARG_OUT_APPLY(ret_min)
PTR_ARG_OUT_APPLY(ret_max)
PTR_ARG_OUT_APPLY(w)
PTR_ARG_OUT_APPLY(h)
PTR_ARG_OUT_APPLY(d)

%include "DCommon.h"
%include "DImage.hpp"
%include "DImage.hxx"
%include "DImage.h"
%include "DBaseImage.h"
%include "DTypes.hpp"
/*%include "D_BaseOperations.h" */
%include "DBaseImageOperations.hpp"
%include "DBaseLineOperations.hpp"
%include "DLineArith.hpp"
%include "DImageArith.hpp"
%include "DImageDraw.hpp"
%include "DLineHistogram.hpp"
%include "DImageHistogram.hpp"




TEMPLATE_WRAP_FUNC(createImage);

TEMPLATE_WRAP_FUNC_CROSS2(copy);

TEMPLATE_WRAP_FUNC(inv);
TEMPLATE_WRAP_FUNC(fill);
TEMPLATE_WRAP_FUNC(add);
TEMPLATE_WRAP_FUNC(addNoSat);
TEMPLATE_WRAP_FUNC(sub);
TEMPLATE_WRAP_FUNC(subNoSat);
TEMPLATE_WRAP_FUNC(mul);
TEMPLATE_WRAP_FUNC(mulNoSat);
TEMPLATE_WRAP_FUNC(div);

TEMPLATE_WRAP_FUNC(sup);
TEMPLATE_WRAP_FUNC(inf);
TEMPLATE_WRAP_FUNC(low);
TEMPLATE_WRAP_FUNC(lowOrEqu);
TEMPLATE_WRAP_FUNC(grt);
TEMPLATE_WRAP_FUNC(grtOrEqu);
TEMPLATE_WRAP_FUNC(logicAnd);
TEMPLATE_WRAP_FUNC(logicOr);
TEMPLATE_WRAP_FUNC(test);

TEMPLATE_WRAP_FUNC(vol);
TEMPLATE_WRAP_FUNC(minVal);
TEMPLATE_WRAP_FUNC(maxVal);
TEMPLATE_WRAP_FUNC(rangeVal);


TEMPLATE_WRAP_FUNC(thresh);
TEMPLATE_WRAP_FUNC(stretchHist);

TEMPLATE_WRAP_FUNC(drawRectangle);


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

