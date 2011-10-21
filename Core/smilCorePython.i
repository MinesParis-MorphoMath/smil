



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


%include "DCommon.h"
%include "DImage.hpp"
%include "DImage.hxx"
%include "DImage.h"
%include "DBaseImage.h"
%include "D_Types.h"
/*%include "D_BaseOperations.h" */
%include "DBaseImageOperations.hpp"
%include "DBaseLineOperations.hpp"
%include "DLineArith.hpp"
%include "DImageArith.hpp"




TEMPLATE_WRAP_FUNC(createImage);

TEMPLATE_WRAP_FUNC(copy);
TEMPLATE_WRAP_FUNC(inv);
TEMPLATE_WRAP_FUNC(fill);
TEMPLATE_WRAP_FUNC(add);
TEMPLATE_WRAP_FUNC(addNoSat);
TEMPLATE_WRAP_FUNC(sub);
TEMPLATE_WRAP_FUNC(subNoSat);

TEMPLATE_WRAP_FUNC(sup);
TEMPLATE_WRAP_FUNC(inf);

TEMPLATE_WRAP_FUNC(vol);





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

