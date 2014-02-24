
#include <Python.h>

#include "DVtkInterface.hpp"
#include "DTest.h"


using namespace smil;

class Test_Python_Import : public TestCase
    {
	virtual void run()
	{
	  Py_Initialize();

	  PyObject *_main = PyImport_ImportModule("__main__");
	  PyObject *globals = PyModule_GetDict(_main);
	  
	  PyRun_String("import vtk", Py_file_input, globals, NULL);
	  PyRun_SimpleString("vIm = vtk.vtkImageData()");
	  PyRun_SimpleString("vIm.SetExtent(0, 49, 0, 49, 0, 49)");
	  PyRun_SimpleString("vIm.SetScalarTypeToUnsignedShort()");
	  PyRun_SimpleString("vIm.SetNumberOfScalarComponents(1)");
	  PyRun_SimpleString("vIm.AllocateScalars()");

	  PyObject *pyobj = PyDict_GetItem(globals, PyString_FromString( "vIm" ));

	  VtkInt<UINT8> sIm(pyobj);
	  TEST_ASSERT(sIm.isAllocated());
	  TEST_ASSERT(sIm.getWidth()==50);
	  
	  Py_Finalize();
	    
	}
    };

int main(int argc, char *argv[])
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Python_Import);
      
      return ts.run();
  
}

