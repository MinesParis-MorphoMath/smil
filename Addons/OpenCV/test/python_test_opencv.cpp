
#include "Python.h"

#include "DOpenCVInterface.hpp"
#include "Core/include/DTest.h"


using namespace smil;

class Test_Python_Import : public TestCase
{
    virtual void run()
    {
      Py_Initialize();

      PyObject *_main = PyImport_ImportModule("__main__");
      PyObject *globals = PyModule_GetDict(_main);
      
      PyRun_SimpleString("import cv");
      PyRun_SimpleString("cvIm = cv.CreateImage((256,127), 8, 1)");

      PyObject *pyobj = PyDict_GetItem(globals, PyUnicode_FromString( "cvIm" ));
      
      OpenCVInt<UINT8> cvIm(pyobj);
      
      TEST_ASSERT(cvIm.isAllocated());
      TEST_ASSERT(cvIm.getWidth()==256 && cvIm.getHeight()==127);
      
      
      Py_Finalize();
        
    }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Python_Import);
      
      return ts.run();
  
}

