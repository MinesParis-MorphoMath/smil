
#include "DVtkInterface.hpp"
#include "Core/include/DTest.h"


using namespace smil;

class Test_Import : public TestCase
{
    virtual void run()
    {
        vtkImageData *imData = vtkImageData::New();
        imData->SetExtent(0, 49, 0, 49, 0, 49);
        imData->SetScalarTypeToUnsignedShort();
        imData->SetNumberOfScalarComponents(1); // image holds one value intensities
        imData->AllocateScalars(); // allocate
        
        VtkInt<UINT8> sIm(imData);
        TEST_ASSERT(sIm.getWidth()==50);
        TEST_ASSERT(sIm.getHeight()==50);
        TEST_ASSERT(sIm.getDepth()==50);
        
        imData->Delete();
    }
};

#if defined Py_PYCONFIG_H 
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

          PyObject *pyobj = PyDict_GetItem(globals, PyString_FromString( "mIm" ));

          VtkInt<UINT8> sIm(pyobj);
          TEST_ASSERT(sIm.isAllocated());
          TEST_ASSERT(sIm.getWidth()==50);
          
          Py_Finalize();
            
        }
    };
#endif // Py_PYCONFIG_H

int main()
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Import);
      
#ifdef WRAP_PYTHON
      ADD_TEST(ts, Test_Python_Import);
#endif // WRAP_PYTHON
      
      return ts.run();
  
}

