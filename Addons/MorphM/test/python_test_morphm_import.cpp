

#include <boost/python.hpp>

#include "DMorphMImage.hpp"
#include "Core/include/DCore.h"

using namespace smil;

#define PY_TEST_SCRIPT "\
import MorpheePython as mp \n\
mIm = mp.createImage(mp.dataCategory.dtScalar, mp.scalarDataType.sdtUINT8) \n\
mIm.setSize(256,127) \n\
mIm.setColorInfo(mp.colorInfo.ciMonoSpectral) \n\
mIm.allocateImage()"

class Test_Python_Import : public TestCase
{
    virtual void run()
    {
      Py_Initialize();

      PyObject *_main = PyImport_ImportModule("__main__");
      PyObject *globals = PyModule_GetDict(_main);
      
      PyRun_String("import sys", Py_file_input, globals, NULL);
      PyRun_SimpleString((string("sys.path.append(\"") + MORPHEE_LIBRARY_DIR + "\")").c_str());
      
      PyRun_SimpleString(PY_TEST_SCRIPT);

      PyObject *pyobj = PyDict_GetItem(globals, PyUnicode_FromString( "mIm" ));
      
      MorphmInt<UINT8> mIm(pyobj);
      TEST_ASSERT(mIm.isAllocated());
      TEST_ASSERT(mIm.getWidth()==256 && mIm.getHeight()==127);
      
      
      Py_Finalize();
        
    }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Python_Import);
      
      return ts.run();
  
}

