 
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/image_T_impl.hpp>
#include <morphee/image/include/imageInterface.hpp>

// #ifdef SWIGPYTHON
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/call_method.hpp>

// Base de pythonExt
#include <boost/python/return_value_policy.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
// #endif // SWIGPYTHON

#include "DMorphMImage.hpp"
#include <morphee/pythonExt/include/pymImageLib_T.hpp>

class _imInterface : public morphee::ImageInterface
{
};

int main()
{
    typedef boost::python::class_<morphee::ImageInterface, boost::noncopyable> iclass;
	
    // Initialize Python
    Py_Initialize();

    // Get the main namespace/module up-and-running
    boost::python::object main_module = boost::python::import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");

    PyRun_SimpleString("import sys");
    PyRun_SimpleString((string("sys.path.append(\"") + MORPHEE_LIBRARY_DIR + "\")").c_str());
    
    PyObject *dict= PyDict_New();
    boost::python::exec("import MorpheePython as mp", main_namespace);
    boost::python::exec("mIm = mp.createImage(mp.dataCategory.dtScalar, mp.scalarDataType.sdtUINT8)", main_namespace);
    boost::python::exec("mIm.setSize(256,256)", main_namespace);
    boost::python::exec("mIm.setColorInfo(mp.colorInfo.ciMonoSpectral)", main_namespace);
    boost::python::exec("mIm.allocateImage()", main_namespace);

    PyObject *pyobj = PyDict_GetItem(main_namespace.ptr(), PyString_FromString( "mIm" ));
    
//     iclass *ic = boost::python::extract<iclass*>(pyobj);
//     ImageInterface *imInt = boost::python::extract<ImageInterface *>(pyobj);
      PyObject *arglist = Py_BuildValue((char*)"()");
      PyObject *result = PyEval_CallObject(pyobj, arglist);
      Py_DECREF(arglist);
      Py_DECREF(pyobj);
    
//     IMorpheeImage *imi;
     ImageInterface *imInt = (ImageInterface *)(result);
    
//     morphee::ImageInterface *ii =  (morphee::ImageInterface *)ic->ptr();
//     morphee::ImageInterface::coordinate_system s = ii->getSize();
    
    morphee::Image<UINT8> *tIm = dynamic_cast< morphee::Image<UINT8>*>(imInt);
    morphee::ImageInterface::coordinate_system s = tIm->getSize();
    
    tIm->setSize(10,10);
    cout << long(tIm->rawPointer()) << endl;
    cout << tIm->getXSize() << "," << tIm->getYSize() << "," << tIm->getZSize() << endl;
    cout << s[0] << "," << s[1] << "," << s[2] << endl;
    

    Py_Finalize();
    return 0;
  
}
