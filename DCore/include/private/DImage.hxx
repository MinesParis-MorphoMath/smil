#ifndef _IMAGE_HXX
#define _IMAGE_HXX

// template <>
// const char *getImageDataTypeAsString<UINT8>(Image<UINT8> &im)
// {
//     return "UINT8 (unsigned char)";
// }


template <class T>
Image<T>::Image()
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init(); 
}

template <class T>
Image<T>::Image(UINT w, UINT h, UINT d)
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init(); 
    setSize(w, h, d);
}

template <class T>
Image<T>::Image(Image<T> &rhs, bool cloneit)
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
    if (cloneit) clone(rhs);
    else setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
}

template <class T>
template <class T2>
Image<T>::Image(Image<T2> &rhs, bool cloneit)
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
    if (cloneit) clone(rhs);
    else setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
}


template <class T>
Image<T>::~Image()
{ 
    deallocate();
#ifdef USE_QT
    delete viewer;
//     viewer = new ImageViewer();
#endif // USE_QT
    
}



template <class T>
void Image<T>::init() 
{ 
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    dataTypeSize = sizeof(pixelType); 
    
#ifdef USE_QT
    viewer = new ImageViewerWidget();
//     viewer = new ImageViewer();
#endif // USE_QT
}

template <class T>
inline void Image<T>::modified()
{ 
#ifdef USE_QT
    if (viewer->isVisible())
      updateViewerData();
#endif // USE_QT    
}


#ifdef USE_QT    

template <class T>
inline void Image<T>::setName(const char *name)
{ 	
    viewer->setName(name);
}

template <class T>
inline void Image<T>::show(const char *name)
{ 
    if (name)
      setName(name);
    updateViewerData();
    viewer->show();
//     qapp->exec();
}

template <class T>
inline void Image<T>::updateViewerData()
{ 
}
template <>
inline void Image<UINT8>::updateViewerData()
{ 
    viewer->loadFromData(pixels, width, height);
}

#endif // USE_QT    


template <class T>
inline Image<T>& Image<T>::clone(Image<T> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), isAlloc);
    if (isAlloc)
      memcpy(pixels, rhs.getPixels(), pixelCount*sizeof(T));
    modified();
    return *this;
}

template <class T>
template <class T2>
inline Image<T>& Image<T>::clone(Image<T2> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), isAlloc);
    if (isAlloc)
      copyIm(rhs, *this);
    modified();
    return *this;
}

template <class T>
inline Image<T>& Image<T>::clone(void)
{ 
    static Image<T> newIm(*this, true);
    return newIm;
}

template <class T>
void Image<T>::setSize(int w, int h, int d, bool doAllocate)
{
    if (w==width && h==height && d==depth)
	return;
    
    if (allocated) deallocate();
    
    width = w;
    height = h;
    depth = d;
    
    sliceCount = d;
    lineCount = sliceCount * h;
    pixelCount = lineCount * w;
    
    if (doAllocate) allocate();
    modified();
}

template <class T>
inline RES_T Image<T>::allocate(void)
{
    if (allocated)
	return RES_ERR_BAD_ALLOCATION;
    
    pixels = new pixelType[pixelCount];
    
    restruct();
    
    allocated = true;
    
    return RES_OK;
}

template <class T>
RES_T Image<T>::restruct(void)
{
    if (slices)
	delete[] slices;
    if (lines)
	delete[] lines;
    
    lines =  new lineType[lineCount];
    slices = new sliceType[sliceCount];
    
    lineType *cur_line = lines;
    sliceType *cur_slice = slices;
    
    int pixelsPerSlice = width * height;
    
    for (int k=0; k<(int)depth; k++, cur_slice++)
    {
      *cur_slice = cur_line;
      
      for (int j=0; j<(int)height; j++, cur_line++)
	*cur_line = pixels + k*pixelsPerSlice + j*width;
    }
	
    // Calc. line (mis)alignment
    int n = SIMD_VEC_SIZE / sizeof(T);
    int w = width%SIMD_VEC_SIZE;
    for (int i=0;i<n;i++)
      lineAlignment[i] = (SIMD_VEC_SIZE - (i*w)%SIMD_VEC_SIZE)%SIMD_VEC_SIZE;
    
    return RES_OK;
}

template <class T>
int Image<T>::getLineAlignment(UINT l)
{
    return lineAlignment[l%(SIMD_VEC_SIZE/sizeof(T))];
}

template <class T>
RES_T Image<T>::deallocate(void)
{
    if (!allocated)
	return RES_OK;
    
    if (slices)
	delete[] slices;
    if (lines)
	delete[] lines;
    if (pixels)
		delete[] pixels;
    
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    allocated = false;
    
    return RES_OK;
}

template <class T>
void Image<T>::printSelf(bool displayPixVals)
{
    if (depth>1)
    {
      cout << "3D image" << endl;
      cout << "Size: " << width << "x" << height << "x" << depth << endl;
    }
    else
    {
      cout << "2D image" << endl;
      cout << "Size: " << width << "x" << height << endl;
    }
    
    if (allocated) cout << "Allocated (" << pixelCount*sizeof(T) << " bits)" << endl;
    else cout << "Not allocated" << endl;
    
    if (!displayPixVals)
      return;
    
    cout << "Pixels value:" << endl;
    sliceType *cur_slice;
    lineType *cur_line;
    pixelType *cur_pixel;
    
    UINT i, j, k;
    
    for (k=0, cur_slice = slices; k<depth; k++, cur_slice++)
    {
      cur_line = *cur_slice;
      for (j=0, cur_line = *cur_slice; j<height; j++, cur_line++)
      {
	for (i=0, cur_pixel = *cur_line; i<width; i++, cur_pixel++)
	  cout << (double)*cur_pixel << "  ";
	cout << endl;
      }
      cout << endl;
    }
    cout << endl;
}


// OPERATORS

template <class T>
Image<T>& Image<T>::operator = (Image<T> &rhs)
{
    cout << "= op" << endl;
    this->clone(rhs);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (Image<T> &rhs)
{
    copyIm(rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (T value)
{
    fillIm(*this, value);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator ~()
{
    static Image<T> newIm(*this);
    invIm(*this, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator + (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    addIm(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator + (T value)
{
    static Image<T> newIm(*this);
    addIm(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator += (Image<T> &rhs)
{
    addIm(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator += (T value)
{
    addIm(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator - (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    subIm(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator - (T value)
{
    static Image<T> newIm(*this);
    subIm(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator -= (Image<T> &rhs)
{
    subIm(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator -= (T value)
{
    subIm(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator < (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    lowIm(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator < (T value)
{
    static Image<T> newIm(*this);
    lowIm(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator > (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    grtIm(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator > (T value)
{
    static Image<T> newIm(*this);
    grtIm(*this, value, newIm);
    return newIm;
}



#endif // _IMAGE_HXX
