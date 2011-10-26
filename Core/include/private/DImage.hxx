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
    if (viewer)
	delete viewer;
//     viewer = new ImageViewer();
    
}



template <class T>
void Image<T>::init() 
{ 
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    dataTypeSize = sizeof(pixelType); 
    
//     viewer = new ImageViewerWidget();
//     viewer = new ImageViewer();
     viewer = NULL;
     name = NULL;
}

template <class T>
inline void Image<T>::modified()
{ 
    if (viewer && viewer->isVisible())
      updateViewerData();
}



template <class T>
inline void Image<T>::setName(const char *_name)
{ 	
    name = _name;
    if (viewer)
	viewer->setName(_name);
}

template <class T>
inline void Image<T>::updateViewerData()
{ 
    if (viewer)
	viewer->loadFromData(pixels, width, height);
}



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
      copy(rhs, *this);
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
    
//     pixels = createAlignedBuffer<T>(pixelCount);
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
    {
      lineAlignment[i] = (SIMD_VEC_SIZE - (i*w)%SIMD_VEC_SIZE)%SIMD_VEC_SIZE;
//       cout << i << " " << lineAlignment[i] << endl;
    }
    
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
void Image<T>::printSelf(ostream &os, bool displayPixVals)
{
    if (name)
      os << name << endl;
    
    if (depth>1)
      os << "3D image" << endl;
    else
      os << "2D image" << endl;

    T val;
    os << "Data type: " << getDataTypeAsString(val) << endl;
    
    if (depth>1)
      os << "Size: " << width << "x" << height << "x" << depth << endl;
    else
      os << "Size: " << width << "x" << height << endl;
    
    if (allocated) os << "Allocated (" << pixelCount*sizeof(T) << " bits)" << endl;
    else os << "Not allocated" << endl;
    
   
    if (displayPixVals)
    {
	os << "Pixels value:" << endl;
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
	      os << (double)*cur_pixel << "  ";
	    os << endl;
	  }
	  os << endl;
	}
	os << endl;
    }
    
    cout << endl;   
}

template <class T>
void Image<T>::printSelf(bool displayPixVals)
{
    printSelf(std::cout, displayPixVals);
}



// OPERATORS

template <class T>
void operator << (ostream &os, Image<T> &im)
{
    im.printSelf(os);
}

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
    copy(rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (T value)
{
    fill(*this, value);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator ~()
{
    static Image<T> newIm(*this);
    inv(*this, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator + (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    add(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator + (T value)
{
    static Image<T> newIm(*this);
    add(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator += (Image<T> &rhs)
{
    add(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator += (T value)
{
    add(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator - (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    sub(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator - (T value)
{
    static Image<T> newIm(*this);
    sub(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator -= (Image<T> &rhs)
{
    sub(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator -= (T value)
{
    sub(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator < (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    low(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator < (T value)
{
    static Image<T> newIm(*this);
    low(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator > (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    grt(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator > (T value)
{
    static Image<T> newIm(*this);
    grt(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator << (const T *tab)
{
    for (int i=0;i<pixelCount;i++)
      pixels[i] = tab[i];
    modified();
}


#endif // _IMAGE_HXX
