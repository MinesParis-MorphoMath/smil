#ifndef _BASE_LINE_OPERATIONS_HPP
#define _BASE_LINE_OPERATIONS_HPP


#include "DImage.hpp"
#include "DBasePixelOperations.hpp"

struct stat;

template <class T> class Image;


struct Point
{
  int x;
  int y;
  int z;
};

enum seType { stGeneric, stHSE };

class StrElt
{
  public:
    StrElt(UINT s=1) : seT(stGeneric), size(s) 
    {
    }
vector<Point> points;
    inline void addPoint(int x, int y, int z=0)
    {
	Point p;
	p.x = x;
	p.y = y;
	p.z = z;
	points.push_back(p);
    }
    inline StrElt& operator()(int s=1)
    {
	this->size = s;
	return *this;
    }
    bool odd;
    UINT size;
    virtual seType getType() { return seT; }
    seType seT;
};

class hSE : public StrElt
{
  public:
    hSE(UINT s=1) 
    {
	seT = stHSE;
	size = s;
	odd = true;
	addPoint(0,0);
	addPoint(1,0);
	addPoint(-1,0);
	addPoint(-1,1);
	addPoint(0,1);
	addPoint(-1,-1);
	addPoint(0,-1);
    }
};

class sSE : public StrElt
{
  public:
    sSE(UINT s=1) : StrElt(s)
    {
	odd = false;
	addPoint(0,0);
	addPoint(1,0);
	addPoint(1,1);
	addPoint(0,1);
	addPoint(-1,1);
	addPoint(-1,0);
	addPoint(-1,-1);
	addPoint(0,-1);
	addPoint(1,-1);
    }
};

#define DEFAULT_SE hSE()


// Base abstract struct of line unary function
template <class T>
struct unaryLineFunctionBase
{
    virtual void _exec(T *lineIn, int size, T *lineOut) = 0;
    virtual inline void operator()(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
};


// Base abstract struct of line binary function
template <class T>
struct binaryLineFunctionBase
{
    virtual void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut) = 0;
    virtual inline void operator()(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
};

class lineFunctionBase
{
};

template <class T, class unaryPixelFunction_T>
struct unaryLineFunction
{
    static unaryPixelFunction_T pixelFunction;
    
    static void _exec(T *lineIn, int size, T *lineOut)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(lineIn[i], lineOut[i]);
    }
    static void _exec(T *lineInOut, int size, T value)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(value, lineInOut[i]);
    }
    
    inline void operator()(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
    inline void operator()(T *lineInOut, int size, T value) { _exec(lineInOut, size, value); }
    unaryLineFunction() {}
    unaryLineFunction(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
    unaryLineFunction(T *lineInOut, int size, T value) { _exec(lineInOut, size, value); }
};


template <class T, class binaryPixelFunction_T>
struct binaryLineFunction
{
    static binaryPixelFunction_T pixelFunction;
    static void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(lineIn1[i], lineIn2[i], lineOut[i]);
    }
    static void _exec(T *lineIn, T value, int size, T *lineOut)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(lineIn[i], value, lineOut[i]);
    }
    inline void operator()(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
    inline void operator()(T *lineIn, T value, int size, T *lineOut) { _exec(lineIn, value, size, lineOut); }
};


#endif
