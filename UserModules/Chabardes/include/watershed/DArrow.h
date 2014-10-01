#ifndef _ARROW_H_
#define _ARROW_H_

namespace smil {

template <class T1, class T2>
struct equSupLine3 : public tertiaryLineFunctionBase<T1>
{
	equSupLine3 () : trueVal (ImDtTypes<T2>::max()), falseVal (0) {}

	typedef typename Image<T1>::lineType lineType1;
	typedef typename Image<T2>::lineType lineType2;

	T2 trueVal, falseVal;

	inline void operator()(const lineType1 lIn1, const lineType1 lIn2, const size_t size, lineType2 lOut) {
		return _exec(lIn1, lIn2, size, lOut);
	}
	inline void _exec(const lineType1 lIn1, const lineType1 lIn2, const size_t size, lineType2 lOut) {
		T2 _trueVal (trueVal), _falseVal (falseVal);
		for (size_t i=0; i<size; i++) {
			lOut[i] |= lIn1 == lIn2 ? _trueVal : _falseVal;
		}
	}

};

template <class T, class arrowT=UINT8>
class arrowSteepestFunction : public unaryMorphImageFunctionBase<T, arrowT>
{
	public:
		typedef unaryMorphImageFunctionBase<T, arrowT> parentClass;
		typedef typename parentClass::imageInType imageInType;
		typedef typename imageInType::lineType lineInType;
		typedef typename imageInType::lineType sliceInType;
		typedef typename imageInType::volType volInType;
		typedef typename parentClass::imageOutType imageArrowType;
		typedef typename imageArrowType::lineType lineArrowType;
		typedef typename imageArrowType::sliceType sliceArrowType;
		typedef typename imageArrowType::volType volArrowType;

	arrowSteepestFunction (T border=numeric_limits<T>::min()) : borderValue (border), unaryMorphImageFunctionBase<T, arrowT> () 
		{ 
		}

	virtual RES_T _exec (const imageInType &in, imageArrowType &imOut, const StrElt &se) ;
	inline void extract_translated_line (const imageInType &in, const int &x, const int &y, const int &z, lineInType outBuf) {
		if (z<0 || z>= int(in.getDepth()) || y<0 || y>=int(in.getHeight()))
			copyLine<T> (borderBuf, lineLen, outBuf);
		else
			shiftLine<T>(in.getSlices()[z][y], x, lineLen, outBuf, borderValue);
	}
	private:
		T borderValue;
		lineInType borderBuf, cpBuf, minBuf, flagBuf, nullBuf;
		size_t lineLen;

		lowLine<T> low;
		infLine<T> inf;
		testLine<T,arrowT> test;
		equSupLine3<T, arrowT> equSup;
};

template <class T, class arrowT>
RES_T arrowSteepestFunction<T, arrowT>::_exec (const imageInType &in, imageArrowType &arrow, const StrElt &se) {
	ASSERT_ALLOCATED (&in, &arrow) ;
	ASSERT_SAME_SIZE (&in, &arrow) ;

	if (!areAllocated (&in, &arrow, NULL))
		return RES_ERR_BAD_ALLOCATION;

	UINT sePtsNumber = se.points.size();
	if( sePtsNumber == 0 )
		return RES_OK;


	size_t nSlices = in.getSliceCount();
	size_t nLines = in.getHeight();
	lineLen = in.getWidth () ;

	volInType srcSlices = in.getSlices();
	volArrowType destSlices = arrow.getSlices();
	lineInType *srcLines;
	lineArrowType *destLines;

	bool oddSe = se.odd, oddLine = 0;
	size_t x,y,z;

	borderBuf = ImDtTypes<T>::createLine (lineLen);
	cpBuf = ImDtTypes<T>::createLine (lineLen);
	minBuf = ImDtTypes<T>::createLine (lineLen);

	fillLine<T>(nullBuf, lineLen, arrowT(0));

	for (size_t s=0; s<nSlices; ++s) {
		srcLines = srcSlices[s];
		destLines = destSlices[s];
		if (oddSe)
			oddLine = s%2!=0;

		for (size_t l=0; l<nLines; ++l) {
			lineInType lineIn = srcLines[l];
			lineArrowType lineArrow = destLines[l];
			fillLine<arrowT> (lineArrow, lineLen, arrowT(0));
			copyLine<T> (lineIn, lineLen, minBuf);

			for (UINT p=0; p<sePtsNumber; ++p) {
				x = - se.points[p].x + oddLine;
				y = l + se.points[p].y;
				z = s + se.points[p].z;
				
				equSup.trueVal = (1UL << p);

				extract_translated_line (in, x, y, z, cpBuf) ;

				low._exec (cpBuf, minBuf, lineLen, flagBuf);
				inf._exec (cpBuf, minBuf, lineLen, minBuf);
				test._exec (flagBuf, nullBuf, lineArrow, lineLen, lineArrow);
				equSup._exec (cpBuf, minBuf, lineLen, lineArrow);
			}
			if (oddSe)
				oddLine = !oddLine;
		}
	}

	// Now we Have the arrow to the minimum neighbors.
	// Process equivalent path. 
}

template <class T, class arrowT>
RES_T arrowSteepest (const Image<T> &in, Image<arrowT> &arrow, const StrElt &se, T borderValue=numeric_limits<T>::min()) {
	arrowSteepestFunction<T, arrowT> iFunc (borderValue) ;
	return iFunc (in, arrow, se) ;
}

}

#endif // _ARROW_H_
