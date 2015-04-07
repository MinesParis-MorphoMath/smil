/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_MORPHO_ARROW_HPP
#define _D_MORPHO_ARROW_HPP

#include "DMorphImageOperations.hpp"
#include "DMorphoHierarQ.hpp"

namespace smil
{
   /**
    * \ingroup Morpho
    * \defgroup Arrow
    * @{
    */


    template <class T, class lineFunction_T>
    class unaryMorphArrowImageFunction : public MorphImageFunction<T, lineFunction_T>
    {
    public:
        typedef MorphImageFunction<T, lineFunction_T> parentClass;
        typedef Image<T> imageType;
        typedef typename imageType::lineType lineType;
        typedef typename imageType::sliceType sliceType;
        typedef typename imageType::volType volType;
        
        unaryMorphArrowImageFunction(T border=numeric_limits<T>::min()) 
          : MorphImageFunction<T, lineFunction_T>(border) 
        {
        }
        virtual RES_T _exec_single(const imageType &imIn, imageType &imOut, const StrElt &se);
        virtual RES_T _exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se);
    };


    template <class T, class lineFunction_T>
    RES_T unaryMorphArrowImageFunction<T, lineFunction_T>::_exec_single(const imageType &imIn, imageType &imOut, const StrElt &se)
    {
        return _exec_single_generic(imIn, imOut, se);
    }

    template <class T, class lineFunction_T>
    RES_T unaryMorphArrowImageFunction<T, lineFunction_T>::_exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        if (&imIn==&imOut)
        {
            Image<T> tmpIm = imIn;
            return _exec_single_generic(tmpIm, imOut, se);
        }
        
        if (!areAllocated(&imIn, &imOut, NULL))
          return RES_ERR_BAD_ALLOCATION;

        UINT sePtsNumber = se.points.size();
        if (sePtsNumber==0)
            return RES_OK;
        
        size_t nSlices = imIn.getSliceCount();
        size_t nLines = imIn.getHeight();

        
        volType srcSlices = imIn.getSlices();
        volType destSlices = imOut.getSlices();
        
        int nthreads = Core::getInstance()->getNumberOfThreads();
        lineType *_bufs = this->createAlignedBuffers(2*nthreads, this->lineLen);

        size_t l;

        for (size_t s=0;s<nSlices;s++)
        {
            lineType *srcLines = srcSlices[s];
            lineType *destLines = destSlices[s];
            
      #ifdef USE_OPEN_MP
          #pragma omp parallel num_threads(nthreads)
      #endif
          {
            
            bool oddSe = se.odd, oddLine = 0;
            
            size_t x, y, z;
            lineFunction_T arrowLineFunction;
            
            lineType tmpBuf = _bufs[0];
            lineType tmpBuf2 = _bufs[nthreads];
      #ifdef USE_OPEN_MP
            int tid = omp_get_thread_num();
            tmpBuf = _bufs[tid];
            tmpBuf2 = _bufs[tid+nthreads];
      #endif // _OPENMP
            
        #pragma omp for
            for (l=0;l<nLines;l++)
            {
                lineType lineIn  = srcLines[l];
                lineType lineOut = destLines[l];

                oddLine = oddSe && l%2;
                
                fillLine<T>(tmpBuf2, this->lineLen, 0);
                
                for (UINT p=0;p<sePtsNumber;p++)
                {
                    y = l + se.points[p].y;
                    x = - se.points[p].x - (oddLine && (y+1)%2);
                    z = s + se.points[p].z;

                    arrowLineFunction.trueVal = (1UL << p);
                    this->_extract_translated_line(&imIn, x, y, z, tmpBuf);
                    arrowLineFunction._exec(lineIn, tmpBuf, this->lineLen, tmpBuf2);
                }
                copyLine<T>(tmpBuf2, this->lineLen, lineOut);
             }
          }  // pragma omp parallel
        }

        imOut.modified();

            return RES_OK;
    }


    template <class T>
    RES_T arrowLow(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
        unaryMorphArrowImageFunction<T, lowSupLine<T> > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowLowOrEqu(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
        unaryMorphArrowImageFunction<T, lowOrEquSupLine<T> > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowGrt(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
        unaryMorphArrowImageFunction<T, grtSupLine<T> > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowGrtOrEqu(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
        unaryMorphArrowImageFunction<T, grtOrEquSupLine<T> > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowEqu(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
        unaryMorphArrowImageFunction<T, equSupLine<T> > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    /**
    * Arrow operator
    * 
    * \param imIn
    * \param operation "==", ">", ">=", "<" or "<="
    * \param imOut
    * \param se
    */
    template <class T>
    RES_T arrow(const Image<T> &imIn, const char *operation, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
        if (strcmp(operation, "==")==0)
          return arrowEqu(imIn, imOut, se, borderValue);
        else if (strcmp(operation, ">")==0)
          return arrowGrt(imIn, imOut, se, borderValue);
        else if (strcmp(operation, ">=")==0)
          return arrowGrtOrEqu(imIn, imOut, se, borderValue);
        else if (strcmp(operation, "<")==0)
          return arrowLow(imIn, imOut, se, borderValue);
        else if (strcmp(operation, "<=")==0)
          return arrowLowOrEqu(imIn, imOut, se, borderValue);
          
        else return RES_ERR;
    }

    /**
     * Propagation Functor on a Arrow Image
     *
     */
    template <class arrowT, class statutT, class outT, class containerType = STD_Queue<size_t> >
    class arrowPropagate {

        protected:
            containerType q;

            virtual bool testAndAssign (statutT &/*pS*/, outT &pO) 
            {
                if (pO != propagationValue) 
                {
                   // pS = numeric_limits<statutT>::max();
                    pO = propagationValue;
                    return true;
                }
                return false;
            }

        public:
            outT propagationValue;

            typedef Image<arrowT> arrowIT;
            typedef Image<statutT> statutIT;
            typedef Image<outT>  outIT;
            typedef typename arrowIT::lineType arrowLT;
            typedef typename statutIT::lineType statutLT;
            typedef typename outIT::lineType outLT;

            arrowPropagate () {}
            ~arrowPropagate () {}

            RES_T _exec ( const Image<arrowT> &imArrow, Image<statutT> &imStatut, Image<outT> &imOut, const StrElt &se, const size_t &offset ) 
            {
                arrowLT arrowP = imArrow.getPixels ();
                statutLT statutP = imStatut.getPixels ();
                outLT outP = imOut.getPixels ();

                bool oddLine;
                size_t size[3]; imArrow.getSize (size);
                UINT sePtsNumber = se.points.size ();

                size_t x, x0, y, y0, z, z0;
                arrowT arrow;

                size_t o, nb_o;

                q.push (offset);
                do
                {
                    o = q.front();
                    q.pop();
                    z0 = o / (size[1] * size[0]);
                    y0 = (o % (size[1]*size[0])) / size[0];
                    x0 = o % size[0];
                    oddLine = se.odd && y0%2;

                    for (UINT p=0; p<sePtsNumber; ++p)
                    {
                        arrow = (1UL << p);
                        if (arrowP[o] & arrow) 
                        {
                            x = x0 + se.points[p].x;
                            y = y0 + se.points[p].y;
                            z = z0 + se.points[p].z;
                            if (oddLine)
                                x += (y+1)%2;
                            nb_o = x + y*size[0] + z*size[1]*size[0];
                            if ( x < size[0] &&
                                 y < size[1] && 
                                 z < size[2]) 
                                if (testAndAssign (statutP[nb_o], outP[nb_o]))
                                    q.push(nb_o);
                       }
                    }
                       
                } while (!q.empty());
                return RES_OK;
            }
            inline RES_T operator ()( const Image<arrowT> &imArrow, Image<statutT> &imStatut, Image<outT> &imOut, const StrElt &se, const size_t &offset ) 
            {
                return _exec (imArrow, imStatut, imOut, se, offset);
            }
            inline RES_T operator ()(const Image<arrowT> &imArrow, Image<outT> &imOut, const StrElt &se, const size_t &offset) 
            {
                return _exec (imArrow, imOut, imOut, se, offset);
            }
    };

    template < class T1,
	class T2 > struct equSupLine3:public tertiaryLineFunctionBase < T1 >
    {
	equSupLine3 (  ):trueVal ( ImDtTypes < T2 >::max (  ) ),
	    falseVal ( 0 )
	{
	}

	typedef typename Image < T1 >::lineType lineType1;
	typedef typename Image < T2 >::lineType lineType2;

	T2 trueVal, falseVal;

	inline void operator      (  ) ( const lineType1 lIn1,
					 const lineType1 lIn2,
					 const size_t size, lineType2 lOut )
	{
	    return _exec ( lIn1, lIn2, size, lOut );
	}
	inline void _exec ( const lineType1 lIn1,
			    const lineType1 lIn2,
			    const size_t size, lineType2 lOut )
	{
	    T2 _trueVal ( trueVal ), _falseVal ( falseVal );

	    for ( size_t i = 0; i < size; i++ )
	      {
		  lOut[i] += ( lIn1[i] == lIn2[i] ) ? _trueVal : _falseVal;
	      }
	}
    };


    template < class T, class arrowT = UINT8 > 
    class arrowMinFunction : public MorphImageFunctionBase<T, arrowT>
    {
      public:
	typedef MorphImageFunctionBase < T, arrowT > parentClass;
	typedef typename parentClass::imageInType imageInType;
	typedef typename imageInType::lineType lineInType;
	typedef typename imageInType::lineType sliceInType;
	typedef typename imageInType::volType volInType;
	typedef typename parentClass::imageOutType imageArrowType;
	typedef typename imageArrowType::lineType lineArrowType;
	typedef typename imageArrowType::sliceType sliceArrowType;
	typedef typename imageArrowType::volType volArrowType;

	arrowMinFunction ( T border = numeric_limits < T >::max (  ) ):borderValue ( border ), MorphImageFunctionBase < T, arrowT > (  )
	{
	    borderValue = border;
	}

	virtual RES_T _exec ( const imageInType & in,
			      imageArrowType & imOut, const StrElt & se );
      private:
	T borderValue;
	size_t lineLen;

    };

    template < class T, class arrowT >
    RES_T arrowMinFunction < T, arrowT >::_exec ( const imageInType & in,
						  imageArrowType & arrow,
						  const StrElt & se )
    {
	ASSERT_ALLOCATED ( &in, &arrow );
	ASSERT_SAME_SIZE ( &in, &arrow );

	if ( !areAllocated ( &in, &arrow, NULL ) )
	    return RES_ERR_BAD_ALLOCATION;

	StrElt cpSe = se.noCenter ();

	UINT sePtsNumber = cpSe.points.size (  );

	if ( sePtsNumber == 0 )
	    return RES_OK;

	size_t nSlices = in.getSliceCount (  );
	size_t nLines = in.getHeight (  );

	lineLen = in.getWidth (  );

	volInType srcSlices = in.getSlices (  );
	volArrowType destSlices = arrow.getSlices (  );
	lineInType *srcLines;
	lineArrowType *destLines;

	bool oddSe = cpSe.odd, oddLine = 0;

	#pragma omp parallel private(oddLine)
	{
	T* borderBuf = ImDtTypes < T >::createLine ( lineLen );
	T* cpBuf = ImDtTypes < T >::createLine ( lineLen );
	T* minBuf = ImDtTypes < T >::createLine ( lineLen );
	T* nullBuf = ImDtTypes < T >::createLine ( lineLen );
	T* flagBuf = ImDtTypes < T >::createLine ( lineLen );
 	lowLine < T > low;
	infLine < T > inf;
	testLine < T, arrowT > test;
	equSupLine3 < T, arrowT > equSup;
	equLine < T > equ;

	fillLine < T > ( nullBuf, lineLen, arrowT ( 0 ) );
	fillLine < T > ( borderBuf, lineLen, T ( borderValue ) );
	equ.trueVal = 0;
	equ.falseVal = numeric_limits < T >::max (  );

	lineInType lineIn;
	lineArrowType lineArrow;
	size_t x, y, z;

	for ( size_t s = 0; s < nSlices; ++s )
        {
	  srcLines = srcSlices[s];
	  destLines = destSlices[s];

	  #pragma omp for
	  for ( size_t l = 0; l < nLines; ++l )
	  {
	        oddLine = oddSe && l %2;

		lineIn = srcLines[l];
		lineArrow = destLines[l];

		fillLine < arrowT > ( lineArrow, lineLen, arrowT ( 0 ) );
		copyLine < T > ( lineIn, lineLen, minBuf );

		for ( UINT p = 0; p < sePtsNumber; ++p )
	        {
		    y = l + cpSe.points[p].y;
		    x = -cpSe.points[p].x - (oddLine && (y+1)%2);
		    z = s + cpSe.points[p].z;

		    equSup.trueVal = ( 1UL << p );


  		    if ( z >= nSlices || y >= nLines )
		        copyLine < T > ( borderBuf, lineLen, cpBuf );
		    else
		        shiftLine < T > ( srcLines[y], x, lineLen, cpBuf, borderValue );

		    low._exec ( cpBuf, minBuf, lineLen, flagBuf );
		    inf._exec ( cpBuf, minBuf, lineLen, minBuf );
		    test._exec ( flagBuf, nullBuf, lineArrow, lineLen,
			       lineArrow );
		    equSup._exec ( cpBuf, minBuf, lineLen, lineArrow );

		}
	    }
	}
	}

	return RES_OK;
    };

    template < class T > RES_T arrowMin ( const Image < T > &im,
							Image < T >
							&arrow,
							const StrElt & se,
							T borderValue =
							numeric_limits <
							T >::max (  ) )
    {
	arrowMinFunction < T, T > iFunc ( borderValue );
	return iFunc ( im, arrow, se );
    }

    template < class T, class arrowT = UINT8 > 
    class arrowMinStepFunction : public MorphImageFunctionBase<T, arrowT>
    {
      public:
	typedef MorphImageFunctionBase < T, arrowT > parentClass;
	typedef typename parentClass::imageInType imageInType;
	typedef typename imageInType::lineType lineInType;
	typedef typename imageInType::lineType sliceInType;
	typedef typename imageInType::volType volInType;
	typedef typename parentClass::imageOutType imageArrowType;
	typedef typename imageArrowType::lineType lineArrowType;
	typedef typename imageArrowType::sliceType sliceArrowType;
	typedef typename imageArrowType::volType volArrowType;

	arrowMinStepFunction ( T border = numeric_limits < T >::max (  ) ):borderValue ( border ), MorphImageFunctionBase < T, arrowT > (  )
	{
	    borderValue = border;
	}

	virtual RES_T _exec ( const imageInType & in,
			      imageArrowType & imOut, const StrElt & se );
      private:
	T borderValue;
	size_t lineLen;

    };

    template < class T, class arrowT >
    RES_T arrowMinStepFunction < T, arrowT >::_exec ( const imageInType & in,
						  imageArrowType & arrow,
						  const StrElt & se )
    {
	ASSERT_ALLOCATED ( &in, &arrow );
	ASSERT_SAME_SIZE ( &in, &arrow );

	if ( !areAllocated ( &in, &arrow, NULL ) )
	    return RES_ERR_BAD_ALLOCATION;

	StrElt cpSe = se.noCenter ();

	UINT sePtsNumber = cpSe.points.size (  );

	if ( sePtsNumber == 0 )
	    return RES_OK;

	size_t nSlices = in.getSliceCount (  );
	size_t nLines = in.getHeight (  );
	lineLen = in.getWidth (  );

	volInType srcSlices = in.getSlices (  );
	volArrowType destSlices = arrow.getSlices (  );
	lineInType *srcLines;
	lineArrowType *destLines;

	bool oddSe = cpSe.odd, oddLine = 0;

	#pragma omp parallel private(oddLine)
	{
	T* borderBuf = ImDtTypes < T >::createLine ( lineLen );
	T* cpBuf = ImDtTypes < T >::createLine ( lineLen );
	T* minBuf = ImDtTypes < T >::createLine ( lineLen );
	T* nullBuf = ImDtTypes < T >::createLine ( lineLen );
	T* flagBuf = ImDtTypes < T >::createLine ( lineLen );
 	lowLine < T > low;
	infLine < T > inf;
	testLine < T, arrowT > test;
	equSupLine3 < T, arrowT > equSup;
	equLine < T > equ;

	fillLine < T > ( nullBuf, lineLen, arrowT ( 0 ) );
	fillLine < T > ( borderBuf, lineLen, T ( borderValue ) );
	equ.trueVal = 0;
	equ.falseVal = numeric_limits < T >::max (  );

	lineInType lineIn;
	lineArrowType lineArrow;
	size_t x, y, z, i;

	for ( size_t s = 0; s < nSlices; ++s )
        {
	  srcLines = srcSlices[s];
	  destLines = destSlices[s];

	  #pragma omp for
	  for ( size_t l = 0; l < nLines; ++l )
	  {
	        oddLine = oddSe && l %2;

		lineIn = srcLines[l];
		lineArrow = destLines[l];

		fillLine < arrowT > ( lineArrow, lineLen, arrowT ( 0 ) );
		copyLine < T > ( borderBuf, lineLen, minBuf );

		for ( UINT p = 0; p < sePtsNumber; ++p )
	        {
		    y = l + cpSe.points[p].y;
		    x = -cpSe.points[p].x - (oddLine && (y+1)%2);
		    z = s + cpSe.points[p].z;

		    equSup.trueVal = ( 1UL << p );


  		    if ( z >= nSlices || y >= nLines )
		        copyLine < T > ( borderBuf, lineLen, cpBuf );
		    else
		        shiftLine < T > ( srcLines[y], x, lineLen, cpBuf, borderValue );

		    for (i=0; i<lineLen; ++i)
			flagBuf[i] = cpBuf[i] >= lineIn[i] ? 255 : 0;
		    for (i=0; i<lineLen; ++i)
			cpBuf[i] = flagBuf[i] ? cpBuf[i] : borderBuf[i];

		    low._exec ( cpBuf, minBuf, lineLen, flagBuf );
		    inf._exec ( cpBuf, minBuf, lineLen, minBuf );
		    test._exec ( flagBuf, nullBuf, lineArrow, lineLen,
			       lineArrow );
		    equSup._exec ( cpBuf, minBuf, lineLen, lineArrow );

		}
	    }
	}
	}

	return RES_OK;
    };

    template < class T > RES_T arrowMinStep ( const Image < T > &im,
							Image < T >
							&arrow,
							const StrElt & se,
							T borderValue =
							numeric_limits <
							T >::max (  ) )
    {
	arrowMinStepFunction < T, T > iFunc ( borderValue );
	return iFunc ( im, arrow, se );
    }


/** \} */

} // namespace smil


#endif // _D_MORPHO_ARROW_HPP

