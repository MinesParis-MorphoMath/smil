/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
    * \defgroup Arrow Arrow Graphs
    * @{
    */


    template <class T_in, class lineFunction_T, class T_out=T_in>
    class unaryMorphArrowImageFunction : public MorphImageFunction<T_in, lineFunction_T, T_out>
    {
    public:
        typedef MorphImageFunction<T_in, lineFunction_T, T_out> parentClass;
        
        typedef Image<T_in> imageInType;
        typedef typename ImDtTypes<T_in>::lineType lineInType;
        typedef typename ImDtTypes<T_in>::sliceType sliceInType;
        typedef typename ImDtTypes<T_in>::volType volInType;
        
        typedef Image<T_out> imageOutType;
        typedef typename ImDtTypes<T_out>::lineType lineOutType;
        typedef typename ImDtTypes<T_out>::sliceType sliceOutType;
        typedef typename ImDtTypes<T_out>::volType volOutType;
        
        unaryMorphArrowImageFunction(T_in border=numeric_limits<T_in>::min(), T_out /*_initialValue*/ = ImDtTypes<T_out>::min()) 
          : MorphImageFunction<T_in, lineFunction_T, T_out>(border) 
        {
        }
        virtual RES_T _exec_single(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
    };


    template <class T_in, class lineFunction_T, class T_out>
    RES_T unaryMorphArrowImageFunction<T_in, lineFunction_T, T_out>::_exec_single(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        if ((void*)&imIn==(void*)&imOut)
        {
            Image<T_in> tmpIm = imIn;
            return _exec_single(tmpIm, imOut, se);
        }
        
        if (!areAllocated(&imIn, &imOut, NULL))
          return RES_ERR_BAD_ALLOCATION;

        UINT sePtsNumber = se.points.size();
        if (sePtsNumber==0)
            return RES_OK;
        
        size_t nSlices = imIn.getSliceCount();
        size_t nLines = imIn.getHeight();
        
        this->lineLen = imIn.getWidth();

        
        volInType srcSlices = imIn.getSlices();
        volOutType destSlices = imOut.getSlices();
        
        int nthreads = Core::getInstance()->getNumberOfThreads();
        typename ImDtTypes<T_in>::vectorType vec(this->lineLen);
        typename ImDtTypes<T_in>::matrixType bufsIn(nthreads, typename ImDtTypes<T_in>::vectorType(this->lineLen));
        typename ImDtTypes<T_out>::matrixType bufsOut(nthreads, typename ImDtTypes<T_out>::vectorType(this->lineLen));
        
        size_t l;

        for (size_t s=0;s<nSlices;s++)
        {
            lineInType *srcLines = srcSlices[s];
            lineOutType *destLines = destSlices[s];
            
      #ifdef USE_OPEN_MP
          #pragma omp parallel num_threads(nthreads)
      #endif
          {
            
            bool oddSe = se.odd, oddLine = 0;
            
            size_t x, y, z;
            lineFunction_T arrowLineFunction;
            
            int tid = 0;
            
      #ifdef USE_OPEN_MP
            tid = omp_get_thread_num();
      #endif // _OPENMP
            lineInType tmpBuf = bufsIn[tid].data();
            lineOutType tmpBuf2 = bufsOut[tid].data();
            
      #ifdef USE_OPEN_MP
        #pragma omp for
      #endif // USE_OPEN_MP
            for (l=0;l<nLines;l++)
            {
                lineInType lineIn  = srcLines[l];
                lineOutType lineOut = destLines[l];

                oddLine = oddSe && l%2;
                
                fillLine<T_out>(tmpBuf2, this->lineLen, T_out(0));
                
                for (UINT p=0;p<sePtsNumber;p++)
                {
                    y = l + se.points[p].y;
                    x = - se.points[p].x - (oddLine && (y+1)%2);
                    z = s + se.points[p].z;

                    arrowLineFunction.trueVal = (1UL << p);
                    this->_extract_translated_line(&imIn, x, y, z, tmpBuf);
                    arrowLineFunction._exec(lineIn, tmpBuf, this->lineLen, tmpBuf2);
                }
                copyLine<T_out>(tmpBuf2, this->lineLen, lineOut);
             }
          }  // pragma omp parallel
        }

        imOut.modified();

            return RES_OK;
    }


    template <class T_in, class T_out>
    RES_T arrowLow(const Image<T_in> &imIn, Image<T_out> &imOut, const StrElt &se=DEFAULT_SE, T_in borderValue=numeric_limits<T_in>::min())
    {
        unaryMorphArrowImageFunction<T_in, lowSupLine<T_in, T_out>, T_out > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T_in, class T_out>
    RES_T arrowLowOrEqu(const Image<T_in> &imIn, Image<T_out> &imOut, const StrElt &se=DEFAULT_SE, T_in borderValue=numeric_limits<T_in>::min())
    {
        unaryMorphArrowImageFunction<T_in, lowOrEquSupLine<T_in, T_out>, T_out > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T_in, class T_out>
    RES_T arrowGrt(const Image<T_in> &imIn, Image<T_out> &imOut, const StrElt &se=DEFAULT_SE, T_in borderValue=numeric_limits<T_in>::min())
    {
        unaryMorphArrowImageFunction<T_in, grtSupLine<T_in, T_out>, T_out > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T_in, class T_out>
    RES_T arrowGrtOrEqu(const Image<T_in> &imIn, Image<T_out> &imOut, const StrElt &se=DEFAULT_SE, T_in borderValue=numeric_limits<T_in>::min())
    {
        unaryMorphArrowImageFunction<T_in, grtOrEquSupLine<T_in, T_out>, T_out > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    template <class T_in, class T_out>
    RES_T arrowEqu(const Image<T_in> &imIn, Image<T_out> &imOut, const StrElt &se=DEFAULT_SE, T_in borderValue=numeric_limits<T_in>::min())
    {
        unaryMorphArrowImageFunction<T_in, equSupLine<T_in, T_out>, T_out > iFunc(borderValue);
        return iFunc(imIn, imOut, se);
    }

    /**
    * Arrow operator
    * 
    * \param imIn
    * \param operation "==", ">", ">=", "<" or "<="
    * \param imOut
    * \param se
    * \param borderValue
    */
    template <class T_in, class T_out>
    RES_T arrow(const Image<T_in> &imIn, const char *operation, Image<T_out> &imOut, const StrElt &se=DEFAULT_SE, T_in borderValue=numeric_limits<T_in>::min())
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
            virtual ~arrowPropagate () {}

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

    template < class T1, class T2, class T_out=T2>
    struct equSupLine3:public binaryLineFunctionBase < T1, T2, T_out>
    {
        equSupLine3 () : 
                trueVal ( ImDtTypes < T_out >::max (  ) ), falseVal ( 0 ) {}

        T_out trueVal, falseVal;

        typedef binaryLineFunctionBase<T1, T2, T_out> parentClass;
        typedef typename parentClass::lineType1 lineType1;
        typedef typename parentClass::lineType2 lineType2;
        typedef typename parentClass::lineOutType lineOutType;

        inline void operator () (const lineType1 lIn1,
                         const lineType2 lIn2,
                         const size_t size, lineOutType lOut )
        {
            return _exec ( lIn1, lIn2, size, lOut );
        }
        virtual void _exec ( const lineType1 lIn1,
                    const lineType2 lIn2,
                    const size_t size, lineOutType lOut )
        {
            T_out _trueVal ( trueVal ), _falseVal ( falseVal );

            for (size_t i = 0; i < size; i++)
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

	UINT sePtsNumber = se.points.size (  );

	if ( sePtsNumber == 0 )
	    return RES_OK;

	size_t nSlices = in.getSliceCount (  );
	size_t nLines = in.getHeight (  );

	lineLen = in.getWidth (  );

	volInType srcSlices = in.getSlices (  );
	volArrowType destSlices = arrow.getSlices (  );
	lineInType *srcLines;
	lineArrowType *destLines;

	bool oddSe = se.odd, oddLine = 0;
       
    #ifdef USE_OPEN_MP
    #pragma omp parallel private(oddLine)
    #endif // USE_OPEN_MP
	{
	T* borderBuf = ImDtTypes < T >::createLine ( lineLen );
	T* cpBuf = ImDtTypes < T >::createLine ( lineLen );
	T* minBuf = ImDtTypes < T >::createLine ( lineLen );
	T* nullBuf = ImDtTypes < T >::createLine ( lineLen );
	T* flagBuf = ImDtTypes < T >::createLine ( lineLen );
 	lowLine < T > low;
	infLine < T > inf;
	testLine < T, arrowT > test;
	equSupLine3 < T, T, arrowT > _equSup;

	fillLine < T > ( nullBuf, lineLen, arrowT ( 0 ) );
	fillLine < T > ( borderBuf, lineLen, T ( borderValue ) );

	lineInType lineIn;
	lineArrowType lineArrow;
	size_t x, y, z;

	for ( size_t s = 0; s < nSlices; ++s )
    {
	    srcLines = srcSlices[s];
	    destLines = destSlices[s];

	    #ifdef USE_OPEN_MP
	    #pragma omp for
	    #endif // USE_OPEN_MP
	    for ( size_t l = 0; l < nLines; ++l )
	    {
	        oddLine = oddSe && l %2;

		    lineIn = srcLines[l];
		    lineArrow = destLines[l];

		    fillLine < arrowT > ( lineArrow, lineLen, arrowT ( 0 ) );
		    copyLine < T > ( lineIn, lineLen, minBuf );

		    for ( UINT p = 0; p < sePtsNumber; ++p )
	        {
		        y = l + se.points[p].y;
                x = -se.points[p].x - (oddLine && (y+1)%2);
                z = s + se.points[p].z;

                _equSup.trueVal = ( 1UL << p );


                if ( z >= nSlices || y >= nLines )
                    copyLine < T > ( borderBuf, lineLen, cpBuf );
                else
                    shiftLine < T > ( srcLines[y], x, lineLen, cpBuf, borderValue );

                low._exec ( cpBuf, minBuf, lineLen, flagBuf );
                inf._exec ( cpBuf, minBuf, lineLen, minBuf );
                test._exec ( flagBuf, nullBuf, lineArrow, lineLen, lineArrow );
                _equSup._exec ( cpBuf, minBuf, lineLen, lineArrow );

		    }
	    }
	}

    }

	return RES_OK;
}

    template < class T, class arrowT > RES_T arrowMin ( const Image < T > &im,
							Image < arrowT >
							&arrow,
							const StrElt & se,
							T borderValue =
							numeric_limits <
							T >::max (  ) )
    {
	arrowMinFunction < T, arrowT > iFunc ( borderValue );
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

    #ifdef USE_OPEN_MP
	#pragma omp parallel private(oddLine)
    #endif // USE_OPEN_MP
	{
	T* borderBuf = ImDtTypes < T >::createLine ( lineLen );
	T* cpBuf = ImDtTypes < T >::createLine ( lineLen );
	T* minBuf = ImDtTypes < T >::createLine ( lineLen );
	T* nullBuf = ImDtTypes < T >::createLine ( lineLen );
	T* flagBuf = ImDtTypes < T >::createLine ( lineLen );
 	lowLine < T > low;
	infLine < T > inf;
	testLine < T, arrowT > test;
	equSupLine3 < T, T, arrowT> _equSup;
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

    #ifdef USE_OPEN_MP
	  #pragma omp for
    #endif // USE_OPEN_MP
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

		    _equSup.trueVal = ( 1UL << p );


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
		    _equSup._exec ( cpBuf, minBuf, lineLen, lineArrow );

		}
	    }
	}
	}

	return RES_OK;
    }

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

