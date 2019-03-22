#ifndef _ARROW_HPP_
#define _ARROW_HPP_

#include "Morpho/include/DMorpho.h"
#include "DGenerateLocales.hxx"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonArrow  
   *
   * @{
   */

  // SIMD version of WP2 - Nifty Revised.
  /**
   * @brief hammingWeight SIMD version of WP2 - Nifty Revised.
   *
   */
  template <class T> RES_T hammingWeight(const Image<T> &_im_, Image<T> &_out_)
  {
    typedef Image<T> imI;
    typedef Image<T> outI;
    typedef typename imI::lineType imL;
    typedef typename outI::lineType outL;
    typedef typename imI::volType imV;
    typedef typename outI::volType outV;

    size_t S[3];
    _im_.getSize(S);

    T m1Val = (~0);
    m1Val /= 3;
    T m2Val = (~0);
    m2Val /= 5;
    T m4Val = (~0);
    m4Val /= 17;

    imV srcSlices   = _im_.getSlices();
    outV destSlices = _out_.getSlices();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
    {
      T *m1 = ImDtTypes<T>::createLine(S[0]);
      fillLine<T>(m1, S[0], m1Val);
      T *m2 = ImDtTypes<T>::createLine(S[0]);
      fillLine<T>(m2, S[0], m2Val);
      T *m4 = ImDtTypes<T>::createLine(S[0]);
      fillLine<T>(m4, S[0], m4Val);
      T *finaland = ImDtTypes<T>::createLine(S[0]);
      fillLine<T>(finaland, S[0], 0x7F);

      size_t _d, _h, _w;
      imL *srcLines;
      outL *destLines;
      outL buf  = ImDtTypes<T>::createLine(S[0]);
      outL buf2 = ImDtTypes<T>::createLine(S[0]);
      imL im;
      outL out;

      unsigned TEST_BITS = sizeof(T) * 8;

      subNoSatLine<T> snsl;
      addNoSatLine<T> ansl;
      bitAndLine<T> andl;
      rightShiftLine<T> rsl;

      for (_d = 0; _d < S[2]; ++_d) {
        srcLines  = srcSlices[_d];
        destLines = destSlices[_d];

#pragma omp for
        for (_h = 0; _h < S[1]; ++_h) {
          im  = srcLines[_h];
          out = destLines[_h];

          for (_w = 0; _w < S[0]; ++_w)
            buf[_w] = im[_w] >> 1;
          rsl._exec(im, T(1), S[0], buf);
          andl(buf, m1, S[0], buf);
          snsl(im, buf, S[0], out);

          for (_w = 0; _w < S[0]; ++_w)
            buf[_w] = im[_w] >> 2;
          rsl._exec(out, T(2), S[0], buf);
          andl(buf, m2, S[0], buf);
          andl(out, m2, S[0], buf2);
          ansl(buf2, buf, S[0], out);

          for (_w = 0; _w < S[0]; ++_w)
            buf[_w] = im[_w] >> 4;
          rsl._exec(out, T(4), S[0], buf);
          ansl(out, buf, S[0], out);
          andl(out, m4, S[0], out);

          if (TEST_BITS > 8) {
            for (_w = 0; _w < S[0]; ++_w)
              buf[_w] = im[_w] >> 8;
            rsl._exec(out, T(8), S[0], buf);
            ansl(out, buf, S[0], out);
          }
          if (TEST_BITS > 16) {
            for (_w = 0; _w < S[0]; ++_w)
              buf[_w] = im[_w] >> 16;
            rsl._exec(out, T(16), S[0], buf);
            ansl(out, buf, S[0], out);
          }
          /*                    if (TEST_BITS > 32)
                              {
                                  for (_w=0; _w<S[0]; ++_w)
                                      buf[_w] = im[_w] >> 32;
                                  rsl._exec (out, T(32), S[0], buf);
                                  ansl (out, buf, S[0], out);
                              }
          */
          andl(out, finaland, S[0], out);
        }
      }
    }

    return RES_OK;
  }

  /**
   * hammingWeight
   * @brief hammingWeight SIMD version of WP2 - Nifty Revised.
   */
  RES_T hammingWeight(const Image<UINT8> &_im_, Image<UINT8> &_out_)
  {
    typedef Image<UINT8> imI;
    typedef typename imI::lineType imL;
    typedef typename imI::volType imV;

    size_t S[3];
    _im_.getSize(S);

    UINT8 m1Val = 0x55;
    UINT8 m2Val = 0x33;
    UINT8 m4Val = 0xF;

    imV srcSlices  = _im_.getSlices();
    imV destSlices = _out_.getSlices();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();

#pragma omp parallel num_threads(nthreads)
    {
      UINT8 *m1 = ImDtTypes<UINT8>::createLine(S[0]);
      fillLine<UINT8>(m1, S[0], m1Val);
      UINT8 *m2 = ImDtTypes<UINT8>::createLine(S[0]);
      fillLine<UINT8>(m2, S[0], m2Val);
      UINT8 *m4 = ImDtTypes<UINT8>::createLine(S[0]);
      fillLine<UINT8>(m4, S[0], m4Val);
      UINT8 *finaland = ImDtTypes<UINT8>::createLine(S[0]);
      fillLine<UINT8>(finaland, S[0], 0x7F);

      size_t _d, _h, _w;
      imL im;
      imL out;
      imL *srcLines;
      imL *destLines;
      imL buf  = ImDtTypes<UINT8>::createLine(S[0]);
      imL buf2 = ImDtTypes<UINT8>::createLine(S[0]);

      subNoSatLine<UINT8> snsl;
      addNoSatLine<UINT8> ansl;
      bitAndLine<UINT8> andl;
      rightShiftLine<UINT8> rsl;

      for (_d = 0; _d < S[2]; ++_d) {
        srcLines  = srcSlices[_d];
        destLines = destSlices[_d];

#pragma omp for
        for (_h = 0; _h < S[1]; ++_h) {
          im  = srcLines[_h];
          out = destLines[_h];

          for (_w = 0; _w < S[0]; ++_w)
            buf[_w] = im[_w] >> 1;
          rsl._exec(im, (UINT8) 1, S[0], buf);
          andl(buf, m1, S[0], buf);
          snsl(im, buf, S[0], out);

          for (_w = 0; _w < S[0]; ++_w)
            buf[_w] = im[_w] >> 2;
          rsl._exec(out, (UINT8) 2, S[0], buf);
          andl(buf, m2, S[0], buf);
          andl(out, m2, S[0], buf2);
          ansl(buf2, buf, S[0], out);

          for (_w = 0; _w < S[0]; ++_w)
            buf[_w] = im[_w] >> 4;
          rsl._exec(out, (UINT8) 4, S[0], buf);
          ansl(out, buf, S[0], out);
          andl(out, m4, S[0], out);

          andl(out, finaland, S[0], out);
        }
      }
    }

    return RES_OK;
  }

  /**
   * arrowComplement
   *
   */
  template <class T1, class T2>
  RES_T arrowComplement(const Image<T1> &_im_, Image<T2> &_out_,
                        const StrElt &s)
  {
    typedef Image<T1> imI;
    typedef typename imI::lineType imL;
    typedef Image<T2> outI;
    typedef typename outI::lineType outL;
    typedef typename outI::volType outV;

    size_t S[3];
    _im_.getSize(S);

    StrElt se  = s.noCenter();
    StrElt tse = se.transpose();
    vector<UINT> inverses;
    generateInverses(inverses, se);

    imL imP         = _im_.getPixels();
    outV destSlices = _out_.getSlices();
    outL *destLines;

    UINT sePtsNumber = se.points.size();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
    {
      outL cpyBuf = ImDtTypes<T2>::createLine(S[0]);
      imL buf     = ImDtTypes<T1>::createLine(S[0]);

      imL borderBuf = ImDtTypes<T1>::createLine(S[0]);
      fillLine<T1>(borderBuf, S[0], 0);

      T1 flag, flag2;
      size_t _d, _h, _w;
      bool _odd = 0;
      size_t x, y, z;
      UINT _pts;

      imL im = ImDtTypes<T1>::createLine(S[0]);
      outL out;

      for (_d = 0; _d < S[2]; ++_d) {
        destLines = destSlices[_d];

#pragma omp for
        for (_h = 0; _h < S[1]; ++_h) {
          _odd = se.odd && _h % 2;

          out = destLines[_h];
          fillLine<T2>(cpyBuf, S[0], 0);
          for (_pts = 0; _pts < sePtsNumber; ++_pts) {
            flag  = (1 << _pts);
            flag2 = (1 << inverses[_pts]);
            y     = _h + tse.points[_pts].y;
            x     = -tse.points[_pts].x - (_odd && (y + 1) % 2);
            z     = _d + tse.points[_pts].z;
            if (z >= S[2] || y >= S[1]) {
              copyLine<T1>(borderBuf, S[0], im);
            } else {
              shiftLine<T1>(imP + y * S[0] + z * S[0] * S[1], x, S[0], im, 0);
            }

            for (_w = 0; _w < S[0]; ++_w)
              buf[_w] = im[_w] & flag;
            for (_w = 0; _w < S[0]; ++_w)
              buf[_w] = buf[_w] > 0 ? flag2 : 0;
            for (_w = 0; _w < S[0]; ++_w)
              cpyBuf[_w] += buf[_w];
          }
          copyLine<T2>(cpyBuf, S[0], out);
        }
      }
    }
    return RES_OK;
  }

  /*
   * binaryMorphArrowImageFunction
   *
   */
  template <class T_in, class lineFunction_T, class T_out = T_in>
  class binaryMorphArrowImageFunction
      : public MorphImageFunction<T_in, lineFunction_T, T_out>
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

    binaryMorphArrowImageFunction(
        T_in border             = numeric_limits<T_in>::min(),
        T_out /*_initialValue*/ = ImDtTypes<T_out>::min())
        : MorphImageFunction<T_in, lineFunction_T, T_out>(border)
    {
    }
    virtual RES_T _exec_single(const imageInType &imIn,
                               const imageInType &imIn2, imageOutType &imOut,
                               const StrElt &se);
  };

  /**
   * binaryMorphArrowImageFunction
   *
   */
  template <class T_in, class lineFunction_T, class T_out>
  RES_T
  binaryMorphArrowImageFunction<T_in, lineFunction_T, T_out>::_exec_single(
      const imageInType &imIn, const imageInType &imIn2, imageOutType &imOut,
      const StrElt &se)
  {
    ASSERT_ALLOCATED(&imIn, &imIn2);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if ((void *) &imIn == (void *) &imOut) {
      Image<T_in> tmpIm = imIn;
      return _exec_single(tmpIm, imIn2, imOut, se);
    } else if ((void *) &imIn2 == (void *) &imOut) {
      Image<T_in> tmpIm = imIn;
      return _exec_single(imIn, tmpIm, imOut, se);
    }

    if (!areAllocated(&imIn, &imIn2, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    UINT sePtsNumber = se.points.size();
    if (sePtsNumber == 0)
      return RES_OK;

    size_t nSlices = imIn.getSliceCount();
    size_t nLines  = imIn.getHeight();

    this->initialize(imIn, imOut, se);
    this->lineLen = imIn.getWidth();

    // JOE volInType srcSlices = imIn.getSlices();
    volInType src2Slices  = imIn2.getSlices();
    volOutType destSlices = imOut.getSlices();

    int nthreads = Core::getInstance()->getNumberOfThreads();
    typename ImDtTypes<T_in>::vectorType vec(this->lineLen);
    typename ImDtTypes<T_in>::matrixType bufsIn(
        nthreads, typename ImDtTypes<T_in>::vectorType(this->lineLen));
    typename ImDtTypes<T_out>::matrixType bufsOut(
        nthreads, typename ImDtTypes<T_out>::vectorType(this->lineLen));

    size_t l;

    for (size_t s = 0; s < nSlices; s++) {
      // JOE lineInType *srcLines = srcSlices[s];
      lineInType *src2Lines  = src2Slices[s];
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
        lineInType tmpBuf   = bufsIn[tid].data();
        lineOutType tmpBuf2 = bufsOut[tid].data();

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
        for (l = 0; l < nLines; l++) {
          lineInType lineIn   = src2Lines[l];
          lineOutType lineOut = destLines[l];

          oddLine = oddSe && l % 2;

          fillLine<T_out>(tmpBuf2, this->lineLen, T_out(0));

          for (UINT p = 0; p < sePtsNumber; p++) {
            y = l + se.points[p].y;
            x = -se.points[p].x - (oddLine && (y + 1) % 2);
            z = s + se.points[p].z;

            arrowLineFunction.trueVal = (1UL << p);
            this->_extract_translated_line(&imIn, x, y, z, tmpBuf);
            arrowLineFunction._exec(lineIn, tmpBuf, this->lineLen, tmpBuf2);
          }
          copyLine<T_out>(tmpBuf2, this->lineLen, lineOut);
        }
      } // pragma omp parallel
    }

    imOut.modified();

    return RES_OK;
  }

  /**
   * arrowLowDual
   *
   */
  template <class T_in, class T_out>
  RES_T arrowLowDual(const Image<T_in> &imIn, const Image<T_in> &imIn2,
                     Image<T_out> &imOut, const StrElt &se = DEFAULT_SE,
                     T_in borderValue = numeric_limits<T_in>::min())
  {
    binaryMorphArrowImageFunction<T_in, lowSupLine<T_in, T_out>, T_out> iFunc(
        borderValue);
    return iFunc._exec_single(imIn, imIn2, imOut, se);
  }

  /**
   * arrowLowOrEquDual
   *
   */
  template <class T_in, class T_out>
  RES_T arrowLowOrEquDual(const Image<T_in> &imIn, const Image<T_in> &imIn2,
                          Image<T_out> &imOut, const StrElt &se = DEFAULT_SE,
                          T_in borderValue = numeric_limits<T_in>::min())
  {
    binaryMorphArrowImageFunction<T_in, lowOrEquSupLine<T_in, T_out>, T_out>
        iFunc(borderValue);
    return iFunc._exec_single(imIn, imIn2, imOut, se);
  }

  /**
   * arrowGrtDual
   *
   */
  template <class T_in, class T_out>
  RES_T arrowGrtDual(const Image<T_in> &imIn, const Image<T_in> &imIn2,
                     Image<T_out> &imOut, const StrElt &se = DEFAULT_SE,
                     T_in borderValue = numeric_limits<T_in>::min())
  {
    binaryMorphArrowImageFunction<T_in, grtSupLine<T_in, T_out>, T_out> iFunc(
        borderValue);
    return iFunc._exec_single(imIn, imIn2, imOut, se);
  }

  /**
   * arrowGrtOrEquDual
   *
   */
  template <class T_in, class T_out>
  RES_T arrowGrtOrEquDual(const Image<T_in> &imIn, const Image<T_in> &imIn2,
                          Image<T_out> &imOut, const StrElt &se = DEFAULT_SE,
                          T_in borderValue = numeric_limits<T_in>::min())
  {
    binaryMorphArrowImageFunction<T_in, grtOrEquSupLine<T_in, T_out>, T_out>
        iFunc(borderValue);
    return iFunc._exec_single(imIn, imIn2, imOut, se);
  }

  /**
   * arrowEquDual
   *
   */
  template <class T_in, class T_out>
  RES_T arrowEquDual(const Image<T_in> &imIn, const Image<T_in> &imIn2,
                     Image<T_out> &imOut, const StrElt &se = DEFAULT_SE,
                     T_in borderValue = numeric_limits<T_in>::min())
  {
    binaryMorphArrowImageFunction<T_in, equSupLine<T_in, T_out>, T_out> iFunc(
        borderValue);
    return iFunc._exec_single(imIn, imIn2, imOut, se);
  }

  /**
   * arrowDual
   *
   */
  template <class T_in, class T_out>
  RES_T arrowDual(const Image<T_in> &imIn, const Image<T_in> &imIn2,
                  const char *operation, Image<T_out> &imOut,
                  const StrElt &se = DEFAULT_SE,
                  T_in borderValue = numeric_limits<T_in>::min())
  {
    if (strcmp(operation, "==") == 0)
      return arrowEquDual(imIn, imIn2, imOut, se, borderValue);
    else if (strcmp(operation, ">") == 0)
      return arrowGrtDual(imIn, imIn2, imOut, se, borderValue);
    else if (strcmp(operation, ">=") == 0)
      return arrowGrtOrEquDual(imIn, imIn2, imOut, se, borderValue);
    else if (strcmp(operation, "<") == 0)
      return arrowLowDual(imIn, imIn2, imOut, se, borderValue);
    else if (strcmp(operation, "<=") == 0)
      return arrowLowOrEquDual(imIn, imIn2, imOut, se, borderValue);

    else
      return RES_ERR;
  }

  /** @} */
} // namespace smil

#endif // _ARROW_HPP_
