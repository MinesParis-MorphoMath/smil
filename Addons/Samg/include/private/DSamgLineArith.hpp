#ifndef _D_CHABARDES_LINE_ARITH_HPP_
#define _D_CHABARDES_LINE_ARITH_HPP_

namespace smil
{
  template <class T>
  struct supCstLine : public unaryLineFunctionBase<T> {
    typedef typename unaryLineFunctionBase<T>::lineType lineType;
    T threshold, trueVal, falseVal;
    supCstLine() : threshold(0), trueVal(0), falseVal(ImDtTypes<T>::max())
    {
    }
    virtual void _exec(const lineType lIn, const size_t size, lineType lOut)
    {
      for (size_t i = 0; i < size; ++i)
        lOut[i] = lIn[i] > threshold ? trueVal : falseVal;
    }
  };

  template <class T>
  struct addNoSatCstLine : public unaryLineFunctionBase<T> {
    addNoSatCstLine() : val(1)
    {
    }

    T                                                   val;
    typedef typename unaryLineFunctionBase<T>::lineType lineType;
    virtual void _exec(const lineType lIn1, const size_t size, lineType lOut)
    {
      for (size_t i = 0; i < size; i++)
        lOut[i] = lIn1[i] + val;
    }
  };

  template <class T>
  struct equLines {
    equLines()
    {
    }
    void _exec(const T *lIn1, const T *lIn2, const T *lIn3, const T *lIn4,
               const size_t size, T *lOut)
    {
      for (size_t i = 0; i < size; ++i) {
        lOut[i] = lIn1[i] == lIn2[i] ? lIn3[i] : lIn4[i];
      }
    }
  };

  template <class T>
  struct equLinesMasked {
    equLinesMasked(size_t lineLen)
    {
      eq.trueVal  = ImDtTypes<T>::max();
      eq.falseVal = 0;
    }
    inline void _exec(T *lIn1, T *lIn2, T *lIn3, T *lIn4, const size_t size,
                      T *lOut)
    {
      T *buf = ImDtTypes<T>::createLine(size);
      eq._exec(lIn1, lIn2, size, buf);
      tes._exec(buf, lIn3, lIn4, size, lOut);
    }

  private:
    equLine<T>     eq;
    testLine<T, T> tes;
  };
  /*
      template <class T1, class T2>
      struct equLines : public tertiaryLineFunctionBase<T1,T2,T2,T2>
      {
          equLines() {}

          typedef typename tertiaryLineFunctionBase<T1,T2,T2,T2>::lineType1
     lineType1; typedef typename
     tertiaryLineFunctionBase<T1,T2,T2,T2>::lineOutType lineType2;

          virtual void _exec(const lineType1 lIn1, const lineType2 lIn2, const
     lineType2 lIn3, const size_t size, lineType2 lOut)
          {
              for (size_t i=0;i<size;i++)
                  lOut[i] = lIn1[i] == lIn2[i] ? lIn2[i] : lIn3[i];
          }
      };

      template <class T>
      struct equLines2 : public binaryLineFunctionBase<T>
      {
          equLines2() {}

          typedef typename binaryLineFunctionBase<T>::lineType1 lineType1;
          typedef typename binaryLineFunctionBase<T>::lineOutType lineType2;

          virtual void _exec(const lineType1 lIn1, const lineType2 lIn2, const
     size_t size, lineType2 lOut)
          {
              for (size_t i=0;i<size;i++)
                  lOut[i] = lOut[i] == lIn1[i] ? lIn2[i] : lOut[i];
          }
      };
  */
  template <class T1, class T2>
  struct supLines : public binaryLineFunctionBase<T1, T1, T2> {
    typedef typename binaryLineFunctionBase<T1, T1, T2>::lineType1   lineType1;
    typedef typename binaryLineFunctionBase<T1, T1, T2>::lineOutType lineType2;

    virtual void _exec(const lineType1 lIn1, const lineType1 lIn2,
                       const size_t size, lineType2 lOut)
    {
      for (size_t i = 0; i < size; i++)
        lOut[i] = lIn1[i] > lIn2[i] ? lIn2[i] : lOut[i];
    }
  };

} // namespace smil

#endif // _D_CHABARDES_LINE_ARITH_HPP_
