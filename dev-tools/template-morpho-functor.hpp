
  /*
   *  ######  #    #  #    #   ####    #####   ####   #####    ####
   *  #       #    #  ##   #  #    #     #    #    #  #    #  #
   *  #####   #    #  # #  #  #          #    #    #  #    #   ####
   *  #       #    #  #  # #  #          #    #    #  #####        #
   *  #       #    #  #   ##  #    #     #    #    #  #   #   #    #
   *  #        ####   #    #   ####      #     ####   #    #   ####
   */
  /* @devdoc */
  /** @cond */
  template <class T1, class T2, class compOperatorT = std::equal_to<T1>>
  class labelFunctGeneric : public MorphImageFunctionBase<T1, T2>
  {
  public:
    typedef MorphImageFunctionBase<T1, T2> parentClass;
    typedef typename parentClass::imageInType imageInType;
    typedef typename parentClass::imageOutType imageOutType;

    size_t getLabelNbr()
    {
      return real_labels;
    }

    virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut,
                             const StrElt &se)
    {
      parentClass::initialize(imIn, imOut, se);
      fill(imOut, T2(0));
      labels          = 0;
      real_labels     = 0;
      max_value_label = ImDtTypes<T2>::max();
      return RES_OK;
    }

#if PROCESS_OFFSET
    virtual RES_T processImage(const imageInType &imIn,
                               imageOutType & /*imOut*/, const StrElt & /*se*/)
    {
      this->pixelsIn = imIn.getPixels();

      size_t nbPixels = imIn.getPixelCount();
      for (size_t i = 0; i < nbPixels; i++) {
        if (this->pixelsOut[i] == T2(0)) {
          vector<int> dum;
          processPixel(i, dum);
        }
      }
      return RES_OK;
    }

    virtual void processPixel(size_t pointOffset,
                              SMIL_UNUSED vector<int> &dOffsets)
    {
      T1 pVal = this->pixelsIn[pointOffset];

      if (pVal == T1(0) || this->pixelsOut[pointOffset] != T2(0))
        return;

      queue<size_t> propagation;
      int x, y, z, n_x, n_y, n_z;
      IntPoint p;

      ++real_labels;
      ++labels;
      if (labels == max_value_label)
        labels = 1;
      this->pixelsOut[pointOffset] = (T2) labels;
      propagation.push(pointOffset);

      bool oddLine = 0;
      size_t curOffset, nbOffset;

      while (!propagation.empty()) {
        curOffset = propagation.front();
        pVal      = this->pixelsIn[curOffset];

        z = curOffset / (this->imSize[1] * this->imSize[0]);
        y = (curOffset - z * this->imSize[1] * this->imSize[0]) /
            this->imSize[0];
        x = curOffset - y * this->imSize[0] -
            z * this->imSize[1] * this->imSize[0];

        oddLine = this->oddSe && (y % 2);

        for (UINT i = 0; i < this->sePointNbr; ++i) {
          p   = this->sePoints[i];
          n_x = x + p.x;
          n_y = y + p.y;
          n_x += (oddLine && ((n_y + 1) % 2) != 0);
          n_z      = z + p.z;
          nbOffset = n_x + (n_y) * this->imSize[0] +
                     (n_z) * this->imSize[1] * this->imSize[0];
          if (nbOffset != curOffset && n_x >= 0 &&
              n_x < (int) this->imSize[0] && n_y >= 0 &&
              n_y < (int) this->imSize[1] && n_z >= 0 &&
              n_z < (int) this->imSize[2] &&
              this->pixelsOut[nbOffset] != labels &&
              compareFunc(this->pixelsIn[nbOffset], pVal)) {
            this->pixelsOut[nbOffset] = T2(labels);
            propagation.push(nbOffset);
          }
        }
        propagation.pop();
      }
    }
#else
    virtual RES_T processImage(const imageInType &imIn,
                               imageOutType & /*imOut*/, const StrElt & /*se*/)
    {
      IntPoint pt;

      for (pt.z = 0; pt.z < this->imSize[2], pt.z++) {
        for (pt.y = 0; pt.y < this->imSize[1], pt.y++) {
          for (pt.x = 0; pt.x < this->imSize[0], pt.x++) {
            vector<int> dum;
            processPixel(pt, dum);
          }
        }
      }
      return RES_OK;
    }

    virtual void processPixel(IntPoint &pt, SMIL_UNUSED vector<int> &dOffsets)
    {
      T1 pVal = this->pixelsIn[pointOffset];

      if (pVal == T1(0) || this->pixelsOut[pointOffset] != T2(0))
        return;

      queue<size_t> propagation;

      ++real_labels;
      ++labels;
      if (labels == max_value_label)
        labels = 1;

      this->pixelsOut[pointOffset] = (T2) labels;

      propagation.push(pt);

      while (!propagation.empty()) {
        int x, y, z, n_x, n_y, n_z;
        IntPoint p;

        bool oddLine = 0;
        size_t curOffset, nbOffset;

        curOffset = propagation.front();
        pVal      = this->pixelsIn[curOffset];


        z = curOffset / (this->imSize[1] * this->imSize[0]);
        y = (curOffset - z * this->imSize[1] * this->imSize[0]) /
            this->imSize[0];
        x = curOffset - y * this->imSize[0] -
            z * this->imSize[1] * this->imSize[0];

        oddLine = this->oddSe && (y % 2);

        for (UINT i = 0; i < this->sePointNbr; ++i) {
          IntPoint sePt;
          IntPoint nl;

          sePt = this->sePoints[i];

          nl = pt + sePt;
          if (oddLine && ((nl.y + 1) % 2) != 0)
            nl.x += 1;

          nbOffset = n_x + (n_y) * this->imSize[0] +
                     (n_z) * this->imSize[1] * this->imSize[0];
          if (nbOffset != curOffset && n_x >= 0 &&
              n_x < (int) this->imSize[0] && n_y >= 0 &&
              n_y < (int) this->imSize[1] && n_z >= 0 &&
              n_z < (int) this->imSize[2] &&
              this->pixelsOut[nbOffset] != labels &&
              compareFunc(this->pixelsIn[nbOffset], pVal)) {
            this->pixelsOut[nbOffset] = T2(labels);
            propagation.push(nbOffset);
          }
        }
        propagation.pop();
      }
    }
#endif

    compOperatorT compareFunc;

  protected:
    T2 labels;
    size_t real_labels;
    T2 max_value_label;
  };



  template <class T1, class T2>
  size_t label(const Image<T1> &imIn, Image<T2> &imOut,
               const StrElt &se = DEFAULT_SE)
  {
    if ((void *) &imIn == (void *) &imOut) {
      // clone
      Image<T1> tmpIm(imIn, true);
      return label(tmpIm, imOut);
    }

    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    labelFunctGeneric<T1, T2> f;

    ASSERT((f._exec(imIn, imOut, se) == RES_OK), 0);

    size_t lblNbr = f.getLabelNbr();

    if (lblNbr > size_t(ImDtTypes<T2>::max()))
      std::cerr << "Label number exceeds data type max!" << std::endl;

    return lblNbr;
  }




label UINT8 1024x1024 7.28 msecs
lambdaLabel UINT8 1024x1024 7.27 msecs
fastLabel UINT8 1024x1024 15.01 msecs
labelWithArea UINT8 1024x1024 13.09 msecs

label UINT8 1024x1024 6.98 msecs
lambdaLabel UINT8 1024x1024 6.98 msecs
fastLabel UINT8 1024x1024 16.10 msecs
labelWithArea UINT8 1024x1024 12.90 msecs
