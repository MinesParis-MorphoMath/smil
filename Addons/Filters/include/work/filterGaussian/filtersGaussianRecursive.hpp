#ifndef MORPHEE_FILTERS_GAUSSIAN_FAST_T_HPP
#define MORPHEE_FILTERS_GAUSSIAN_FAST_T_HPP

#include <iostream>
#include <cmath>

#include <morphee/common/include/commonTypes.hpp>

namespace morphee
{
  namespace filters
  {
    namespace
    {
      typedef enum {
        GaussianDirectionHorizontal,
        GaussianDirectionVertical
      } eGaussianDirection;

      template <typename Real, int S = 2> class GaussRecFilter
      {
      public:
        typedef Real value_type;

        GaussRecFilter()
        {
        }
        //~GaussRecFilter()
        //{}

        value_type ExpoFilter2(const int ord, Real alpha); // ordre de
                                                           // derivation

        void recursiveFilter2(Real *input, Real *yp, Real *ym, int len, int inc,
                              int incOut, int dim, Real kNorm, int type,
                              Real *output) const;

        friend std::ostream &operator<<(std::ostream &, const GaussRecFilter &);

      private:
        Real m_np[S];
        Real m_nm[S];
        Real m_d[S];
      };

      template <typename Real, int S>
      typename GaussRecFilter<Real, S>::value_type
      GaussRecFilter<Real, S>::ExpoFilter2(
          const int ord, /* ordre de derivation */
          Real alpha)
      // retourne double kNorm ; facteur de normalisation */
      // et remplit le filtre
      {
        value_type k, t1, t2, t4, t7;
        // value_type alpha=1.0;//*******parametre a modifier ??

        value_type expa  = std::exp(-alpha);
        value_type exp2a = expa * expa;

        m_d[0] = -2.0 * expa;
        m_d[1] = exp2a;

        switch (ord) {
        case 0:
          k = 1 - expa;
          k = k * k / (1.0 + 2.0 * alpha * expa - exp2a);

          m_np[0] = k;
          m_np[1] = k * (alpha - 1.0) * expa;

          m_nm[0] = k * expa * (alpha + 1.0);
          m_nm[1] = -k * exp2a;
          return 1.0;
        case 1:
          m_np[0] = 0.0;
          m_np[1] = 1.0;

          m_nm[0] = 1.0;
          m_nm[1] = 0.0;
          t1      = std::exp(alpha);
          t2      = t1 + 1.0;
          t4      = t1 * t1;
          t7      = 1 / (t4 * t1 - 3.0 * t4 + 3.0 * t1 - 1.0);
          return -expa / (t1 * t2 * t7 + t1 / (-1.0 + t1) + t4 * t2 * t7 -
                          2.0 * t4 / (1.0 - 2.0 * t1 + t4));
        case 2:
          m_np[0] = exp(alpha);        /* Constant defined by Deriche */
          m_np[1] = m_np[0] * m_np[0]; /*  in its implementation so that */
          m_nm[0] =
              m_np[0] * m_np[1]; /* the second derivative of a quadratic */
          m_nm[1] =
              (1. - exp2a) / (2. * alpha * expa); /* function is equal to 2. */
          m_np[0] = 2 * (m_nm[0] - 3 * (m_np[1] - m_np[0]) - 1.) /
                    (m_nm[0] + 3. * (m_np[1] + m_np[0]) + 1.);
          m_np[1] = m_np[0] * (m_nm[1] * alpha + 1.) * expa;
          m_nm[0] = m_np[0] * (-1. + m_nm[1] * alpha) * expa;
          m_nm[1] = m_np[0] * exp2a;
          m_np[0] *= -1.;
          return 1.0;
          /*   np[0]=(1.-exp2a)/(2.*alpha*expa);*/ /* Constants defined in the
                                                    */
          /*   np[1]=-(np[0]*alpha+1)*expa; */ /* article so that the response
                                                */
          /*   nm[0]= (1.-np[0]*alpha)*expa;   */ /* to a constant signal is
                                                     null.*/
                                                  /*   nm[1]= -exp2a; */
                                                  /*   np[0]=1.; */
        default:
          throw("Error in ExpoFilter2, invalid order of derivation");
        } // switch derivation order
      }

      template <typename Real, int S>
      void GaussRecFilter<Real, S>::recursiveFilter2(
          Real *input, Real *yp, Real *ym, int len, int inc, int incOut,
          int dim, Real kNorm, /* normalization coeff */
          int type,            /* 0: odd filter, 1: even */
          Real *output) const
      // inc vaut soit 1 (cas HORIZ) soit largeur_image (cas VERT) dans les
      // deux cas, on se deplace d'un pixel dans un sens ou l'autre
      {
        int k;
        Real *inp2, *inp1, out;

        /*    Initialization of the recursive constants.  */
        Real d0 = m_d[0];
        Real d1 = m_d[1];

        /*    Initialization of filter's constants.   */
        Real n0 = m_np[0];
        Real n1 = m_np[1];

        /*    Side initialisation.        */
        Real *inp = input; // position de depart ?
        Real in1  = *inp;
        inp += inc;
        Real in0 = *inp;
        inp += inc;
        /*
         *    Normally here we have:
         *
         *          in0  = input[inc]; (in0=input[k])
         *          in1  = input[0];   (in1=input[k-1])
         *         out0  = output[0];  (out0=output[k-2])
         *         out1  = output[inc]; (out1=output[k-1])
         *
         */

        Real *outp = yp; // yp reste fixe, outp se promene
        Real out0 = *outp++ = n0 * in1;
        Real out1 = *outp++ = n0 * in0 + n1 * in1 - d0 * out0; // OK

        /*    Computation of the causal part of the filter.   */
        for (k = 2; k < len; k++) {
          in1 = in0;
          in0 = /*(double)*/ *inp;
          out = *outp++ = n0 * in0 + n1 * in1 - d0 * out1 - d1 * out0;
          out0          = out1;
          out1          = out;
          inp += inc;
        } // le resultat est dans *outp

        /*    Computation of the anti-causal part of the filter.  */

        /*    Initialisation of filter's constants.   */
        n0 = m_nm[0];
        n1 = m_nm[1];

        /*    Side initialisation.        */

        /*
         *    Normally here we have:
         *
         *          in0  = input[len(input)-inc]; (input[k+2])
         *          in1  = input[len(input)-2*inc];(input[k+1])
         *          out1 = output[k+2]
         *          out0 = output[k+1]
         *    So Only inp needs to be initialized.
         *
         */

        inp  = inp - 3 * inc;
        outp = &ym[len - 1]; // on met outp a la fin de ym, et on le fait
                             // reculer
        out1 = *outp-- = 0;
        out0 = *outp-- = n0 * in0; //???

        /*    Computation of the recursive part of the filter.        */
        for (k = len - 3; k >= 0; k--) {
          out = *outp-- =
              n0 * in1 + n1 * in0 - d0 * out0 - d1 * out1; // outp recule et
          in0  = in1; // remplit ym a l'envers
          in1  = /*(double)*/ *inp;
          out1 = out0;
          out0 = out;
          inp -= inc;
        }

        /*    Compute the result by adding causal and anticausal parts.    */

        // outputFloat = output;//valeur de retour
        inp2 = yp;
        inp1 = ym;
        if (type == 1) // somme des deux tableauxm yp et ym
          for (k = 0; k < len; k++) {
            *output = kNorm * ((*inp2++) - (*inp1++));
            output += incOut;
          }
        else
          for (k = 0; k < len; k++) {
            *output = (*inp2++) + (*inp1++);
            output += incOut;
          }
        // delete(yp);
      }

      template <typename Real, int S>
      std::ostream &operator<<(std::ostream &os,
                               const GaussRecFilter<Real, S> &grf)
      {
        os << "np: " << grf.m_np[0] << ' ' << grf.m_np[1] << '\n';
        os << "nm: " << grf.m_nm[0] << ' ' << grf.m_nm[1] << '\n';
        os << "d : " << grf.m_d[0] << ' ' << grf.m_d[1];

        return os;
      }

      template <class T>
      void DerivExpoRec1D(T *input, const int width, const int height,
                          const int ord, int filterRadius,
                          const eGaussianDirection dir, T *output)
      {
        // double np[2],nm[2],d[2]; /* coefficients du filtre */
        T kNorm; /* normalisation du filtre */
        int dim;
        int k;
        int localLen, localIncIn,
            localIncOut; // parametres de la fenetre de lissage
        int smoothLen, smoothIncIn, smoothIncOut;
        int type = ord % 2;

        T *inp = input;

        // ATTENTION pour des raisons de commodite,cette fonction est marquee
        // template mais ne fonctionne qu'avec des grey_image<double>

        switch (dir) {
        case GaussianDirectionHorizontal:
          dim        = width;
          localLen   = height;
          localIncIn = width; // imgi.largeur();
          localIncOut =
              dim /**sizeof(double)*/; // attention quand on passera en double*
          smoothLen   = dim;
          smoothIncIn = smoothIncOut = 1;
          break;
        case GaussianDirectionVertical:
          dim          = height; // HAUTEUR_IMAGE(imgd);
          localLen     = width;  // LARGEUR_IMAGE(imgd);
          localIncIn   = 1;
          localIncOut  = 1 /*sizeof(double)*/; // meme remarque
          smoothLen    = dim;
          smoothIncIn  = width;  // LARGEUR_IMAGE(imgi);
          smoothIncOut = height; // LARGEUR_IMAGE(imgd);
          break;
        default:
          throw("Invalid direction");
        }

        /* Derivation et filtrage */

        // filtre
        GaussRecFilter<T> f;
        // T alpha = 1. / (2. * filterRadius * filterRadius);
        // T alpha = 2.0 ; //(T)filterRadius / 4.;
        // T alpha =  36. / ( std::pow(2. * filterRadius +1,2. ) ) ;
        T alpha = 5.;

        std::cout << " alpha = " << alpha << "\n";
        kNorm = f.ExpoFilter2(ord, alpha);

        // initialisation des buffers yp et  ym:
        T *yp;

        if (!(yp = new (std::nothrow) T[2 * dim])) {
          throw("Pb d'allocation dans recursiveFilter2 pour yp");
        }
        T *ym = &(yp[dim]);

        for (k = 0; k < localLen; k++, inp += localIncIn) {
          f.recursiveFilter2(inp, yp, ym, smoothLen, smoothIncIn, smoothIncOut,
                             dim, kNorm, type, output);
          output += localIncOut;
        }
        delete[] yp;
      } // DerivExpoRec1D

    } // namespace

    template <class ImageReal>
    RES_C ImGaussianRecursive_Helper(ImageReal &imIn, int filterRadius,
                                     ImageReal &imOut)
    {
      ImageReal tmp = imIn.getSame();

      DerivExpoRec1D(imIn.rawPointer(), imIn.getXSize(), imIn.getYSize(), 0,
                     filterRadius, GaussianDirectionHorizontal,
                     tmp.rawPointer());
      DerivExpoRec1D(tmp.rawPointer(), imIn.getXSize(), imIn.getYSize(), 0,
                     filterRadius, GaussianDirectionVertical,
                     imOut.rawPointer());

      return RES_OK;
    }

  } // namespace filters

} // namespace morphee

#endif // MORPHEE_FILTERS_GAUSSIAN_FAST_T_HPP
