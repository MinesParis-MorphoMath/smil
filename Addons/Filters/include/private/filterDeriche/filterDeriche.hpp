#ifndef __DERICHE_FILTER_T_HPP__
#define __DERICHE_FILTER_T_HPP__

namespace smil
{
  /**
   * @author Vincent Morard
   * @brief Deriche edge detection:
   *	R. Deriche, Using Canny's criteria to derive a recursively implemented
   *optimal edge detector, Int. J. Computer Vision, Vol. 1, pp. 167-187, April
   *1987.
   */

  //**************************************************************************
  // Deriche
  // Ce fichier regroupe certaines fonctions permettant d'effectuer la
  // détection de coutour de Deriche. On ne calculera dans ce fichier que
  // l'image lissée par le filtre récursif de Deriche. OUTLINE corespond au
  // nombre de pixel que l'on ne prendra pas en compte lors de la convolution.
  // Aussi, pour pouvoir effectuer la détection de contour à tous les pixels y
  // compris ceux du bord de l'image, on augmentera la taille de l'image de
  // 2*OUTLINE. C'est le rôle des fonctions embded et debed
  //**************************************************************************

#define OUTLINE 25
#define WINDOW_SIZE 7

  //**************************************************************************
  // embed
  // Cette fonction retourne un pointeur sur un float qui de l'image initial
  // ou l'on a ajouté des bordure de largeur et de hauteur de Width
  //**************************************************************************
  template <typename T1>
  double *embed(T1 *imIn, int W, int H, int size, int *NewW, int *NewH)
  {
    int i, j, I, J;

    double *Img = 0;

    size += 2;
    *NewW = W + 2 * size;
    *NewH = H + 2 * size;
    Img   = new double[(*NewW) * (*NewH)];
    if (Img == 0)
      return 0;

    for (i = 0; i < *NewW; i++)
      for (j = 0; j < *NewH; j++) {
        I                    = (i - size + W) % W;
        J                    = (j - size + H) % H;
        Img[i + j * (*NewW)] = (double) imIn[I + J * W];
      }

    return Img;
  }

  //**************************************************************************
  // debed
  // Cette fonction retourne inscrit l'image obtenue dans le buffer Dest en
  // enlevant les bordures ajoutées
  //**************************************************************************
  template <typename T2>
  void debed(T2 *imOut, int W, int H, int size, double *Img, int NewW, int NewH)
  {
    int i, j;
    size += 2;
    for (i = size; i < NewW - size; i++)
      for (j = size; j < NewH - size; j++)
        imOut[(i - size) + (j - size) * W] = (T2) Img[i + j * NewW];
  }

  //**************************************************************************
  // ApplySmoothing_Horizontal
  // On calcule ici le flou suivant la direction horizontale. On calcule tout
  // d'abord les valeurs des coefficients de la fonction définie par Deriche.
  // On doit faire attention à la gestion des pixels des bordures: il faut
  // faire leur initialisation avant de lancer la procédure récursive. La
  // sortie y correspond à la somme des composantes causales et des
  // composantes anticausales.
  //**************************************************************************
  void ApplySmoothing_Horizontal(double *x, double *y, double *Causal,
                                 double *AntiCausal, int W, int H, double Alpha)
  {
    double c1, c2, a1, a2, g;
    double s1, s2, s;
    int i, j;

    g = ((1 - exp(-Alpha)) * (1 - exp(-Alpha))) /
        (1 + 2 * Alpha * exp(-Alpha) - exp(-2 * Alpha));
    a1 = exp(-Alpha);
    a2 = exp(-2 * Alpha);
    c1 = Alpha + 1;
    c2 = Alpha - 1;

    for (j = 0; j < H; j++) {
      s2 = x[(W - 1) + j * W];
      s1 = s2;

      for (i = 0; i < 2;
           i++) { // les 2 derniers pixels (-> on duplique le bord)
        s  = s1;
        s1 = g * a1 * c1 * x[W - 1 + j * W] - g * a2 * x[W - 1 + j * W] +
             2 * a1 * s1 - a2 * s2;
        s2 = s;
      }
      for (i = W - 3; i >= 0; i--) {
        s  = s1;
        s1 = g * a1 * c1 * x[i + 1 + j * W] - g * a2 * x[i + 2 + j * W] +
             2 * a1 * s1 - a2 * s2; // initialisation (lissage anticausal)
        s2 = s;
      }
      Causal[0 + j * W] = s1; // recopie du point dans les 2 premieres colonnes.
      Causal[1 + j * W] = s2;

      for (i = 2; i < W; i++) {
        Causal[i + j * W] =
            (float) (g * x[i + j * W] + g * a1 * c2 * x[i - 1 + j * W] +
                     2 * a1 * Causal[i - 1 + j * W] -
                     a2 * Causal[i - 2 + j * W]); // lissage causal
        s1 = s;
        s1 = g * x[i + j * W] + g * a1 * c2 * x[i - 1 + j * W] + 2 * a1 * s1 -
             a2 * s2; // deuxieme lissage causal pour initialisation
        s2 = s;       // de l'anticausal
      }

      AntiCausal[W - 1 + j * W] = s1; // recopie du point d'initialisation
      AntiCausal[W - 2 + j * W] = s2; // dans les 2 dernieres colonnes.

      for (i = W - 3; i >= 0; i--)
        AntiCausal[i + j * W] =
            (double) (g * a1 * c1 * x[i + 1 + j * W] -
                      g * a2 * x[i + 2 + j * W] +
                      2 * a1 * AntiCausal[i + 1 + j * W] -
                      a2 * AntiCausal[i + 2 + j * W]); // lissage anticausal
    }

    for (i = 0; i < W * H; i++)
      y[i] = Causal[i] + AntiCausal[i];
  }

  //**************************************************************************
  // ApplySmoothing_Vertical
  // Meme fonction que ApplySmoothing_Horizontal sauf que les composante sont
  // créées suivant un axe vertical.
  //**************************************************************************
  void ApplySmoothing_Vertical(double *x, double *y, double *Causal,
                               double *AntiCausal, int W, int H, double Alpha)
  {
    float c1, c2, a1, a2, g;
    double s1, s2, s;
    int i, j;

    g  = (float) (((1 - exp(-Alpha)) * (1 - exp(-Alpha))) /
                 (1 + 2 * Alpha * exp(-Alpha) - exp(-2 * Alpha)));
    a1 = (float) exp(-Alpha);
    a2 = (float) exp(-2 * Alpha);
    c1 = (float) (Alpha + 1);
    c2 = (float) (Alpha - 1);

    for (i = 0; i < W; i++) {
      s2 = x[i + (H - 1) * W];
      s1 = s2;

      for (j = 0; j < 2;
           j++) { // les 2 derniers pixels (-> on duplique le bord)
        s  = s1;
        s1 = g * a1 * c1 * x[i + (H - 1) * W] - g * a2 * x[i + (H - 1) * W] +
             2 * a1 * s1 - a2 * s2;
        s2 = s;
      }
      for (j = H - 3; j >= 0; j--) {
        s  = s1;
        s1 = g * a1 * c1 * x[i + (j + 1) * W] - g * a2 * x[i + (j + 2) * W] +
             2 * a1 * s1 - a2 * s2; // initialisation (lissage anticausal)
        s2 = s;
      }
      Causal[i]     = s1; // recopie du point dans les 2 premieres colonnes.
      Causal[i + W] = s2;

      for (j = 2; j < H; j++) {
        Causal[i + j * W] =
            (double) (g * x[i + j * W] + g * a1 * c2 * x[i + (j - 1) * W] +
                      2 * a1 * Causal[i + (j - 1) * W] -
                      a2 * Causal[i + (j - 2) * W]); // lissage causal
        s1 = s;
        s1 = g * x[i + j * W] + g * a1 * c2 * x[i + (j - 1) * W] + 2 * a1 * s1 -
             a2 * s2; // deuxieme lissage causal pour initialisation
        s2 = s;       // de l'anticausal
      }

      AntiCausal[i + (H - 1) * W] = s1; // recopie du point d'initialisation
      AntiCausal[i + (H - 2) * W] = s2; // dans les 2 dernieres colonnes.

      for (j = H - 3; j >= 0; j--)
        AntiCausal[i + j * W] =
            (g * a1 * c1 * x[i + (j + 1) * W] - g * a2 * x[i + (j + 2) * W] +
             2 * a1 * AntiCausal[i + (j + 1) * W] -
             a2 * AntiCausal[i + (j + 2) * W]); // lissage anticausal
    }

    for (i = 0; i < W * H; i++)
      y[i] = Causal[i] + AntiCausal[i];
  }

  //**************************************************************************
  // ComputeSmoothing
  // Cette fonction permet de calculer récursivement (au sens traitement du
  // signal) l'image Lisser suivant les critères définis par Deriche. On
  // calcule donc les composantes causales et anticausales de l'image
  // d'origine et on fait la somme de ces deux composantes.
  //**************************************************************************
  bool ComputeSmoothing(double *x, double *y, int W, int H, double Alpha)
  {
    double *Causal, *AntiCausal;

    Causal = new double[W * H];
    if (Causal == 0)
      return 0;

    AntiCausal = new double[W * H];
    if (AntiCausal == 0) {
      delete[] Causal;
      return 0;
    }

    ApplySmoothing_Horizontal(x, y, Causal, AntiCausal, W, H, Alpha);
    ApplySmoothing_Vertical(y, y, Causal, AntiCausal, W, H, Alpha);

    delete[] Causal;
    delete[] AntiCausal;
    return 1;
  }

  //**************************************************************************
  // ComputeBli:
  // On fait la différence des deux images et on compare le resultat à 0
  // ImgBli est donc une image composé uniquement de 0 et de 1
  //**************************************************************************
  bool *ComputeBli(double *ImgFiltrer, double *ImgBuf, int W, int H)
  {
    int i, j;
    bool *ImgBli = new bool[W * H];
    if (ImgBli == 0)
      return 0;

    // On prend la difference entre l'image lisse et l'image originale.
    // On calcule l'ImgBli en mettant a 1 tous les pixels ou le Laplacian est
    // positif. 0 sinon
    for (i = 0; i < W; i++)
      for (j = 0; j < H; j++) {
        ImgBli[i + j * W] = 0;
        if (i < OUTLINE || i >= W - OUTLINE || j < OUTLINE || j >= H - OUTLINE)
          continue;
        ImgBli[i + j * W] = ((ImgFiltrer[i + j * W] - ImgBuf[i + j * W]) > 0.0);
      }

    return ImgBli;
  }

  //**************************************************************************
  // IsCandidateEdge
  // On regarde le pixel voisin en on regarde s'il y a un franchissement de
  // zero. On effectue donc la multiplication des 2 pixels et on compare à 0
  //**************************************************************************
  bool IsCandidateEdge(bool *Buff, double *Orig, int W, int H, int i, int j)
  {
    // a positive z-c must have a positive 1st derivative,where positive z-c
    // means the second derivative goes from + to - as we cross the edge

    if (Buff[i + j * W] == 1 && Buff[i + 1 + j * W] == 0)
      return (Orig[i + 1 + j * W] - Orig[i - 1 + j * W] > 0
                  ? 1
                  : 0); // positive z-c

    else if (Buff[i + j * W] == 1 && Buff[i + (j + 1) * W] == 0)
      return (Orig[i + (j + 1) * W] - Orig[i + (j - 1) * W] > 0
                  ? 1
                  : 0); // positive z-c

    else if (Buff[i + j * W] == 1 && Buff[i - 1 + j * W] == 0)
      return (Orig[i + 1 + j * W] - Orig[i - 1 + j * W] < 0
                  ? 1
                  : 0); // negative z-c

    else if (Buff[i + j * W] == 1 && Buff[i + (j - 1) * W] == 0)
      return (Orig[i + (j + 1) * W] - Orig[i + (j - 1) * W] < 0
                  ? 1
                  : 0); // negative z-c

    return 0; // not a z-c
  }

  //**************************************************************************
  // ComputeAdaptativeGradient
  // On calcule un seuil pour chaque pixel. Le seuil sera determiner grace aux
  // pixels voisins présent dans la fenêtre WINDOW_SIZE
  //**************************************************************************
  double ComputeAdaptativeGradient(bool *Bli, double *Orig, int W, int H, int k,
                                   int l)
  {
    int i, j;
    double SumOn, SumOff;
    double AvgOn, AvgOff;
    int NbOn, NbOff;

    SumOn  = 0.0;
    SumOff = 0.0;
    NbOn   = 0;
    NbOff  = 0;

    // On regarde par rapport aux pixels voisins, le nombre de pixel a 1 et 0
    for (i = (-WINDOW_SIZE / 2); i <= (WINDOW_SIZE / 2); i++)
      for (j = (-WINDOW_SIZE / 2); j <= (WINDOW_SIZE / 2); j++) {
        if (Bli[k + i + (l + j) * W]) {
          SumOn += Orig[k + i + (l + j) * W];
          NbOn++;
        } else {
          SumOff += Orig[k + i + (l + j) * W];
          NbOff++;
        }
      }

    if (SumOff)
      AvgOff = SumOff / (double) NbOff;
    else
      AvgOff = 0.0;

    if (SumOn)
      AvgOn = SumOn / (double) NbOn;
    else
      AvgOn = 0.0;
    return abs(AvgOff - AvgOn);
  }

  //**************************************************************************
  // LocateZeroCrossings
  // Cette fonction permettra de déterminer pour tous les pixels de l'image
  // s'il est un pixel appartenant à un contour ou non.
  //**************************************************************************
  void LocateZeroCrossings(double *Orig, double *BufFiltrer, bool *ImgBli,
                           int W, int H)
  {
    int i, j;

    for (i = 0; i < W; i++)
      for (j = 0; j < H; j++) {
        // On ignore les pixels que l'on a ajoute pour le calcule
        if (i < OUTLINE || i >= W - OUTLINE || j < OUTLINE || j >= H - OUTLINE)
          Orig[i + j * W] = 0.0;

        // On verifie si ce pixel est un "zero crossing" pour le Laplacian
        else if (IsCandidateEdge(ImgBli, BufFiltrer, W, H, i, j)) {
          // On Calcule le gradian adaptatif
          Orig[i + j * W] =
              ComputeAdaptativeGradient(ImgBli, BufFiltrer, W, H, i, j);

        } else
          Orig[i + j * W] = 0.0;
      }
  }

  //**************************************************************************
  // DericheFilter
  // Tout le traitement est effectué dans cette fonction. On reçoit en entrée
  // l'image étendue avec de nouvelles bordures ainsi que la nouvelle Largeur
  // et la nouvelle hauteur de l'image. On calcule donc l'image filtrée,
  // l'image BLI et on effectue la détection des zéros.
  //**************************************************************************
  bool DericheFilter(double *Img, int W, int H, double Alpha)
  {
    double *BufFiltrer = 0;
    bool *ImgBli       = 0;

    BufFiltrer = new double[W * H];
    if (BufFiltrer == 0)
      return 0;

    if (ComputeSmoothing(Img, BufFiltrer, W, H, Alpha) == 0) {
      delete[] BufFiltrer;
      return 0;
    }

    // On calule une image qui sera egale a 1 lorsque le Laplacian est
    // positif. 0 sinon
    ImgBli = ComputeBli(BufFiltrer, Img, W, H);

    // Detection des contours en localisant les passages a 0
    LocateZeroCrossings(Img, BufFiltrer, ImgBli, W, H);

    delete[] BufFiltrer;
    delete[] ImgBli;
    return 1;
  }

  //************************************************************************
  // Deriche
  // C'est le point d'entrée de cette méthode, (fonction export). Il prend en
  // argument l'image source et l'image destination ainsi que la valeur du
  // paramètre alpha du traitement.
  //************************************************************************
  template <typename T1, typename T2>
  RES_T Deriche(T1 *imIn, int W, int H, double Alpha, T2 *imOut)
  {
    int NewW, NewH;

    // On ajoute des bords de largeur et de hauteur OUTLINE pour éviter les
    // effets de bord.

    double *Img = embed(imIn, W, H, OUTLINE, &NewW, &NewH);
    if (Img == 0)
      return RES_ERROR_MEMORY;

    // Start
    if (DericheFilter(Img, NewW, NewH, Alpha) == 0) {
      delete[] Img;
      return RES_ERROR_MEMORY;
    }

    // On supprime les bordures créées.
    debed(imOut, W, H, OUTLINE, Img, NewW, NewH);

    delete[] Img;
    return RES_OK;
  }

  template <class T>
  RES_T ImDericheEdgeDetection(const Image<T> *imIn, const double Alpha,
                               Image<T> *imOut)
  {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);

      size_t s[3];
      imIn.getSize(s);
      
      // TODO: check that image is 2D
      if (s[2] > 1) {
        // Error : this is a 3D image
      }

      typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
      typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();
      
      return Deriche(bufferIn, Alpha, bufferOut);
  }
} // namespace smil
#endif
