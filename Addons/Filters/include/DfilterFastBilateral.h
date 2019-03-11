#ifndef _DFAST_FILTER_H_
#define _DFAST_FILTER_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   AddonFilters
   * @defgroup  AddonFastFilter        Fast Bilateral Filter (2D)
   *
   * @brief A 2D Fast Bilateral Filter
   *
   * Filtrage par filtre bilateral rapide, version non exacte
   * * T(E) Vint 3D
   * * W,H,Z(E) taille Vint
   * * methodS,methodG(E) estimator 1->Gauss, !1->Tukey pour filtrage
   * spatial S et niveau de gris G
   * * nS(E) taille voisinage (à l'extérieur, Gaussienne nulle)
   * *  EcTSx,y,z(E) ecart type Gaussienne filtrage spatial suivant axes 
   * x, y et z
   * * EcTGx,y,z(E) Ecart type Gaussienne
   * filtrage niveau de gris suivant axes x, y et z
   * * Retour: Vint 3D 
   *
   * Remarque:
   * * calcul inexact (car le filtre n'est pas décomposable), mais les résultats
   * sont corrects et beaucoup plus rapide 
   * * Possibilité de gérer différement le
   * filtrage suivant les axes 
   * * valeur typique EcTS=3ou5 EcTG=20ou40 
   * * Equivalence Gauss - Tukey: Ect - Ect*Rac(5)
   * * Exemple de paramètres correct:
   *   * D = FastBilateralFilter(D, 2, 2, 10, 3, 3, 3);
   * ou
   *   * D = FastBilateralFilter(D, 2, 2, 10, 5, 5, 5); (un peu plus lent
   * mais resultat encore meilleurs)
   *
   * @author Vincent Morard / Jose-Marcio Martins da Cruz
   * @{ */



  /** @brief FastBilateral filter : Smooth the picture while conserving the
   * edges
   * @param[in]  imIn the initial image
   * @param[in]	Method : 1 for a gaussian window. Otherwize it is a tukey
   * window.
   * @param[in]  nS : size of the neigbourhood (Outside this window, the gaussian
   * is null) (common value 5)
   * @param[in]  EctS : standard deviation (std) for the spatial filtering
   * (common value : 3 or 5)
   * @param[in]  EctG : standard deviation (std) for the gray level filtering
   * (common value :20 or 40)
   * @param[out] imOut : Result of the bilateral filter of size Lenght
   */
  template <class T1, class T2>
  RES_T ImFastBilateralFilter(const Image<T1> &imIn, const UINT8 Method,
                              const UINT8 nS, const UINT32 EctS,
                              const UINT32 EctG, Image<T2> &imOut);

  /** @} */
  //Description: filtrage par filtre bilateral rapide, version non exacte 
  //Param: T(E) Vint 3D
  //Param: W,H,Z(E) taille Vint
  //Param: methodS,methodG(E) estimator 1->Gauss, !1->Tukey pour filtrage spatial S et niveau de gris G
  //Param: nS(E) taille voisinage (à l'extérieur, Gaussienne nulle)
  //Param: EcTSx,y,z(E) ecart type Gaussienne filtrage spatial suivant axes x, y et z
  //Param: EcTGx,y,z(E) Ecart type Gaussienne filtrage niveau de gris suivant axes x, y et z
  //Retour: Vint 3D
  //Remarque: calcul inexact (car le filtre n'est pas décomposable), mais les résultats sont corrects et beaucoup plus rapide
  //Possibilité de gérer différement le filtrage suivant les axes
  //valeur typique EcTS=3ou5 EcTG=20ou40
  //Equivalence Gauss - Tukey: Ect - Ect*Rac(5)
  //Exemple de paramètres correct:
  //D=FastBilateralFilter(D,W,H,Z,2,2,10,3,3,3,20,20,20);
  //ou
  //D=FastBilateralFilter(D,W,H,Z,2,2,10,5,5,5,20,20,20); (un peu plus lent mais resultat encore meilleurs)

} // namespace smil

#include "private/filterFastBilateral/filterFastBilateral.hpp"

#endif // _DFAST_FILTER_H_
