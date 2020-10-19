#ifndef __FAST_AREA_OPENIN_MAX_TREE_T_HPP__
#define __FAST_AREA_OPENIN_MAX_TREE_T_HPP__

#include "Core/include/DCore.h"

// Interface to SMIL Vincent Morard
// Date : 7 march 2011

/* maxtree3a.c
 * May 26, 2005  Erik R. Urbach
 * Email: erik@cs.rug.nl
 * Max-tree with a single attribute parameter and an optional template
 * Attribute: I/(A^2) (default) and others (start program without arguments for
 *            a complete list of attributes available)
 * Decision: Min, Direct, Max, Subtractive (default)
 * Input images: raw (P5) and plain (P2) PGM 8-bit gray-scale images
 * Output image: raw (P5) PGM 8-bit gray-scale image
 * Compilation: gcc -ansi -pedantic -Wall -O3 -o maxtree3a maxtree3a.c
 *
 * Related papers:
 * [1] E. J. Breen and R. Jones.
 *     Attribute openings, thinnings and granulometries.
 *     Computer Vision and Image Understanding.
 *     Vol.64, No.3, Pages 377-389, 1996.
 * [2] P. Salembier and A. Oliveras and L. Garrido.
 *     Anti-extensive connected operators for image and sequence processing.
 *     IEEE Transactions on Image Processing,
 *     Vol.7, Pages 555-570, 1998.
 * [3] E. R. Urbach and M. H. F. Wilkinson.
 *     Shape-Only Granulometries and Grey-Scale Shape Filters.
 *     Proceedings of the ISMM2002,
 *     Pages 305-314, 2002.
 * [4] E. R. Urbach and J. B. T. M. Roerdink and M. H. F. Wilkinson.
 *     Connected Rotation-Invariant Size-Shape Granulometries.
 *     Proceedings of the 17th Int. Conf. Pat. Rec.,
 *     Vol.1, Pages 688-691, 2004.
 */

namespace smil
{
#define NUMLEVELS 256
#define CONNECTIVITY 4
#ifdef PI
#undef PI
#endif
#define PI 3.14159265358979323846
#define bzero(b, len) (memset((b), '\0', (len)), (void) 0)

  typedef unsigned char ubyte;
  typedef unsigned int uint;
  typedef unsigned long ulong;

#ifdef MIN
#undef MIN
#endif
#define MIN(a, b) ((a <= b) ? (a) : (b))
#ifdef MAX
#undef MAX
#endif
#define MAX(a, b) ((a >= b) ? (a) : (b))

  typedef struct ImageGray ImageGray;

  struct ImageGray {
    ulong Width;
    ulong Height;
    ubyte *Pixmap;
  };

  typedef struct HQueue {
    ulong *Pixels;
    ulong Head;
    ulong Tail; /* First free place in queue, or -1 when the queue is full */
  } HQueue;

  typedef struct MaxNode MaxNode;

  struct MaxNode {
    ulong Parent;
    ulong Area;
    void *Attribute;
    ubyte Level;
    ubyte NewLevel; /* gray level after filtering */
  };

/* Status stores the information of the pixel status: the pixel can be
 * NotAnalyzed, InTheQueue or assigned to node k at level h. In this
 * last case Status(p)=k. */
#define ST_NotAnalyzed -1
#define ST_InTheQueue -2

  typedef struct MaxTree MaxTree;

  struct MaxTree {
    long *Status;
    ulong *NumPixelsBelowLevel;
    ulong *NumNodesAtLevel; /* Number of nodes C^k_h at level h */
    MaxNode *Nodes;
    void *(*NewAuxData)(ulong, ulong);
    void (*AddToAuxData)(void *, ulong, ulong);
    void (*MergeAuxData)(void *, void *);
    void (*DeleteAuxData)(void *);
  };

  void MaxTreeDelete(MaxTree *mt);

  /****** Typedefs and functions for area attributes
   * ******************************/

  typedef struct AreaData {
    ulong Area;
  } AreaData;

  void *NewAreaData(SMIL_UNUSED ulong x, SMIL_UNUSED ulong y)
  {
    AreaData *areadata;

    areadata       = (AreaData *) malloc(sizeof(AreaData));
    areadata->Area = 1;
    return (areadata);
  } /* NewAreaData */

  void DeleteAreaData(void *areaattr)
  {
    free(areaattr);
  } /* DeleteAreaData */

  void AddToAreaData(void *areaattr, SMIL_UNUSED ulong x, SMIL_UNUSED ulong y)
  {
    AreaData *areadata = (AreaData *) areaattr;

    areadata->Area++;
  } /* AddToAreaData */

  void MergeAreaData(void *areaattr, void *childattr)
  {
    AreaData *areadata  = (AreaData *) areaattr;
    AreaData *childdata = (AreaData *) childattr;

    areadata->Area += childdata->Area;
  } /* MergeAreaData */

  double AreaAttribute(void *areaattr)
  {
    AreaData *areadata = (AreaData *) areaattr;
    double area;

    area = areadata->Area;
    return (area);
  } /* AreaAttribute */

  /****** Typedefs and functions for minimum enclosing rectangle attributes
   * *******/

  typedef struct EnclRectData {
    ulong MinX;
    ulong MinY;
    ulong MaxX;
    ulong MaxY;
  } EnclRectData;

  void *NewEnclRectData(ulong x, ulong y)
  {
    EnclRectData *rectdata;

    rectdata       = (EnclRectData *) malloc(sizeof(EnclRectData));
    rectdata->MinX = rectdata->MaxX = x;
    rectdata->MinY = rectdata->MaxY = y;
    return (rectdata);
  } /* NewEnclRectData */

  void DeleteEnclRectData(void *rectattr)
  {
    free(rectattr);
  } /* DeleteEnclRectData */

  void AddToEnclRectData(void *rectattr, ulong x, ulong y)
  {
    EnclRectData *rectdata = (EnclRectData *) rectattr;

    rectdata->MinX = MIN(rectdata->MinX, x);
    rectdata->MinY = MIN(rectdata->MinY, y);
    rectdata->MaxX = MAX(rectdata->MaxX, x);
    rectdata->MaxY = MAX(rectdata->MaxY, y);
  } /* AddToEnclRectData */

  void MergeEnclRectData(void *rectattr, void *childattr)
  {
    EnclRectData *rectdata  = (EnclRectData *) rectattr;
    EnclRectData *childdata = (EnclRectData *) childattr;

    rectdata->MinX = MIN(rectdata->MinX, childdata->MinX);
    rectdata->MinY = MIN(rectdata->MinY, childdata->MinY);
    rectdata->MaxX = MAX(rectdata->MaxX, childdata->MaxX);
    rectdata->MaxY = MAX(rectdata->MaxY, childdata->MaxY);
  } /* MergeEnclRectData */

  double EnclRectAreaAttribute(void *rectattr)
  {
    EnclRectData *rectdata = (EnclRectData *) rectattr;
    double area;

    area = (rectdata->MaxX - rectdata->MinX + 1) *
           (rectdata->MaxY - rectdata->MinY + 1);
    return (area);
  } /* EnclRectAreaAttribute */

  double EnclRectDiagAttribute(void *rectattr)
  /* Computes the square of the length of the diagonal */
  {
    EnclRectData *rectdata = (EnclRectData *) rectattr;
    double minx, miny, maxx, maxy, l;

    minx = rectdata->MinX;
    miny = rectdata->MinY;
    maxx = rectdata->MaxX;
    maxy = rectdata->MaxY;
    l    = (maxx - minx + 1) * (maxx - minx + 1) +
        (maxy - miny + 1) * (maxy - miny + 1);
    return (l);
  } /* EnclRectDiagAttribute */

  /****** Typedefs and functions for moment of inertia attributes
   * **************************/

  typedef struct InertiaData {
    ulong Area;
    double SumX, SumY, SumX2, SumY2;
  } InertiaData;

  void *NewInertiaData(ulong x, ulong y)
  {
    InertiaData *inertiadata;

    inertiadata        = (InertiaData *) malloc(sizeof(InertiaData));
    inertiadata->Area  = 1;
    inertiadata->SumX  = x;
    inertiadata->SumY  = y;
    inertiadata->SumX2 = x * x;
    inertiadata->SumY2 = y * y;
    return (inertiadata);
  } /* NewInertiaData */

  void DeleteInertiaData(void *inertiaattr)
  {
    free(inertiaattr);
  } /* DeleteInertiaData */

  void AddToInertiaData(void *inertiaattr, ulong x, ulong y)
  {
    InertiaData *inertiadata = (InertiaData *) inertiaattr;

    inertiadata->Area++;
    inertiadata->SumX += x;
    inertiadata->SumY += y;
    inertiadata->SumX2 += x * x;
    inertiadata->SumY2 += y * y;
  } /* AddToInertiaData */

  void MergeInertiaData(void *inertiaattr, void *childattr)
  {
    InertiaData *inertiadata = (InertiaData *) inertiaattr;
    InertiaData *childdata   = (InertiaData *) childattr;

    inertiadata->Area += childdata->Area;
    inertiadata->SumX += childdata->SumX;
    inertiadata->SumY += childdata->SumY;
    inertiadata->SumX2 += childdata->SumX2;
    inertiadata->SumY2 += childdata->SumY2;
  } /* MergeInertiaData */

  double InertiaAttribute(void *inertiaattr)
  {
    InertiaData *inertiadata = (InertiaData *) inertiaattr;
    double area, inertia;

    area    = inertiadata->Area;
    inertia = inertiadata->SumX2 + inertiadata->SumY2 -
              (inertiadata->SumX * inertiadata->SumX +
               inertiadata->SumY * inertiadata->SumY) /
                  area +
              area / 6.0;
    return (inertia);
  } /* InertiaAttribute */

  double InertiaDivA2Attribute(void *inertiaattr)
  {
    InertiaData *inertiadata = (InertiaData *) inertiaattr;
    double inertia, area;

    area    = (double) (inertiadata->Area);
    inertia = inertiadata->SumX2 + inertiadata->SumY2 -
              (inertiadata->SumX * inertiadata->SumX +
               inertiadata->SumY * inertiadata->SumY) /
                  area +
              area / 6.0;
    return (inertia * 2.0 * PI / (area * area));
  } /* InertiaDivA2Attribute */

  double MeanXAttribute(void *inertiaattr)
  {
    InertiaData *inertiadata = (InertiaData *) inertiaattr;
    double area, sumx;

    area = inertiadata->Area;
    sumx = inertiadata->SumX;
    return (sumx / area);
  } /* MeanXAttribute */

  double MeanYAttribute(void *inertiaattr)
  {
    InertiaData *inertiadata = (InertiaData *) inertiaattr;
    double area, sumy;

    area = inertiadata->Area;
    sumy = inertiadata->SumY;
    return (sumy / area);
  } /* MeanYAttribute */

  /****** Image create/read/write functions ******************************/

  ImageGray *ImageGrayCreate(ulong width, ulong height)
  {
    ImageGray *img;

    img = (ImageGray *) malloc(sizeof(ImageGray));
    if (img == NULL)
      return (NULL);
    img->Width  = width;
    img->Height = height;
    img->Pixmap = (ubyte *) malloc(width * height);
    if (img->Pixmap == NULL) {
      free(img);
      return (NULL);
    }
    return (img);
  } /* ImageGrayCreate */

  void ImageGrayDelete(ImageGray *img)
  {
    free(img->Pixmap);
    free(img);
  } /* ImageGrayDelete */

  void ImageGrayInit(ImageGray *img, ubyte h)
  {
    memset(img->Pixmap, h, (img->Width) * (img->Height));
  } /* ImageGrayInit */

  /****** Max-tree routines ******************************/

  HQueue *HQueueCreate(ulong imgsize, ulong *numpixelsperlevel)
  {
    HQueue *hq;
    int i;

    hq = (HQueue *) calloc(NUMLEVELS, sizeof(HQueue));
    if (hq == NULL)
      return (NULL);
    hq->Pixels = (ulong *) calloc(imgsize, sizeof(ulong));
    if (hq->Pixels == NULL) {
      free(hq);
      return (NULL);
    }
    hq->Head = hq->Tail = 0;
    for (i = 1; i < NUMLEVELS; i++) {
      hq[i].Pixels = hq[i - 1].Pixels + numpixelsperlevel[i - 1];
      hq[i].Head = hq[i].Tail = 0;
    }
    return (hq);
  } /* HQueueCreate */

  void HQueueDelete(HQueue *hq)
  {
    free(hq->Pixels);
    free(hq);
  } /* HQueueDelete */

#define HQueueFirst(hq, h) (hq[h].Pixels[hq[h].Head++])
#define HQueueAdd(hq, h, p) hq[h].Pixels[hq[h].Tail++] = p
#define HQueueNotEmpty(hq, h) (hq[h].Head != hq[h].Tail)

  int GetNeighbors(ubyte *shape, ulong imgwidth, ulong imgsize, ulong p,
                   ulong *neighbors)
  {
    ulong x;
    int n = 0;

    x = p % imgwidth;
    if ((x < (imgwidth - 1)) && (shape[p + 1]))
      neighbors[n++] = p + 1;
    if ((p >= imgwidth) && (shape[p - imgwidth]))
      neighbors[n++] = p - imgwidth;
    if ((x > 0) && (shape[p - 1]))
      neighbors[n++] = p - 1;
    p += imgwidth;
    if ((p < imgsize) && (shape[p]))
      neighbors[n++] = p;
    return (n);
  } /* GetNeighbors */

  int MaxTreeFlood(MaxTree *mt, HQueue *hq, ulong *numpixelsperlevel,
                   bool *nodeatlevel, ImageGray *img, ubyte *shape, int h,
                   ulong *thisarea, void **thisattr)
  /* Returns value >=NUMLEVELS if error */
  {
    ulong neighbors[CONNECTIVITY];
    ubyte *pixmap;
    void *attr = NULL, *childattr;
    ulong imgwidth, imgsize, p, q, idx, x, y;
    ulong area = *thisarea, childarea;
    MaxNode *node;
    int numneighbors, i;
    int m;

    imgwidth = img->Width;
    imgsize  = imgwidth * (img->Height);
    pixmap   = img->Pixmap;
    while (HQueueNotEmpty(hq, h)) {
      area++;
      p = HQueueFirst(hq, h);
      x = p % imgwidth;
      y = p / imgwidth;
      if (attr)
        mt->AddToAuxData(attr, x, y);
      else {
        attr = mt->NewAuxData(x, y);
        if (attr == NULL)
          return (NUMLEVELS);
        if (*thisattr)
          mt->MergeAuxData(attr, *thisattr);
      }
      mt->Status[p] = mt->NumNodesAtLevel[h];
      numneighbors  = GetNeighbors(shape, imgwidth, imgsize, p, neighbors);
      for (i = 0; i < numneighbors; i++) {
        q = neighbors[i];
        if (mt->Status[q] == ST_NotAnalyzed) {
          HQueueAdd(hq, pixmap[q], q);
          mt->Status[q]          = ST_InTheQueue;
          nodeatlevel[pixmap[q]] = true;
          if (pixmap[q] > pixmap[p]) {
            m         = pixmap[q];
            childarea = 0;
            childattr = NULL;
            do {
              m = MaxTreeFlood(mt, hq, numpixelsperlevel, nodeatlevel, img,
                               shape, m, &childarea, &childattr);
              if (m >= NUMLEVELS) {
                mt->DeleteAuxData(attr);
                return (m);
              }
            } while (m != h);
            area += childarea;
            mt->MergeAuxData(attr, childattr);
          }
        }
      }
    }
    mt->NumNodesAtLevel[h] = mt->NumNodesAtLevel[h] + 1;
    m                      = h - 1;
    while ((m >= 0) && (nodeatlevel[m] == false))
      m--;
    if (m >= 0) {
      node =
          mt->Nodes + (mt->NumPixelsBelowLevel[h] + mt->NumNodesAtLevel[h] - 1);
      node->Parent = mt->NumPixelsBelowLevel[m] + mt->NumNodesAtLevel[m];
    } else {
      idx          = mt->NumPixelsBelowLevel[h];
      node         = mt->Nodes + idx;
      node->Parent = idx;
    }
    node->Area      = area;
    node->Attribute = attr;
    node->Level     = h;
    nodeatlevel[h]  = false;
    *thisarea       = area;
    *thisattr       = attr;
    return (m);
  } /* MaxTreeFlood */

  MaxTree *MaxTreeCreate(ImageGray *img, ImageGray *templateImg,
                         void *(*newauxdata)(ulong, ulong),
                         void (*addtoauxdata)(void *, ulong, ulong),
                         void (*mergeauxdata)(void *, void *),
                         void (*deleteauxdata)(void *))
  {
    ulong numpixelsperlevel[NUMLEVELS];
    bool nodeatlevel[NUMLEVELS];
    HQueue *hq;
    MaxTree *mt;
    ubyte *pixmap = img->Pixmap;
    void *attr    = NULL;
    ulong imgsize, p, m = 0, area = 0;
    int l;

    /* Allocate structures */
    mt = (MaxTree *) malloc(sizeof(MaxTree));
    if (mt == NULL)
      return (NULL);
    imgsize    = (img->Width) * (img->Height);
    mt->Status = (long *) calloc((size_t) imgsize, sizeof(long));
    if (mt->Status == NULL) {
      free(mt);
      return (NULL);
    }
    mt->NumPixelsBelowLevel = (ulong *) calloc(NUMLEVELS, sizeof(ulong));
    if (mt->NumPixelsBelowLevel == NULL) {
      free(mt->Status);
      free(mt);
      return (NULL);
    }
    mt->NumNodesAtLevel = (ulong *) calloc(NUMLEVELS, sizeof(ulong));
    if (mt->NumNodesAtLevel == NULL) {
      free(mt->NumPixelsBelowLevel);
      free(mt->Status);
      free(mt);
      return (NULL);
    }
    mt->Nodes = (MaxNode *) calloc((size_t) imgsize, sizeof(MaxNode));
    if (mt->Nodes == NULL) {
      free(mt->NumNodesAtLevel);
      free(mt->NumPixelsBelowLevel);
      free(mt->Status);
      free(mt);
      return (NULL);
    }

    /* Initialize structures */
    for (p = 0; p < imgsize; p++)
      mt->Status[p] = ST_NotAnalyzed;
    bzero(nodeatlevel, NUMLEVELS * sizeof(bool));
    bzero(numpixelsperlevel, NUMLEVELS * sizeof(ulong));
    /* Following bzero is redundant, array is initialized by calloc */
    /* bzero(mt->NumNodesAtLevel, NUMLEVELS*sizeof(ulong)); */
    for (p = 0; p < imgsize; p++)
      numpixelsperlevel[pixmap[p]]++;
    mt->NumPixelsBelowLevel[0] = 0;
    for (l = 1; l < NUMLEVELS; l++) {
      mt->NumPixelsBelowLevel[l] =
          mt->NumPixelsBelowLevel[l - 1] + numpixelsperlevel[l - 1];
    }
    hq = HQueueCreate(imgsize, numpixelsperlevel);
    if (hq == NULL) {
      free(mt->Nodes);
      free(mt->NumNodesAtLevel);
      free(mt->NumPixelsBelowLevel);
      free(mt->Status);
      free(mt);
      return (NULL);
    }

    /* Find pixel m which has the lowest intensity l in the image */
    for (p = 0; p < imgsize; p++) {
      if (pixmap[p] < pixmap[m])
        m = p;
    }
    l = pixmap[m];

    /* Add pixel m to the queue */
    nodeatlevel[l] = true;
    HQueueAdd(hq, l, m);
    mt->Status[m] = ST_InTheQueue;

    /* Build the Max-tree using a flood-fill algorithm */
    mt->NewAuxData    = newauxdata;
    mt->AddToAuxData  = addtoauxdata;
    mt->MergeAuxData  = mergeauxdata;
    mt->DeleteAuxData = deleteauxdata;
    l = MaxTreeFlood(mt, hq, numpixelsperlevel, nodeatlevel, img,
                     templateImg->Pixmap, l, &area, &attr);

    if (l >= NUMLEVELS)
      MaxTreeDelete(mt);
    HQueueDelete(hq);
    return (mt);
  } /* MaxTreeCreate */

  void MaxTreeDelete(MaxTree *mt)
  {
    void *attr;
    ulong i;
    int h;

    for (h = 0; h < NUMLEVELS; h++) {
      for (i = 0; i < mt->NumNodesAtLevel[h]; i++) {
        attr = mt->Nodes[mt->NumPixelsBelowLevel[h] + i].Attribute;
        if (attr)
          mt->DeleteAuxData(attr);
      }
    }
    free(mt->Nodes);
    free(mt->NumNodesAtLevel);
    free(mt->NumPixelsBelowLevel);
    free(mt->Status);
    free(mt);
  } /* MaxTreeDelete */

  void MaxTreeFilterMin(MaxTree *mt, ImageGray *img, ImageGray *templateImg,
                        ImageGray *out, double (*attribute)(void *),
                        double lambda)
  {
    MaxNode *node, *parnode;
    ubyte *shape = templateImg->Pixmap;
    ulong i, idx, parent;
    int l;

    for (l = 0; l < NUMLEVELS; l++) {
      for (i = 0; i < mt->NumNodesAtLevel[l]; i++) {
        idx     = mt->NumPixelsBelowLevel[l] + i;
        node    = &(mt->Nodes[idx]);
        parent  = node->Parent;
        parnode = &(mt->Nodes[parent]);
        if ((idx != parent) && (((*attribute)(node->Attribute) < lambda) ||
                                (parnode->Level != parnode->NewLevel))) {
          node->NewLevel = parnode->NewLevel;
        } else
          node->NewLevel = node->Level;
      }
    }
    for (i = 0; i < (img->Width) * (img->Height); i++) {
      if (shape[i]) {
        idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
        out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
    }
  } /* MaxTreeFilterMin */

  void MaxTreeFilterDirect(MaxTree *mt, ImageGray *img, ImageGray *templateImg,
                           ImageGray *out, double (*attribute)(void *),
                           double lambda)
  {
    MaxNode *node;
    ubyte *shape = templateImg->Pixmap;
    ulong i, idx, parent;
    int l;

    for (l = 0; l < NUMLEVELS; l++) {
      for (i = 0; i < mt->NumNodesAtLevel[l]; i++) {
        idx    = mt->NumPixelsBelowLevel[l] + i;
        node   = &(mt->Nodes[idx]);
        parent = node->Parent;
        if ((idx != parent) && ((*attribute)(node->Attribute) < lambda)) {
          node->NewLevel = mt->Nodes[parent].NewLevel;
        } else
          node->NewLevel = node->Level;
      }
    }
    for (i = 0; i < (img->Width) * (img->Height); i++) {
      if (shape[i]) {
        idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
        out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
    }
  } /* MaxTreeFilterDirect */

  void MaxTreeFilterMax(MaxTree *mt, ImageGray *img, ImageGray *templateImg,
                        ImageGray *out, double (*attribute)(void *),
                        double lambda)
  {
    MaxNode *node;
    ubyte *shape = templateImg->Pixmap;
    ulong i, idx, parent;
    int l;

    for (l = 0; l < NUMLEVELS; l++) {
      for (i = 0; i < mt->NumNodesAtLevel[l]; i++) {
        idx    = mt->NumPixelsBelowLevel[l] + i;
        node   = &(mt->Nodes[idx]);
        parent = node->Parent;
        if ((idx != parent) && ((*attribute)(node->Attribute) < lambda)) {
          node->NewLevel = mt->Nodes[parent].NewLevel;
        } else
          node->NewLevel = node->Level;
      }
    }
    for (l = NUMLEVELS - 1; l > 0; l--) {
      for (i = 0; i < mt->NumNodesAtLevel[l]; i++) {
        idx    = mt->NumPixelsBelowLevel[l] + i;
        node   = &(mt->Nodes[idx]);
        parent = node->Parent;
        if ((idx != parent) && (node->NewLevel == node->Level)) {
          mt->Nodes[parent].NewLevel = mt->Nodes[parent].Level;
        }
      }
    }
    for (i = 0; i < (img->Width) * (img->Height); i++) {
      if (shape[i]) {
        idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
        out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
    }
  } /* MaxTreeFilterMax */

  void MaxTreeFilterSubtractive(MaxTree *mt, ImageGray *img,
                                ImageGray *templateImg, ImageGray *out,
                                double (*attribute)(void *), double lambda)
  {
    MaxNode *node, *parnode;
    ubyte *shape = templateImg->Pixmap;
    ulong i, idx, parent;
    int l;

    for (l = 0; l < NUMLEVELS; l++) {
      for (i = 0; i < mt->NumNodesAtLevel[l]; i++) {
        idx     = mt->NumPixelsBelowLevel[l] + i;
        node    = &(mt->Nodes[idx]);
        parent  = node->Parent;
        parnode = &(mt->Nodes[parent]);
        if ((idx != parent) && ((*attribute)(node->Attribute) < lambda)) {
          node->NewLevel = parnode->NewLevel;
        } else
          node->NewLevel = ((int) (node->Level)) + ((int) (parnode->NewLevel)) -
                           ((int) (parnode->Level));
      }
    }
    for (i = 0; i < (img->Width) * (img->Height); i++) {
      if (shape[i]) {
        idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
        out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
    }
  } /* MaxTreeFilterSubtractive */

  typedef struct AttribStruct AttribStruct;

  struct AttribStruct {
    char const *Name;
    void *(*NewData)(ulong, ulong);
    void (*DeleteData)(void *);
    void (*AddToData)(void *, ulong, ulong);
    void (*MergeData)(void *, void *);
    double (*Attribute)(void *);
  };

#define NUMATTR 7

  AttribStruct Attribs[NUMATTR] = {
      {"Area", NewAreaData, DeleteAreaData, AddToAreaData, MergeAreaData,
       AreaAttribute},
      {"Area of min. enclosing rectangle", NewEnclRectData, DeleteEnclRectData,
       AddToEnclRectData, MergeEnclRectData, EnclRectAreaAttribute},
      {"Square of diagonal of min. enclosing rectangle", NewEnclRectData,
       DeleteEnclRectData, AddToEnclRectData, MergeEnclRectData,
       EnclRectDiagAttribute},
      {"Moment of Inertia", NewInertiaData, DeleteInertiaData, AddToInertiaData,
       MergeInertiaData, InertiaAttribute},
      {"(Moment of Inertia) / (area)^2", NewInertiaData, DeleteInertiaData,
       AddToInertiaData, MergeInertiaData, InertiaDivA2Attribute},
      {"Mean X position", NewInertiaData, DeleteInertiaData, AddToInertiaData,
       MergeInertiaData, MeanXAttribute},
      {"Mean Y position", NewInertiaData, DeleteInertiaData, AddToInertiaData,
       MergeInertiaData, MeanYAttribute}};

  typedef struct DecisionStruct DecisionStruct;

  struct DecisionStruct {
    char const *Name;
    void (*Filter)(MaxTree *, ImageGray *, ImageGray *, ImageGray *,
                   double (*attribute)(void *), double);
  };

#define NUMDECISIONS 4

  DecisionStruct Decisions[NUMDECISIONS] = {
      {"Min", MaxTreeFilterMin},
      {"Direct", MaxTreeFilterDirect},
      {"Max", MaxTreeFilterMax},
      {"Subtractive", MaxTreeFilterSubtractive},
  };

  template <class T1, class T2>
  RES_T ImAreaOpening_MaxTree(const Image<T1> &imIn, int size, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    MaxTree *mt;
    int attrib = 0, decision = 2;
    int W                                  = imIn.getWidth();
    int H                                  = imIn.getHeight();
    typename Image<T1>::lineType bufferIn  = imIn.getPixels();
    typename Image<T2>::lineType bufferOut = imOut.getPixels();

    ImageGray img, *templateImg, out;
    img.Height = H;
    img.Width  = W;
    img.Pixmap = bufferIn;

    out.Height = H;
    out.Width  = W;
    out.Pixmap = bufferOut;

    templateImg = ImageGrayCreate(img.Width, img.Height);
    if (templateImg)
      ImageGrayInit(templateImg, NUMLEVELS - 1);

    mt = MaxTreeCreate(&img, templateImg, Attribs[attrib].NewData,
                       Attribs[attrib].AddToData, Attribs[attrib].MergeData,
                       Attribs[attrib].DeleteData);
    Decisions[decision].Filter(mt, &img, templateImg, &out,
                               Attribs[attrib].Attribute, size);
    MaxTreeDelete(mt);
    ImageGrayDelete(templateImg);

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImInertiaThinning_MaxTree(const Image<T1> &imIn, double size,
                                  Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    MaxTree *mt;
    int attrib = 4, decision = 2;
    int W                                  = imIn.getWidth();
    int H                                  = imIn.getHeight();
    typename Image<T1>::lineType bufferIn  = imIn.getPixels();
    typename Image<T2>::lineType bufferOut = imOut.getPixels();

    ImageGray img, *templateImg, out;
    img.Height = H;
    img.Width  = W;
    img.Pixmap = bufferIn;

    out.Height = H;
    out.Width  = W;
    out.Pixmap = bufferOut;

    templateImg = ImageGrayCreate(img.Width, img.Height);
    if (templateImg)
      ImageGrayInit(templateImg, NUMLEVELS - 1);

    mt = MaxTreeCreate(&img, templateImg, Attribs[attrib].NewData,
                       Attribs[attrib].AddToData, Attribs[attrib].MergeData,
                       Attribs[attrib].DeleteData);
    Decisions[decision].Filter(mt, &img, templateImg, &out,
                               Attribs[attrib].Attribute, size);
    MaxTreeDelete(mt);
    ImageGrayDelete(templateImg);

    return RES_OK;
  }

} // namespace smil

#endif
