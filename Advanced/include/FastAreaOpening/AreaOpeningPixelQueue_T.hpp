#ifndef __FAST_AREA_OPENIN_PIXEL_QUEUE_T_HPP__
#define __FAST_AREA_OPENIN_PIXEL_QUEUE_T_HPP__

#include "Core/include/DCore.h"

/// Interface to SMIL  Vincent Morard
/// Date : 7 march 2011

/// This program computes the grey scale area closing using
/// Vincent's priority-queue algorithm.
/// The input is a grey scale image (im), and its output is
/// a grey-scale image (out).
/// The time complexity of this algorithm is quadratic in the
/// number of pixels, and AlogA in the area A of the closing.
///
/// (c) Arnold Meijster and Michael Wilkinson
///

namespace smil
{
#define MAXGREYVAL 256
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

  typedef int greyval;
  typedef greyval **GImage;

  int *Queue;
  int head, tail;

  /********** Heap Management **************************************/
  typedef struct {
    long *heap;
    long N, HeapMax;
  } PixelHeap;

  void InitPixelHeap(PixelHeap *p_heap, int lambda)
  {
    p_heap->heap    = (long *) malloc((lambda + 1) * sizeof(long));
    p_heap->N       = 0;
    p_heap->HeapMax = lambda;
  }

  void ExitPixelHeap(PixelHeap *p_heap)
  {
    free(p_heap->heap);
    p_heap->N = 0;
  }

  void InsertInPixelHeap(PixelHeap *p_heap, greyval *im, long pixel)
  {
    long child, parent, gval = im[pixel];
    p_heap->N++;
    child               = p_heap->N;
    p_heap->heap[child] = pixel;
    while ((parent = (child >> 1)) && (im[p_heap->heap[parent]] > gval)) {
      p_heap->heap[child] = p_heap->heap[parent];
      child               = parent;
    }
    p_heap->heap[child] = pixel;
  }

  int RemoveFromPixelHeap(PixelHeap *p_heap, greyval *im)
  {
    long pixel = p_heap->heap[p_heap->N--], temp = p_heap->heap[1],
         gval = im[pixel], child, parent, maxparent = p_heap->N >> 1;

    p_heap->heap[parent = 1] = pixel;
    while (parent <= maxparent) {
      child = parent << 1;
      if (child < p_heap->N)
        if (im[p_heap->heap[child]] > im[p_heap->heap[child + 1]])
          child++;
      if (gval <= im[p_heap->heap[child]])
        break;
      p_heap->heap[parent] = p_heap->heap[child];
      parent               = child;
    }
    p_heap->heap[parent] = pixel;
    return temp;
  }

  int HeapEmpty(PixelHeap *p_heap)
  {
    return (p_heap->N == 0);
  }

  /********** Area opening according Luc Vincent *******************/

  long FillExtremum(int *Qpos, SMIL_UNUSED int lambda, greyval *im,
                    greyval *label, int width, int height, PixelHeap *p_heap,
                    int *Queue, int Qmax, long *fill_list)
  {
    long area = 0;
    int i = *Qpos, pixel = Queue[i], curneigh, px, MARK = label[pixel],
        uplimit = (height - 1) * width;
    p_heap->N   = 0;
    do {
      label[pixel] = -MARK;

      fill_list[area++] = pixel;
      px                = pixel % width;
      if (pixel >= width) {
        curneigh = pixel - width;
        if ((label[curneigh] != MARK) && (label[curneigh] != -MARK)) {
          InsertInPixelHeap(p_heap, im, curneigh);
          label[curneigh] = -MARK;
        }
      }
      if (pixel < uplimit) {
        curneigh = pixel + width;
        if ((label[curneigh] != MARK) && (label[curneigh] != -MARK)) {
          InsertInPixelHeap(p_heap, im, curneigh);
          label[curneigh] = -MARK;
        }
      }
      if (px > 0) {
        curneigh = pixel - 1;
        if ((label[curneigh] != MARK) && (label[curneigh] != -MARK)) {
          InsertInPixelHeap(p_heap, im, curneigh);
          label[curneigh] = -MARK;
        }
      }
      if (px < width - 1) {
        curneigh = pixel + 1;
        if ((label[curneigh] != MARK) && (label[curneigh] != -MARK)) {
          InsertInPixelHeap(p_heap, im, curneigh);
          label[curneigh] = -MARK;
        }
      }
      i++;
      if (i == Qmax)
        break;
      pixel = Queue[i];
    } while (label[pixel] == MARK);
    *Qpos = i;
    return area;
  }

  void SmallRegionalMinima(greyval *im, greyval *label, int *Queue, int *Qmax,
                           int width, int height, int lambda)
  {
#define REGMAX -2
#define NARM 0
    register int i, j, curneigh, pixel, px, gval, curlab = 1;

    /* Determine regional maxima */
    for (i = 0; i < width * height; i++)
      label[i] = REGMAX;
    *Qmax = 0;
    for (i = 0; i < width * height; i++) {
      if (label[i] == REGMAX) /* not processed yet */
      {
        gval        = im[i];
        label[i]    = curlab;
        head        = *Qmax;
        Queue[head] = i;
        tail        = head + 1;
        while (head != tail) {
          pixel = Queue[head++];
          px    = pixel % width;
          if (pixel >= width) {
            curneigh = pixel - width;
            if (im[curneigh] < gval) {
              label[pixel] = NARM;
              break;
            } else if ((im[curneigh] == gval) && (label[curneigh] == REGMAX)) {
              label[curneigh] = curlab;
              Queue[tail++]   = curneigh;
            }
          }
          if (pixel < (height - 1) * width - 1) {
            curneigh = pixel + width;
            if (im[curneigh] < gval) {
              label[pixel] = NARM;
              break;
            } else if ((im[curneigh] == gval) && (label[curneigh] == REGMAX)) {
              label[curneigh] = curlab;
              Queue[tail++]   = curneigh;
            }
          }
          if (px > 0) {
            curneigh = pixel - 1;
            if (im[curneigh] < gval) {
              label[pixel] = NARM;
              break;
            } else if ((im[curneigh] == gval) && (label[curneigh] == REGMAX)) {
              label[curneigh] = curlab;
              Queue[tail++]   = curneigh;
            }
          }
          if (px < width - 1) {
            curneigh = pixel + 1;
            if (im[curneigh] < gval) {
              label[pixel] = NARM;
              break;
            } else if ((im[curneigh] == gval) && (label[curneigh] == REGMAX)) {
              label[curneigh] = curlab;
              Queue[tail++]   = curneigh;
            }
          }
        }
        if (label[pixel] != NARM) {
          if ((head - *Qmax) < lambda) {
            curlab++;
            *Qmax = head;
          } else
            for (j = *Qmax; j < head; j++)
              label[Queue[j]] = NARM;
        } else {
          head--;
          for (j = *Qmax; j < head; j++)
            label[Queue[j]] = NARM;
          while (head != tail) {
            pixel        = Queue[head++];
            px           = pixel % width;
            label[pixel] = NARM;
            curneigh     = pixel - width;
            if ((pixel >= width) && (im[curneigh] == gval) &&
                (label[curneigh] != NARM)) {
              label[curneigh] = NARM;
              Queue[tail++]   = curneigh;
            }
            curneigh = pixel + width;
            if ((pixel < (height - 1) * width - 1) && (im[curneigh] == gval) &&
                (label[curneigh] != NARM)) {
              label[curneigh] = NARM;
              Queue[tail++]   = curneigh;
            }
            curneigh = pixel - 1;
            if ((px > 0) && (im[curneigh] == gval) &&
                (label[curneigh] != NARM)) {
              label[curneigh] = NARM;
              Queue[tail++]   = curneigh;
            }
            curneigh = pixel + 1;
            if ((px < width - 1) && (im[curneigh] == gval) &&
                (label[curneigh] != NARM)) {
              label[curneigh] = NARM;
              Queue[tail++]   = curneigh;
            }
          }
        }
      }
    }
    return;
  }

  void VincentAreaClosing(int lambda, greyval *im, greyval *label, int width,
                          int height, PixelHeap *p_heap, long *fill_list)
  {
    int Qpos, j, pixel, px,
        // JOE imagesize=width*height,
        uplimit = (height - 1) * width, maxarea, MARK, curneigh, Qmax;
    greyval gval;
    SmallRegionalMinima(im, label, Queue, &Qmax, width, height, lambda);
    Qpos = 0;
    MARK = label[Queue[Qpos]];
    while (Qpos < Qmax) {
      maxarea = FillExtremum(&Qpos, lambda, im, label, width, height, p_heap,
                             Queue, Qmax, fill_list);
      MARK    = -MARK;
      while (maxarea < lambda) {
        gval                 = im[pixel = RemoveFromPixelHeap(p_heap, im)];
        px                   = pixel % width;
        fill_list[maxarea++] = pixel;
        curneigh             = pixel - width;
        if ((pixel >= width) && (label[curneigh] != MARK)) {
          if (gval <= im[curneigh]) {
            label[curneigh] = MARK;
            InsertInPixelHeap(p_heap, im, curneigh);
          } else
            break;
        }
        curneigh = pixel + width;
        if ((pixel < uplimit) && (label[curneigh] != MARK)) {
          if (gval <= im[curneigh]) {
            label[curneigh] = MARK;
            InsertInPixelHeap(p_heap, im, curneigh);
          } else
            break;
        }
        curneigh = pixel - 1;
        if ((px > 0) && (label[curneigh] != MARK)) {
          if (gval <= im[curneigh]) {
            label[curneigh] = MARK;
            InsertInPixelHeap(p_heap, im, curneigh);
          } else
            break;
        }
        curneigh = pixel + 1;
        if ((px < width - 1) && (label[curneigh] != MARK)) {
          if (gval <= im[curneigh]) {
            label[curneigh] = MARK;
            InsertInPixelHeap(p_heap, im, curneigh);
          } else
            break;
        }
      }
      for (j = 0; j < maxarea; j++)
        im[fill_list[j]] = gval;
      if (Qpos < Qmax)
        MARK = label[Queue[Qpos]];
    }
  }

  GImage CreateImage(int width, int height)
  {
    GImage img;
    greyval *buf;
    int i;
    img    = (GImage) malloc(height * sizeof(greyval *));
    buf    = (greyval *) malloc(width * height * sizeof(greyval));
    img[0] = buf;
    for (i = 1; i < height; i++)
      img[i] = img[i - 1] + width;
    return img;
  }

  void ReleaseImage(GImage img)
  {
    free(img[0]);
    free(img);
    img = 0;
  }

  // This algorithm needs an INT32 input and an INT32 output. It can be an
  // inplace transform However, we are working with UINT8 and UINT8 input and
  // output buffer. Hence, we have already converted the input image in INT32.
  // (not const) First, the result of the area op will be store in imIn. Then we
  // convert it into imOut

  // Here T1 = INT32
  template <class T1, class T2>
  RES_T ImAreaClosing_PixelQueue(const Image<T1> &imIn, int size,
                                 Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    Image<INT32> im_32(imIn);
    RES_T res = copy(imIn, im_32);
    if (res != RES_OK)
      return res;

    int W, H;
    W = im_32.getWidth();
    H = im_32.getHeight();

    typename Image<INT32>::lineType bufferIn = im_32.getPixels();

    // Change the way to read the image for the output image
    GImage outputImage;
    outputImage    = (GImage) malloc(H * sizeof(greyval *));
    outputImage[0] = bufferIn;
    for (int i = 1; i < H; i++)
      outputImage[i] = outputImage[i - 1] + W;

    // Create space for the label image
    GImage labelImage = CreateImage(W, H);

    // Allocate the queue
    Queue = (int *) malloc(W * H * sizeof(int));

    PixelHeap p_heap;
    InitPixelHeap(&p_heap, 2 * (size + 1));
    long *fill_list = (long *) malloc(size * sizeof(long));

    VincentAreaClosing(size, outputImage[0], labelImage[0], W, H, &p_heap,
                       fill_list);

    res = copy(im_32, imOut);
    if (res != RES_OK)
      return res;

    // Free the extra memory
    ExitPixelHeap(&p_heap);
    free(outputImage);
    ReleaseImage(labelImage);
    free(Queue);
    free(fill_list);

    return RES_OK;
  }
} // namespace smil

#endif
