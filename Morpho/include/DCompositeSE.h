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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _COMPOSITE_SE_HPP
#define _COMPOSITE_SE_HPP

#include "DStructuringElement.h"
#include <list>

namespace smil
{
  /**
   * @defgroup CompSE Composite Structuring Elements
   * @ingroup StrElt
   *
   * @brief Couples of disjoint Structuring Elements
   *
   * @details These are basically couples of disjoint Structuring Elements for
   * use in @ref ops-hit-or-miss "Hit-or-Miss Morphological Operations" - one
   * for the image foreground and another for the background.
   *
   * @see
   * @ref ops-hit-or-miss "Hit-or-Miss Morphological Operations"
   * @{
   */

  class CompStrEltList;

  /**
   * Composite structuring element
   */
  class CompStrElt : public BaseObject
  {
  public:
    StrElt fgSE;
    StrElt bgSE;

    CompStrElt() : BaseObject("CompStrElt"){};
    ~CompStrElt(){};

    /** CompStrElt - Constructor
     *
     * Build a CompStrElt based on the one passed as parameter
     *
     * @param[in] rhs : Composite Structuring Element
     */
    CompStrElt(const CompStrElt &rhs);

    /** CompStrElt - Constructor
     *
     * Build a CompStrElt based on the two SEs passed as parameter
     *
     * @param[in] fg, bg : Composite Structuring Element
     **/
    CompStrElt(const StrElt &fg, const StrElt &bg);

    /** operator~() Switch foreground/background SE
     *
     * @return a CompStrElm
     */
    CompStrElt operator~();

    /** Counterclockwise rotate SE points
     *
     */
    CompStrElt &rotate(int steps = 1);

    /** operator |
     *
     */
    CompStrEltList operator|(const CompStrElt &rhs);

    /** operator ()
     *
     * Return a CompStrEltList containing the nrot rotations of the CompStrElt
     *
     */
    CompStrEltList operator()(UINT nrot);

    /** printSelf() - print CompStrEltList content
     *
     * @param[in] os : output stream (default : @b std::cout)
     * @param[in] indent : prefix to each printed line (string)
     */
    virtual void printSelf(ostream &os = std::cout, string indent = "") const;
  };

  /**
   * CompStrEltList - A list (vector) of Composite structuring elements
   *
   * @see CompStrElt
   */
  class CompStrEltList : public BaseObject
  {
  public:
    std::vector<CompStrElt> compSeList;
    CompStrEltList() : BaseObject("CompStrEltList")
    {
    }
    CompStrEltList(const CompStrEltList &rhs);
    CompStrEltList(const CompStrElt &compSe);
    CompStrEltList(const CompStrElt &compSe, UINT nrot);
    CompStrEltList operator~();

    /** operator |() - Append composite SEs to current list
     *
     * @b Example:
     * @code{.py}
     * import smilPython as sp
     *
     * csel = sp.CompStrEltList(sp.HMT_sL1())
     * csel |= sp.HMT_sL2()
     *
     * print(csel)
     * @endcode
     *
     */
    CompStrEltList operator|(const CompStrEltList &rhs);

    /** Get the nth CompStrElt
     *
     */
    CompStrElt &operator[](const UINT n)
    {
      return compSeList[n];
    }

    /**
     *
     */
    void add(const CompStrElt &cse);
    /**
     *
     */
    void add(const StrElt &fgse, const StrElt &bgse);

    /** Add as the nrot rotations of the StrElt pair
     *
     * The rotation step is 6/nrot counterclockwise for Hex
     * and 8/nrot for Squ
     */
    void add(const StrElt &fgse, const StrElt &bgse, UINT nrot);
  
    /**
     *
     */
    void add(const CompStrElt &cse, UINT nrot);

    /**
     *
     */
    CompStrEltList &rotate(int steps = 1);
    
    /**
    *
    *
    */
    void setName(string name)
    {
      this->name = name;
    }

    virtual void printSelf(ostream &os = std::cout, string indent = "") const;
  };

  //! Square L1 ([8,1,2], [4,5,6])
  class HMT_sL1 : public CompStrEltList
  {
  public:
    HMT_sL1(UINT nrot = 1)
    {
      this->add(StrElt(false, 3, 8, 1, 2), StrElt(false, 3, 4, 5, 6), nrot);
      this->name = "HMT_sL1";
    }
  };

  //! Square L2 ([1,3], [5,6,7])
  class HMT_sL2 : public CompStrEltList
  {
  public:
    HMT_sL2(UINT nrot = 1)
    {
      this->add(StrElt(false, 2, 1, 3), StrElt(false, 3, 5, 6, 7), nrot);
      this->name = "HMT_sL2";
    }
  };

  //! Hexagonal L ([1,2], [4,5])
  class HMT_hL : public CompStrEltList
  {
  public:
    HMT_hL(UINT nrot = 1)
    {
      this->add(StrElt(true, 2, 1, 2), StrElt(true, 2, 4, 5), nrot);
      this->name = "HMT_hL";
    }
  };

  //! Square M ([1,8], [3,4,5,6])
  class HMT_sM : public CompStrEltList
  {
  public:
    HMT_sM(UINT nrot = 1)
    {
      add(StrElt(false, 2, 1, 8), StrElt(false, 4, 3, 4, 5, 6), nrot);
      this->name = "HMT_sM";
    }
  };

  //! Hexagonal M ([1], [3,4,5])
  class HMT_hM : public CompStrEltList
  {
  public:
    HMT_hM(UINT nrot = 1)
    {
      add(StrElt(true, 1, 1), StrElt(true, 3, 3, 4, 5), nrot);
      this->name = "HMT_hM";
    }
  };

  //! Square D ([3,4,5,6], [1,8])
  class HMT_sD : public CompStrEltList
  {
  public:
    HMT_sD(UINT nrot = 1)
    {
      add(StrElt(false, 4, 3, 4, 5, 6), StrElt(false, 2, 1, 8), nrot);
      this->name = "HMT_sD";
    }
  };

  //! Hexagonal D ([3,4,5], [1])
  class HMT_hD : public CompStrEltList
  {
  public:
    HMT_hD(UINT nrot = 1)
    {
      add(StrElt(true, 3, 3, 4, 5), StrElt(true, 1, 1), nrot);
      this->name = "HMT_hD";
    }
  };

  //! Square E ([3,4,5,6,7], [0])
  class HMT_sE : public CompStrEltList
  {
  public:
    HMT_sE(UINT nrot = 1)
    {
      add(StrElt(false, 5, 3, 4, 5, 6, 7), StrElt(false, 1, 0), nrot);
      this->name = "HMT_sE";
    }
  };

  //! Hexagonal E ([3,4,5,6], [0])
  class HMT_hE : public CompStrEltList
  {
  public:
    HMT_hE(UINT nrot = 1)
    {
      add(StrElt(true, 4, 3, 4, 5, 6), StrElt(true, 1, 0), nrot);
      this->name = "HMT_hE";
    }
  };

  // # Some other specific structuring elements used for multiple points
  // extraction
  //! Square S1 ([3,7], [0,1,5])
  class HMT_sS1 : public CompStrEltList
  {
  public:
    HMT_sS1(UINT nrot = 1)
    {
      add(StrElt(false, 2, 3, 7), StrElt(false, 3, 0, 1, 5), nrot);
      this->name = "HMT_sS1";
    }
  };

  //! Hexagonal S1 ([2,3,5,6], [0,1,4])
  class HMT_hS1 : public CompStrEltList
  {
  public:
    HMT_hS1(UINT nrot = 1)
    {
      add(StrElt(true, 4, 2, 3, 5, 6), StrElt(true, 3, 0, 1, 4), nrot);
      this->name = "HMT_hS1";
    }
  };

  //! Square S2 ([2,5,6,7], [0,1,3])
  class HMT_sS2 : public CompStrEltList
  {
  public:
    HMT_sS2(UINT nrot = 1)
    {
      add(StrElt(false, 4, 2, 5, 6, 7), StrElt(false, 3, 0, 1, 3), nrot);
      this->name = "HMT_sS2";
    }
  };

  //! Hexagonal S2 ([2,4,5,6], [0,1,3])
  class HMT_hS2 : public CompStrEltList
  {
  public:
    HMT_hS2(UINT nrot = 1)
    {
      add(StrElt(true, 4, 2, 4, 5, 6), StrElt(false, 3, 0, 1, 3), nrot);
      this->name = "HMT_hS2";
    }
  };

  // # Special pattern used to perform SKIZ
  //! Square S3 ([3,4,5,6,7], [1])
  class HMT_sS3 : public CompStrEltList
  {
  public:
    HMT_sS3(UINT nrot = 1)
    {
      add(StrElt(false, 5, 3, 4, 5, 6, 7), StrElt(false, 1, 1), nrot);
      this->name = "HMT_sS3";
    }
  };

  // Isolated points detection
  //! Square I ([0], [1,2,3,4,5,6,7,8])
  class HMT_sI : public CompStrEltList
  {
  public:
    HMT_sI()
    {
      add(StrElt(false, 1, 0), StrElt(false, 8, 1, 2, 3, 4, 5, 6, 7, 8));
      this->name = "HMT_sI";
    }
  };

  //! Hexagonal I ([0], [1,2,3,4,5,6])
  class HMT_hI : public CompStrEltList
  {
  public:
    HMT_hI()
    {
      add(StrElt(true, 1, 0), StrElt(true, 6, 1, 2, 3, 4, 5, 6));
      this->name = "HMT_hI";
    }
  };

  //! Square line end ([0,1], [3,4,5,6,7])
  class HMT_sLineEnd : public CompStrEltList
  {
  public:
    HMT_sLineEnd(UINT nrot = 1)
    {
      add(StrElt(false, 2, 0, 1), StrElt(false, 5, 3, 4, 5, 6, 7), nrot);
      this->name = "HMT_sLineEnd";
    }
  };

  //! Square line junction ([0,1,4,6],[] and [0,1,3,5],[])
  class HMT_sLineJunc : public CompStrEltList
  {
  public:
    HMT_sLineJunc(UINT nrot = 1)
    {
      add(StrElt(false, 4, 0, 1, 4, 6), StrElt(false, 1, 5), nrot);
      add(StrElt(false, 4, 0, 1, 3, 5), StrElt(false, 2, 2, 4), nrot);
      this->name = "HMT_sLineJunc";
    }
  };

  /** @} */

} // namespace smil

#endif // _COMPOSITE_SE_HPP
