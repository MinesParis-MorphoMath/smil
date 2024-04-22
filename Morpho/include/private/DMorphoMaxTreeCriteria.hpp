/* __HEAD__
 * Copyright (c) 2017-2024, Centre de Morphologie Mathematique
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - XX/XX/2017 - by Beatriz Marcotegui
 *     Just created it...A
 *
 * __HEAD__ - Stop here !
 */

#ifndef MORPHO_MAX_TREE_CRITERIA_H_
#define MORPHO_MAX_TREE_CRITERIA_H_

#include <complex>
#include <memory>
#include <queue>

#include "DMorphoHierarQ.hpp" //BMI

namespace smil
{

  /// Generic criterion for the max-tree. A user-defined criterion should be
  /// derived from this class.
  template <class tAttType>
  class GenericCriterion
  {
  public:
    GenericCriterion()
    {
    }

    virtual ~GenericCriterion()
    {
    }

  public:
    virtual void initialize()                                           = 0;
    virtual void reset()                                                = 0;
    virtual void merge(GenericCriterion *other_criteron)                = 0;
    virtual void update(const size_t x, const size_t y, const size_t z) = 0;
    virtual bool operator<(const tAttType &other_attribute)             = 0;
    tAttType     getAttributeValue()
    {
      compute();
      return attribute_value_;
    }

  protected:
    virtual void compute() = 0;

  protected:
    tAttType attribute_value_;
  };

  /// Area criterion. Useful for Area Opening/Closing algorithms based on
  /// max-tree.
  class AreaCriterion : public GenericCriterion<size_t>
  {
  public:
    AreaCriterion()
    {
      initialize();
    }

    virtual ~AreaCriterion()
    {
    }

  public:
    virtual void initialize()
    {
      attribute_value_ = 1;
    }

    virtual void reset()
    {
      attribute_value_ = 0;
    }

    virtual void merge(GenericCriterion *other_criteron)
    {
      attribute_value_ +=
          dynamic_cast<AreaCriterion &>(*other_criteron).attribute_value_;
    }

    virtual void update(SMIL_UNUSED const size_t x, SMIL_UNUSED const size_t y,
                        SMIL_UNUSED const size_t z)
    {
      attribute_value_ += 1;
    }
    virtual bool operator<(const size_t &other_attribute)
    {
      return (attribute_value_ < other_attribute);
    }

  protected:
    virtual void compute()
    {
    }
  };

  /// Height criterion. Useful for Height Opening/Closing algorithms based on
  /// max-tree.
  class HeightCriterion : public GenericCriterion<size_t>
  {
  public:
    HeightCriterion()
    {
      initialize();
    }

    virtual ~HeightCriterion()
    {
    }

  public:
    virtual void initialize()
    {
      attribute_value_ = 0;
      // lowest instead of min in Andres code
      y_max_ = std::numeric_limits<size_t>::min();
      y_min_ = std::numeric_limits<size_t>::max();
    }

    virtual void reset()
    {
      initialize();
    }

    virtual void merge(GenericCriterion *other_criteron)
    {
      y_max_ = std::max(
          y_max_, dynamic_cast<HeightCriterion &>(*other_criteron).y_max_);
      y_min_ = std::min(
          y_min_, dynamic_cast<HeightCriterion &>(*other_criteron).y_min_);
    }

    virtual void update(SMIL_UNUSED const size_t x, const size_t y,
                        SMIL_UNUSED const size_t z)
    {
      y_max_ = std::max(y_max_, y);
      y_min_ = std::min(y_min_, y);
    }
    virtual bool operator<(const size_t &other_attribute)
    {
      return (attribute_value_ < other_attribute);
    }

  protected:
    virtual void compute()
    {
      attribute_value_ = y_max_ - y_min_ + 1;
    }

  private:
    // BMI: int in Andres code
    size_t y_max_;
    size_t y_min_;
  };

  /// Width criterion. Useful for Width Opening/Closing algorithms based on
  /// max-tree.
  class WidthCriterion : public GenericCriterion<size_t>
  {
  public:
    WidthCriterion()
    {
      initialize();
    }

    virtual ~WidthCriterion()
    {
    }

  public:
    virtual void initialize()
    {
      attribute_value_ = 0;
      // lowest instead of min in Andres code
      x_max_ = std::numeric_limits<size_t>::min();
      x_min_ = std::numeric_limits<size_t>::max();
    }

    virtual void reset()
    {
      initialize();
    }

    virtual void merge(GenericCriterion *other_criteron)
    {
      x_max_ = std::max(x_max_,
                        dynamic_cast<WidthCriterion &>(*other_criteron).x_max_);
      x_min_ = std::min(x_min_,
                        dynamic_cast<WidthCriterion &>(*other_criteron).x_min_);
    }

    virtual void update(const size_t x, SMIL_UNUSED const size_t y,
                        SMIL_UNUSED const size_t z)
    {
      x_max_ = std::max(x_max_, x);
      x_min_ = std::min(x_min_, x);
    }
    virtual bool operator<(const size_t &other_attribute)
    {
      return (attribute_value_ < other_attribute);
    }

  protected:
    virtual void compute()
    {
      attribute_value_ = x_max_ - x_min_ + 1;
    }

  private:
    // BMI: int in Andres code
    size_t x_max_;
    size_t x_min_;
  };

  struct HA {
    size_t H;
    size_t A;
  };
  struct HWA {
    size_t H;
    size_t W;
    size_t A;
  };

  /// HeightArea criterion. Useful for Height Opening/Closing algorithms based
  /// on max-tree.
  class HACriterion : public GenericCriterion<HA>
  {
  public:
    HACriterion()
    {
      initialize();
    }

    virtual ~HACriterion()
    {
    }

  public:
    virtual void initialize()
    {
      attribute_value_.H = 1;
      attribute_value_.A = 0;

      // lowest instead of min in Andres code
      y_max_ = std::numeric_limits<size_t>::min();
      y_min_ = std::numeric_limits<size_t>::max();
    }

    virtual void reset()
    {
      initialize();
    }

    virtual void merge(GenericCriterion *other_criteron)
    {
      attribute_value_.A +=
          dynamic_cast<HACriterion &>(*other_criteron).getAttributeValue().A;

      y_max_ =
          std::max(y_max_, dynamic_cast<HACriterion &>(*other_criteron).y_max_);
      y_min_ =
          std::min(y_min_, dynamic_cast<HACriterion &>(*other_criteron).y_min_);
    }

    virtual void update(SMIL_UNUSED const size_t x, const size_t y,
                        SMIL_UNUSED const size_t z)
    {
      attribute_value_.A += 1;
      y_max_ = std::max(y_max_, y);
      y_min_ = std::min(y_min_, y);
    }
    virtual bool operator<(const HA &other_attribute)
    {
      return (attribute_value_.H < other_attribute.H);
    }

  protected:
    virtual void compute()
    {
      attribute_value_.H = y_max_ - y_min_ + 1;
    }

  private:
    // BMI: int in Andres code
    size_t y_max_;
    size_t y_min_;
  };

  /// HeightArea criterion. Useful for Height Opening/Closing algorithms based
  /// on max-tree.
  class HWACriterion : public GenericCriterion<HWA>
  {
  public:
    HWACriterion()
    {
      initialize();
    }

    virtual ~HWACriterion()
    {
    }

  public:
    virtual void initialize()
    {
      attribute_value_.H = 1;
      attribute_value_.W = 1;
      attribute_value_.A = 0;

      // lowest instead of min in Andres code
      x_max_ = std::numeric_limits<size_t>::min();
      x_min_ = std::numeric_limits<size_t>::max();

      y_max_ = std::numeric_limits<size_t>::min();
      y_min_ = std::numeric_limits<size_t>::max();
    }

    virtual void reset()
    {
      initialize();
    }

    virtual void merge(GenericCriterion *other_criteron)
    {
      attribute_value_.A +=
          dynamic_cast<HWACriterion &>(*other_criteron).getAttributeValue().A;

      x_max_ = std::max(x_max_,
                        dynamic_cast<HWACriterion &>(*other_criteron).x_max_);
      x_min_ = std::min(x_min_,
                        dynamic_cast<HWACriterion &>(*other_criteron).x_min_);

      y_max_ = std::max(y_max_,
                        dynamic_cast<HWACriterion &>(*other_criteron).y_max_);
      y_min_ = std::min(y_min_,
                        dynamic_cast<HWACriterion &>(*other_criteron).y_min_);
    }

    virtual void update(const size_t x, const size_t y,
                        SMIL_UNUSED const size_t z)
    {
      attribute_value_.A += 1;
      x_max_ = std::max(x_max_, x);
      x_min_ = std::min(x_min_, x);

      // AttributeOpen with this criterion would be a Height Opening
      y_max_ = std::max(y_max_, y);
      y_min_ = std::min(y_min_, y);
    }
    virtual bool operator<(const HWA &other_attribute)
    {
      return (attribute_value_.H < other_attribute.H);
    }

  protected:
    virtual void compute()
    {
      attribute_value_.W = x_max_ - x_min_ + 1;
      attribute_value_.H = y_max_ - y_min_ + 1;
    }

  private:
    // BMI: int in Andres code
    size_t x_max_;
    size_t x_min_;

    size_t y_max_;
    size_t y_min_;
  };

} // namespace smil

#endif // MORPHO_MAX_TREE_CRITERIA_H_
