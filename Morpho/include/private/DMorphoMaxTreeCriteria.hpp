#ifndef MORPHO_MAX_TREE_ATTRIBUTES_H_
#define MORPHO_MAX_TREE_ATTRIBUTES_H_

#include <complex>
#include <memory>
#include <queue>


#include "DMorphoHierarQ.hpp"//BMI

namespace smil
{

/// Generic criterion for the max-tree. A user-defined criterion should be derived from this class.
template<class tAttType>
class GenericCriterion
{
public:
  GenericCriterion(){}

  virtual ~GenericCriterion(){}

public:
  virtual void initialize() = 0;
  virtual void reset() = 0;
  virtual void merge(GenericCriterion* other_criteron) = 0;
  virtual void update(const size_t x, const size_t y,const size_t z) = 0;
  tAttType getAttributeValue()
  {
    compute();
    return attribute_value_;
  }

protected:
  virtual void compute() = 0;

protected:
  tAttType attribute_value_;

};

/// Area criterion. Useful for Area Opening/Closing algorithms based on max-tree.
class AreaCriterion : public GenericCriterion<size_t>
{
public:
  AreaCriterion(){}

  virtual ~AreaCriterion(){}

public:
  virtual void initialize()
  {
    attribute_value_ = 1;
  }

  virtual void reset()
  {
    attribute_value_ = 0;
  }

  virtual void merge(GenericCriterion* other_criteron)
  {
    attribute_value_ += dynamic_cast<AreaCriterion&>(*other_criteron).attribute_value_;
  }

  virtual void update(SMIL_UNUSED const size_t x, SMIL_UNUSED const size_t y,
                      SMIL_UNUSED const size_t z) {
    attribute_value_ += 1;
  }

protected:
  virtual void compute(){}
};

/// Height criterion. Useful for Height Opening/Closing algorithms based on max-tree.
class HeightCriterion : public GenericCriterion<size_t>
{
public:
  HeightCriterion(){}

  virtual ~HeightCriterion(){}

public:
  virtual void initialize()
  {
    attribute_value_ = 0;
    y_max_ = std::numeric_limits<size_t>::min();//lowest instead of min in Andres code
    y_min_ = std::numeric_limits<size_t>::max();
  }

  virtual void reset()
  {
    initialize();
  }

  virtual void merge(GenericCriterion* other_criteron)
  {
    y_max_ = std::max(y_max_, dynamic_cast<HeightCriterion&>(*other_criteron).y_max_);
    y_min_ = std::min(y_min_, dynamic_cast<HeightCriterion&>(*other_criteron).y_min_);
  }

  virtual void update(SMIL_UNUSED const size_t x, const size_t y,
                      SMIL_UNUSED const size_t z) {
    y_max_ = std::max(y_max_, y);
    y_min_ = std::min(y_min_, y);
  }

protected:
  virtual void compute()
  {
    attribute_value_ = y_max_ - y_min_ + 1;
  }

private:
  size_t y_max_;// BMI: int in Andres code
  size_t y_min_;// BMI: int in Andres code
};


} // namespace smil


#endif // MORPHO_MAX_TREE_ATTRIBUTES_H_
