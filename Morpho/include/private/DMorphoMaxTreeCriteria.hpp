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
  virtual bool operator < (const tAttType& other_attribute) = 0;
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
  AreaCriterion(){initialize();}

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

  virtual void update(SMIL_UNUSED const size_t x, SMIL_UNUSED const size_t y, SMIL_UNUSED const size_t z)
  {
    attribute_value_ += 1;
  }
  virtual bool operator < (const size_t& other_attribute){
    return (attribute_value_ < other_attribute);
  }
protected:  

  virtual void compute(){}
};

/// Height criterion. Useful for Height Opening/Closing algorithms based on max-tree.
class HeightCriterion : public GenericCriterion<size_t>
{
public:
  HeightCriterion(){initialize();}

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

  virtual void update(SMIL_UNUSED const size_t x, const size_t y, SMIL_UNUSED const size_t z)
  {

    y_max_ = std::max(y_max_, y);
    y_min_ = std::min(y_min_, y);
  }
  virtual bool operator < (const size_t& other_attribute){
    return (attribute_value_ < other_attribute);
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



/// Width criterion. Useful for Width Opening/Closing algorithms based on max-tree.
class WidthCriterion : public GenericCriterion<size_t>
{
public:
  WidthCriterion(){initialize();}

  virtual ~WidthCriterion(){}

public:
  virtual void initialize()
  {
    attribute_value_ = 0;
    x_max_ = std::numeric_limits<size_t>::min();//lowest instead of min in Andres code
    x_min_ = std::numeric_limits<size_t>::max();
  }

  virtual void reset()
  {
    initialize();
  }

  virtual void merge(GenericCriterion* other_criteron)
  {
    x_max_ = std::max(x_max_, dynamic_cast<WidthCriterion&>(*other_criteron).x_max_);
    x_min_ = std::min(x_min_, dynamic_cast<WidthCriterion&>(*other_criteron).x_min_);
  }

  virtual void update(const size_t x, SMIL_UNUSED const size_t y, SMIL_UNUSED const size_t z)
  {
    x_max_ = std::max(x_max_, x);
    x_min_ = std::min(x_min_, x);
  }
  virtual bool operator < (const size_t & other_attribute){
    return (attribute_value_ < other_attribute);
  }

protected:
  virtual void compute()
  {
    attribute_value_ = x_max_ - x_min_ + 1;
  }

private:
  size_t x_max_;// BMI: int in Andres code
  size_t x_min_;// BMI: int in Andres code
};




struct HA{
    size_t H;
    size_t A;
  };
struct HWA{
    size_t H;
    size_t W;
    size_t A;
  };

/// HeightArea criterion. Useful for Height Opening/Closing algorithms based on max-tree.
  class HACriterion : public GenericCriterion< HA>
{
public:
  HACriterion(){initialize();}

  virtual ~HACriterion(){}

public:
  virtual void initialize()
  {

    attribute_value_.H = 1;
    attribute_value_.A = 0;

    y_max_ = std::numeric_limits<size_t>::min();//lowest instead of min in Andres code
    y_min_ = std::numeric_limits<size_t>::max();
  }

  virtual void reset()
  {
    initialize();
  }

  virtual void merge(GenericCriterion* other_criteron)
  {
    attribute_value_.A += dynamic_cast<HACriterion&>(*other_criteron).getAttributeValue().A;

    y_max_ = std::max(y_max_, dynamic_cast<HACriterion&>(*other_criteron).y_max_);
    y_min_ = std::min(y_min_, dynamic_cast<HACriterion&>(*other_criteron).y_min_);

  }

  virtual void update(SMIL_UNUSED const size_t x, const size_t y, SMIL_UNUSED const size_t z)
  {
    attribute_value_.A += 1;
    y_max_ = std::max(y_max_, y);
    y_min_ = std::min(y_min_, y);
  }
  virtual bool operator < (const HA& other_attribute){// AttributeOpen with this criterion would be a Height Opening
    return (attribute_value_.H < other_attribute.H);
  }

protected:
  virtual void compute()
  {
    attribute_value_.H = y_max_ - y_min_ + 1;
  }

private:
  size_t y_max_;// BMI: int in Andres code
  size_t y_min_;// BMI: int in Andres code
};

/// HeightArea criterion. Useful for Height Opening/Closing algorithms based on max-tree.
  class HWACriterion : public GenericCriterion< HWA >
{
public:
  HWACriterion(){initialize();}

  virtual ~HWACriterion(){}

public:
  virtual void initialize()
  {

    attribute_value_.H = 1;
    attribute_value_.W = 1;
    attribute_value_.A = 0;

    x_max_ = std::numeric_limits<size_t>::min();//lowest instead of min in Andres code
    x_min_ = std::numeric_limits<size_t>::max();

    y_max_ = std::numeric_limits<size_t>::min();//lowest instead of min in Andres code
    y_min_ = std::numeric_limits<size_t>::max();
  }

  virtual void reset()
  {
    initialize();
  }

  virtual void merge(GenericCriterion* other_criteron)
  {
    attribute_value_.A += dynamic_cast<HWACriterion&>(*other_criteron).getAttributeValue().A;

    x_max_ = std::max(x_max_, dynamic_cast<HWACriterion&>(*other_criteron).x_max_);
    x_min_ = std::min(x_min_, dynamic_cast<HWACriterion&>(*other_criteron).x_min_);

    y_max_ = std::max(y_max_, dynamic_cast<HWACriterion&>(*other_criteron).y_max_);
    y_min_ = std::min(y_min_, dynamic_cast<HWACriterion&>(*other_criteron).y_min_);

  }

  virtual void update(const size_t x, const size_t y, SMIL_UNUSED const size_t z)
  {
    attribute_value_.A += 1;
    x_max_ = std::max(x_max_, x);
    x_min_ = std::min(x_min_, x);

    y_max_ = std::max(y_max_, y);
    y_min_ = std::min(y_min_, y);
  }
  virtual bool operator < (const HWA& other_attribute){// AttributeOpen with this criterion would be a Height Opening
    return (attribute_value_.H < other_attribute.H);
  }

protected:
  virtual void compute()
  {
    attribute_value_.W = x_max_ - x_min_ + 1;
    attribute_value_.H = y_max_ - y_min_ + 1;
  }

private:
  size_t x_max_;// BMI: int in Andres code
  size_t x_min_;

  size_t y_max_;
  size_t y_min_;
};




} // namespace smil


#endif // MORPHO_MAX_TREE_ATTRIBUTES_H_




