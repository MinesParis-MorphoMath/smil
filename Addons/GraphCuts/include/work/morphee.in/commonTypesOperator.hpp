

#ifndef _MORPHEE_TYPES_OPERATOR_HPP_
#define _MORPHEE_TYPES_OPERATOR_HPP_

#include <vector>
#include <map>
#include <algorithm>

#include <typeinfo>
#include <limits>
#include <assert.h>
#include <morphee/common/include/commonException.hpp>
#include <morphee/common/include/commonTypes.hpp>

namespace morphee
{
  // forward declarations
  template <class T> struct DataTraits;
  class ImageInterface;

  template <typename T> struct pixel_2 {
    typedef T value_type;
    typedef pixel_2<T> this_type;
    T channel1, channel2;
    enum e_dimension { dimension = 2 };

  public:
    // constructeurs, explicites
    explicit pixel_2()
    {
    }
    // en conflit avec les conversions implicites de CVariant
    // explicit pixel_2(const T& c): channel1(c), channel2(c), channel3(c){};
    pixel_2(const T &c1, const T &c2) : channel1(c1), channel2(c2)
    {
    }

    template <typename T2> operator pixel_2<T2>() const
    {
      pixel_2<T2> out(static_cast<T2>(channel1), static_cast<T2>(channel2));
      return out;
    }
    inline const this_type &operator=(const T &rhs)
    {
      channel1 = rhs;
      channel2 = rhs;
      return *this;
    }
    const value_type operator[](unsigned int channel) const
    {
      if (channel >= 2) {
        throw(morphee::MException(
            "pixel_2::operator[] const: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      switch (channel) {
      case 0:
        return channel1;
      case 1:
        return channel2;
      default:
        throw(morphee::MException(
            "pixel_2::operator[] const: Unhandled BIG ERROR !!"));
      }
    }
    value_type &operator[](unsigned int channel)
    {
      if (channel >= 2) {
        throw(morphee::MException(
            "pixel_2::operator[]: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      switch (channel) {
      case 0:
        return channel1;
      case 1:
        return channel2;
      default:
        throw(
            morphee::MException("pixel_2::operator[]: Unhandled BIG ERROR !!"));
      }
    }
    inline const this_type &operator&=(const this_type &rhs)
    {
      channel1 &= rhs.channel1;
      channel2 &= rhs.channel2;
      return *this;
    }
    inline const this_type &operator&=(const value_type rhs)
    {
      channel1 &= rhs;
      channel2 &= rhs;
      return *this;
    }
    inline const this_type &operator|=(const this_type &rhs)
    {
      channel1 |= rhs.channel1;
      channel2 |= rhs.channel2;
      return *this;
    }
    inline const this_type &operator|=(const value_type rhs)
    {
      channel1 |= rhs;
      channel2 |= rhs;
      return *this;
    }
    template <class T2>
    inline const this_type &operator+=(const pixel_2<T2> &rhs)
    {
      channel1 += rhs.channel1;
      channel2 += rhs.channel2;
      return *this;
    }
    template <class T2>
    inline const this_type &operator*=(const pixel_2<T2> &rhs)
    {
      channel1 = static_cast<value_type>(channel1 * rhs.channel1);
      channel2 = static_cast<value_type>(channel2 * rhs.channel2);
      return *this;
    }
    template <class T2>
    inline const this_type &operator-=(const pixel_2<T2> &rhs)
    {
      channel1 -= rhs.channel1;
      channel2 -= rhs.channel2;
      return *this;
    }
    template <class T2>
    inline const this_type &operator/=(const pixel_2<T2> &rhs)
    {
      channel1 = static_cast<value_type>(channel1 / rhs.channel1);
      channel2 = static_cast<value_type>(channel2 / rhs.channel2);
      return *this;
    }
    template <class T2>
    inline const this_type &operator=(const pixel_2<T2> &rhs)
    {
      channel1 = static_cast<T>(rhs.channel1);
      channel2 = static_cast<T>(rhs.channel2);
      return *this;
    }
    inline bool operator==(const this_type &rhs) const
    {
      return channel1 == rhs.channel1 && channel2 == rhs.channel2;
    }
    inline bool operator!=(const this_type &rhs) const
    {
      return channel1 != rhs.channel1 || channel2 != rhs.channel2;
    }

    inline void swap(this_type &_pix)
    {
      std::swap(_pix.channel1, channel1);
      std::swap(_pix.channel2, channel2);
    }

    // "Je parle avec la personne la plus intelligente dans la pièce, c'est une
    // habitude de vieil homme" (Le Retour du Roi, J.R.R. Tolkien)
  };

  template <typename T> struct pixel_3 {
    typedef T value_type;
    typedef pixel_3<T> this_type;
    T channel1, channel2, channel3;
    enum e_dimension { dimension = 3 };

  public:
    // constructeurs, explicites
    explicit pixel_3(){};
    // en conflit avec les conversions implicites de CVariant
    // explicit pixel_3(const T& c): channel1(c), channel2(c), channel3(c){};
    pixel_3(const T &c1, const T &c2, const T &c3)
        : channel1(c1), channel2(c2), channel3(c3){};

    template <typename T2> operator vector_N<T2>() const
    {
      vector_N<T2> out;
      out.values[0] = static_cast<T2>(channel1);
      out.values[1] = static_cast<T2>(channel2);
      out.values[2] = static_cast<T2>(channel3);
      return out;
    }

    template <typename T2> operator pixel_3<T2>() const
    {
      pixel_3<T2> out(static_cast<T2>(channel1), static_cast<T2>(channel2),
                      static_cast<T2>(channel3));
      return out;
    }
    inline const this_type &operator=(const value_type &rhs)
    {
      channel1 = rhs;
      channel2 = rhs;
      channel3 = rhs;
      return *this;
    }
    const value_type operator[](unsigned int channel) const
    {
      if (channel >= 3) {
        throw(morphee::MException(
            "pixel_3::operator[] const: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      switch (channel) {
      case 0:
        return channel1;
      case 1:
        return channel2;
      case 2:
        return channel3;
      default:
        throw(morphee::MException(
            "pixel_3::operator[] const: Unhandled BIG ERROR !!"));
      }
    }
    value_type &operator[](unsigned int channel)
    {
      if (channel >= 3) {
        throw(morphee::MException(
            "pixel_3::operator[]: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      switch (channel) {
      case 0:
        return channel1;
      case 1:
        return channel2;
      case 2:
        return channel3;
      default:
        throw(
            morphee::MException("pixel_3::operator[]: Unhandled BIG ERROR !!"));
      }
    }

    inline const this_type &operator&=(const this_type &rhs)
    {
      channel1 &= rhs.channel1;
      channel2 &= rhs.channel2;
      channel3 &= rhs.channel3;
      return *this;
    }
    inline const this_type &operator&=(const value_type rhs)
    {
      channel1 &= rhs;
      channel2 &= rhs;
      channel3 &= rhs;
      return *this;
    }
    inline const this_type &operator|=(const this_type &rhs)
    {
      channel1 |= rhs.channel1;
      channel2 |= rhs.channel2;
      channel3 |= rhs.channel3;
      return *this;
    }
    inline const this_type &operator|=(const value_type rhs)
    {
      channel1 |= rhs;
      channel2 |= rhs;
      channel3 |= rhs;
      return *this;
    }
    template <class T2>
    inline const this_type &operator+=(const pixel_3<T2> &rhs)
    {
      channel1 += rhs.channel1;
      channel2 += rhs.channel2;
      channel3 += rhs.channel3;
      return *this;
    }
    template <class T2>
    inline const this_type &operator*=(const pixel_3<T2> &rhs)
    {
      channel1 = static_cast<value_type>(channel1 * rhs.channel1);
      channel2 = static_cast<value_type>(channel2 * rhs.channel2);
      channel3 = static_cast<value_type>(channel3 * rhs.channel3);
      return *this;
    }
    template <class T2>
    inline const this_type &operator-=(const pixel_3<T2> &rhs)
    {
      channel1 -= rhs.channel1;
      channel2 -= rhs.channel2;
      channel3 -= rhs.channel3;
      return *this;
    }
    template <class T2>
    inline const this_type &operator/=(const pixel_3<T2> &rhs)
    {
      channel1 = static_cast<value_type>(channel1 / rhs.channel1);
      channel2 = static_cast<value_type>(channel2 / rhs.channel2);
      channel3 = static_cast<value_type>(channel3 / rhs.channel3);
      return *this;
    }
    template <class T2>
    inline const this_type &operator=(const pixel_3<T2> &rhs)
    {
      channel1 = static_cast<T>(rhs.channel1);
      channel2 = static_cast<T>(rhs.channel2);
      channel3 = static_cast<T>(rhs.channel3);
      return *this;
    }
    inline bool operator==(const this_type &rhs) const
    {
      return channel1 == rhs.channel1 && channel2 == rhs.channel2 &&
             channel3 == rhs.channel3;
    }
    inline bool operator!=(const this_type &rhs) const
    {
      return channel1 != rhs.channel1 || channel2 != rhs.channel2 ||
             channel3 != rhs.channel3;
    }

    inline void swap(this_type &_pix)
    {
      std::swap(_pix.channel1, channel1);
      std::swap(_pix.channel2, channel2);
      std::swap(_pix.channel3, channel3);
    }

    // "Je parle avec la personne la plus intelligente dans la pièce, c'est une
    // habitude de vieil homme" (Le Retour du Roi, J.R.R. Tolkien)
  };

  template <typename T> struct pixel_4 {
    typedef T value_type;
    typedef pixel_4<T> this_type;
    T channel1, channel2, channel3, channel4;
    enum e_dimension { dimension = 4 };

  public:
    // constructeurs, explicites
    explicit pixel_4(){};
    pixel_4(const T &c1, const T &c2, const T &c3, const T &c4)
        : channel1(c1), channel2(c2), channel3(c3), channel4(c4){};

    template <typename T2> operator vector_N<T2>() const
    {
      vector_N<T2> out;
      out.values[0] = static_cast<T2>(channel1);
      out.values[1] = static_cast<T2>(channel2);
      out.values[2] = static_cast<T2>(channel3);
      out.values[3] = static_cast<T2>(channel4);
      return out;
    }

    template <typename T2> operator pixel_4<T2>() const
    {
      pixel_4<T2> out;
      out.channel1 = static_cast<T2>(channel1);
      out.channel2 = static_cast<T2>(channel2);
      out.channel3 = static_cast<T2>(channel3);
      out.channel4 = static_cast<T2>(channel4);
      return out;
    }
    inline const this_type &operator=(const value_type &rhs)
    {
      channel1 = rhs;
      channel2 = rhs;
      channel3 = rhs;
      channel4 = rhs;
      return *this;
    }

    const value_type operator[](unsigned int channel) const
    {
      if (channel >= 4) {
        throw(morphee::MException(
            "pixel_4::operator[] const: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      switch (channel) {
      case 0:
        return channel1;
      case 1:
        return channel2;
      case 2:
        return channel3;
      case 3:
        return channel4;
      default:
        throw(morphee::MException(
            "pixel_4::operator[] const: Unhandled BIG ERROR !!"));
      }
    }
    value_type &operator[](unsigned int channel)
    {
      if (channel >= 4) {
        throw(morphee::MException(
            "pixel_4::operator[]: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      switch (channel) {
      case 0:
        return channel1;
      case 1:
        return channel2;
      case 2:
        return channel3;
      case 3:
        return channel4;
      default:
        throw(
            morphee::MException("pixel_4::operator[]: Unhandled BIG ERROR !!"));
      }
    }

    inline const this_type &operator&=(const this_type &rhs)
    {
      channel1 &= rhs.channel1;
      channel2 &= rhs.channel2;
      channel3 &= rhs.channel3;
      channel4 &= rhs.channel4;
      return *this;
    }
    inline const this_type &operator&=(const T rhs)
    {
      channel1 &= rhs;
      channel2 &= rhs;
      channel3 &= rhs;
      channel4 &= rhs;
      return *this;
    }
    inline const this_type &operator|=(const this_type &rhs)
    {
      channel1 |= rhs.channel1;
      channel2 |= rhs.channel2;
      channel3 |= rhs.channel3;
      channel4 |= rhs.channel4;
      return *this;
    }
    inline const this_type &operator|=(const T rhs)
    {
      channel1 |= rhs;
      channel2 |= rhs;
      channel3 |= rhs;
      channel4 |= rhs;
      return *this;
    }
    template <class T2>
    inline const this_type &operator=(const pixel_4<T2> &rhs)
    {
      channel1 = static_cast<T>(rhs.channel1);
      channel2 = static_cast<T>(rhs.channel2);
      channel3 = static_cast<T>(rhs.channel3);
      channel4 = static_cast<T>(rhs.channel4);
      return *this;
    }
    template <class T2>
    inline const this_type &operator+=(const pixel_4<T2> &rhs)
    {
      channel1 += rhs.channel1;
      channel2 += rhs.channel2;
      channel3 += rhs.channel3;
      channel4 += rhs.channel4;
      return *this;
    }
    template <class T2>
    inline const this_type &operator*=(const pixel_4<T2> &rhs)
    {
      channel1 *= rhs.channel1;
      channel2 *= rhs.channel2;
      channel3 *= rhs.channel3;
      channel4 *= rhs.channel4;
      return *this;
    }
    template <class T2>
    inline const this_type &operator/=(const pixel_4<T2> &rhs)
    {
      channel1 /= rhs.channel1;
      channel2 /= rhs.channel2;
      channel3 /= rhs.channel3;
      channel4 /= rhs.channel4;
      return *this;
    }
    template <class T2>
    inline const this_type &operator-=(const pixel_4<T2> &rhs)
    {
      channel1 -= rhs.channel1;
      channel2 -= rhs.channel2;
      channel3 -= rhs.channel3;
      channel4 -= rhs.channel4;
      return *this;
    }
    inline bool operator==(const this_type &rhs) const
    {
      return channel1 == rhs.channel1 && channel2 == rhs.channel2 &&
             channel3 == rhs.channel3 && channel4 == rhs.channel4;
    }
    inline bool operator!=(const this_type &rhs) const
    {
      return channel1 != rhs.channel1 || channel2 != rhs.channel2 ||
             channel3 != rhs.channel3 || channel4 != rhs.channel4;
    }

    inline void swap(this_type &_pix)
    {
      std::swap(_pix.channel1, channel1);
      std::swap(_pix.channel2, channel2);
      std::swap(_pix.channel3, channel3);
      std::swap(_pix.channel4, channel4);
    }
  };

  //! Template class for pixels of any dimension
  template <typename T, int dim> struct pixel_N {
    typedef T value_type;
    typedef pixel_N<T, dim> this_type;
    T values[dim];
    enum e_dimension { dimension = dim };

  public:
    // constructeurs, explicites
    explicit pixel_N()
    {
    }
    pixel_N(T val[dim])
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] = val[k];
      }
    }
    /*explicit */ pixel_N(const this_type &p)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] = p.values[k];
      }
    }

    template <typename T2> operator pixel_N<T2, dim>() const
    {
      pixel_N<T2, dim> out;
      for (unsigned int k = 0; k < dim; k++) {
        out.values[k] = values[k];
      }
      return out;
    }
    inline const this_type &operator=(const value_type &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] = rhs;
      }
      return *this;
    }

    const value_type &operator[](unsigned int channel) const
    {
      if (channel >= dim) {
        throw(morphee::MException(
            "pixel_N::operator[] const : Trying to access a channel outside "
            "the maximal dimension of the pixel"));
      }
      return values[channel];
    }
    value_type &operator[](unsigned int channel)
    {
      if (channel >= dim) {
        throw(morphee::MException(
            "pixel_N::operator[]: Trying to access a channel outside the "
            "maximal dimension of the pixel"));
      }
      return values[channel];
    }

    inline const this_type &operator&=(const this_type &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] &= rhs.values[k];
      }
      return *this;
    }
    inline const this_type &operator&=(const T rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] &= rhs;
      }
      return *this;
    }
    inline const this_type &operator|=(const this_type &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] |= rhs.values[k];
      }
      return *this;
    }
    inline const this_type &operator|=(const T rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] |= rhs;
      }
      return *this;
    }

    template <class T2>
    inline const this_type &operator=(const pixel_N<T2, dim> &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] = rhs.values[k];
      }
      return *this;
    }
    template <class T2>
    inline const this_type &operator+=(const pixel_N<T2, dim> &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] += rhs.values[k];
      }
      return *this;
    }
    template <class T2>
    inline const this_type &operator*=(const pixel_N<T2, dim> &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] *= rhs.values[k];
      }
      return *this;
    }
    template <class T2>
    inline const this_type &operator/=(const pixel_N<T2, dim> &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] /= rhs.values[k];
      }
      return *this;
    }
    template <class T2>
    inline const this_type &operator-=(const pixel_N<T2, dim> &rhs)
    {
      for (unsigned int k = 0; k < dim; k++) {
        values[k] -= rhs.values[k];
      }
      return *this;
    }
    inline bool operator==(const this_type &rhs) const
    {
      for (unsigned int k = 0; k < dim; k++) {
        if (values[k] != rhs.values[k])
          return false;
      }
      return true;
    }
    inline bool operator!=(const this_type &rhs) const
    {
      return !operator==(rhs);
    }

    inline void swap(this_type &_pix)
    {
      // Raffi: je ne sais pas si on peut faire directement swap(values,
      // pix.values), mais je ne pense pas
      for (unsigned int k = 0; k < dim; k++) {
        std::swap(values[k], _pix.values[k]);
      }
    }
  };

  //! Template class for pixels of any dimension
  template <typename T> struct vector_N {
    typedef T value_type;
    typedef vector_N<T> this_type;
    std::map<int, T> values;

  public:
    // constructeurs, explicites
    explicit vector_N()
    {
    }
    vector_N(const T &c1)
    {
      add(c1);
    }
    vector_N(const T &c1, const T &c2)
    {
      add(c1);
      add(c2);
    }
    vector_N(const T &c1, const T &c2, const T &c3)
    {
      add(c1);
      add(c2);
      add(c3);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4, const T &c5)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
      add(c5);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4, const T &c5,
             const T &c6)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
      add(c5);
      add(c6);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4, const T &c5,
             const T &c6, const T &c7)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
      add(c5);
      add(c6);
      add(c7);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4, const T &c5,
             const T &c6, const T &c7, const T &c8)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
      add(c5);
      add(c6);
      add(c7);
      add(c8);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4, const T &c5,
             const T &c6, const T &c7, const T &c8, const T &c9)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
      add(c5);
      add(c6);
      add(c7);
      add(c8);
      add(c9);
    }
    vector_N(const T &c1, const T &c2, const T &c3, const T &c4, const T &c5,
             const T &c6, const T &c7, const T &c8, const T &c9, const T &c10)
    {
      add(c1);
      add(c2);
      add(c3);
      add(c4);
      add(c5);
      add(c6);
      add(c7);
      add(c8);
      add(c9);
      add(c10);
    }
    template <typename T2> operator vector_N<T2>() const
    {
      vector_N<T2> out;
      typename std::map<int, T>::const_iterator it, itend = values.end();
      for (it = values.begin(); it != itend; ++it)
        out.values[(*it).first] = static_cast<T2>((*it).second);
      return out;
    }
    void add(T value)
    {
      values[static_cast<int>(values.size())] = value;
    }
    void add(int index, T value)
    {
      values[index] = value;
    }
    const value_type operator[](unsigned int index) const
    {
      if (values.find(index) == values.end()) {
        throw(morphee::MException(
            "vector_N::operator[]: The specified index does not exist"));
      }
      return values[index];
    }
    value_type &operator[](unsigned int index)
    {
      if (values.find(index) == values.end()) {
        throw(morphee::MException(
            "vector_N::operator[]: The specified index does not exist"));
      }
      return values[index];
    }
    inline bool operator==(const this_type &rhs) const
    {
      if (values.size() != rhs.values.size())
        return false;
      typename std::map<int, T>::iterator it1    = values.begin(),
                                          it2    = rhs.values.begin(),
                                          itend1 = values.end();
      for (; it1 != itend1; ++it1, ++it2) {
        if ((*it1).first != (*it2).first)
          return false;
        if ((*it1).second != (*it2).second)
          return false;
      }
      return true;
    }
    inline bool operator!=(const this_type &rhs) const
    {
      return !(this == rhs);
    }
    inline const this_type &operator&=(const this_type &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1       = values.begin(),
                                          itend1    = values.end();
      typename std::map<int, T>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second &= (*it2).second;
      return *this;
    }
    inline const this_type &operator&=(const value_type rhs)
    {
      typename std::map<int, T>::iterator it1    = values.begin(),
                                          itend1 = values.end();
      for (; it1 != itend1; ++it1)
        (*it1).second &= rhs;
      return *this;
    }
    inline const this_type &operator|=(const this_type &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1       = values.begin(),
                                          itend1    = values.end();
      typename std::map<int, T>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second |= (*it2).second;
      return *this;
    }
    inline const this_type &operator|=(const value_type rhs)
    {
      typename std::map<int, T>::iterator it1    = values.begin(),
                                          itend1 = values.end();
      for (; it1 != itend1; ++it1)
        (*it1).second |= rhs;
      return *this;
    }
    template <class T2>
    inline const this_type &operator+=(const vector_N<T2> &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1        = values.begin(),
                                          itend1     = values.end();
      typename std::map<int, T2>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second += static_cast<value_type>((*it2).second);
      return *this;
    }
    template <class T2>
    inline const this_type &operator*=(const vector_N<T2> &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1        = values.begin(),
                                          itend1     = values.end();
      typename std::map<int, T2>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second *= static_cast<value_type>((*it2).second);
      return *this;
    }
    template <class T2>
    inline const this_type &operator-=(const vector_N<T2> &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1        = values.begin(),
                                          itend1     = values.end();
      typename std::map<int, T2>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second -= static_cast<value_type>((*it2).second);
      return *this;
    }
    template <class T2>
    inline const this_type &operator/=(const vector_N<T2> &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1        = values.begin(),
                                          itend1     = values.end();
      typename std::map<int, T2>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second /= static_cast<value_type>((*it2).second);
      return *this;
    }
    inline const this_type &operator=(const T &rhs)
    {
      typename std::map<int, T>::iterator it1    = values.begin(),
                                          itend1 = values.end();
      for (; it1 != itend1; ++it1)
        (*it1).second = rhs;
      return *this;
    }
    template <class T2>
    inline const this_type &operator=(const vector_N<T2> &rhs)
    {
      assert(values.size() == rhs.values.size());
      typename std::map<int, T>::iterator it1        = values.begin(),
                                          itend1     = values.end();
      typename std::map<int, T2>::const_iterator it2 = rhs.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        (*it1).second = static_cast<value_type>((*it2).second);
      return *this;
    }
    inline void swap(this_type &_pix)
    {
      typename std::map<int, T>::iterator it1       = values.begin(),
                                          itend1    = values.end();
      typename std::map<int, T>::const_iterator it2 = _pix.values.begin();
      for (; it1 != itend1; ++it1, ++it2)
        std::swap((*it2).second, (*it1).second);
      return *this;
    }
  };

  template <class T1, class T2>
  inline pixel_2<T1> operator+(const pixel_2<T1> &lhs, const pixel_2<T2> &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_2<T1> operator+(const pixel_2<T1> &lhs, const T2 &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_2<T1> operator-(const pixel_2<T1> &lhs, const pixel_2<T2> &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_2<T1> operator-(const pixel_2<T1> &lhs, const T2 &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_2<T1> operator*(const pixel_2<T1> &lhs, const T2 &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_2<T1> operator*(const pixel_2<T1> &lhs, const pixel_2<T2> &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_2<T1> operator/(const pixel_2<T1> &lhs, const pixel_2<T2> &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_2<T1> operator/(const pixel_2<T1> &lhs, const T2 &rhs)
  {
    pixel_2<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T>
  inline pixel_2<T> operator&(const pixel_2<T> &lhs, const pixel_2<T> &rhs)
  {
    pixel_2<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_2<T> operator&(const pixel_2<T> &lhs, const T rhs)
  {
    pixel_2<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_2<T> operator|(const pixel_2<T> &lhs, const pixel_2<T> &rhs)
  {
    pixel_2<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_2<T> operator|(const pixel_2<T> &lhs, const T rhs)
  {
    pixel_2<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_2<T1> operator+(const T2 &lhs, const pixel_2<T1> &rhs)
  {
    return operator+(rhs, lhs);
  };
  template <class T1, class T2>
  inline pixel_2<T1> operator*(const T2 &lhs, const pixel_2<T1> &rhs)
  {
    return rhs.operator*(lhs);
  };
  template <class T1, class T2>
  inline pixel_2<T1> operator-(const T2 &lhs, const pixel_2<T1> &rhs)
  {
    return operator-(rhs, lhs);
  };

  template <class T1, class T2>
  inline pixel_3<T1> operator+(const pixel_3<T1> &lhs, const pixel_3<T2> &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_3<T1> operator+(const pixel_3<T1> &lhs, const T2 &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_3<T1> operator-(const pixel_3<T1> &lhs, const pixel_3<T2> &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_3<T1> operator-(const pixel_3<T1> &lhs, const T2 &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_3<T1> operator*(const pixel_3<T1> &lhs, const T2 &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_3<T1> operator*(const pixel_3<T1> &lhs, const pixel_3<T2> &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_3<T1> operator/(const pixel_3<T1> &lhs, const pixel_3<T2> &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_3<T1> operator/(const pixel_3<T1> &lhs, const T2 &rhs)
  {
    pixel_3<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T>
  inline pixel_3<T> operator&(const pixel_3<T> &lhs, const pixel_3<T> &rhs)
  {
    pixel_3<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_3<T> operator&(const pixel_3<T> &lhs, const T rhs)
  {
    pixel_3<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_3<T> operator|(const pixel_3<T> &lhs, const pixel_3<T> &rhs)
  {
    pixel_3<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_3<T> operator|(const pixel_3<T> &lhs, const T rhs)
  {
    pixel_3<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_3<T1> operator+(const T2 &lhs, const pixel_3<T1> &rhs)
  {
    return operator+(rhs, lhs);
  };
  template <class T1, class T2>
  inline pixel_3<T1> operator*(const T2 &lhs, const pixel_3<T1> &rhs)
  {
    return rhs.operator*(lhs);
  };
  template <class T1, class T2>
  inline pixel_3<T1> operator-(const T2 &lhs, const pixel_3<T1> &rhs)
  {
    return operator-(rhs, lhs);
  };
  // template <class T> inline pixel_3<T> operator/(const pixel_3<T>& lhs, const
  // T& rhs){return operator*(lhs, 1./rhs);};
  // Oh le bug:
  // template <class T> inline pixel_3<T> operator/(const T& lhs, const
  // pixel_3<T>& rhs){return operator/(rhs, lhs);};

  template <class T1, class T2>
  inline pixel_4<T1> operator+(const pixel_4<T1> &lhs, const pixel_4<T2> &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator+(const pixel_4<T1> &lhs, const T2 &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator-(const pixel_4<T1> &lhs, const pixel_4<T2> &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator-(const pixel_4<T1> &lhs, const T2 &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator*(const pixel_4<T1> &lhs, const pixel_4<T2> &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator*(const pixel_4<T1> &lhs, const T2 &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator/(const pixel_4<T1> &lhs, const pixel_4<T2> &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline pixel_4<T1> operator/(const pixel_4<T1> &lhs, const T2 &rhs)
  {
    pixel_4<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_4<T> operator&(const pixel_4<T> &lhs, const pixel_4<T> &rhs)
  {
    pixel_4<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_4<T> operator&(const pixel_4<T> &lhs, const T &rhs)
  {
    pixel_4<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_4<T> operator|(const pixel_4<T> &lhs, const pixel_4<T> &rhs)
  {
    pixel_4<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }
  template <class T>
  inline pixel_4<T> operator|(const pixel_4<T> &lhs, const T &rhs)
  {
    pixel_4<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline pixel_4<T1> operator+(const T2 &lhs, const pixel_4<T1> &rhs)
  {
    return operator+(rhs, lhs);
  };
  template <class T1, class T2>
  inline pixel_4<T1> operator*(const T2 &lhs, const pixel_4<T1> &rhs)
  {
    return rhs.operator*(lhs);
  };
  template <class T1, class T2>
  inline pixel_4<T1> operator-(const T2 &lhs, const pixel_4<T1> &rhs)
  {
    return operator-(rhs, lhs);
  };
  // template <class T> inline pixel_4<T> operator/(const pixel_4<T>& lhs, const
  // T& rhs){return operator*(lhs, 1./rhs);};

  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator+(const pixel_N<T1, dim> &lhs,
                                    const pixel_N<T2, dim> &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator+(const pixel_N<T1, dim> &lhs, const T2 &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator-(const pixel_N<T1, dim> &lhs,
                                    const pixel_N<T2, dim> &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator-(const pixel_N<T1, dim> &lhs, const T2 &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }

  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator*(const pixel_N<T1, dim> &lhs,
                                    const pixel_N<T2, dim> &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator*(const pixel_N<T1, dim> &lhs, const T2 &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator/(const pixel_N<T1, dim> &lhs,
                                    const pixel_N<T2, dim> &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator/(const pixel_N<T1, dim> &lhs, const T2 &rhs)
  {
    pixel_N<T1, dim> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T, int dim>
  inline pixel_N<T, dim> operator&(const pixel_N<T, dim> &lhs,
                                   const pixel_N<T, dim> &rhs)
  {
    pixel_N<T, dim> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T, int dim>
  inline pixel_N<T, dim> operator&(const pixel_N<T, dim> &lhs, const T rhs)
  {
    pixel_N<T, dim> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T, int dim>
  inline pixel_N<T, dim> operator|(const pixel_N<T, dim> &lhs,
                                   const pixel_N<T, dim> &rhs)
  {
    pixel_N<T, dim> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }
  template <class T, int dim>
  inline pixel_N<T, dim> operator|(const pixel_N<T, dim> &lhs, const T rhs)
  {
    pixel_N<T, dim> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }

  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator+(const T2 &lhs, const pixel_N<T1, dim> &rhs)
  {
    return operator+(rhs, lhs);
  };
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator*(const T2 &lhs, const pixel_N<T1, dim> &rhs)
  {
    return rhs.operator*(lhs);
  };
  template <class T1, class T2, int dim>
  inline pixel_N<T1, dim> operator-(const T2 &lhs, const pixel_N<T1, dim> &rhs)
  {
    return operator-(rhs, lhs);
  };

  template <class T1, class T2>
  inline vector_N<T1> operator+(const vector_N<T1> &lhs,
                                const vector_N<T2> &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline vector_N<T1> operator+(const vector_N<T1> &lhs, const T2 &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp += rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline vector_N<T1> operator-(const vector_N<T1> &lhs,
                                const vector_N<T2> &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline vector_N<T1> operator-(const vector_N<T1> &lhs, const T2 &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp -= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline vector_N<T1> operator*(const vector_N<T1> &lhs, const T2 &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline vector_N<T1> operator*(const vector_N<T1> &lhs,
                                const vector_N<T2> &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp *= rhs;
    return t_temp;
  }
  template <class T1, class T2>
  inline vector_N<T1> operator/(const vector_N<T1> &lhs,
                                const vector_N<T2> &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline vector_N<T1> operator/(const vector_N<T1> &lhs, const T2 &rhs)
  {
    vector_N<T1> t_temp(lhs);
    t_temp /= rhs;
    return t_temp;
  }

  template <class T>
  inline vector_N<T> operator&(const vector_N<T> &lhs, const vector_N<T> &rhs)
  {
    vector_N<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline vector_N<T> operator&(const vector_N<T> &lhs, const T rhs)
  {
    vector_N<T> t_temp(lhs);
    t_temp &= rhs;
    return t_temp;
  }
  template <class T>
  inline vector_N<T> operator|(const pixel_3<T> &lhs, const vector_N<T> &rhs)
  {
    vector_N<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }
  template <class T>
  inline vector_N<T> operator|(const vector_N<T> &lhs, const T rhs)
  {
    vector_N<T> t_temp(lhs);
    t_temp |= rhs;
    return t_temp;
  }

  template <class T1, class T2>
  inline vector_N<T1> operator+(const T2 &lhs, const vector_N<T1> &rhs)
  {
    return operator+(rhs, lhs);
  };
  template <class T1, class T2>
  inline vector_N<T1> operator*(const T2 &lhs, const vector_N<T1> &rhs)
  {
    return rhs.operator*(lhs);
  };
  template <class T1, class T2>
  inline vector_N<T1> operator-(const T2 &lhs, const vector_N<T1> &rhs)
  {
    return operator-(rhs, lhs);
  };

  //------------------------------------------------------
  // Déplacé de morpheeTypes.hpp
  template <class T>
  std::ostream &operator<<(std::ostream &os, const pixel_3<T> &p)
  {
    os << p.channel1;
    os << " ";
    os << p.channel2;
    os << " ";
    os << p.channel3;
    return os;
  }

  template <>
  __MCom std::ostream &operator<<(std::ostream &os, const pixel_3<UINT8> &p);
  template <>
  __MCom std::ostream &operator<<(std::ostream &os, const pixel_3<INT8> &p);

  template <class T> std::istream &operator>>(std::istream &is, pixel_3<T> &p)
  {
    is >> p.channel1;
    is >> p.channel2;
    is >> p.channel3;
    return is;
  }
  template <>
  __MCom std::istream &operator>>(std::istream &is, pixel_3<UINT8> &p);
  template <>
  __MCom std::istream &operator>>(std::istream &is, pixel_3<INT8> &p);

  template <class T>
  std::ostream &operator<<(std::ostream &os, const pixel_4<T> &p)
  {
    os << p.channel1 << " " << p.channel2 << " " << p.channel3 << p.channel4;
    return os;
  }

  template <>
  __MCom std::ostream &operator<<(std::ostream &os, const pixel_4<UINT8> &p);
  template <>
  __MCom std::ostream &operator<<(std::ostream &os, const pixel_4<INT8> &p);

  template <class T> std::istream &operator>>(std::istream &is, pixel_4<T> &p)
  {
    is >> p.channel1;
    is >> p.channel2;
    is >> p.channel3;
    is >> p.channel4;
    return is;
  }
  template <>
  __MCom std::istream &operator>>(std::istream &is, pixel_4<UINT8> &p);
  template <>
  __MCom std::istream &operator>>(std::istream &is, pixel_4<INT8> &p);

  template <class T, int dim>
  std::ostream &operator<<(std::ostream &os, const pixel_N<T, dim> &p)
  {
    for (int i = 0; i < dim; i++)
      os << p[i];
    return os;
  }
  template <class T, int dim>
  std::istream &operator>>(std::istream &is, const pixel_N<T, dim> &p)
  {
    for (int i = 0; i < dim; i++)
      is >> p[i];
    return is;
  }
  //----------------------------------------------------------

  //! Dot product between two pixel_3
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_DotProduct(const pixel_3<T> &p1, const pixel_3<T> &p2)
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a1 = p1.channel1;
    a1 *= p2.channel1;
    acc_t a2 = p1.channel2;
    a2 *= p2.channel2;
    a1 += a2;
    a2 = p1.channel3;
    a2 *= p2.channel3;
    a1 += a2;

    return a1;
  }

  //! Dot product between two pixel_4
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_DotProduct(const pixel_4<T> &p1, const pixel_4<T> &p2)
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a1 = p1.channel1;
    a1 *= p2.channel1;
    acc_t a2 = p1.channel2;
    a2 *= p2.channel2;
    a1 += a2;
    a2 = p1.channel3;
    a2 *= p2.channel3;
    a1 += a2;
    a2 = p1.channel4;
    a2 *= p2.channel4;
    a1 += a2;

    return a1;
  }

  //! L1 (Manhattan) norm of a pixel_3
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Norm_L1(const pixel_3<T> &p) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a;
    if (p.channel1 >= DataTraits<T>::default_value::background())
      a = p.channel1;
    else
      a = -p.channel1;
    if (p.channel2 >= DataTraits<T>::default_value::background())
      a += p.channel2;
    else
      a -= p.channel2;
    if (p.channel3 >= DataTraits<T>::default_value::background())
      a += p.channel3;
    else
      a -= p.channel3;
    return a;
  }

  //! L1 (Manhattan) norm of a pixel_4
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Norm_L1(const pixel_4<T> &p) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a;
    if (p.channel1 >= DataTraits<T>::default_value::background())
      a = p.channel1;
    else
      a = -p.channel1;
    if (p.channel2 >= DataTraits<T>::default_value::background())
      a += p.channel2;
    else
      a -= p.channel2;
    if (p.channel3 >= DataTraits<T>::default_value::background())
      a += p.channel3;
    else
      a -= p.channel3;
    if (p.channel4 >= DataTraits<T>::default_value::background())
      a += p.channel4;
    else
      a -= p.channel4;
    return a;
  }

  //! L1 (Manhattan) distance - general version
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Distance_L1(const T &p1, const T &p2) throw()
  {
    return std::abs(p1 - p2);
  }

  //! L1 (Manhattan) distance between two pixel_3
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Distance_L1(const pixel_3<T> &p1, const pixel_3<T> &p2) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a1, a2;
    if (p1.channel1 > p2.channel1) {
      a1 = p1.channel1;
      a1 -= p2.channel1;
    } else {
      a1 = p2.channel1;
      a1 -= p1.channel1;
    }

    if (p1.channel2 > p2.channel2) {
      a2 = p1.channel2;
      a2 -= p2.channel2;
    } else {
      a2 = p2.channel2;
      a2 -= p1.channel2;
    }
    a1 += a2;

    if (p1.channel3 > p2.channel3) {
      a2 = p1.channel3;
      a2 -= p2.channel3;
    } else {
      a2 = p2.channel3;
      a2 -= p1.channel3;
    }
    a1 += a2;

    return a1;
  }

  //! L1 (Manhattan) distance between two pixel_4
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Distance_L1(const pixel_4<T> &p1, const pixel_4<T> &p2) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a1, a2;
    if (p1.channel1 > p2.channel1) {
      a1 = p1.channel1;
      a1 -= p2.channel1;
    } else {
      a1 = p2.channel1;
      a1 -= p1.channel1;
    }

    if (p1.channel2 > p2.channel2) {
      a2 = p1.channel2;
      a2 -= p2.channel2;
    } else {
      a2 = p2.channel2;
      a2 -= p1.channel2;
    }
    a1 += a2;

    if (p1.channel3 > p2.channel3) {
      a2 = p1.channel3;
      a2 -= p2.channel3;
    } else {
      a2 = p2.channel3;
      a2 -= p1.channel3;
    }
    a1 += a2;
    if (p1.channel4 > p2.channel4) {
      a2 = p1.channel4;
      a2 -= p2.channel4;
    } else {
      a2 = p2.channel4;
      a2 -= p1.channel4;
    }
    a1 += a2;
    return a1;
  }

  //! Squared L2 (euclidian) norm of a pixel_3 (no call to sqrt)
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Norm_L2_squared(const pixel_3<T> &p) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a1 = p.channel1;
    a1 *= a1;
    acc_t a2 = p.channel2;
    a2 *= a2;
    a1 += a2;
    a2 = p.channel3;
    a2 *= a2;
    a1 += a2;
    return a1;
  }
  //! Squared L2 (euclidian) norm of a pixel_4 (no call to sqrt)
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Norm_L2_squared(const pixel_4<T> &p) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;
    acc_t a1 = p.channel1;
    a1 *= a1;
    acc_t a2 = p.channel2;
    a2 *= a2;
    a1 += a2;
    a2 = p.channel3;
    a2 *= a2;
    a1 += a2;
    a2 = p.channel4;
    a2 *= a2;
    a1 += a2;
    return a1;
  }

  //! L2 (euclidian) norm of a pixel_3
  template <typename T>
  inline const typename DataTraits<T>::float_accumulator_type
  t_Norm_L2(const pixel_3<T> &p) throw()
  {
    return sqrt(static_cast<typename DataTraits<T>::float_accumulator_type>(
        t_Norm_L2_squared(p)));
  }
  //! L2 (euclidian) norm of a pixel_4
  template <typename T>
  inline const typename DataTraits<T>::float_accumulator_type
  t_Norm_L2(const pixel_4<T> &p) throw()
  {
    return sqrt(static_cast<typename DataTraits<T>::float_accumulator_type>(
        t_Norm_L2_squared(p)));
  }

  //! Squared L2 (euclidian) distance between two pixel_3
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Distance_L2_squared(const pixel_3<T> &p1, const pixel_3<T> &p2) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;

    acc_t a1 = p1.channel1;

    a1 -= p2.channel1;
    a1 *= a1;

    acc_t a2 = p1.channel2;
    a2 -= p2.channel2;
    a2 *= a2;
    a1 += a2; // Raffi : on évite la création inutile d'objets temporaires

    a2 = p1.channel3;
    a2 -= p2.channel3;
    a2 *= a2;
    a1 += a2;

    return a1;
  }

  //! Squared L2 (euclidian) distance between two pixel_4
  template <typename T>
  inline const typename DataTraits<T>::accumulator_type
  t_Distance_L2_squared(const pixel_4<T> &p1, const pixel_4<T> &p2) throw()
  {
    typedef typename DataTraits<T>::accumulator_type acc_t;

    acc_t a1 = p1.channel1;

    a1 -= p2.channel1;
    a1 *= a1;

    acc_t a2 = p1.channel2;
    a2 -= p2.channel2;
    a2 *= a2;
    a1 += a2; // Raffi : on évite la création inutile d'objets temporaires

    a2 = p1.channel3;
    a2 -= p2.channel3;
    a2 *= a2;
    a1 += a2;

    a2 = p1.channel4;
    a2 -= p2.channel4;
    a2 *= a2;
    a1 += a2;

    return a1;
  }

  //! L2 (euclidian) distance between two pixel_3
  template <typename T>
  inline const typename DataTraits<T>::float_accumulator_type
  t_Distance_L2(const pixel_3<T> &p1, const pixel_3<T> &p2) throw()
  {
    return sqrt(static_cast<typename DataTraits<T>::float_accumulator_type>(
        t_Distance_L2_squared(p1, p2)));
  }

  //! L2 (euclidian) distance between two pixel_4
  template <typename T>
  inline const typename DataTraits<T>::float_accumulator_type
  t_Distance_L2(const pixel_4<T> &p1, const pixel_4<T> &p2) throw()
  {
    return sqrt(static_cast<typename DataTraits<T>::float_accumulator_type>(
        t_Distance_L2_squared(p1, p2)));
  }

  //! L1 (Manhattan) distance  (functor version)
  template <class T1, class T2 = T1,
            class TOut = typename DataTraits<
                typename T1::value_type>::float_accumulator_type>
  struct s_Distance_L1 {
    typedef TOut value_type;
    inline value_type operator()(const T1 &t_1, const T2 &t_2) const throw()
    {
      return static_cast<TOut>(t_Distance_L1(t_1, t_2));
    }
  };

  //! L2 (euclidian) distance (functor version)
  template <class T1, class T2 = T1,
            class TOut = typename DataTraits<
                typename T1::value_type>::float_accumulator_type>
  struct s_Distance_L2 {
    typedef TOut value_type;
    inline value_type operator()(const T1 &t_1, const T2 &t_2) const throw()
    {
      return static_cast<TOut>(t_Distance_L2(t_1, t_2));
    }
  };

  //! Linfinity distance (supremum among the marginal channel differences)
  template <typename T>
  inline const T t_Distance_LInfinity(const pixel_3<T> &p1,
                                      const pixel_3<T> &p2)
  {
    T a1;
    if (p1.channel1 > p2.channel1) {
      a1 = p1.channel1;
      a1 -= p2.channel1;
    } else {
      a1 = p2.channel1;
      a1 -= p1.channel1;
    }

    T a2;
    if (p1.channel2 > p2.channel2) {
      a2 = p1.channel2;
      a2 -= p2.channel2;
    } else {
      a2 = p2.channel2;
      a2 -= p1.channel2;
    }
    a1 = std::max(a2, a1);

    if (p1.channel3 > p2.channel3) {
      a2 = p1.channel3;
      a2 -= p2.channel3;
    } else {
      a2 = p2.channel3;
      a2 -= p1.channel3;
    }
    a1 = std::max(a2, a1);

    return a1;
  }

  //! Linfinity distance (supremum among the marginal channel differences)
  //! (functor version)
  template <class T1, class T2 = T1, class TOut = T1>
  struct s_Distance_LInfinity {
    typedef TOut value_type;
    inline value_type operator()(const T1 &t_1, const T2 &t_2) const throw()
    {
      return static_cast<TOut>(t_Distance_LInfinity(t_1, t_2));
    }
  };

  namespace error
  {
    //! Tell whether the type T can be stored in one of
    //! variantInfo's fields. @see s_variantInfo and @see CVariant
    template <typename T> struct isTypeHandled {
    };

#ifndef MORPHEE_DOXYGEN_SKIP_TEMPLATE_SPECIALIZATION
    // Romain: il y a une boucle d'inclusions entre commonTypesOperator et
    // commonVariantInfo
    typedef struct s_variantInfo variantInfo;
    template <> struct isTypeHandled<variantInfo *> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<CVariant> {
      typedef bool isHandled;
    };

    template <> struct isTypeHandled<Label> {
      typedef bool isHandled;
    };

    template <> struct isTypeHandled<UINT8> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<UINT16> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<UINT32> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<INT8> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<INT16> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<INT32> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<F_SIMPLE> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<F_DOUBLE> {
      typedef bool isHandled;
    };

#ifdef HAVE_64BITS
    template <> struct isTypeHandled<INT64> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<UINT64> {
      typedef bool isHandled;
    };
#endif
    template <> struct isTypeHandled<std::string> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<ImageInterface *> {
      typedef bool isHandled;
    };

    template <> struct isTypeHandled<std::complex<UINT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<UINT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<UINT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<INT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<INT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<INT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<F_SIMPLE>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<std::complex<F_DOUBLE>> {
      typedef bool isHandled;
    };

    template <> struct isTypeHandled<pixel_3<UINT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<UINT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<UINT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<INT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<INT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<INT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<F_SIMPLE>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<F_DOUBLE>> {
      typedef bool isHandled;
    };

#ifdef HAVE_64BITS
    template <> struct isTypeHandled<pixel_3<INT64>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_3<UINT64>> {
      typedef bool isHandled;
    };
#endif

    template <> struct isTypeHandled<pixel_4<UINT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<UINT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<UINT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<INT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<INT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<INT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<F_SIMPLE>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<F_DOUBLE>> {
      typedef bool isHandled;
    };
#ifdef HAVE_64BITS
    template <> struct isTypeHandled<pixel_4<INT64>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<pixel_4<UINT64>> {
      typedef bool isHandled;
    };
#endif

    template <> struct isTypeHandled<vector_N<UINT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<UINT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<UINT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<INT8>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<INT16>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<INT32>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<F_SIMPLE>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<F_DOUBLE>> {
      typedef bool isHandled;
    };
#ifdef HAVE_64BITS
    template <> struct isTypeHandled<vector_N<INT64>> {
      typedef bool isHandled;
    };
    template <> struct isTypeHandled<vector_N<UINT64>> {
      typedef bool isHandled;
    };
#endif

#endif // MORPHEE_DOXYGEN_SKIP_TEMPLATE_SPECIALIZATION
  }    // namespace error

  /*!
   * @brief A type-conversion tool
   * @ingroup data_types
   *
   * This class is used to link an input type T to various related types and
   * values. For instance, the accumulator_type should be used when adding many
   * instances of type T (e.g. measuring the volume)) while the
   * float_accumulator_type should be used when adding these instances and
   * dividing afterwards (e.g. measuring an average).
   *
   * It can be used this way:
   * @code
   * // In a sum-like function:
   * typename DataTraits<T>::accumulator_type sum=0; // the "typename" is
   * necessary when in a template class/function
   *
   * // When initializing an image to a default value:
   * t_ImSetConstant( im, DataTraits<T>::default_value::background() );
   *
   * @endcode
   */
  template <class T> struct DataTraits {
    //! The dataCategory corresponding to this type (dtScalar, dtPixel3, etc)
    static const dataCategory dc = dtNone;
    //! The scalar type of this type (UINT8, 16, 32, F_SIMPLE/DOUBLE, etc)
    static const scalarDataType sdt = sdtNone;

    //! The type itself
    typedef T _self;
    //! A type suitable for adding instances of type T
    typedef T accumulator_type;
    //! A type suitable for adding instances of type T and dividing afterwards
    //! (average !)
    typedef T float_accumulator_type;

    // generation d'erreur à la compilation: propre
    typedef typename error::isTypeHandled<T>::isHandled do_compiletime_error;

    struct default_value {
      //! Default value for the background, used in ImLabel and similar tools.
      //! Usually it's zero (or equivalent).
      static const _self background() throw()
      {
        return _self();
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  }; // struct DataTraits

#ifndef MORPHEE_DOXYGEN_SKIP_TEMPLATE_SPECIALIZATION

  //****************** vecteur *******************
  template <class T> struct DataTraits<std::vector<T>> {
    static const dataCategory dc    = dtArray;
    static const scalarDataType sdt = sdtObject;

    //		typedef F_DOUBLE	float_accumulator_type;
    // typedef std::vector<morphee::variantInfo*>		accumulator_type;
  };

  //******************  pair *******************
  template <class K, class V> struct DataTraits<std::pair<K, V>> {
    static const dataCategory dc    = dtArray;
    static const scalarDataType sdt = sdtObject;

    //		typedef F_DOUBLE	float_accumulator_type;
    // typedef std::vector<morphee::variantInfo*>		accumulator_type;
  };
  //*******************  map  ********************
  template <class K, class V> struct DataTraits<std::map<K, V>> {
    // FIXME: c'est moche ce truc, mais je sais pas si Raffi aimerait qu'on
    // mette un dtMap en plus, c'est vrai qu'avec une map on crée un dtArray,
    // alors...
    static const dataCategory dc    = dtArray;
    static const scalarDataType sdt = sdtObject;
  };

  template <> struct DataTraits<std::string> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtSTR;
  };
  template <> struct DataTraits<ImageInterface *> {
    static const dataCategory dc    = dtImage;
    static const scalarDataType sdt = sdtObject;
  };

  //****************** variantInfo *******************
  /*	template<class T>
      struct DataTraits< variantInfo* >
    {
      static const dataCategory	dc	= dtVariant;
      static const scalarDataType sdt = sdtNone;

  //		typedef F_DOUBLE	float_accumulator_type;
      //typedef std::vector<morphee::variantInfo*>		accumulator_type;
    };*/

  template <> struct DataTraits<CVariant> {
    static const dataCategory dc    = dtCVariant;
    static const scalarDataType sdt = sdtObject;
  };
  //****************** Scalaires *******************

  // Romain: TODO
  //
#ifdef HAVE_64BITS
  typedef INT64 signed_accumulator_type;
  typedef UINT64 unsigned_accumulator_type;
#else
  typedef INT32 signed_accumulator_type;
  typedef UINT32 unsigned_accumulator_type;
#endif

  template <> struct DataTraits<UINT8> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtUINT8;

    typedef UINT8 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef unsigned_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;

    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };

  template <> struct DataTraits<INT8> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtINT8;

    typedef INT8 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef signed_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };
  template <> struct DataTraits<UINT16> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtUINT16;

    typedef UINT16 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef unsigned_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };
  template <> struct DataTraits<INT16> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtINT16;

    typedef INT16 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef signed_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };

  template <> struct DataTraits<UINT32> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtUINT32;

    typedef UINT32 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef unsigned_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };
  template <> struct DataTraits<INT32> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtINT32;

    typedef INT32 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef signed_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };

#ifdef HAVE_64BITS
  template <> struct DataTraits<UINT64> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtUINT64;

    typedef UINT64 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef unsigned_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };

  template <> struct DataTraits<INT64> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtINT64;

    typedef INT64 _self;
    typedef F_DOUBLE float_accumulator_type;
    typedef signed_accumulator_type accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };
#endif

  template <> struct DataTraits<F_SIMPLE> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtFloat;

    typedef F_SIMPLE _self;
    typedef F_DOUBLE accumulator_type;       // Careful, there is a trap ! :)
    typedef F_DOUBLE float_accumulator_type; // Careful, there is a trap ! :)
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };

  template <> struct DataTraits<F_DOUBLE> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtDouble;

    typedef F_DOUBLE _self;
    typedef F_DOUBLE accumulator_type;
    typedef F_DOUBLE float_accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(0);
      }
      static const accumulator_type max_value() throw()
      {
        return _self(std::numeric_limits<_self>::max());
      }
    };
  };

  // Label
  template <> struct DataTraits<Label> {
    static const dataCategory dc    = dtScalar;
    static const scalarDataType sdt = sdtLabel;

    typedef Label _self;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      static const _self background() throw()
      {
        return _self(labelCandidate);
      }
    };
  };

  //**************** Pixel_3 **********************

  template <> struct DataTraits<pixel_3<UINT8>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtUINT8;

    typedef pixel_3<UINT8> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<UINT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<pixel_3<INT8>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtINT8;

    typedef pixel_3<INT8> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<INT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_3<UINT16>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtUINT16;

    typedef pixel_3<UINT16> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<UINT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<pixel_3<INT16>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtINT16;

    typedef pixel_3<INT16> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<INT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_3<UINT32>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtUINT32;

    typedef pixel_3<UINT32> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<UINT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<pixel_3<INT32>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtINT32;

    typedef pixel_3<INT32> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<INT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

#ifdef HAVE_64BITS
  template <> struct DataTraits<pixel_3<UINT64>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtUINT64;

    typedef pixel_3<UINT64> _self;
    typedef pixel_3<F_DOUBLE>
        float_accumulator_type; // FIXME: 64 bits F_QUAD ? :)
    typedef pixel_3<DataTraits<INT64>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_3<INT64>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtINT64;

    typedef pixel_3<INT64> _self;
    typedef pixel_3<F_DOUBLE>
        float_accumulator_type; // FIXME: 64 bits F_QUAD ? :)
    typedef pixel_3<DataTraits<INT64>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

#endif

  template <> struct DataTraits<pixel_3<F_SIMPLE>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtFloat;

    typedef pixel_3<F_SIMPLE> _self;
    typedef pixel_3<F_SIMPLE> float_accumulator_type;
    typedef pixel_3<DataTraits<F_SIMPLE>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_3<F_DOUBLE>> {
    static const dataCategory dc    = dtPixel3;
    static const scalarDataType sdt = sdtDouble;

    typedef pixel_3<F_DOUBLE> _self;
    typedef pixel_3<F_DOUBLE> float_accumulator_type;
    typedef pixel_3<DataTraits<F_DOUBLE>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  //**************** Pixel_4 **********************
  template <> struct DataTraits<pixel_4<UINT8>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtUINT8;

    typedef pixel_4<UINT8> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<DataTraits<UINT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<pixel_4<INT8>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtINT8;

    typedef pixel_4<INT8> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<DataTraits<INT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_4<UINT16>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtUINT16;

    typedef pixel_4<UINT16> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<DataTraits<UINT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<pixel_4<INT16>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtINT16;

    typedef pixel_4<INT16> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<DataTraits<INT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_4<UINT32>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtUINT32;

    typedef pixel_4<UINT32> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<DataTraits<UINT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<pixel_4<INT32>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtINT32;

    typedef pixel_4<INT32> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<DataTraits<INT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

#ifdef HAVE_64BITS
  template <> struct DataTraits<pixel_4<UINT64>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtUINT64;

    typedef pixel_4<UINT64> _self;
    typedef pixel_4<F_DOUBLE>
        float_accumulator_type; // FIXME: 64 bits F_QUAD ? :)
    typedef pixel_4<DataTraits<INT64>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_4<INT64>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtINT64;

    typedef pixel_4<INT64> _self;
    typedef pixel_4<F_DOUBLE>
        float_accumulator_type; // FIXME: 64 bits F_QUAD ? :)
    typedef pixel_4<DataTraits<INT64>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
#endif

  template <> struct DataTraits<pixel_4<F_SIMPLE>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtFloat;

    typedef pixel_4<F_SIMPLE> _self;
    typedef pixel_4<F_SIMPLE> float_accumulator_type;
    typedef pixel_4<F_SIMPLE> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<pixel_4<F_DOUBLE>> {
    static const dataCategory dc    = dtPixel4;
    static const scalarDataType sdt = sdtDouble;

    typedef pixel_4<F_DOUBLE> _self;
    typedef pixel_4<F_DOUBLE> float_accumulator_type;
    typedef pixel_4<F_DOUBLE> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0), _value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  //******************* COMPLEX ********************
  template <> struct DataTraits<std::complex<UINT8>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtUINT8;

    typedef std::complex<UINT8> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<UINT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<std::complex<INT8>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtINT8;

    typedef std::complex<INT8> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<INT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<std::complex<UINT16>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtUINT16;

    typedef std::complex<UINT16> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<UINT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<std::complex<INT16>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtINT16;

    typedef std::complex<INT16> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<INT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<std::complex<UINT32>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtUINT32;

    typedef std::complex<UINT32> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<UINT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };
  template <> struct DataTraits<std::complex<INT32>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtINT32;

    typedef std::complex<INT32> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<INT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  /*
  template<> struct DataTraits< std::complex<UINT64> > {
    static const dataCategory dc = dtScalar;
    static const scalarDataType sdt= sdtUINT64;
  };
  template<> struct DataTraits< std::complex<INT64> > {
    static const dataCategory dc = dtScalar;
    static const scalarDataType sdt= sdtINT64;
  };
  */

  template <> struct DataTraits<std::complex<F_SIMPLE>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtFloat;

    typedef std::complex<F_SIMPLE> _self;
    typedef std::complex<F_SIMPLE> float_accumulator_type;
    typedef std::complex<DataTraits<F_SIMPLE>::accumulator_type>
        accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  template <> struct DataTraits<std::complex<F_DOUBLE>> {
    static const dataCategory dc    = dtCOMPLEX;
    static const scalarDataType sdt = sdtDouble;

    typedef std::complex<F_DOUBLE> _self;
    typedef std::complex<F_DOUBLE> float_accumulator_type;
    typedef std::complex<DataTraits<F_DOUBLE>::accumulator_type>
        accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      static const _self background() throw()
      {
        return _self(_value(0), _value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()),
            accumulator_type::value_type(
                std::numeric_limits<_self::value_type>::max()));
      }
    };
  };

  //******************* VECTOR ********************
  template <> struct DataTraits<vector_N<UINT8>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtUINT8;

    typedef vector_N<UINT8> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<UINT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };
  template <> struct DataTraits<vector_N<INT8>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtINT8;

    typedef vector_N<INT8> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<INT8>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };

  template <> struct DataTraits<vector_N<UINT16>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtUINT16;

    typedef vector_N<UINT16> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<UINT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };
  template <> struct DataTraits<vector_N<INT16>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtINT16;

    typedef vector_N<INT16> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<INT16>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };

  template <> struct DataTraits<vector_N<UINT32>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtUINT32;

    typedef vector_N<UINT32> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<UINT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };
  template <> struct DataTraits<vector_N<INT32>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtINT32;

    typedef vector_N<INT32> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<INT32>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };

  /*
  template<> struct DataTraits< vector_N<UINT64> > {
    static const dataCategory dc = dtVector;
    static const scalarDataType sdt= sdtUINT64;
  };
  template<> struct DataTraits< vector_N<INT64> > {
    static const dataCategory dc = dtVector;
    static const scalarDataType sdt= sdtINT64;
  };
  */

  template <> struct DataTraits<vector_N<F_SIMPLE>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtFloat;

    typedef vector_N<F_SIMPLE> _self;
    typedef vector_N<F_SIMPLE> float_accumulator_type;
    typedef vector_N<DataTraits<F_SIMPLE>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };

  template <> struct DataTraits<vector_N<F_DOUBLE>> {
    static const dataCategory dc    = dtVector;
    static const scalarDataType sdt = sdtDouble;

    typedef vector_N<F_DOUBLE> _self;
    typedef vector_N<F_DOUBLE> float_accumulator_type;
    typedef vector_N<DataTraits<F_DOUBLE>::accumulator_type> accumulator_type;
    typedef error::isTypeHandled<_self>::isHandled do_compiletime_error;
    struct default_value {
      typedef _self::value_type _value;
      /*static const _self background() throw()
      {
        return _self(_value(0),_value(0));
      }
      static const accumulator_type max_value() throw()
      {
        return accumulator_type(
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()),
          accumulator_type::value_type(std::numeric_limits<_self::value_type>::max()));
      }*/
    };
  };

#endif // MORPHEE_DOXYGEN_SKIP_TEMPLATE_SPECIALIZATION

  template <class __Im> struct ImageTraits {
  };

  // Romain: finalement c'est un struct, pas une fonction pour pouvoir faire des
  // spécialisations partielles. C'est parfois moche le C++, je sais...

  // Types scalaires:
  template <typename T> struct NameTraits {
    static const std::string name()
    {
      return std::string("Unsupported type: ") + std::string(typeid(T).name());
    }
    // const std::string name(){return
    // getScalarUnderstandableName(DataTraits<>::sdt);} ????
  };

#ifndef MORPHEE_DOXYGEN_SKIP_TEMPLATE_SPECIALIZATION
  template <> struct NameTraits<INT8> {
    static const std::string name()
    {
      return "INT8";
    }
  };
  template <> struct NameTraits<UINT8> {
    static const std::string name()
    {
      return "UINT8";
    }
  };
  template <> struct NameTraits<INT16> {
    static const std::string name()
    {
      return "INT16";
    }
  };
  template <> struct NameTraits<UINT16> {
    static const std::string name()
    {
      return "UINT16";
    }
  };
  template <> struct NameTraits<INT32> {
    static const std::string name()
    {
      return "INT32";
    }
  };
  template <> struct NameTraits<UINT32> {
    static const std::string name()
    {
      return "UINT32";
    }
  };
#ifdef HAVE_64BITS
  template <> struct NameTraits<INT64> {
    static const std::string name()
    {
      return "INT64";
    }
  };
  template <> struct NameTraits<UINT64> {
    static const std::string name()
    {
      return "UINT64";
    }
  };
#endif
  template <> struct NameTraits<F_SIMPLE> {
    static const std::string name()
    {
      return "F_SIMPLE";
    }
  };
  template <> struct NameTraits<F_DOUBLE> {
    static const std::string name()
    {
      return "F_DOUBLE";
    }
  };
  template <> struct NameTraits<CVariant> {
    static const std::string name()
    {
      return "CVariant";
    }
  };
  template <> struct NameTraits<Label> {
    static const std::string name()
    {
      return "Label";
    }
  };

  // Types composés:

  template <class T> struct NameTraits<std::complex<T>> {
    static const std::string name()
    {
      return "std::complex<" + NameTraits<T>::name() + std::string("> ");
    }
  };
  template <class T> struct NameTraits<pixel_3<T>> {
    static const std::string name()
    {
      return "pixel_3<" + NameTraits<T>::name() + std::string("> ");
    }
  };
  template <class T> struct NameTraits<pixel_4<T>> {
    static const std::string name()
    {
      return "pixel_4<" + NameTraits<T>::name() + std::string("> ");
    }
  };
  template <class T> struct NameTraits<vector_N<T>> {
    static const std::string name()
    {
      return "vector_N<" + NameTraits<T>::name() + std::string("> ");
    }
  };
  template <class T> struct NameTraits<std::vector<T>> {
    static const std::string name()
    {
      return "std::vector<" + NameTraits<T>::name() + std::string("> ");
    }
  };
#endif // MORPHEE_DOXYGEN_SKIP_TEMPLATE_SPECIALIZATION

} // namespace morphee

#ifndef MORPHEE_DOXYGEN_SKIP_INTERNAL_STUFF

// specialisation de std::less pour pixel_3 et pixel_4 avec un ordre
// lexicographique quelconque
namespace std
{
  template <class T>
  struct less<morphee::pixel_3<T>>
      : public binary_function<morphee::pixel_3<T>, morphee::pixel_3<T>, bool> {
    typedef morphee::pixel_3<T> Type;
    bool operator()(const Type &_Left, const Type &_Right) const
    {
      if (_Left.channel1 != _Right.channel1)
        return (_Left.channel1 < _Right.channel1);
      if (_Left.channel2 != _Right.channel2)
        return (_Left.channel2 < _Right.channel2);
      return (_Left.channel3 < _Right.channel3);
    }
  };

  template <class T>
  struct less<morphee::pixel_4<T>>
      : public binary_function<morphee::pixel_4<T>, morphee::pixel_4<T>, bool> {
    typedef morphee::pixel_4<T> Type;
    bool operator()(const Type &_Left, const Type &_Right) const
    {
      if (_Left.channel1 != _Right.channel1)
        return (_Left.channel1 < _Right.channel1);
      if (_Left.channel2 != _Right.channel2)
        return (_Left.channel2 < _Right.channel2);
      if (_Left.channel3 != _Right.channel3)
        return (_Left.channel3 < _Right.channel3);
      return (_Left.channel4 < _Right.channel4);
    }
  };
} // namespace std
#endif // MORPHEE_DOXYGEN_SKIP_INTERNAL_STUFF

#endif // _MORPHEE_TYPES_OPERATOR_HPP_
