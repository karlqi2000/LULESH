#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

template <class T>
class Vector_h;

template <class T>
class Vector_d;

// host vector
template <class T>
class Vector_h : public std::vector<T> {
 public:
  // Constructors
  Vector_h() {}
  inline Vector_h(int N) : std::vector<T>(N) {}
  inline Vector_h(int N, T v) : std::vector<T>(N, v) {}
  inline Vector_h(const Vector_h<T> &a) : std::vector<T>(a) {}
  inline Vector_h(const Vector_d<T> &a) : std::vector<T>(a.begin(), a.end()) {}

  template <typename OtherVector>
  inline void copy(const OtherVector &a) {
    this->assign(a.begin(), a.end());
  }

  inline Vector_h<T> &operator=(const Vector_h<T> &a) {
    copy(a);
    return *this;
  }
  inline Vector_h<T> &operator=(const Vector_d<T> &a) {
    copy(a);
    return *this;
  }

  inline T *raw() {
    if (bytes() > 0)
      return dpct::get_raw_pointer(this->data());
    else
      return 0;
  }

  inline const T *raw() const {
    if (bytes() > 0)
      return dpct::get_raw_pointer(this->data());
    else
      return 0;
  }

  inline size_t bytes() const { return this->size() * sizeof(T); }
};

// device vector
template <class T>
class Vector_d : public dpct::device_vector<T> {
 public:
  Vector_d() {}
  inline Vector_d(int N) : dpct::device_vector<T>(N) {}
  inline Vector_d(int N, T v) : dpct::device_vector<T>(N, v) {}
  inline Vector_d(const Vector_d<T> &a) : dpct::device_vector<T>(a) {}
  inline Vector_d(const Vector_h<T> &a) : dpct::device_vector<T>(a) {}

  template <typename OtherVector>
  inline void copy(const OtherVector &a) {
    this->assign(a.begin(), a.end());
  }

  inline Vector_d<T> &operator=(const Vector_d<T> &a) {
    copy(a);
    return *this;
  }
  inline Vector_d<T> &operator=(const Vector_h<T> &a) {
    copy(a);
    return *this;
  }

  inline T *raw() {
    if (bytes() > 0)
      return dpct::get_raw_pointer(this->data());
    else
      return 0;
  }

  inline const T *raw() const {
    if (bytes() > 0)
      return dpct::get_raw_pointer(this->data());
    else
      return 0;
  }

  inline size_t bytes() const { return this->size() * sizeof(T); }
};
