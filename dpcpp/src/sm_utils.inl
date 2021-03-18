#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR "l"
#else
#define __PTR "r"
#endif

#if DPCT_COMPATIBILITY_TEMP >= 700
#define FULL_MASK 0xffffffff
#endif

namespace utils {

// ====================================================================================================================
// Atomics.
// ====================================================================================================================

static __dpct_inline__ void atomic_add(float *address, float value) {
  /*
  DPCT1039:0: The generated code assumes that "address" points to the global
  memory address space. If it points to a local memory address space, replace
  "dpct::atomic_fetch_add" with "dpct::atomic_fetch_add<float,
  sycl::access::address_space::local_space>".
  */
  dpct::atomic_fetch_add(address, value);
}

static __dpct_inline__ void atomic_add(double *address, double value) {
  unsigned long long *address_as_ull = (unsigned long long *)address;
  unsigned long long old = sycl::detail::bit_cast<long long>(address[0]),
                     assumed;
  do {
    assumed = old;
    /*
    DPCT1039:1: The generated code assumes that "address_as_ull" points to the
    global memory address space. If it points to a local memory address space,
    replace "dpct::atomic_compare_exchange_strong" with
    "dpct::atomic_compare_exchange_strong<unsigned long long,
    sycl::access::address_space::local_space>".
    */
    old = dpct::atomic_compare_exchange_strong(
        address_as_ull, assumed,
        (unsigned long long)(sycl::detail::bit_cast<long long>(
            value + sycl::detail::bit_cast<double>(assumed))));
  } while (assumed != old);
}

// ====================================================================================================================
// Bit tools.
// ====================================================================================================================

static __dpct_inline__ int bfe(int src, int num_bits) {
  unsigned mask;
  /*
  DPCT1053:2: Migration of device assembly code is not supported.
  */
  asm("bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits));
  return mask;
}

static __dpct_inline__ int bfind(int src) {
  int msb;
  /*
  DPCT1053:3: Migration of device assembly code is not supported.
  */
  asm("bfind.u32 %0, %1;" : "=r"(msb) : "r"(src));
  return msb;
}

static __dpct_inline__ int bfind(unsigned long long src) {
  int msb;
  /*
  DPCT1053:4: Migration of device assembly code is not supported.
  */
  asm("bfind.u64 %0, %1;" : "=r"(msb) : "l"(src));
  return msb;
}

static __dpct_inline__ unsigned long long brev(unsigned long long src) {
  unsigned long long rev;
  /*
  DPCT1053:5: Migration of device assembly code is not supported.
  */
  asm("brev.b64 %0, %1;" : "=l"(rev) : "l"(src));
  return rev;
}

// ====================================================================================================================
// Warp tools.
// ====================================================================================================================

static __dpct_inline__ int lane_id() {
  int id;
  /*
  DPCT1053:6: Migration of device assembly code is not supported.
  */
  asm("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

static __dpct_inline__ int lane_mask_lt() {
  int mask;
  /*
  DPCT1053:7: Migration of device assembly code is not supported.
  */
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

static __dpct_inline__ int warp_id(sycl::nd_item<3> item_ct1) {
  return item_ct1.get_local_id(2) >> 5;
}

// ====================================================================================================================
// Loads.
// ====================================================================================================================

enum Ld_mode { LD_AUTO = 0, LD_CA, LD_CG, LD_TEX, LD_NC };

template <Ld_mode Mode>
struct Ld {};

template <>
struct Ld<LD_AUTO> {
  template <typename T>
  static __dpct_inline__ T load(const T *ptr) {
    return *ptr;
  }
};

template <>
struct Ld<LD_NC> {
  template <typename T>
  static __dpct_inline__ T load(const T *ptr) {
    return __ldg(ptr);
  }
};

// ====================================================================================================================
// Shuffle.
// ====================================================================================================================
static __dpct_inline__ float shfl(float r, int lane, int warp_size,
                                  sycl::nd_item<3> item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  return __shfl_sync(FULL_MASK, r, lane, warp_size);
#elif DPCT_COMPATIBILITY_TEMP >= 300
  return item_ct1.get_sub_group().shuffle(r, lane);
#else
  return 0.0f;
#endif
}

static __dpct_inline__ double shfl(double r, int lane, int warp_size,
                                   sycl::nd_item<3> item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  int hi = __shfl_sync(FULL_MASK, __double2hiint(r), lane, warp_size);
  int lo = __shfl_sync(FULL_MASK, __double2loint(r), lane, warp_size);
  return __hiloint2double(hi, lo);
#elif DPCT_COMPATIBILITY_TEMP >= 300
  /*
  DPCT1017:19: The  call is used instead of the __double2hiint call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  int hi = item_ct1.get_sub_group().shuffle(dpct::cast_double_to_int(r), lane);
  /*
  DPCT1017:20: The  call is used instead of the __double2loint call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  int lo = item_ct1.get_sub_group().shuffle(dpct::cast_double_to_int(r, false),
                                            lane);
  /*
  DPCT1017:21: The  call is used instead of the __hiloint2double call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  return dpct::cast_ints_to_double(hi, lo);
#else
  return 0.0;
#endif
}

static __dpct_inline__ float shfl_xor(float r, int mask, int warp_size,
                                      sycl::nd_item<3> item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  return __shfl_xor_sync(FULL_MASK, r, mask, warp_size);
#elif DPCT_COMPATIBILITY_TEMP >= 300
  return item_ct1.get_sub_group().shuffle_xor(r, mask);
#else
  return 0.0f;
#endif
}

static __dpct_inline__ double shfl_xor(double r, int mask, int warp_size,
                                       sycl::nd_item<3> item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  int hi = __shfl_xor_sync(__double2hiint(r), mask, warp_size);
  int lo = __shfl_xor_sync(__double2loint(r), mask, warp_size);
  return __hiloint2double(hi, lo);
#elif DPCT_COMPATIBILITY_TEMP >= 300
  /*
  DPCT1017:22: The  call is used instead of the __double2hiint call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  int hi =
      item_ct1.get_sub_group().shuffle_xor(dpct::cast_double_to_int(r), mask);
  /*
  DPCT1017:23: The  call is used instead of the __double2loint call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  int lo = item_ct1.get_sub_group().shuffle_xor(
      dpct::cast_double_to_int(r, false), mask);
  /*
  DPCT1017:24: The  call is used instead of the __hiloint2double call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  return dpct::cast_ints_to_double(hi, lo);
#else
  return 0.0;
#endif
}

static __dpct_inline__ float shfl_down(float r, int offset,
                                       sycl::nd_item<3> item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  return __shfl_down_sync(FULL_MASK, r, offset);
#elif DPCT_COMPATIBILITY_TEMP >= 300
  return item_ct1.get_sub_group().shuffle_down(r, offset);
#else
  return 0.0f;
#endif
}

static __dpct_inline__ double shfl_down(double r, int offset,
                                        sycl::nd_item<3> item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  int hi = __shfl_down_sync(FULL_MASK, __double2hiint(r), offset);
  int lo = __shfl_down_sync(FULL_MASK, __double2loint(r), offset);
  return __hiloint2double(hi, lo);
#elif DPCT_COMPATIBILITY_TEMP >= 300
  /*
  DPCT1017:25: The  call is used instead of the __double2hiint call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  int hi = item_ct1.get_sub_group().shuffle_down(dpct::cast_double_to_int(r),
                                                 offset);
  /*
  DPCT1017:26: The  call is used instead of the __double2loint call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  int lo = item_ct1.get_sub_group().shuffle_down(
      dpct::cast_double_to_int(r, false), offset);
  /*
  DPCT1017:27: The  call is used instead of the __hiloint2double call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  return dpct::cast_ints_to_double(hi, lo);
#else
  return 0.0;
#endif
}

// ====================================================================================================================
// Warp-level reductions.
// ====================================================================================================================

struct Add {
  template <typename Value_type>
  static __dpct_inline__ Value_type eval(Value_type x, Value_type y) {
    return x + y;
  }
};

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 300

template <int NUM_THREADS_PER_ITEM, int WARP_SIZE>
struct Warp_reduce_pow2 {
  template <typename Operator, typename Value_type>
  static __inline__ Value_type execute(Value_type x,
                                       sycl::nd_item<3> item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= NUM_THREADS_PER_ITEM; mask >>= 1)
      x = Operator::eval(x, shfl_xor(x, mask));
    return x;
  }
};

template <int NUM_THREADS_PER_ITEM, int WARP_SIZE>
struct Warp_reduce_linear {
  template <typename Operator, typename Value_type>
  static __inline__ Value_type execute(Value_type x,
                                       sycl::nd_item<3> item_ct1) {
    const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
    int my_lane_id = utils::lane_id();
#pragma unroll
    for (int i = 1; i < NUM_STEPS; ++i) {
      Value_type y = shfl_down(x, i * NUM_THREADS_PER_ITEM, item_ct1);
      if (my_lane_id < NUM_THREADS_PER_ITEM) x = Operator::eval(x, y);
    }
    return x;
  }
};

#else

template <int NUM_THREADS_PER_ITEM, int WARP_SIZE>
struct Warp_reduce_pow2 {
  template <typename Operator, typename Value_type>
  static __device__ __inline__ Value_type execute(volatile Value_type *smem,
                                                  Value_type x) {
    int my_lane_id = utils::lane_id();
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset >= NUM_THREADS_PER_ITEM;
         offset >>= 1)
      if (my_lane_id < offset)
        smem[threadIdx.x] = x = Operator::eval(x, smem[threadIdx.x + offset]);
    return x;
  }
};

template <int NUM_THREADS_PER_ITEM, int WARP_SIZE>
struct Warp_reduce_linear {
  template <typename Operator, typename Value_type>
  static __device__ __inline__ Value_type execute(volatile Value_type *smem,
                                                  Value_type x) {
    const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
    int my_lane_id = utils::lane_id();
#pragma unroll
    for (int i = 1; i < NUM_STEPS; ++i)
      if (my_lane_id < NUM_THREADS_PER_ITEM)
        smem[threadIdx.x] = x =
            Operator::eval(x, smem[threadIdx.x + i * NUM_THREADS_PER_ITEM]);
    return x;
  }
};

#endif

// ====================================================================================================================

template <int NUM_THREADS_PER_ITEM, int WARP_SIZE = 32>
struct Warp_reduce : public Warp_reduce_pow2<NUM_THREADS_PER_ITEM, WARP_SIZE> {
};

template <int WARP_SIZE>
struct Warp_reduce<3, WARP_SIZE> : public Warp_reduce_linear<3, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<4, WARP_SIZE> : public Warp_reduce_linear<4, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<5, WARP_SIZE> : public Warp_reduce_linear<5, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<6, WARP_SIZE> : public Warp_reduce_linear<6, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<7, WARP_SIZE> : public Warp_reduce_linear<7, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<9, WARP_SIZE> : public Warp_reduce_linear<9, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<10, WARP_SIZE> : public Warp_reduce_linear<10, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<11, WARP_SIZE> : public Warp_reduce_linear<11, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<12, WARP_SIZE> : public Warp_reduce_linear<12, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<13, WARP_SIZE> : public Warp_reduce_linear<13, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<14, WARP_SIZE> : public Warp_reduce_linear<14, WARP_SIZE> {};

template <int WARP_SIZE>
struct Warp_reduce<15, WARP_SIZE> : public Warp_reduce_linear<15, WARP_SIZE> {};

// ====================================================================================================================

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 300

template <int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type>
static __dpct_inline__ Value_type warp_reduce(Value_type x) {
  return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>(x);
}

#else

template <int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type>
static __device__ __forceinline__ Value_type
warp_reduce(volatile Value_type *smem, Value_type x) {
  return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>(smem, x);
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace utils
