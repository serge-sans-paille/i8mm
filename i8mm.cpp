#include "gemmology/gemmology.h"
#include "xsimd/xsimd.hpp"
#include <cassert>
#include <chrono> // For timing
#include <cstdint>
#include <iomanip> // For formatted output
#include <iostream>
#include <vector>

using vuint8_t = xsimd::batch<uint8_t>;
using vint8_t = xsimd::batch<int8_t>;
using vint16_t = xsimd::batch<int16_t>;
using vint32_t = xsimd::batch<int32_t>;
using vuint32_t = xsimd::batch<uint32_t>;


/**
 * Naive implementation
 */
__attribute__((noinline))
void NaiveMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, int32_t *output) {
  std::fill(output, output+ rowsA * colsB, 0);
  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t k = 0; k < width; ++k) {
      for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
        output[rowIndex * colsB + colIndex] += inputMatrixA[rowIndex * width + k] * inputMatrixB[k * colsB + colIndex];
      }
    }
  }
}

/**
 * Sequential implementation
 */
__attribute__((noinline))
void TransposedMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, int32_t *output) {
  int8_t * transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);
  for (size_t k = 0; k < width; ++k) {
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      transposedB[colIndex * width + k] = inputMatrixB[k * colsB + colIndex];
    }
  }

  std::fill(output, output+ rowsA * colsB, 0);
  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      for (size_t k = 0; k < width; ++k) {
        output[rowIndex * colsB + colIndex] += inputMatrixA[rowIndex * width + k] * transposedB[colIndex * width + k];
      }
    }
  }

  free(transposedB);
}

__attribute__((noinline))
void VecSatMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, int32_t *output) {
  int8_t * transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);
  for (size_t k = 0; k < width; ++k) {
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      transposedB[colIndex * width + k] = inputMatrixB[k * colsB + colIndex];
    }
  }

  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex ) {
      vint32_t vacc = 0;
      for (size_t k = 0; k < width; k += vint8_t::size) {
        vacc = gemmology::maddw(vuint8_t::load_unaligned(&inputMatrixA[rowIndex * width + k]),
                                vint8_t::load_unaligned(&transposedB[colIndex * width + k]),
                                vacc);
      }
      output[rowIndex * colsB + colIndex] = reduce_add(vacc);
    }
  }

  free(transposedB);
}

__attribute__((noinline))
void VecMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, int32_t *output) {
  int8_t * transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);
  for (size_t k = 0; k < width; ++k) {
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
      transposedB[colIndex * width + k] = inputMatrixB[k * colsB + colIndex];
    }
  }

  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < colsB; ++colIndex ) {
      vint32_t vacc = 0;
      for (size_t k = 0; k < width; k += vint8_t::size) {
        vuint8_t vinputMatrixA = vuint8_t::load_unaligned(&inputMatrixA[rowIndex * width + k]);
        vint8_t vtransposedB = vint8_t::load_unaligned(&transposedB[colIndex * width + k]);
        vacc = gemmology::maddw(vinputMatrixA & vuint8_t(+0xA), vtransposedB, vacc);
        vacc = gemmology::maddw(vinputMatrixA & vuint8_t(~0xA), vtransposedB, vacc);
      }
      output[rowIndex * colsB + colIndex] = reduce_add(vacc);
    }
  }

  free(transposedB);
}

__attribute__((noinline))
void VecSatLayoutMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                        size_t rowsA, size_t width, size_t colsB, int32_t *output) {

  int8_t * transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);
  for (size_t k = 0; k < width; k += 4) {
    for (size_t colIndex = 0; colIndex < colsB; colIndex += 4 * vint32_t::size) {
      vint8_t vinputMatrixB0 = vint8_t::load_unaligned(&inputMatrixB[(k + 0) * colsB + colIndex]);
      vint8_t vinputMatrixB1 = vint8_t::load_unaligned(&inputMatrixB[(k + 1) * colsB + colIndex]);
      vint8_t vinputMatrixB2 = vint8_t::load_unaligned(&inputMatrixB[(k + 2) * colsB + colIndex]);
      vint8_t vinputMatrixB3 = vint8_t::load_unaligned(&inputMatrixB[(k + 3) * colsB + colIndex]);

      vint16_t vinputMatrixB_lo0 = xsimd::bit_cast<vint16_t>(zip_lo(vinputMatrixB0, vinputMatrixB1));
      vint16_t vinputMatrixB_lo1 = xsimd::bit_cast<vint16_t>(zip_lo(vinputMatrixB2, vinputMatrixB3));

      vint16_t vinputMatrixB_hi0 = xsimd::bit_cast<vint16_t>(zip_hi(vinputMatrixB0, vinputMatrixB1));
      vint16_t vinputMatrixB_hi1 = xsimd::bit_cast<vint16_t>(zip_hi(vinputMatrixB2, vinputMatrixB3));

      xsimd::bit_cast<vint8_t>(zip_lo(vinputMatrixB_lo0, vinputMatrixB_lo1)).store_unaligned(&transposedB[(k+0) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_hi(vinputMatrixB_lo0, vinputMatrixB_lo1)).store_unaligned(&transposedB[(k+1) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_lo(vinputMatrixB_hi0, vinputMatrixB_hi1)).store_unaligned(&transposedB[(k+2) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_hi(vinputMatrixB_hi0, vinputMatrixB_hi1)).store_unaligned(&transposedB[(k+3) * colsB + colIndex]);
    }
  }

  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < colsB; colIndex += 4 * vint32_t::size) {
      vint32_t vacc[4] = {};
      for (size_t k = 0; k < width; k += 4) {
        vuint8_t vinputMatrixA = xsimd::bitwise_cast<uint8_t>(vuint32_t(*(uint32_t*)(inputMatrixA + rowIndex * width + k)));
        vacc[0] = gemmology::maddw(
            vinputMatrixA,
            vint8_t::load_unaligned(&transposedB[(k + 0) * colsB + colIndex]),
            vacc[0]);
        vacc[1] = gemmology::maddw(
            vinputMatrixA,
            vint8_t::load_unaligned(&transposedB[(k + 1) * colsB + colIndex]),
            vacc[1]);
        vacc[2] = gemmology::maddw(
            vinputMatrixA,
            vint8_t::load_unaligned(&transposedB[(k + 2) * colsB + colIndex]),
            vacc[2]);
        vacc[3] = gemmology::maddw(
            vinputMatrixA,
            vint8_t::load_unaligned(&transposedB[(k + 3) * colsB + colIndex]),
            vacc[3]);
      }
      vacc[0].store_aligned(&output[rowIndex * colsB + colIndex + 0 * vint32_t::size]);
      vacc[1].store_aligned(&output[rowIndex * colsB + colIndex + 1 * vint32_t::size]);
      vacc[2].store_aligned(&output[rowIndex * colsB + colIndex + 2 * vint32_t::size]);
      vacc[3].store_aligned(&output[rowIndex * colsB + colIndex + 3 * vint32_t::size]);
    }
  }

  free(transposedB);
}

__attribute__((noinline))
void VecLayoutMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, int32_t *output) {

  int8_t * transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);
  for (size_t k = 0; k < width; k += 4) {
    for (size_t colIndex = 0; colIndex < colsB; colIndex += 4 * vint32_t::size) {
      vint8_t vinputMatrixB0 = vint8_t::load_unaligned(&inputMatrixB[(k + 0) * colsB + colIndex]);
      vint8_t vinputMatrixB1 = vint8_t::load_unaligned(&inputMatrixB[(k + 1) * colsB + colIndex]);
      vint8_t vinputMatrixB2 = vint8_t::load_unaligned(&inputMatrixB[(k + 2) * colsB + colIndex]);
      vint8_t vinputMatrixB3 = vint8_t::load_unaligned(&inputMatrixB[(k + 3) * colsB + colIndex]);

      vint16_t vinputMatrixB_lo0 = xsimd::bit_cast<vint16_t>(zip_lo(vinputMatrixB0, vinputMatrixB1));
      vint16_t vinputMatrixB_lo1 = xsimd::bit_cast<vint16_t>(zip_lo(vinputMatrixB2, vinputMatrixB3));

      vint16_t vinputMatrixB_hi0 = xsimd::bit_cast<vint16_t>(zip_hi(vinputMatrixB0, vinputMatrixB1));
      vint16_t vinputMatrixB_hi1 = xsimd::bit_cast<vint16_t>(zip_hi(vinputMatrixB2, vinputMatrixB3));

      xsimd::bit_cast<vint8_t>(zip_lo(vinputMatrixB_lo0, vinputMatrixB_lo1)).store_unaligned(&transposedB[(k+0) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_hi(vinputMatrixB_lo0, vinputMatrixB_lo1)).store_unaligned(&transposedB[(k+1) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_lo(vinputMatrixB_hi0, vinputMatrixB_hi1)).store_unaligned(&transposedB[(k+2) * colsB + colIndex]);
      xsimd::bit_cast<vint8_t>(zip_hi(vinputMatrixB_hi0, vinputMatrixB_hi1)).store_unaligned(&transposedB[(k+3) * colsB + colIndex]);
    }
  }

  for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
    for (size_t colIndex = 0; colIndex < colsB; colIndex += 4 * vint32_t::size) {
      vint32_t vacc[4] = {};
      for (size_t k = 0; k < width; k += 4) {
        vuint8_t vinputMatrixA = xsimd::bitwise_cast<uint8_t>(vuint32_t(*(uint32_t*)(inputMatrixA + rowIndex * width + k)));
        vint8_t vtransposedB0 = vint8_t::load_unaligned(&transposedB[(k + 0) * colsB + colIndex]);
        vint8_t vtransposedB1 = vint8_t::load_unaligned(&transposedB[(k + 1) * colsB + colIndex]);
        vint8_t vtransposedB2 = vint8_t::load_unaligned(&transposedB[(k + 2) * colsB + colIndex]);
        vint8_t vtransposedB3 = vint8_t::load_unaligned(&transposedB[(k + 3) * colsB + colIndex]);

        vacc[0] = gemmology::maddw(vinputMatrixA & vuint8_t(+0xA), vtransposedB0, vacc[0]);
        vacc[0] = gemmology::maddw(vinputMatrixA & vuint8_t(~0xA), vtransposedB0, vacc[0]);
        vacc[1] = gemmology::maddw(vinputMatrixA & vuint8_t(+0xA), vtransposedB1, vacc[1]);
        vacc[1] = gemmology::maddw(vinputMatrixA & vuint8_t(~0xA), vtransposedB1, vacc[1]);
        vacc[2] = gemmology::maddw(vinputMatrixA & vuint8_t(+0xA), vtransposedB2, vacc[2]);
        vacc[2] = gemmology::maddw(vinputMatrixA & vuint8_t(~0xA), vtransposedB2, vacc[2]);
        vacc[3] = gemmology::maddw(vinputMatrixA & vuint8_t(+0xA), vtransposedB3, vacc[3]);
        vacc[3] = gemmology::maddw(vinputMatrixA & vuint8_t(~0xA), vtransposedB3, vacc[3]);
      }
      vacc[0].store_aligned(&output[rowIndex * colsB + colIndex + 0 * vint32_t::size]);
      vacc[1].store_aligned(&output[rowIndex * colsB + colIndex + 1 * vint32_t::size]);
      vacc[2].store_aligned(&output[rowIndex * colsB + colIndex + 2 * vint32_t::size]);
      vacc[3].store_aligned(&output[rowIndex * colsB + colIndex + 3 * vint32_t::size]);
    }
  }

  free(transposedB);
}

/**
 * gemmology implementation
 */
__attribute__((noinline))
void GemmologyMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                 size_t rowsA, size_t width, size_t colsB, int32_t *output) {
  int8_t * transposedB;
  posix_memalign((void**)&transposedB, 64, width * colsB);

  gemmology::SequentialExecutionEngine engine;
  gemmology::PrepareBQuantized(inputMatrixB, transposedB, width, colsB);
  gemmology::Shift::Multiply(inputMatrixA, transposedB, rowsA, width, colsB,
                             gemmology::callbacks::Write(output), engine);

  free(transposedB);
}


int main(int argc, char **argv) {
  const size_t rowsA = 128, width=64, colsB=256;

  size_t count = 1000;

  uint8_t *inputMatrixA;
  int8_t *inputMatrixB;
  int32_t *output;

  posix_memalign((void**)&inputMatrixA, 64, rowsA * width);
  posix_memalign((void**)&inputMatrixB, 64, width * colsB);
  posix_memalign((void**)&output, 64, rowsA * colsB * sizeof(int32_t));

  for(size_t i = 0; i < rowsA; ++i)
    for(size_t j = 0; j < width; ++j)
      inputMatrixA[i * width + j] = i * width + j;

  for(size_t i = 0; i < width; ++i)
    for(size_t j = 0; j < colsB; ++j)
      inputMatrixB[i * colsB + j] = (i * colsB + j) % 255 - 127;

  if (argc == 2) {
    if (std::strcmp(argv[1], "--check") == 0) {
      int32_t *output_ref;

      posix_memalign((void **)&output_ref, 64, rowsA * colsB * sizeof(int32_t));

      NaiveMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output_ref);

      TransposedMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
      if (!std::equal(output, output + rowsA * colsB, output_ref)) {
        std::cerr << "failed comparison for TransposedMatMul\n";
        return 1;
      }

      VecSatMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
      if (!std::equal(output, output + rowsA * colsB, output_ref)) {
        std::cerr << "failed comparison for VecSatMatMul\n";
        return 1;
      }

      VecMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
      if (!std::equal(output, output + rowsA * colsB, output_ref)) {
        std::cerr << "failed comparison for VecMatMul\n";
        return 1;
      }

      VecSatLayoutMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
      if (!std::equal(output, output + rowsA * colsB, output_ref)) {
        std::cerr << "failed comparison for VecSatLayoutMatMul\n";
        return 1;
      }

      VecLayoutMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
      if (!std::equal(output, output + rowsA * colsB, output_ref)) {
        std::cerr << "failed comparison for VecLayoutMatMul\n";
        return 1;
      }

      GemmologyMatMul(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
      if (!std::equal(output, output + rowsA * colsB, output_ref)) {
        std::cerr << "failed comparison for GemmologyMatMul\n";
        return 1;
      }
      std::cerr << "ok\n";
      return 0;
    }
    return 1;
  }


  typedef void (*mm_t)(const uint8_t *, const int8_t *, size_t, size_t, size_t, int32_t *);
  struct { const char* name; mm_t matrix_multiplier;} matrix_multipliers[] = {
    {"           naive mat mul", NaiveMatMul},
    {"      transposed mat mul", TransposedMatMul},
    {"             vec mat mul", VecMatMul},
    {"      vec layout mat mul", VecLayoutMatMul},
    {"       gemmology mat mul", GemmologyMatMul},
    {"       vec + sat mat mul", VecSatMatMul},
    {"vec + sat layout mat mul", VecSatLayoutMatMul},
  };

  for(auto [name, matrix_multiplier] : matrix_multipliers) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < count; ++i)
      matrix_multiplier(inputMatrixA, inputMatrixB, rowsA, width, colsB, output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << name << ": " << duration << " microseconds" << std::endl;
  }

  return 0;
}
