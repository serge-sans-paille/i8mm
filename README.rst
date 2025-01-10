A Small Study of Byte Matrix Multiply
=====================================

It's quite common in machine learning operations to multiply a matrix of
unsigned byte by a matrix of signed byte. Don't ask me why, but that's the case.
And it turns out it's an interesting computation kernel to optimize, and that's
what we're going to discuss in this article.

Disclaimer 0: this is not a research paper, it's just a small study. It's very
likely that this work is well behind the state-of-the-art research, but it was
an interesting trip.

Disclaimer 1: I maintain https://github.com/mozilla/gemmology and
https://github.com/xtensor-stack/xsimd. The former is a port of
https://github.com/kpu/intgemm and the latter is an abstraction library for SIMD
operations. The former is based on the latter. So it's only natural we're going
to use those two as baseline and support library for this article.

Disclaimer 2: I'm running the experiment on my laptop, which has access to `AVX
VNNI
<https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI,_AVX-IFMA>`_
instructions, an instruction set designed to support byte matrix multiplication.


Naive Implementation
--------------------

.. code-block:: c++

    __attribute__((noinline))
    void NaiveMatMul(const uint8_t *inputMatrixA, const int8_t *inputMatrixB,
                     size_t rowsA, size_t width, size_t colsB, int32_t *output)
    {
      std::fill(output, output+ rowsA * colsB, 0);
      for (size_t rowIndex = 0; rowIndex < rowsA; ++rowIndex) {
        for (size_t k = 0; k < width; ++k) {
          for (size_t colIndex = 0; colIndex < colsB; ++colIndex) {
            output[rowIndex * colsB + colIndex] += inputMatrixA[rowIndex * width + k] * inputMatrixB[k * colsB + colIndex];
          }
        }
      }
    }

Compiling this code with ``clang++ -O2 -mavxvnni i8mm.cpp -o i8mm -I ../xsimd/include -I..`` leads to interesting assembly:

.. code-block::

    4013b0:       c4 e2 7d 21 4c 01 e8    vpmovsxbd -0x18(%rcx,%rax,1),%ymm1
    4013b7:       c4 e2 7d 21 54 01 f0    vpmovsxbd -0x10(%rcx,%rax,1),%ymm2
    4013be:       c4 e2 7d 21 5c 01 f8    vpmovsxbd -0x8(%rcx,%rax,1),%ymm3
    4013c5:       c4 e2 7d 21 24 01       vpmovsxbd (%rcx,%rax,1),%ymm4
    4013cb:       c5 f5 f5 c8             vpmaddwd %ymm0,%ymm1,%ymm1
    4013cf:       c5 ed f5 d0             vpmaddwd %ymm0,%ymm2,%ymm2
    4013d3:       c5 e5 f5 d8             vpmaddwd %ymm0,%ymm3,%ymm3
    4013d7:       c5 dd f5 e0             vpmaddwd %ymm0,%ymm4,%ymm4
    4013db:       c4 c1 75 fe 4c 82 a0    vpaddd -0x60(%r10,%rax,4),%ymm1,%ymm1
    4013e2:       c4 c1 6d fe 54 82 c0    vpaddd -0x40(%r10,%rax,4),%ymm2,%ymm2
    4013e9:       c4 c1 65 fe 5c 82 e0    vpaddd -0x20(%r10,%rax,4),%ymm3,%ymm3
    4013f0:       c4 c1 5d fe 24 82       vpaddd (%r10,%rax,4),%ymm4,%ymm4
    4013f6:       c4 c1 7e 7f 4c 82 a0    vmovdqu %ymm1,-0x60(%r10,%rax,4)
    4013fd:       c4 c1 7e 7f 54 82 c0    vmovdqu %ymm2,-0x40(%r10,%rax,4)
    401404:       c4 c1 7e 7f 5c 82 e0    vmovdqu %ymm3,-0x20(%r10,%rax,4)
    40140b:       c4 c1 7e 7f 24 82       vmovdqu %ymm4,(%r10,%rax,4)
    401411:       48 83 c0 20             add    $0x20,%rax
    401415:       48 39 c7                cmp    %rax,%rdi
    401418:       75 96                   jne    4013b0 <_Z11NaiveMatMulPKhPKammmPi+0x1a0>

Which basically means the compiler has been able to generate vector instructions
for this basic kernel. It's a pretty decent vectorization but it does not use
the ``vpdpbusd`` instruction from AVX VNNI. The whole goal of this article is to
use this instruction.


``vpdpbusd``
------------

This instruction is described by Intel as `such
<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=1866,4200,2685&avxnewtechs=AVX_VNNI>`_:

| Synopsis
|
| __m128i _mm_dpbusd_epi32 (__m128i src, __m128i a, __m128i b)
|
| #include <immintrin.h>
|
| Instruction: vpdpbusd xmm, xmm, xmm
|
| CPUID Flags: AVX_VNNI
|
| Description
|
| Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding signed 8-bit integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the corresponding 32-bit integer in src, and store the packed 32-bit results in dst.
|
| Operation
|
| FOR j := 0 to 3
|     tmp1.word := Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
|     tmp2.word := Signed(ZeroExtend16(a.byte[4*j+1]) * SignExtend16(b.byte[4*j+1]))
|     tmp3.word := Signed(ZeroExtend16(a.byte[4*j+2]) * SignExtend16(b.byte[4*j+2]))
|     tmp4.word := Signed(ZeroExtend16(a.byte[4*j+3]) * SignExtend16(b.byte[4*j+3]))
|     dst.dword[j] := src.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
| ENDFOR

The important part is that it sums up adjacent integers after point-to-point
multiplication, which is probably why the Clang compiler does not generate them.


Transposition and ``vpdpbusd``
------------------------------

A naive way to present the right layout so that ``vpdpbusd`` can be used is to
transpose the ``inputMatrixB``. It leads to the following code, using the
``gemmology`` abstraction for ``vpdpbusd``, namely ``maddw``, and the ``xsimd``
abstraction to sum each element of a vector register, namely ``reduce_add``:

.. code-block:: c++

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

This runs faster than the naive implementation, but it's spending a lot of time
in the transposition. Too much time, while we don't actually need a transposed
``inputMatrixB``, we can rely on a simpler layout, which should appear in the
following figures. The original scalar product looks like this:

                                                      b00 b10 b20 b30
                                                      b01 b11 b21 b31
                                                      b02 b12 b22 b32
                                                      b03 b13 b23 b33
                                                      b04 b14 b24 b34
                                                      b05 b15 b25 b35
                                                      b06 b16 b26 b36
                                                      b07 b17 b27 b37
                                                      b08 b18 b28 b38
                                                      b09 b19 b29 b39
                                                      b0A b1A b2A b3A
                                                      b0B b1B b2B b3B
                                                      b0C b1C b2C b3C
                                                      b0D b1D b2D b3D
                                                      b0E b1E b2E b3E
                                                      b0F b1F b2F b3F

    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aA aB aC aD aE aF

But to benefit from ``vpdpbusd`` we can do some partial transpose:

                                                      b00 b01 b02 b03
                                                      b10 b11 b12 b13
                                                      b20 b21 b22 b23
                                                      b30 b31 b32 b33

                                                      b04 b05 b06 b07
                                                      b14 b15 b16 b17
                                                      b24 b25 b26 b27
                                                      b34 b35 b36 b37

                                                      b08 b09 b0A b0B
                                                      b18 b19 b1A b1B
                                                      b28 b29 b2A b2B
                                                      b38 b39 b3A b3B

                                                      b0C b0D b0E b0F
                                                      b1C b1D b1E b1F
                                                      b2C b2D b2E b2F
                                                      b3C b3D b3E b3F

    a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aA aB aC aD aE aF

This leads to the slightly more complex following code, but it's faster and
that's what we want:

.. code-block:: c++

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

The inner assembly loop is very clean:

.. code-block::

    401ff0:       c4 82 7d 58 64 15 00    vpbroadcastd 0x0(%r13,%r10,1),%ymm4
    401ff7:       c4 c2 5d 50 19          {vex} vpdpbusd (%r9),%ymm4,%ymm3
    401ffc:       c4 82 5d 50 14 31       {vex} vpdpbusd (%r9,%r14,1),%ymm4,%ymm2
    402002:       c4 82 5d 50 0c 71       {vex} vpdpbusd (%r9,%r14,2),%ymm4,%ymm1
    402008:       c4 c2 5d 50 04 01       {vex} vpdpbusd (%r9,%rax,1),%ymm4,%ymm0
    40200e:       49 83 c2 04             add    $0x4,%r10
    402012:       49 01 c9                add    %rcx,%r9
    402015:       4d 39 fa                cmp    %r15,%r10
    402018:       72 d6                   jb     401ff0 <_Z18VecSatLayoutMatMulPKhPKammmPi+0x1b0>

It is actually more efficient if unrolled (look at all this free registers!), but that would make the example more
complex, so I'm not doing it here :-)

Small note on ``gemmology::maddw``
----------------------------------

``gemmology::maddw`` provides an abstraction over the instruction ``vpdpbusd``,
so that it can be used on machines with AVX VNNI as well as machines with Neon
with i8mm--easy, they provide the equivalent for arm architecture for registers
of 128 bits, or ssse3--more complex.

The implementation of ``gemmology::maddw`` is more complex because it basically
needs to do the widening, the temporary point-to-point multiplication and the
adjacent summation. Its implementation on ssse3 is the following:

.. code-block:: c++

    template <class Arch>
    inline xsimd::batch<int16_t, Arch>
    madd(xsimd::batch<uint8_t, Arch> x, xsimd::batch<int8_t, Arch> y,
         xsimd::kernel::requires_arch<xsimd::ssse3>) {
      return _mm_maddubs_epi16(x, y);
    }


    template <class Arch>
    inline xsimd::batch<int32_t, Arch>
    madd(xsimd::batch<int16_t, Arch> x, xsimd::batch<int16_t, Arch> y,
         xsimd::kernel::requires_arch<xsimd::sse2>) {
      return _mm_madd_epi16(x, y);
    }

    template <class Arch>
    inline xsimd::batch<int32_t, Arch>
    maddw(xsimd::batch<uint8_t, Arch> x, xsimd::batch<int8_t, Arch> y,
          xsimd::batch<int32_t, Arch> z,
          xsimd::kernel::requires_arch<xsimd::generic>) {
      return z + madd(xsimd::batch<int16_t, Arch>(1), madd(x, y, Arch{}), Arch{});
    }

Which is relatively fast *but* there is an intermediate sum of two ``int16_t``
integers done with saturation through ``_mm_maddubs_epi16``, with a potential
data loss (if one takes extreme values for the inputs). This can be circumvented
by masking the upper bit and doing ``maddw`` twice, as in ``maddw(x & 0xa0, y,
maddw(x & 0x7f, y, z))``. But of course this is slower :-).

This approximation is `relatively common in machine learning
<https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#m256_add_dpbusd_epi32>`_, and people have
been `bitten by this
<https://github.com/official-stockfish/Stockfish/pull/3261>`_.

People interested in operations on int8 in the context of meachine learning can
dive into this `OneAPI article
<https://oneapi-src.github.io/oneDNN/v1.5/dev_guide_int8_computations.html>`_.

Benchmarks
----------

The ``i8mm.cpp`` code associated to this article can be used to compare the
implementation mentioned in this article (I pruned some of the output not
discussed in this article):

.. list-table::

   * - naive mat mul
     - gemmology mat mul
     - vec mat mul
     - vec layout mat mul
   * - 177197 microseconds
     - 46322 microseconds
     - 118356 microseconds
     - 35333 microseconds


Interestingly the layout described in this article is more efficient than the one
used in ``gemmology`` \o/.

Looking at the assembly of the inner loop for gemmology, it's close to ours (but
unrolled):

.. code-block::

    4024cb:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)
    4024d0:       c4 41 7d 6f 04 2a       vmovdqa (%r10,%rbp,1),%ymm8
    4024d6:       c4 e2 3d 50 a4 ea 20    {vex} vpdpbusd -0xe0(%rdx,%rbp,8),%ymm8,%ymm4
    4024dd:       ff ff ff
    4024e0:       c4 e2 3d 50 bc ea 40    {vex} vpdpbusd -0xc0(%rdx,%rbp,8),%ymm8,%ymm7
    4024e7:       ff ff ff
    4024ea:       c4 e2 3d 50 ac ea 60    {vex} vpdpbusd -0xa0(%rdx,%rbp,8),%ymm8,%ymm5
    4024f1:       ff ff ff
    4024f4:       c4 e2 3d 50 74 ea 80    {vex} vpdpbusd -0x80(%rdx,%rbp,8),%ymm8,%ymm6
    4024fb:       c4 e2 3d 50 54 ea a0    {vex} vpdpbusd -0x60(%rdx,%rbp,8),%ymm8,%ymm2
    402502:       c4 e2 3d 50 5c ea c0    {vex} vpdpbusd -0x40(%rdx,%rbp,8),%ymm8,%ymm3
    402509:       c4 e2 3d 50 4c ea e0    {vex} vpdpbusd -0x20(%rdx,%rbp,8),%ymm8,%ymm1
    402510:       c4 e2 3d 50 04 ea       {vex} vpdpbusd (%rdx,%rbp,8),%ymm8,%ymm0
    402516:       48 83 c5 20             add    $0x20,%rbp
    40251a:       48 ff c8                dec    %rax
    40251d:       75 b1                   jne    4024d0 <_Z15GemmologyMatMulPKhPKammmPi+0x180>

The difference lies in the accumulation, which is straight forward in our case:

.. code-block:: c++

    40203a:       c4 c1 7d 7f 19          vmovdqa %ymm3,(%r9)
    40203f:       c4 c1 7d 7f 51 20       vmovdqa %ymm2,0x20(%r9)
    402045:       c4 c1 7d 7f 49 40       vmovdqa %ymm1,0x40(%r9)
    40204b:       c4 c1 7d 7f 41 60       vmovdqa %ymm0,0x60(%r9)

While it requires extra reduction and some data movement for gemmology:

.. code-block::

    40252e:       c4 e2 5d 02 e7          vphaddd %ymm7,%ymm4,%ymm4
    402533:       c4 e2 55 02 ee          vphaddd %ymm6,%ymm5,%ymm5
    402538:       c4 e2 5d 02 e5          vphaddd %ymm5,%ymm4,%ymm4
    40253d:       c4 e2 6d 02 d3          vphaddd %ymm3,%ymm2,%ymm2
    402542:       c4 e2 75 02 c0          vphaddd %ymm0,%ymm1,%ymm0
    402547:       c4 e2 6d 02 c0          vphaddd %ymm0,%ymm2,%ymm0
    40254c:       c4 e3 5d 46 c8 21       vperm2i128 $0x21,%ymm0,%ymm4,%ymm1
    402552:       c4 e3 5d 02 c0 f0       vpblendd $0xf0,%ymm0,%ymm4,%ymm0
    402558:       c5 f5 fe c0             vpaddd %ymm0,%ymm1,%ymm0
    40255c:       c5 fd 7f 00             vmovdqa %ymm0,(%rax)

Going Further
-------------

Gemmology inherits from intgemm the notion of prepared matrices, where the data
layout update is actually done in a routine, so that the computation could be
faster. For instance the above benchmark without the layout change becomes:


.. list-table::

   * - naive mat mul
     - gemmology mat mul
     - vec mat mul
     - vec layout mat mul
   * - 179300 microseconds
     - 46438 microseconds
     - 131477 microseconds
     - 34667 microseconds

So even without the data layout change, our approach is better than gemmology's,
cool.

There may be ways to include this research in gemmology, but gemmology provides
other operators, and those need to be made compatible with the new layout.
Future works?


Thanks
------

Special thanks to `Gian-Carlo Pascutto <https://github.com/gcp>`_ for the
detailed feedback on this article, and to `Sylvestre Ledru
<https://sylvestre.ledru.info/>`_ and `Tarek Ziade
<https://github.com/tarekziade>`_ for the proof reading.
