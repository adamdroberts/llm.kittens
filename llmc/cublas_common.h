/*
cublas_common.h — kept as a header-existence shim for source-compatibility with
llm.c. llm.kittens v1 does NOT use cuBLAS or cuBLASLt: every matmul goes through
the ThunderKittens H100 GEMM kernel in llmc/tk/gemm_h100.cuh, called from
llmc/matmul.cuh. There is intentionally nothing here.
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H
// (intentionally empty)
#endif // CUBLAS_COMMON_H
