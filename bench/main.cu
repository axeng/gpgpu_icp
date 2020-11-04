#include <benchmark/benchmark.h>

#include "cpu/parser/parser.hh"
#include "cpu/icp/icp.hh"

#include "gpu_1/parser/parser.hh"
#include "gpu_1/icp/icp.hh"

#include "gpu_2/parser/parser.hh"
#include "gpu_2/icp/icp.hh"

#include "gpu_3/parser/parser.hh"
#include "gpu_3/icp/icp.hh"

#include "gpu_final/parser/parser.hh"
#include "gpu_final/icp/icp.hh"

void run_icp_cpu(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    cpu::parser::matrix_t A;
    bool ret = cpu::parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    cpu::parser::matrix_t B;
    ret = cpu::parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        cpu::parser::matrix_t newP;
        float error = 0;
        state.ResumeTiming();

        cpu::icp::icp_cpu(A, B, newP, error);
    }

    state.counters["frame_rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

void run_icp_gpu_1(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    gpu_1::parser::matrix_host_t A;
    bool ret = gpu_1::parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    gpu_1::parser::matrix_host_t B;
    ret = gpu_1::parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        gpu_1::parser::matrix_host_t newP;
        float error = 0;
        state.ResumeTiming();

        gpu_1::icp::icp_gpu(A, B, newP, error);
    }

    state.counters["frame_rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

void run_icp_gpu_2(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    gpu_2::parser::matrix_host_t A;
    bool ret = gpu_2::parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    gpu_2::parser::matrix_host_t B;
    ret = gpu_2::parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        gpu_2::parser::matrix_host_t newP;
        float error = 0;
        state.ResumeTiming();

        gpu_2::icp::icp_gpu(A, B, newP, error);
    }

    state.counters["frame_rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

void run_icp_gpu_3(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    gpu_3::parser::matrix_host_t A;
    bool ret = gpu_3::parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    gpu_3::parser::matrix_host_t B;
    ret = gpu_3::parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        gpu_3::parser::matrix_host_t newP;
        float error = 0;
        state.ResumeTiming();

        gpu_3::icp::icp_gpu(A, B, newP, error);
    }

    state.counters["frame_rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

void run_icp_gpu_final(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    gpu_final::parser::matrix_host_t A;
    bool ret = gpu_final::parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    gpu_final::parser::matrix_host_t B;
    ret = gpu_final::parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        gpu_final::parser::matrix_host_t newP;
        float error = 0;
        state.ResumeTiming();

        gpu_final::icp::icp_gpu(A, B, newP, error);
    }

    state.counters["frame_rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

// COW TR 1
static void BM_Cow_TR_1_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_1_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_1_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_1_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_1_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

// COW TR 2
static void BM_Cow_TR_2_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

static void BM_Cow_TR_2_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

static void BM_Cow_TR_2_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

static void BM_Cow_TR_2_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

static void BM_Cow_TR_2_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

// HORSE TR 1
static void BM_Horse_TR_1_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

static void BM_Horse_TR_1_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

static void BM_Horse_TR_1_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

static void BM_Horse_TR_1_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

static void BM_Horse_TR_1_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

// HORSE TR 2
static void BM_Horse_TR_2_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}

static void BM_Horse_TR_2_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}

static void BM_Horse_TR_2_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}

static void BM_Horse_TR_2_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}

static void BM_Horse_TR_2_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}

// BUN 045
static void BM_BUN_045_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun045.txt");
}

static void BM_BUN_045_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun045.txt");
}

static void BM_BUN_045_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun045.txt");
}

static void BM_BUN_045_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun045.txt");
}

static void BM_BUN_045_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun045.txt");
}

// BUN 180
static void BM_BUN_180_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun180.txt");
}

static void BM_BUN_180_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun180.txt");
}

static void BM_BUN_180_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun180.txt");
}

static void BM_BUN_180_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun180.txt");
}

static void BM_BUN_180_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun180.txt");
}

// BUN 270
static void BM_BUN_270_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun270.txt");
}

static void BM_BUN_270_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun270.txt");
}

static void BM_BUN_270_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun270.txt");
}

static void BM_BUN_270_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun270.txt");
}

static void BM_BUN_270_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun270.txt");
}

// BUN 315
static void BM_BUN_315_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun315.txt");
}

static void BM_BUN_315_GPU_1(benchmark::State& state)
{
    run_icp_gpu_1(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun315.txt");
}

static void BM_BUN_315_GPU_2(benchmark::State& state)
{
    run_icp_gpu_2(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun315.txt");
}

static void BM_BUN_315_GPU_3(benchmark::State& state)
{
    run_icp_gpu_3(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun315.txt");
}

static void BM_BUN_315_GPU_FINAL(benchmark::State& state)
{
    run_icp_gpu_final(state, "../tests/data_students/bun000.txt", "../tests/data_students/bun315.txt");
}

BENCHMARK(BM_Cow_TR_1_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_1_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_1_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_1_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_1_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_Cow_TR_2_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_2_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_2_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_2_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Cow_TR_2_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_Horse_TR_1_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_1_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_1_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_1_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_1_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_Horse_TR_2_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_2_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_2_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_2_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Horse_TR_2_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_BUN_045_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_045_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_045_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_045_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_045_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_BUN_180_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_180_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_180_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_180_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_180_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_BUN_270_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_270_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_270_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_270_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_270_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_BUN_315_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_315_GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_315_GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_315_GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BUN_315_GPU_FINAL)->Unit(benchmark::kMillisecond)->UseRealTime();
/*BENCHMARK(BM_Horse_TR_1);
BENCHMARK(BM_Horse_TR_2);*/

// Run the benchmark
BENCHMARK_MAIN();