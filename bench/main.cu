#include <benchmark/benchmark.h>

#include "cpu/parser/parser.hh"
#include "cpu/icp/icp.hh"

#include "gpu_full/parser/parser.hh"
#include "gpu_full/icp/icp.hh"

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
        double error = 0;
        state.ResumeTiming();

        cpu::icp::icp_cpu(A, B, newP, error);
    }

    state.SetItemsProcessed(state.iterations());
}

void run_icp_gpu(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    gpu_full::parser::matrix_host_t A;
    bool ret = gpu_full::parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    gpu_full::parser::matrix_host_t B;
    ret = gpu_full::parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        gpu_full::parser::matrix_host_t newP;
        double error = 0;
        state.ResumeTiming();

        gpu_full::icp::icp_gpu(A, B, newP, error);
    }

    state.SetItemsProcessed(state.iterations());
}

static void BM_Cow_TR_1_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_2_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

static void BM_Cow_TR_1_GPU(benchmark::State& state)
{
    run_icp_gpu(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_2_GPU(benchmark::State& state)
{
    run_icp_gpu(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

/*static void BM_Horse_TR_1_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

static void BM_Horse_TR_2_CPU(benchmark::State& state)
{
    run_icp_cpu(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}*/

BENCHMARK(BM_Cow_TR_1_CPU);
BENCHMARK(BM_Cow_TR_2_CPU);
BENCHMARK(BM_Cow_TR_1_GPU);
BENCHMARK(BM_Cow_TR_2_GPU);
/*BENCHMARK(BM_Horse_TR_1);
BENCHMARK(BM_Horse_TR_2);*/

// Run the benchmark
BENCHMARK_MAIN();