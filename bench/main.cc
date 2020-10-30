#include <benchmark/benchmark.h>

#include "cpu/parser/parser.hh"
#include "cpu/icp/icp.hh"

void run_icp(benchmark::State& state, const std::string& file_1, const std::string& file_2)
{
    parser::matrix_t A;
    bool ret = parser::parse_file(file_1, A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    parser::matrix_t B;
    ret = parser::parse_file(file_2, B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return;
    }

    // Start the benchmarking
    for (auto _ : state)
    {
        state.PauseTiming();
        parser::matrix_t newP;
        double error = 0;
        state.ResumeTiming();

        icp::icp(A, B, newP, error);
    }

    state.SetItemsProcessed(state.iterations());
}

static void BM_Cow_TR_1(benchmark::State& state)
{
    run_icp(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

static void BM_Cow_TR_2(benchmark::State& state)
{
    run_icp(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr2.txt");
}

static void BM_Horse_TR_1(benchmark::State& state)
{
    run_icp(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr1.txt");
}

static void BM_Horse_TR_2(benchmark::State& state)
{
    run_icp(state, "../tests/data_students/horse_ref.txt", "../tests/data_students/horse_tr2.txt");
}

BENCHMARK(BM_Cow_TR_1);
BENCHMARK(BM_Cow_TR_2);
BENCHMARK(BM_Horse_TR_1);
BENCHMARK(BM_Horse_TR_2);

// Run the benchmark
BENCHMARK_MAIN();