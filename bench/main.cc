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

static void BM_Simple_Cow(benchmark::State& state)
{
    run_icp(state, "../tests/data_students/cow_ref.txt", "../tests/data_students/cow_tr1.txt");
}

BENCHMARK(BM_Simple_Cow);

// Run the benchmark
BENCHMARK_MAIN();