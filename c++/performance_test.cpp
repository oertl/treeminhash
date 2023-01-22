// #######################################
// # Copyright (C) 2020-2023 Otmar Ertl. #
// # All rights reserved.                #
// #######################################

#include "bitstream_random.hpp"
#include "weighted_minwise_hashing.hpp"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "../bagminhash/c++/weighted_minwise_hashing.hpp"
#include "../dartminhash/dartminhash.hpp"

using namespace std;

class RNGFunction {
  const uint64_t seed;

public:
  typedef tmh::WyrandBitStream RngType;

  RNGFunction(uint64_t seed) : seed(seed) {}

  tmh::WyrandBitStream operator()(uint64_t x) const {
    return tmh::WyrandBitStream(x, seed);
  }

  tmh::WyrandBitStream operator()(uint64_t x, uint64_t y) const {
    return tmh::WyrandBitStream(x, y, seed);
  }
};

template <typename H, typename RESULT_CONSUMER_TYPE>
void testCase(uint64_t dataSize, uint32_t hashSize, uint64_t numCycles,
              const vector<vector<pair<uint64_t, double>>> &testData, H &&h,
              const string &distributionLabel, const string &algorithmLabel,
              const RESULT_CONSUMER_TYPE &resultConsumer) {

  chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
  uint64_t resultAggregate = 0; // calculate some aggregate over entire result
                                // to prevent compiler optimizations
  for (const auto &data : testData) {
    assert(data.size() == dataSize);
    auto result = h(data);
    resultConsumer(resultAggregate, result);
  }
  chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();

  assert(numCycles = testData.size());

  double avgHashTime =
      chrono::duration_cast<chrono::duration<double>>(tEnd - tStart).count() /
      numCycles;

  cout << setprecision(numeric_limits<double>::max_digits10) << scientific;
  cout << algorithmLabel << ";";
  cout << numCycles << ";";
  cout << hashSize << ";";
  cout << dataSize << ";";
  cout << avgHashTime << ";";
  cout << distributionLabel << ";";
  cout << resultAggregate << endl << flush;
}

template <typename DIST, typename GEN>
void testDistribution(DIST &&dist, GEN &rng, const string &distributionLabel,
                      uint32_t hashSize, uint64_t dataSize,
                      uint64_t numCycles) {

  // generate test data
  vector<vector<pair<uint64_t, double>>> testDataPair(numCycles);
  for (uint64_t i = 0; i < numCycles; ++i) {

    vector<pair<uint64_t, double>> dPair(dataSize);
    for (uint64_t j = 0; j < dataSize; ++j) {
      uint64_t data = rng();
      double weight = dist(rng, dataSize);
      assert(weight <= numeric_limits<float>::max());
      dPair[j] = make_pair(data, weight);
    }
    testDataPair[i] = dPair;
  }

  mt19937_64 dartMinHashRng(UINT64_C(0x2538b4b37a60fb5f));
  testCase(
      dataSize, hashSize, numCycles, testDataPair,
      dmh::DartMinHash(dartMinHashRng, hashSize), distributionLabel,
      "DartMinHash",
      [](uint64_t &aggregate, const vector<pair<uint64_t, double>> &result) {
        for (const auto &x : result)
          aggregate += x.first;
      });

  testCase(
      dataSize, hashSize, numCycles, testDataPair,
      [hashSize](const vector<pair<uint64_t, double>> &x) {
        return bmh::bag_min_hash_2<bmh::FloatWeightDiscretization,
                                   bmh::XXHash64>(x, hashSize);
      },
      distributionLabel, "BagMinHash2",
      [](uint64_t &aggregate, const bmh::WeightedHashResult &result) {
        for (const auto &x : result.hashValues)
          aggregate += x;
      });

  if (dataSize <= 10000) {
    testCase(
        dataSize, hashSize, numCycles, testDataPair,
        [hashSize](const vector<pair<uint64_t, double>> &x) {
          return bmh::improved_consistent_weighted_hashing<bmh::XXHash64>(
              x, hashSize);
        },
        distributionLabel, "ICWS",
        [](uint64_t &aggregate, const bmh::WeightedHashResult &result) {
          for (const auto &x : result.hashValues)
            aggregate += x;
        });
  }

  testCase(
      dataSize, hashSize, numCycles, testDataPair,
      tmh::TreeMinHash<RNGFunction>(
          hashSize, RNGFunction(UINT64_C(0x4859cb1af8987719)), 0.5,
          std::numeric_limits<double>::max(), 0.9),
      distributionLabel, "TreeMinHash",
      [](uint64_t &aggregate, const vector<pair<uint64_t, double>> &result) {
        for (const auto &x : result)
          aggregate += x.first;
      });
}

int main(int argc, char *argv[]) {

  uint64_t numCycles = 100;

  assert(argc == 4);
  uint64_t seed = atol(argv[1]);
  uint32_t hashSize = atoi(argv[2]);
  uint64_t dataSize = atol(argv[3]);

  mt19937_64 rng(seed);

  auto exp1 = exponential_distribution<double>(1);
  auto exp1e30 = exponential_distribution<double>(1e30);
  auto exp1em30 = exponential_distribution<double>(1e-30);
  testDistribution([&exp1](auto &rng, uint64_t dataSize) { return exp1(rng); },
                   rng, "exp(1)", hashSize, dataSize, numCycles);
  testDistribution(
      [&exp1e30](auto &rng, uint64_t dataSize) { return exp1e30(rng); }, rng,
      "exp(1E30)", hashSize, dataSize, numCycles);
  testDistribution(
      [&exp1em30](auto &rng, uint64_t dataSize) { return exp1em30(rng); }, rng,
      "exp(1E-30)", hashSize, dataSize, numCycles);
  testDistribution(
      [&exp1](auto &rng, uint64_t dataSize) { return exp1(rng) / dataSize; },
      rng, "exp(n)", hashSize, dataSize, numCycles);
  testDistribution(
      [&exp1](auto &rng, uint64_t dataSize) {
        return exp1(rng) / (1e-6 * dataSize);
      },
      rng, "exp(n*1E-6)", hashSize, dataSize, numCycles);
  testDistribution(
      [&exp1](auto &rng, uint64_t dataSize) {
        return exp1(rng) / (1e6 * dataSize);
      },
      rng, "exp(n*1E6)", hashSize, dataSize, numCycles);

  return 0;
}
