// #######################################
// # Copyright (C) 2020-2023 Otmar Ertl. #
// # All rights reserved.                #
// #######################################

#include "bitstream_random.hpp"
#include "data_generation.hpp"
#include "weighted_minwise_hashing.hpp"

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

template <typename S>
void testCase(const tmh::Weights &w, const string &algorithmDescription,
              uint32_t m, uint64_t numIterations, S &&hashFunctionSupplier) {

  const auto dataSizes = w.getSizes();

  uint64_t seedSize = 256;

  // values from random.org
  seed_seq initialSeedSequence{UINT32_C(0xc9c5e41d), UINT32_C(0x14b77d0b),
                               UINT32_C(0x78ff862e), UINT32_C(0x51d8975e),
                               UINT32_C(0xe6dc72f6), UINT32_C(0x5f64d296),
                               UINT32_C(0xe2946980), UINT32_C(0xea8615eb)};

  mt19937 initialRng(initialSeedSequence);
  vector<uint32_t> seeds(numIterations * seedSize);
  generate(seeds.begin(), seeds.end(), initialRng);

  vector<uint32_t> numEquals(m + 1);

#pragma omp parallel
  {
    auto h = hashFunctionSupplier(m);

#pragma omp for
    for (uint64_t i = 0; i < numIterations; ++i) {

      seed_seq seedSequence(seeds.begin() + i * seedSize,
                            seeds.begin() + (i + 1) * seedSize);
      mt19937_64 rng(seedSequence);

      const pair<vector<pair<uint64_t, double>>, vector<pair<uint64_t, double>>>
          data = generateData(rng, w);

      const vector<pair<uint64_t, double>> &d1 = get<0>(data);
      const vector<pair<uint64_t, double>> &d2 = get<1>(data);

      assert(get<0>(dataSizes) == d1.size());
      assert(get<1>(dataSizes) == d2.size());

      auto h1 = h(d1);
      auto h2 = h(d2);

      uint32_t numEqual = 0;
      for (uint32_t j = 0; j < m; ++j) {
        if (h1[j] == h2[j]) {
          numEqual += 1;
        }
      }

#pragma omp atomic
      numEquals[numEqual] += 1;
    }
  }

  cout << setprecision(numeric_limits<double>::max_digits10) << scientific;
  cout << w.getJw() << ";";
  cout << w.getJn() << ";";
  cout << w.getJp() << ";";
  cout << algorithmDescription << ";";
  cout << w.getLatexDescription() << ";";
  cout << numIterations << ";";
  cout << m << ";";
  cout << w.getId() << ";";
  cout << w.allWeightsZeroOrOne() << ";";
  cout << get<0>(dataSizes) << ";";
  cout << get<1>(dataSizes) << ";";
  cout << get<2>(dataSizes) << endl;

  bool first = true;
  for (uint32_t x : numEquals) {
    if (!first) {
      cout << ";";
    } else {
      first = false;
    }
    cout << x;
  }
  cout << endl;
  cout << flush;
}

void testCase(const tmh::Weights &w, uint32_t hashSize,
              uint64_t numIterations) {

  testCase(w, "BagMinHash2", hashSize, numIterations, [](uint32_t m) {
    return [m](const vector<pair<uint64_t, double>> &x) {
      return bmh::bag_min_hash_2<bmh::FloatWeightDiscretization, bmh::XXHash64>(
                 x, m)
          .hashValues;
    };
  });
  testCase(w, "DartMinHash", hashSize, numIterations, [](uint32_t m) {
    mt19937_64 r(UINT64_C(0x1e402663a58575d1));
    return dmh::DartMinHash(r, m);
  });
  testCase(w, "TreeMinHash", hashSize, numIterations, [](uint32_t m) {
    return tmh::TreeMinHash<RNGFunction>(
        m, RNGFunction(UINT64_C(0x8eddaf1bc709729d)));
  });
}

int main(int argc, char *argv[]) {
  cout << "Jw"
       << ";";
  cout << "Jn"
       << ";";
  cout << "Jp"
       << ";";
  cout << "algorithmDescription"
       << ";";
  cout << "caseDescription"
       << ";";
  cout << "numIterations"
       << ";";
  cout << "hashSize"
       << ";";
  cout << "caseId"
       << ";";
  cout << "isUnweighted"
       << ";";
  cout << "dataSizeA"
       << ";";
  cout << "dataSizeB"
       << ";";
  cout << "dataSizeAB";
  cout << endl;
  cout << "histogramEqualSignatureComponents";
  cout << endl;
  cout << flush;

  uint32_t hashSizes[] = {1,   2,   4,   8,    16,   32,  64,
                          128, 256, 512, 1024, 2048, 4096};

  uint64_t numIterations = 10000;

  vector<tmh::Weights> cases = {tmh::getWeightsCase_075be894225e78f7(),
                                tmh::getWeightsCase_0a92d95c38b0bec5(),
                                tmh::getWeightsCase_29baac0d70950228(),
                                tmh::getWeightsCase_4e8536ff3d0c07af(),
                                tmh::getWeightsCase_52d5eb9e59e690e7(),
                                tmh::getWeightsCase_83f19a65b7f42e88(),
                                tmh::getWeightsCase_ae7f50b05c6ea2dd(),
                                tmh::getWeightsCase_dae81d77e5c7e0c3(),
                                tmh::getWeightsCase_a9415c152258dac1(),
                                tmh::getWeightsCase_431c7f212064fc5d(),
                                tmh::getWeightsCase_8d6bb210472266c3(),
                                tmh::getWeightsCase_8a224349623eeb24()};

  for (uint32_t hashSize : hashSizes) {
    for (const tmh::Weights &w : cases) {
      testCase(w, hashSize, numIterations);
    }
  }

  return 0;
}
