// #######################################
// # Copyright (C) 2020-2023 Otmar Ertl. #
// # All rights reserved.                #
// #######################################

// Bessa, Aline, et al. "Weighted Minwise Hashing Beats Linear Sketching for
// Inner Product Estimation." arXiv preprint arXiv:2301.05811 (2023) describes
// how weighted minwise hashing can be used for inner product estimation. This
// code demonstrates that their approach can be simplified using TreeMinHash
// which is likely faster and does not require the unnatural discretization of
// input vectors.

#include "bitstream_random.hpp"
#include "data_generation.hpp"
#include "weighted_minwise_hashing.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

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

class InnerProductSketch {
  const vector<pair<double, double>> hashValueList;
  double norm;
  const double exponent;

public:
  InnerProductSketch(const vector<pair<double, double>> &hashValueList,
                     double norm, double exponent)
      : hashValueList(hashValueList), norm(norm), exponent(exponent) {}

  static double estimate_inner_product(const InnerProductSketch &sketch1,
                                       const InnerProductSketch &sketch2) {

    assert(sketch1.exponent == sketch2.exponent);
    assert(sketch1.hashValueList.size() == sketch2.hashValueList.size());
    const size_t m = sketch1.hashValueList.size();
    double exponent = sketch1.exponent;

    double estimatorSum = 0;
    double minHashSum = 0;
    for (size_t j = 0; j < m; ++j) {
      auto &[hash1, val1] = sketch1.hashValueList[j];
      auto &[hash2, val2] = sketch2.hashValueList[j];
      if (hash1 == hash2) {
        estimatorSum +=
            (val1 * val2) / pow(min(fabs(val1), fabs(val2)), exponent);
      }
      minHashSum += min(hash1, hash2);
    }

    const double estimatedUnionSize =
        static_cast<double>(m) * static_cast<double>(m) / minHashSum;

    return sketch1.norm * sketch2.norm * (estimatorSum / m) *
           estimatedUnionSize;
  }
};

class InnerProductSketchCalculator {
  tmh::TreeMinHash<RNGFunction> treeMinHash;
  const double exponent;

public:
  InnerProductSketchCalculator(uint32_t m, uint64_t seed, double exponent = 2)
      : treeMinHash(m, RNGFunction(seed),
                    1. // set maximum supported weight equal to 1, as all
                       // weights will be <= 1. due to normalization
                    ),
        exponent(exponent) {}

  InnerProductSketch compute(const vector<pair<uint64_t, double>> &input) {

    unordered_map<uint64_t, double> keyToValueMap(input.cbegin(), input.cend());

    // calculate norm
    const double norm =
        sqrt(accumulate(input.cbegin(), input.cend(), 0., [](auto sum, auto x) {
          return sum + x.second * x.second;
        }));

    // normalized and squared input for TreeMinHash
    vector<pair<uint64_t, double>> keyNormalizedSquaredValueList(input.size());
    transform(input.cbegin(), input.cend(),
              keyNormalizedSquaredValueList.begin(), [&](auto &keyValue) {
                return make_pair(keyValue.first,
                                 pow(fabs(keyValue.second) / norm, exponent));
              });

    // weighted minhash
    const vector<pair<uint64_t, double>> weightedMinHashResult =
        treeMinHash(keyNormalizedSquaredValueList);

    // setup data for inner product sketch
    vector<pair<double, double>> hashValueList(weightedMinHashResult.size());
    transform(weightedMinHashResult.cbegin(), weightedMinHashResult.cend(),
              hashValueList.begin(), [&](auto &keyHash) {
                return make_pair(keyHash.second,
                                 keyToValueMap[keyHash.first] / norm);
              });

    return InnerProductSketch(hashValueList, norm, exponent);
  }
};

int main(int argc, char *argv[]) {

  uint64_t numCycles = 1000;
  uint64_t sizeOfWeightList = 1000;
  uint32_t m = 1 << 14;

  // test different exponents
  vector<double> exponentValues = {0, 0.5, 1, 1.5, 2, 3};

  uint64_t seedForWeights = UINT64_C(0x0c2954b1cb065f32);
  uint64_t seedForKeys = UINT64_C(0x11da3e19c9262418);
  uint64_t seedForTreeMinHash = UINT64_C(0xda47270740231451);

  // generate some weight vectors
  mt19937_64 rngForWeights(seedForWeights);
  uniform_real_distribution<double> nonZeroValueDistribution1(-2.3, 1);
  uniform_real_distribution<double> nonZeroValueDistribution2(-3, 10);
  bernoulli_distribution zeroDistribution1(0.1);
  bernoulli_distribution zeroDistribution2(0.2);
  double trueInnerProduct = 0.;
  vector<pair<double, double>> weights(sizeOfWeightList);
  for (uint64_t i = 0; i < sizeOfWeightList; ++i) {
    double weight1 = zeroDistribution1(rngForWeights)
                         ? 0.
                         : nonZeroValueDistribution1(rngForWeights);
    double weight2 = zeroDistribution2(rngForWeights)
                         ? 0.
                         : nonZeroValueDistribution2(rngForWeights);
    weights[i] = {weight1, weight2};
    trueInnerProduct += weight1 * weight2;
  }

  for (double exponent : exponentValues) {

    mt19937_64 rngForKeys(seedForKeys);

    double sumError = 0;
    double sumSquaredError = 0;

    InnerProductSketchCalculator sketch_calculator(m, seedForTreeMinHash,
                                                   exponent);
    for (uint64_t j = 0; j < numCycles; ++j) {

      // create random input by associating random keys with given weights
      vector<pair<uint64_t, double>> input1;
      vector<pair<uint64_t, double>> input2;
      for (uint64_t i = 0; i < weights.size(); ++i) {
        uint64_t key = rngForKeys();
        double weight1 = weights[i].first;
        double weight2 = weights[i].second;
        if (weight1 != 0.)
          input1.emplace_back(key, weight1);
        if (weight2 != 0.)
          input2.emplace_back(key, weight2);
      }

      // compute sketches for inner product
      auto sketch1 = sketch_calculator.compute(input1);
      auto sketch2 = sketch_calculator.compute(input2);

      // estimate inner product
      double estimatedInnerProduct =
          InnerProductSketch::estimate_inner_product(sketch1, sketch2);

      double error = estimatedInnerProduct - trueInnerProduct;
      sumError += error;
      sumSquaredError += error * error;
    }
    const double relativeEstimationBias =
        sumError / fabs(trueInnerProduct) / numCycles;
    const double relativeEstimationRmse =
        sqrt(sumSquaredError) / fabs(trueInnerProduct) / numCycles;

    cout << "exponent = " << exponent << endl;
    cout << "true inner product = " << trueInnerProduct << endl;
    cout << "relative estimation bias = " << relativeEstimationBias << endl;
    cout << "relative estimation rmse = " << relativeEstimationRmse << endl;
    cout << endl;
  }

  return 0;
}
