//#######################################
//# Copyright (C) 2020 Otmar Ertl.      #
//# All rights reserved.                #
//#######################################

#ifndef _TMH_WEIGHTED_MINWISE_HASHING_HPP_
#define _TMH_WEIGHTED_MINWISE_HASHING_HPP_

#include "bitstream_random.hpp"

#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <numeric>

namespace tmh {

template <typename T>
class MaxValueTracker {
    const uint32_t m;
    const uint32_t lastIndex;
    const std::unique_ptr<T[]> values; 

public:
    MaxValueTracker(uint32_t m) : m(m), lastIndex((m << 1) - 2), values(new T[lastIndex+1]) {}

    void reset(const T& infinity) {
        std::fill_n(values.get(), lastIndex + 1, infinity);
    }

    bool update(uint32_t idx, T value) {
        assert(idx < m);
        if (value < values[idx]) {
            while(true) {
                values[idx] = value;
                const uint32_t parentIdx = m + (idx >> 1);
                if (parentIdx > lastIndex) break;
                const uint32_t siblingIdx = idx ^ UINT32_C(1);
                const T siblingValue = values[siblingIdx];
                if (!(siblingValue < values[parentIdx])) break;
                if (value < siblingValue) value = siblingValue;
                idx = parentIdx;
            }
            return true;
        }
        else {
            return false;
        }
    }

    bool isUpdatePossible(T value) const {
        return value < values[lastIndex];
    }

    const T& operator[](uint32_t idx) const {
        return values[idx];
    }

    double max() const {
        return values[lastIndex];
    }
};

class BinaryWeightDiscretization {
public:
    typedef uint8_t index_type;
    typedef uint8_t weight_type;

    static const index_type maxBoundIdx = 1;

    static weight_type getBound(index_type boundIdx) {
        return boundIdx;
    }
};

template<typename I, typename W>
I calculateMaxBoundIdx() {
    static_assert(sizeof(I) == sizeof(W), "index type and weight type do not have same size");
    I i = 0;
    const W w = std::numeric_limits<W>::max();
    memcpy(&i, &w, sizeof(I));
    return i;
}

template<typename W, typename I>
class WeightDiscretization {
public:
    typedef I index_type;
    typedef W weight_type;

    static const index_type maxBoundIdx;

    static weight_type getBound(index_type boundIdx) {
        W f;
        static_assert(std::numeric_limits<W>::is_iec559, "weight_type is not iec559");
        static_assert(sizeof(weight_type) == sizeof(index_type), "weight_type and index_type do not have same size");
        memcpy(&f, &boundIdx, sizeof(index_type));
        return f;
    }
};



template<typename W, typename I>
const typename WeightDiscretization<W, I>::index_type WeightDiscretization<W, I>::maxBoundIdx = calculateMaxBoundIdx<WeightDiscretization<W, I>::index_type, WeightDiscretization<W, I>::weight_type>();

typedef WeightDiscretization<float, uint32_t> FloatWeightDiscretization;

typedef WeightDiscretization<double, uint64_t> DoubleWeightDiscretization;


template <typename D, typename R>
class PoissonProcess {

    double point;
    double weight;
    typename D::index_type weightIdxMin;
    typename D::index_type weightIdxMax;
    typename D::weight_type boundMin;
    typename D::weight_type boundMax;
    uint32_t signatureIdx;
    R rng;

public:

    PoissonProcess(
        double _point,
        double _weight,
        typename D::index_type _weightIdxMin,
        typename D::index_type _weightIdxMax,
        typename D::weight_type _boundMin,
        typename D::weight_type _boundMax,
        R&& _rng
        )
      : point(_point),
        weight(_weight),
        weightIdxMin(_weightIdxMin),
        weightIdxMax(_weightIdxMax),
        boundMin(_boundMin),
        boundMax(_boundMax),
        signatureIdx(std::numeric_limits<uint32_t>::max()),
        rng(std::move(_rng)) {}


    PoissonProcess(R&& _rng, double _weight) :
        PoissonProcess(0., _weight, 0, D::maxBoundIdx, 0, D::getBound(D::maxBoundIdx), std::move(_rng)) {}

    bool splittable() const {
        return weightIdxMax > weightIdxMin + 1;
    }

    bool partiallyRelevant() const {
        return D::getBound(weightIdxMin + 1) <= weight;
    }

    bool fullyRelevant() const {
        return boundMax <= weight;
    }

    uint32_t getIndex() const {
        return signatureIdx;
    }

    double getPoint() const {
        return point;
    }

    void next(uint32_t m) {
        point += getExponential1(rng) / (static_cast<double>(boundMax) - static_cast<double>(boundMin));
        signatureIdx = getUniformLemire(m, rng);
    }

    template<typename RF>
    std::unique_ptr<PoissonProcess> split(const RF& rngFunction, uint64_t d) {

        typename D::index_type weightIdxMid = (weightIdxMin + weightIdxMax) >> 1;

        double boundMid = D::getBound(weightIdxMid);

        bool inheritToLeft = getBernoulli((boundMid - static_cast<double>(boundMin)) / (static_cast<double>(boundMax) - static_cast<double>(boundMin)), rng);

        std::unique_ptr<PoissonProcess> pPrime;

        R rng = rngFunction(d, weightIdxMid);

        if (inheritToLeft) {
            pPrime = std::make_unique<PoissonProcess>(point, weight, weightIdxMid, weightIdxMax, boundMid, boundMax, std::move(rng));
            weightIdxMax = weightIdxMid;
            boundMax = boundMid;
        }
        else {
            pPrime = std::make_unique<PoissonProcess>(point, weight, weightIdxMin, weightIdxMid, boundMin, boundMid, std::move(rng));
            weightIdxMin = weightIdxMid;
            boundMin = boundMid;
        }
        return pPrime;
    }
};

template<typename D, typename H>
struct CmpPoissonProcessPtrs
{
    bool operator()(const std::unique_ptr<PoissonProcess<D,H>>& lhs, const std::unique_ptr<PoissonProcess<D,H>>& rhs) const
    {
        return rhs->getPoint() < lhs->getPoint();
    }
};

template<typename D, typename H>
struct CmpPoissonProcessPtrsInverse
{
    bool operator()(const std::unique_ptr<PoissonProcess<D,H>>& lhs, const std::unique_ptr<PoissonProcess<D,H>>& rhs) const
    {
        return rhs->getPoint() > lhs->getPoint();
    }
};

template<typename D, typename H>
void pushHeap(std::unique_ptr<PoissonProcess<D, H>>& p, std::vector<std::unique_ptr<PoissonProcess<D,H>>>& heap, size_t offset = 0) {
    heap.emplace_back(std::move(p));
    std::push_heap(heap.begin() + offset, heap.end(), CmpPoissonProcessPtrs<D,H>());
}

template<typename D, typename H>
std::unique_ptr<PoissonProcess<D, H>> popHeap(std::vector<std::unique_ptr<PoissonProcess<D,H>>>& heap, size_t offset = 0) {
    std::pop_heap(heap.begin() + offset, heap.end(), CmpPoissonProcessPtrs<D,H>());
    std::unique_ptr<PoissonProcess<D,H>> p = std::move(heap.back());
    heap.pop_back();
    return p;
}

static const uint64_t bagMinHashSeedA = UINT64_C(0xf331e07615a87fd7); // constant from random.org
static const uint64_t bagMinHashSeedB = UINT64_C(0xe224afad0d89c684); // constant from random.org

struct UnaryWeightFunction {
    template<typename X>
    constexpr double operator()(X) const {
        return 1;
    }
};

template<typename D, typename RF>
class BagMinHash1 {

    const uint32_t m;
    const RF& rngFunction;

    typedef typename RF::RngType R;

    std::vector<std::unique_ptr<PoissonProcess<D, R>>> heap;
    std::vector<uint64_t> elements;
    MaxValueTracker<double> h;

    void reset() {
        heap.clear();
        h.reset(std::numeric_limits<double>::infinity());
    }

public:

    BagMinHash1(const uint32_t m, const RF& rngFunction) : m(m), rngFunction(rngFunction),elements(m),h(m) {
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {

        reset();

        for(const auto& x : data) {

            heap.clear();

            double w = x.second;
            if (w < D::getBound(1)) continue;

            const uint64_t d = x.first;

            auto rng = rngFunction(d, 0);
            std::unique_ptr<PoissonProcess<D,R>> p = std::make_unique<PoissonProcess<D,R>>(std::move(rng), w);

            p->next(m);
            if (p->fullyRelevant()) {
                if (h.update(p->getIndex(), p->getPoint())) elements[p->getIndex()] = d;
            }

            while(h.isUpdatePossible(p->getPoint())) {
                while(p->splittable() && p->partiallyRelevant()) {

                    std::unique_ptr<PoissonProcess<D,R>> pPrime = p->split(rngFunction, d);

                    if (p->fullyRelevant()) {
                        if (h.update(p->getIndex(), p->getPoint())) elements[p->getIndex()] = d;
                    }

                    if (pPrime->partiallyRelevant()) {
                        pPrime->next(m);
                        if (pPrime->fullyRelevant()) {
                            if (h.update(pPrime->getIndex(), pPrime->getPoint())) elements[pPrime->getIndex()] = d;
                        }
                        if (h.isUpdatePossible(pPrime->getPoint())) pushHeap(pPrime, heap);
                    }
                }

                if (p->fullyRelevant()) {
                    p->next(m);
                    if (h.update(p->getIndex(), p->getPoint())) elements[p->getIndex()] = d;
                    if (h.isUpdatePossible(p->getPoint())) pushHeap(p, heap);
                }
                if (heap.empty()) break;
                p = popHeap(heap);
            }
        }

        std::vector<std::pair<uint64_t, double>> result(m);
        for (uint32_t k = 0; k < m; ++k) {
            result[k] = std::make_pair(elements[k], h[k]);
        }

        return result;
    }
};


struct Node {
    double lowerBound;
    double invRate;
    double ratio;

    Node(double lowerBound, double midBound, double upperBound) : lowerBound(lowerBound), invRate(1./(upperBound - lowerBound)), ratio(invRate * (midBound - lowerBound)) {}

};

static std::vector<Node> preCalculateTree(double factor, double max) {
    assert(max > 0);
    assert(factor > 0);
    assert(factor < 1);


    struct TmpNode {
        uint32_t minIdx;
        uint32_t midIdx;
        uint32_t maxIdx;
    };

    std::vector<double> boundaries;
    
    boundaries.push_back(max);
    do {
        boundaries.push_back(std::min(factor * boundaries.back(), std::nexttoward(boundaries.back(), 0)));
    } while(boundaries.back() > 0);
    assert(boundaries.back() == 0.);
    std::reverse(boundaries.begin(), boundaries.end());
    uint32_t numNodes = boundaries.size()-1;

    std::vector<uint32_t> counts(2*numNodes - 1);
    for(uint32_t i = 0; i < numNodes; ++i) counts[numNodes-1+i] = 1;
    for(uint32_t idx = numNodes-2; idx != std::numeric_limits<uint32_t>::max(); --idx) {
        uint32_t leftChildIdx = 2*idx + 1;
        uint32_t rightChildIdx = 2*idx + 2;
        counts[idx] = counts[leftChildIdx] + counts[rightChildIdx];
    }
    assert(counts[0] == numNodes);

    std::vector<TmpNode> tmpNodes(2*numNodes - 1);
    tmpNodes[0] = {0, 0, numNodes};
    
    for(uint32_t idx = 0; idx < numNodes-1;++idx) {
        uint32_t leftChildIdx = 2*idx + 1;
        uint32_t rightChildIdx = 2*idx + 2;
        assert(tmpNodes[idx].maxIdx-tmpNodes[idx].minIdx == counts[leftChildIdx] + counts[rightChildIdx]);
        tmpNodes[idx].midIdx = tmpNodes[idx].minIdx + counts[leftChildIdx];
        tmpNodes[leftChildIdx] = {tmpNodes[idx].minIdx, 0, tmpNodes[idx].midIdx};
        tmpNodes[rightChildIdx] = {tmpNodes[idx].midIdx, 0, tmpNodes[idx].maxIdx};

    }

    for(uint32_t idx = 0; idx < numNodes-1;++idx) {
        uint32_t leftChildIdx = 2*idx + 1;
        uint32_t rightChildIdx = 2*idx + 2;
        assert(tmpNodes[leftChildIdx].minIdx == tmpNodes[idx].minIdx);
        assert(tmpNodes[rightChildIdx].maxIdx == tmpNodes[idx].maxIdx);
        assert(tmpNodes[leftChildIdx].maxIdx == tmpNodes[rightChildIdx].minIdx);
    }

    for(uint32_t idx = numNodes-1; idx < 2*numNodes-1; ++idx) {
        assert(tmpNodes[idx].minIdx + 1 == tmpNodes[idx].maxIdx);
        assert(tmpNodes[idx].midIdx == 0);
    }

    std::vector<Node> tree;
    tree.reserve(2*numNodes-1);
    for(uint32_t idx = 0; idx < 2*numNodes-1; ++idx) {
        tree.emplace_back(boundaries[tmpNodes[idx].minIdx], (tmpNodes[idx].midIdx==0)?-1.:boundaries[tmpNodes[idx].midIdx], boundaries[tmpNodes[idx].maxIdx]);
    }

    return tree;


}


template<typename RF>
class TreeMinHash1 {

    typedef typename RF::RngType R;

    struct PoissonProcess {
        double point;
        uint32_t nodeIdx;
        R rng;

        PoissonProcess(double point, uint32_t nodeIdx, R&& rng) : point(point), nodeIdx(nodeIdx), rng(std::move(rng)) {}
    };

    const uint32_t m;
    const RF rngFunction;
    std::vector<PoissonProcess> processes;
    std::vector<PoissonProcess*> heap;
    std::vector<uint64_t> elements;
    MaxValueTracker<double> h;
    std::vector<Node> tree;
    uint32_t numLeafNodes;

    struct CmpPoissonProcessPtrs
    {
        bool operator()(const PoissonProcess* lhs, const PoissonProcess* rhs) const
        {
            return rhs->point < lhs->point;
        }
    };

    void pushHeap(PoissonProcess* p) {
        heap.push_back(p);
        std::push_heap(heap.begin(), heap.end(), CmpPoissonProcessPtrs());
    }

    PoissonProcess* popHeap() {
        std::pop_heap(heap.begin(), heap.end(), CmpPoissonProcessPtrs());
        PoissonProcess* p = heap.back();
        heap.pop_back();
        return p;
    }
    
public:

    TreeMinHash1(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max()) : m(m), rngFunction(rngFunction), elements(m), h(m), tree(preCalculateTree(factor, max)), numLeafNodes((tree.size()+1)/2) {

        heap.reserve(numLeafNodes);
        processes.reserve(numLeafNodes);

    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data, double initialLimit = std::numeric_limits<double>::max()) {

        h.reset(initialLimit);

        for(const auto& x : data) {

            const double w = x.second;
            if (w == 0) continue;

            heap.clear();
            processes.clear();

            const uint64_t d = x.first;

            processes.emplace_back(0., 0, std::move(rngFunction(d, 0)));
            PoissonProcess* p = &(processes.back());

            p->point += getExponential1(p->rng) * tree[0].invRate;
    
            while(h.isUpdatePossible(p->point)) {
                while(p->nodeIdx + 1 < numLeafNodes && tree[p->nodeIdx].lowerBound < w) {

                    const bool inheritToLeft = getBernoulli(tree[p->nodeIdx].ratio, p->rng);
                    p->nodeIdx <<= 1;
                    const uint32_t siblingIdx = p->nodeIdx + 1 + inheritToLeft;
                    p->nodeIdx += 2 - inheritToLeft;
                    const double siblingPoint = p->point + getExponential1(p->rng) * tree[siblingIdx].invRate;
                    if (h.isUpdatePossible(siblingPoint) && tree[siblingIdx].lowerBound < w) {
                        processes.emplace_back(siblingPoint, siblingIdx, rngFunction(d, siblingIdx));
                        pushHeap(&(processes.back()));
                    }
                }
                if (tree[p->nodeIdx].lowerBound < w) {

                    double acceptanceProbability = (w - tree[p->nodeIdx].lowerBound) * tree[p->nodeIdx].invRate;
                    bool accept = getUniformDouble(p->rng) < acceptanceProbability;
                    const uint32_t idx = getUniformLemire(m, p->rng);
                    if (accept) {
                        if (h.update(idx, p->point)) elements[idx] = d;
                    }
                    p->point += getExponential1(p->rng) * tree[p->nodeIdx].invRate;
                    if (h.isUpdatePossible(p->point)) pushHeap(p);
                }

                if (heap.empty()) break;
                p = popHeap();
            }
        }

        std::vector<std::pair<uint64_t, double>> result(m);
        for (uint32_t k = 0; k < m; ++k) {
            result[k] = std::make_pair(elements[k], h[k]);
        }

        return result;
    }
};

template<typename RF>
class TreeMinHash1x {

    typedef typename RF::RngType R;

    struct PoissonProcess {
        double point;
        uint32_t weightIdxMin;
        uint32_t weightIdxMax;
        R rng;

        PoissonProcess(double point, uint32_t weightIdxMin, uint32_t weightIdxMax, R&& rng) : point(point), weightIdxMin(weightIdxMin), weightIdxMax(weightIdxMax), rng(std::move(rng)) {}
    };

    const uint32_t m;
    const RF rngFunction;
    std::vector<PoissonProcess> processes;
    std::vector<PoissonProcess*> heap;
    std::vector<uint64_t> elements;
    MaxValueTracker<double> h;
    std::vector<double> boundaries;

    struct CmpPoissonProcessPtrs
    {
        bool operator()(const PoissonProcess* lhs, const PoissonProcess* rhs) const
        {
            return rhs->point < lhs->point;
        }
    };

    void pushHeap(PoissonProcess* p) {
        heap.push_back(p);
        std::push_heap(heap.begin(), heap.end(), CmpPoissonProcessPtrs());
    }

    PoissonProcess* popHeap() {
        std::pop_heap(heap.begin(), heap.end(), CmpPoissonProcessPtrs());
        PoissonProcess* p = heap.back();
        heap.pop_back();
        return p;
    }

    void reset() {
        h.reset(std::numeric_limits<double>::infinity());
    }
    
    double getBound(uint32_t boundIdx) const {
        return boundaries[boundIdx];
    }

public:

    TreeMinHash1x(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max()) : m(m), rngFunction(rngFunction), elements(m), h(m) {
        assert(max > 0);
        assert(factor > 0);
        assert(factor < 1);
        boundaries.push_back(max);
        do {
            boundaries.push_back(std::min(factor * boundaries.back(), std::nexttoward(boundaries.back(), 0)));
        } while(boundaries.back() > 0);
        assert(boundaries.back() == 0.);
        std::reverse(boundaries.begin(), boundaries.end());
        heap.reserve(boundaries.size()-1);
        processes.reserve(boundaries.size()-1);
        //numNodes = boundaries.size()-1;
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {

        reset();

        for(const auto& x : data) {

            const double w = x.second;
            if (w == 0) continue;

            heap.clear();
            processes.clear();

            const uint64_t d = x.first;

            processes.emplace_back(0., UINT32_C(0), static_cast<uint32_t>(boundaries.size()-1), std::move(rngFunction(d, 0)));
            PoissonProcess* p = &(processes.back());

            p->point += getExponential1(p->rng) / (getBound(p->weightIdxMax) - getBound(p->weightIdxMin));

            while(h.isUpdatePossible(p->point)) {

                while(p->weightIdxMax > p->weightIdxMin + 1 && getBound(p->weightIdxMin) < w) {

                    const uint32_t weightIdxMid = (p->weightIdxMin + p->weightIdxMax + 1) >> 1;
                    const bool inheritToLeft = getBernoulli((getBound(weightIdxMid) - getBound(p->weightIdxMin)) / (getBound(p->weightIdxMax) - getBound(p->weightIdxMin)), p->rng);

                    if (inheritToLeft) {
                        processes.emplace_back(p->point, weightIdxMid, p->weightIdxMax, rngFunction(d, weightIdxMid));
                        p->weightIdxMax = weightIdxMid;
                    }
                    else {
                        processes.emplace_back(p->point, p->weightIdxMin, weightIdxMid, rngFunction(d, weightIdxMid));
                        p->weightIdxMin = weightIdxMid;
                    }
                    PoissonProcess* pPrime = &(processes.back());

                    if (getBound(pPrime->weightIdxMin) < w) {
                        pPrime->point += getExponential1(pPrime->rng) / (getBound(pPrime->weightIdxMax) - getBound(pPrime->weightIdxMin));
                        if (h.isUpdatePossible(pPrime->point)) pushHeap(pPrime);
                    }
                }
                if (getBound(p->weightIdxMin) < w) {
                    double acceptanceProbability = (w - getBound(p->weightIdxMin)) / (getBound(p->weightIdxMax) - getBound(p->weightIdxMin));
                    bool accept = getUniformDouble(p->rng) < acceptanceProbability;
                    const uint32_t idx = getUniformLemire(m, p->rng);
                    if (accept) {
                        if (h.update(idx, p->point)) elements[idx] = d;
                    }
                    p->point += getExponential1(p->rng) / (getBound(p->weightIdxMax) - getBound(p->weightIdxMin));
                    if (h.isUpdatePossible(p->point)) pushHeap(p);
                }

                if (heap.empty()) break;
                p = popHeap();
            }
        }

        std::vector<std::pair<uint64_t, double>> result(m);
        for (uint32_t k = 0; k < m; ++k) {
            result[k] = std::make_pair(elements[k], h[k]);
        }

        return result;
    }
};


template<typename RF>
class TreeMinHash1_NonStreaming {
    TreeMinHash1<RF> treeMinHash;
    const double initialLimitFactor;
public:

    TreeMinHash1_NonStreaming(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
            treeMinHash(m, rngFunction, factor, max), 
            initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m) {

        assert(successProbabilityFirstRun > 0);
        assert(successProbabilityFirstRun <= 1);
        assert(initialLimitFactor > 0);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data, uint64_t* numberOfAttempts = nullptr) {
        double weightSum = 0;
        for(const auto& d:data) weightSum += d.second;

        double limit = initialLimitFactor / weightSum;

        uint64_t numAttempts = 0;
        while(true) {
            auto result = treeMinHash(data, limit);
            numAttempts += 1;

            bool success = std::none_of(result.begin(), result.end(), [limit](const auto& r){return r.second == limit;});

            if (success) {
                if (numberOfAttempts != nullptr) *numberOfAttempts = numAttempts;
                return result;
            }

            limit *= 2;

        }
    }
};


template<typename RF>
class TreeMinHash1a {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    
public:

    TreeMinHash1a(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m) 
    {
        buffer.reserve(numNonLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        std::vector<std::pair<uint64_t, double>> result(m, {UINT64_C(0), limit});

        while(true) {
    
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!(point < limit)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (siblingPoint < limit && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        while(true) {
                            double nextPoint = point + getExponential1(rng) * invRate;
                            bool accept = getUniformDouble(rng) < acceptanceProbability;
                            if (accept) {
                                const uint32_t idx = getUniformLemire(m, rng);
                                if (point < result[idx].second) result[idx] = {d, point};
                            }
                            if(!(nextPoint < limit)) break;
                            if (!accept) getUniformLemire(m, rng);
                            point = nextPoint;
                        }
                    }
                    if (buffer.empty()) break;
                    std::tie(point, nodeIdx) = buffer.back();
                    buffer.pop_back();
                    rng = rngFunction(d, nodeIdx);
                }
            }

            bool success = std::none_of(result.begin(), result.end(), [limit](const auto& r){return r.second == limit;});

            if (success) return result;

            double oldLimit = limit;
            limit += limit;
            std::for_each(result.begin(), result.end(), [oldLimit, limit](auto& d) {if (d.second == oldLimit) d.second = limit;});
        }
    }
};


template<typename RF>
class TreeMinHash1b {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    
public:

    TreeMinHash1b(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m)
    {
        buffer.reserve(numNonLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        std::vector<std::pair<uint64_t, double>> result(m, {UINT64_C(0), limit});

        while(true) {
    
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!(point < limit)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (siblingPoint < limit && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        while(true) {
                            double nextPoint = point + getExponential1(rng) * invRate;
                            const uint32_t idx = getUniformLemire(m, rng);
                            if (point < result[idx].second) {
                                if (!(acceptanceProbability < 1)) {
                                    result[idx] = {d, point};
                                    if(!(nextPoint < limit)) break;
                                    getUniformDouble(rng);
                                } else {
                                    bool accept = getUniformDouble(rng) < acceptanceProbability;
                                    if (accept) {
                                        result[idx] = {d, point};
                                    }
                                    if(!(nextPoint < limit)) break;
                                }
                            } else {
                                if(!(nextPoint < limit)) break;
                                getUniformDouble(rng);
                            }
                            point = nextPoint;
                        }
                    }
                    if (buffer.empty()) break;
                    std::tie(point, nodeIdx) = buffer.back();
                    buffer.pop_back();
                    rng = rngFunction(d, nodeIdx);
                }
            }

            bool success = std::none_of(result.begin(), result.end(), [limit](const auto& r){return r.second == limit;});

            if (success) return result;

            double oldLimit = limit;
            limit += limit;
            std::for_each(result.begin(), result.end(), [oldLimit, limit](auto& d) {if (d.second == oldLimit) d.second = limit;});
        }
    }
};

template<typename RF>
class TreeMinHash1c {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    PermutationStream permutationStream;
    std::vector<double> factors;
    
public:

    TreeMinHash1c(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m),
        permutationStream(m),
        factors(m-1)
    {
        buffer.reserve(numNonLeafNodes);
        for(uint32_t i = 0; i < m - 1; ++i) factors[i] = static_cast<double>(m) / static_cast<double>(m - i - 1);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        std::vector<std::pair<uint64_t, double>> result(m, {UINT64_C(0), limit});

        while(true) {
    
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!(point < limit)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (siblingPoint < limit && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;

                        permutationStream.reset();
                        for(uint32_t k = 0; k < m; ++k) {
                            double nextPoint = (k < factors.size())?point + getExponential1(rng) * invRate * factors[k]:std::numeric_limits<double>::infinity();
                            const uint32_t idx = permutationStream.next(rng);
                            if (point < result[idx].second) {
                                if (!(acceptanceProbability < 1)) {
                                    result[idx] = {d, point};
                                    if(!(nextPoint < limit)) break;
                                    getUniformDouble(rng);
                                } else {
                                    const bool accept = getUniformDouble(rng) < acceptanceProbability;
                                    if (accept) {
                                        result[idx] = {d, point};
                                    } else {
                                        auto rng2 = rngFunction(d, (static_cast<uint64_t>(nodeIdx) << 32) | idx);
                                        const double invRate2 = invRate*m;
                                        while(true) {
                                            point += getExponential1(rng2) * invRate2;
                                            if (point < result[idx].second) {  
                                                const bool accept2 = getUniformDouble(rng2) < acceptanceProbability;
                                                if (accept2) {
                                                    result[idx] = {d, point};
                                                    break;
                                                }
                                            } else {
                                                break;
                                            }
                                        }
                                    }
                                    if(!(nextPoint < limit)) break;
                                }
                            } else {
                                if(!(nextPoint < limit)) break;
                                getUniformDouble(rng);
                            }
                            point = nextPoint;
                        }
                    }
                    if (buffer.empty()) break;
                    std::tie(point, nodeIdx) = buffer.back();
                    buffer.pop_back();
                    rng = rngFunction(d, nodeIdx);
                }
            }

            bool success = std::none_of(result.begin(), result.end(), [limit](const auto& r){return r.second == limit;});

            if (success) return result;

            double oldLimit = limit;
            limit += limit;
            std::for_each(result.begin(), result.end(), [oldLimit, limit](auto& d) {if (d.second == oldLimit) d.second = limit;});
        }
    }
};

template<typename RF>
class TreeMinHash {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    PermutationStream permutationStream;
    std::vector<double> factors;
    
public:

    TreeMinHash(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m),
        permutationStream(m),
        factors(m-1)
    {
        buffer.reserve(numNonLeafNodes);
        for(uint32_t i = 0; i < m - 1; ++i) factors[i] = static_cast<double>(m) / static_cast<double>(m - i - 1);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        std::vector<std::pair<uint64_t, double>> result(m, {UINT64_C(0), limit});

        while(true) {
    
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!(point < limit)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (siblingPoint < limit && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;

                        permutationStream.reset();
                        for(uint32_t k = 0; k < m; ++k) {
                            double nextPoint = (k < factors.size())?point + getExponential1(rng) * invRate * factors[k]:std::numeric_limits<double>::infinity();
                            const uint32_t idx = permutationStream.next(rng);
                            if (point < result[idx].second) {
                                if (!(acceptanceProbability < 1)) {
                                    result[idx] = {d, point};
                                } else {
                                    auto rng2 = rngFunction(d, (static_cast<uint64_t>(nodeIdx) << 32) | idx);
                                    do {
                                        bool accept = getUniformDouble(rng2) < acceptanceProbability;
                                        if (accept) {
                                            result[idx] = {d, point};
                                            break;
                                        }
                                        point += getExponential1(rng2) * invRate * m;
                                    } while (point < result[idx].second);
                                }
                            }
                            if(!(nextPoint < limit)) break;
                            point = nextPoint;
                        }
                    }
                    if (buffer.empty()) break;
                    std::tie(point, nodeIdx) = buffer.back();
                    buffer.pop_back();
                    rng = rngFunction(d, nodeIdx);
                }
            }

            bool success = std::none_of(result.begin(), result.end(), [limit](const auto& r){return r.second == limit;});

            if (success) return result;

            double oldLimit = limit;
            limit += limit;
            std::for_each(result.begin(), result.end(), [oldLimit, limit](auto& d) {if (d.second == oldLimit) d.second = limit;});
        }
    }
};


template<typename RF>
class TreeMinHash1e {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    
public:

    TreeMinHash1e(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m)
    {
        buffer.reserve(numNonLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        std::vector<std::pair<uint64_t, double>> result(m, {UINT64_C(0), limit});

        while(true) {
    
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!(point < limit)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (siblingPoint < limit && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        if (!(acceptanceProbability < 1)) {
                            do {    
                                const uint32_t idx = getUniformLemire(m, rng);
                                if (point < result[idx].second) result[idx] = {d, point};
                                point += getExponential1(rng) * invRate;    
                            } while(point < limit);
                        } else {
                            uint64_t counter = 0;
                            std::unique_ptr<R> rng2;
                            do {
                                const uint32_t idx = getUniformLemire(m, rng);
                                if (point < result[idx].second) {
                                    auto rng2 = rngFunction(d, (static_cast<uint64_t>(nodeIdx) << 32) + counter);
                                    bool accept = getUniformDouble(rng2) < acceptanceProbability;
                                    if (accept) result[idx] = {d, point};
                                }
                                counter += 1;
                                point += getExponential1(rng) * invRate;    
                            } while(point < limit);
                        }
                    }
                    if (buffer.empty()) break;
                    std::tie(point, nodeIdx) = buffer.back();
                    buffer.pop_back();
                    rng = rngFunction(d, nodeIdx);
                }
            }

            bool success = std::none_of(result.begin(), result.end(), [limit](const auto& r){return r.second == limit;});

            if (success) return result;

            double oldLimit = limit;
            limit += limit;
            std::for_each(result.begin(), result.end(), [oldLimit, limit](auto& d) {if (d.second == oldLimit) d.second = limit;});
        }
    }
};


template<typename RF>
class TreeMinHash1a2 {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    MaxValueTracker<double> h;
    
public:

    TreeMinHash1a2(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m),
        h(m)
    {
        buffer.reserve(numNonLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        assert(limit <= std::numeric_limits<double>::infinity());

        std::vector<std::pair<uint64_t, double>> result(m);

        while(true) {

            h.reset(limit);
            
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!h.isUpdatePossible(point)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (h.isUpdatePossible(siblingPoint) && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        while(true) {
                            double nextPoint = point + getExponential1(rng) * invRate;
                            bool accept = getUniformDouble(rng) < acceptanceProbability;
                            if (accept) {
                                const uint32_t idx = getUniformLemire(m, rng);
                                if (h.update(idx, point)) result[idx] = {d, point};
                            }
                            if(!h.isUpdatePossible(nextPoint)) break;
                            if (!accept) getUniformLemire(m, rng);
                            point = nextPoint;
                        };
                    }
                    bool cont = false;
                    while(!buffer.empty()) {
                        std::tie(point, nodeIdx) = buffer.back();
                        buffer.pop_back();
                        if (h.isUpdatePossible(point)) {
                            cont = true;
                            break;
                        }
                    }
                    if (!cont) break;
                    rng = rngFunction(d, nodeIdx);
                }
            }

            if (h.max() < limit) return result;

            limit += limit;
        }
    }
};

template<typename RF>
class TreeMinHash1a3 {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    uint32_t readIdx;
    MaxValueTracker<double> h;
    
public:

    TreeMinHash1a3(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m),
        h(m)
    {
        buffer.reserve(numNonLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;});
        
        double limit = initialLimitFactor / weightSum;

        assert(limit <= std::numeric_limits<double>::infinity());

        std::vector<std::pair<uint64_t, double>> result(m);
        while(true) {

            h.reset(limit);
            
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                readIdx = 0;
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!h.isUpdatePossible(point)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (h.isUpdatePossible(siblingPoint) && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        while(true) {
                            double nextPoint = point + getExponential1(rng) * invRate;
                            bool accept = getUniformDouble(rng) < acceptanceProbability;
                            if (accept) {
                                const uint32_t idx = getUniformLemire(m, rng);
                                if (h.update(idx, point)) result[idx] = {d, point};
                            }
                            if(!h.isUpdatePossible(nextPoint)) break;
                            if (!accept) getUniformLemire(m, rng);
                            point = nextPoint;
                        };
                    }
                    bool cont = false;
                    while(readIdx < buffer.size()) {
                        std::tie(point, nodeIdx) = buffer[readIdx];
                        readIdx += 1;
                        if (h.isUpdatePossible(point)) {
                            cont = true;
                            break;
                        }
                    }
                    if (!cont) break;
                    rng = rngFunction(d, nodeIdx);
                }
            }

            if (h.max() < limit) return result;

            limit += limit;
        }
    }
};



template<typename RF>
class TreeMinHash1a4 {

    typedef typename RF::RngType R;

    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const double initialLimitFactor;
    std::vector<std::pair<double, uint32_t>> buffer;
    MaxValueTracker<double> h;
    
    struct Cmp
    {
        bool operator()(const auto& lhs, const auto& rhs) const
        {
            return rhs.first < lhs.first;
        }
    };
public:

    TreeMinHash1a4(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m),
        h(m)
    {
        buffer.reserve(numNonLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;}); 
        
        double limit = initialLimitFactor / weightSum;

        assert(limit <= std::numeric_limits<double>::infinity());

        std::vector<std::pair<uint64_t, double>> result(m);
        while(true) {

            h.reset(limit);
            
            for(const auto [d,w] : data) {

                if (w == 0) continue;
                buffer.clear();
                uint32_t nodeIdx = 0;
                auto rng = rngFunction(d, nodeIdx);
                double point = getExponential1(rng) * tree[nodeIdx].invRate;
                if (!h.isUpdatePossible(point)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rng);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rng) * tree[siblingIdx].invRate;
                        if (h.isUpdatePossible(siblingPoint) && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx);
                            std::push_heap(buffer.begin(), buffer.end(), Cmp());
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        while(true) {
                            double nextPoint = point + getExponential1(rng) * invRate;
                            bool accept = getUniformDouble(rng) < acceptanceProbability;
                            if (accept) {
                                const uint32_t idx = getUniformLemire(m, rng);
                                if (h.update(idx, point)) result[idx] = {d, point};
                            }
                            if(!h.isUpdatePossible(nextPoint)) break;
                            if (!accept) getUniformLemire(m, rng);
                            point = nextPoint;
                        };
                    }
                    if (buffer.empty()) break;
                    std::pop_heap(buffer.begin(), buffer.end(), Cmp());
                    std::tie(point, nodeIdx) = buffer.back();
                    if (!h.isUpdatePossible(point)) break;
                    buffer.pop_back();
                    rng = rngFunction(d, nodeIdx);
                }
            }

            if (h.max() < limit) return result;

            limit += limit;
        }
    }
};

template<typename RF>
class TreeMinHash1a5 {

    typedef typename RF::RngType R;


    static const uint32_t noRng = std::numeric_limits<uint32_t>::max();
    const uint32_t m;
    const RF rngFunction;
    const std::vector<Node> tree;
    const uint32_t numNonLeafNodes;
    const uint32_t numLeafNodes;
    const double initialLimitFactor;
    std::vector<std::tuple<double, uint32_t, uint32_t>> buffer;
    MaxValueTracker<double> h;
    std::vector<R> rngs;
    
    struct Cmp
    {
        bool operator()(const auto& lhs, const auto& rhs) const
        {
            return std::get<0>(rhs) < std::get<0>(lhs);
        }
    };
public:

    TreeMinHash1a5(const uint32_t m, const RF& rngFunction, double factor = 0.5, double max = std::numeric_limits<double>::max(), double successProbabilityFirstRun = 0.9) : 
        m(m), 
        rngFunction(rngFunction), 
        tree(preCalculateTree(factor, max)), 
        numNonLeafNodes(tree.size() - (tree.size()+1)/2),
        numLeafNodes(tree.size() - numNonLeafNodes),
        initialLimitFactor(-std::log(-std::expm1(std::log(successProbabilityFirstRun) / m))*m),
        h(m)
        
    {
        buffer.reserve(numLeafNodes);
        rngs.reserve(numLeafNodes);
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& data) {
    
        const double weightSum = std::accumulate(data.begin(), data.end(), 0., [](double x, const auto& d) {return x + d.second;}); 
        
        double limit = initialLimitFactor / weightSum;

        assert(limit <= std::numeric_limits<double>::infinity());

        std::vector<std::pair<uint64_t, double>> result(m);
        while(true) {

            h.reset(limit);
            
            for(const auto [d,w] : data) {

                if (w == 0) continue;

                buffer.clear();
                rngs.clear();

                uint32_t nodeIdx = 0;
                uint32_t rngIdx = 0;
                rngs.emplace_back(d, nodeIdx);
                double point = getExponential1(rngs[rngIdx]) * tree[nodeIdx].invRate;
                if (!h.isUpdatePossible(point)) continue;
                
                while(true) {
                    while(nodeIdx < numNonLeafNodes && tree[nodeIdx].lowerBound < w) { // while not leaf do
                        const bool inheritToLeft = getBernoulli(tree[nodeIdx].ratio, rngs[rngIdx]);
                        nodeIdx <<= 1;
                        const uint32_t siblingIdx = nodeIdx + 1 + inheritToLeft;
                        nodeIdx += 2 - inheritToLeft;
                        double siblingPoint = point + getExponential1(rngs[rngIdx]) * tree[siblingIdx].invRate;
                        if (h.isUpdatePossible(siblingPoint) && tree[siblingIdx].lowerBound < w) {
                            buffer.emplace_back(siblingPoint, siblingIdx, noRng);
                            std::push_heap(buffer.begin(), buffer.end(), Cmp());
                        }
                    }
                    if (tree[nodeIdx].lowerBound < w) {
                        const double invRate = tree[nodeIdx].invRate;
                        const double acceptanceProbability = (w - tree[nodeIdx].lowerBound) * invRate;
                        bool accept = getUniformDouble(rngs[rngIdx]) < acceptanceProbability;
                        double nextPoint = point + getExponential1(rngs[rngIdx]) * invRate;
                        if(accept) {
                            const uint32_t idx = getUniformLemire(m, rngs[rngIdx]);
                            if (h.update(idx, point)) result[idx] = {d, point};
                        }
                        if (h.isUpdatePossible(nextPoint)) {
                            if (!accept) getUniformLemire(m, rngs[rngIdx]);
                            buffer.emplace_back(nextPoint, nodeIdx, rngIdx);
                            std::push_heap(buffer.begin(), buffer.end(), Cmp());
                        }
                    }
                    if (buffer.empty()) break;
                    std::pop_heap(buffer.begin(), buffer.end(), Cmp());
                    std::tie(point, nodeIdx, rngIdx) = buffer.back();
                    if (!h.isUpdatePossible(point)) break;
                    buffer.pop_back();
                    if (rngIdx == noRng) {
                        rngIdx = rngs.size();
                        rngs.emplace_back(d, nodeIdx);
                    }
                }
            }

            if (h.max() < limit) return result;

            limit += limit;
        }
    }
};


} // namespace tmh

#endif // _TMH_WEIGHTED_MINWISE_HASHING_HPP_
