#pragma once
#include <queue>     // priority_queue
#include <vector>    // vector

#include "types.hpp" // Float


namespace ngv {
namespace utils {

    struct Record {
        size_t id;
        Float value;
        Record(size_t id, Float value): id(id), value(value) {}
    };

    struct GreaterEqual {
        bool operator()(Record& r1, Record& r2) {
            return r1.value >= r2.value;
        }
    };

    /**
    * Min priority queue of records the capacity of which can be reserved beforehand
    */
    class MinPriorityHeap: public std::priority_queue<Record, std::vector<Record>, GreaterEqual>
    {
        public:

            explicit MinPriorityHeap(size_t capacity) { this->c.reserve(capacity); }
            size_t capacity() const { return this->c.capacity(); }
    };

} // namespace utils
} // namespace ngv
