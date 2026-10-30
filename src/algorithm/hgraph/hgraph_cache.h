/*
    Copyright 2026 Huawei Technologies Co., Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdlib>
#include <limits>
#include <new>
#include <string>

#include "storage/serialization.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

template <typename T>
class Aligned64Allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Propagate the allocator on container move/copy/swap
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind {
        using other = Aligned64Allocator<U>;
    };

    Aligned64Allocator() noexcept = default;
    template <typename U>
    Aligned64Allocator(const Aligned64Allocator<U>&) noexcept {
    }

    // Allocation: enforce 64-byte alignment
    T*
    allocate(std::size_t n) {
        if (n == 0)
            return nullptr;
        constexpr std::size_t kAlign = 64;
        constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();
        if (n > kMaxSize / sizeof(T)) {  // n * sizeof(T) would overflow
            throw std::bad_alloc{};
        }
        const std::size_t bytes = n * sizeof(T);
        if (bytes > kMaxSize - (kAlign - 1)) {  // rounding up would overflow
            throw std::bad_alloc{};
        }
        // The mask must be as wide as std::size_t: ~63u is a 32-bit constant,
        // and allocations over 4GB would be truncated
        const std::size_t aligned = (bytes + kAlign - 1) & ~static_cast<std::size_t>(kAlign - 1);
        void* p = std::aligned_alloc(kAlign, aligned);
        if (!p)
            throw std::bad_alloc{};
        return static_cast<T*>(p);
    }

    void
    deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    // Object construction and destruction (handled by std::allocator_traits since C++17)
    template <typename U, typename... Args>
    void
    construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void
    destroy(U* p) noexcept {
        p->~U();
    }
};

class HGraphCache {
public:
    explicit HGraphCache(Allocator* allocator);

    ~HGraphCache() = default;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    std::vector<std::string>
    GetNeighbors(const std::string& source_id) const;

public:
    Allocator* const allocator_;

    // mapping from inner_id to source_id string
    // Vector<std::string> source_ids_;
    std::vector<std::string, Aligned64Allocator<std::string>> source_ids_;

    // mapping from source_id to neighbor inner_ids,
    // neighbors_[source_id][0] is self inner_id,
    // neighbors_[source_id][1...] are neighbor inner_ids
    UnorderedMap<std::string, Vector<InnerIdType>> neighbors_;
};
}  // namespace vsag
