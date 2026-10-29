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

#include "compressed_graph_datacell.h"

#include <algorithm>

#include "graph_datacell_parameter.h"
#include "index_common_param.h"
#include "vsag_exception.h"

namespace vsag {

CompressedGraphDataCell::CompressedGraphDataCell(const GraphInterfaceParamPtr& graph_param,
                                                 const IndexCommonParam& common_param)
    : CompressedGraphDataCell(
          std::dynamic_pointer_cast<CompressedGraphDatacellParameter>(graph_param), common_param) {
}

CompressedGraphDataCell::CompressedGraphDataCell(const CompressedGraphDatacellParamPtr& graph_param,
                                                 const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()), neighbor_sets_(allocator_) {
    this->maximum_degree_ = graph_param->max_degree_;
    this->max_capacity_ = 0;
    GraphInterface::allocator_ = common_param.allocator_.get();
    if (graph_param->support_duplicate_) {
        this->InitDuplicateTracker();
    }
    if (graph_param->use_reverse_edges_) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "CompressedGraphDataCell does not support reverse edges");
    }
}

CompressedGraphDataCell::~CompressedGraphDataCell() {
    for (auto& encoder : neighbor_sets_) {
        if (encoder) {
            encoder->Clear(allocator_);
            encoder.reset();
        }
    }
}

void
CompressedGraphDataCell::InsertNeighborsById(InnerIdType id,
                                             const Vector<InnerIdType>& neighbor_ids) {
    if (neighbor_ids.size() > this->maximum_degree_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("insert neighbors count {} more than {}",
                                        neighbor_ids.size(),
                                        this->maximum_degree_));
    }

    Vector<InnerIdType> tmp(neighbor_ids.begin(), neighbor_ids.end(), allocator_);
    std::sort(tmp.begin(), tmp.end());

    std::unique_ptr<EliasFanoEncoder> new_node;
    if (not tmp.empty()) {
        new_node = std::make_unique<EliasFanoEncoder>();
        new_node->Encode(tmp, max_capacity_, allocator_);
    }
    if (neighbor_sets_[id]) {
        neighbor_sets_[id]->Clear(allocator_);
    }
    neighbor_sets_[id] = std::move(new_node);

    InnerIdType current = total_count_.load();
    while (current < id + 1 && !total_count_.compare_exchange_weak(current, id + 1)) {
    }
}

uint32_t
CompressedGraphDataCell::GetNeighborSize(InnerIdType id) const {
    return (neighbor_sets_[id]) ? neighbor_sets_[id]->Size() : 0;
}

void
CompressedGraphDataCell::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    neighbor_ids.clear();
    if (GetNeighborSize(id) > 0) {
        neighbor_sets_[id]->DecompressAll(neighbor_ids);
    }
}

void
CompressedGraphDataCell::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);

    auto vertex_num = this->neighbor_sets_.size();
    StreamWriter::WriteObj(writer, vertex_num);
    for (InnerIdType id = 0; id < vertex_num; id++) {
        if (GetNeighborSize(id) == 0) {
            uint8_t zero = 0;
            StreamWriter::WriteObj(writer, zero);
        } else {
            const EliasFanoEncoder& encoder = *neighbor_sets_[id];
            StreamWriter::WriteObj(writer, encoder.num_elements);
            StreamWriter::WriteObj(writer, encoder.low_bits_width);
            StreamWriter::WriteObj(writer, encoder.low_bits_size);
            StreamWriter::WriteObj(writer, encoder.high_bits_size);
            for (uint64_t j = 0; j < encoder.low_bits_size + encoder.high_bits_size; j++) {
                StreamWriter::WriteObj(writer, encoder.bits[j]);
            }
        }
    }
}

void
CompressedGraphDataCell::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    uint64_t vertex_num;
    StreamReader::ReadObj(reader, vertex_num);
    Resize(vertex_num);
    for (uint64_t id = 0; id < vertex_num; ++id) {
        uint8_t num_elements = 0;
        StreamReader::ReadObj(reader, num_elements);
        if (num_elements > 0) {
            this->neighbor_sets_[id] = std::make_unique<EliasFanoEncoder>();
            EliasFanoEncoder& encoder = *this->neighbor_sets_[id];
            encoder.num_elements = num_elements;
            StreamReader::ReadObj(reader, encoder.low_bits_width);
            StreamReader::ReadObj(reader, encoder.low_bits_size);
            StreamReader::ReadObj(reader, encoder.high_bits_size);

            encoder.bits = static_cast<uint64_t*>(allocator_->Allocate(
                (encoder.low_bits_size + encoder.high_bits_size) * sizeof(uint64_t)));
            for (uint64_t j = 0; j < encoder.low_bits_size + encoder.high_bits_size; j++) {
                StreamReader::ReadObj(reader, encoder.bits[j]);
            }
        }
    }
}

void
CompressedGraphDataCell::Resize(InnerIdType new_size) {
    if (new_size < this->max_capacity_) {
        return;
    }
    neighbor_sets_.resize(new_size);
    this->max_capacity_ = new_size;
    if (this->duplicate_tracker_ != nullptr) {
        this->duplicate_tracker_->Resize(new_size);
    }
}

bool
CompressedGraphDataCell::CheckIdExists(InnerIdType id) const {
    return id < neighbor_sets_.size() && neighbor_sets_[id] != nullptr;
}

uint64_t
CompressedGraphDataCell::GetMemoryUsage() const {
    auto memory = sizeof(CompressedGraphDataCell);
    memory += neighbor_sets_.size() * sizeof(std::nullptr_t);
    for (const auto& encoder : neighbor_sets_) {
        if (encoder) {
            memory += encoder->SizeInBytes();
        }
    }
    return static_cast<uint64_t>(memory);
}

Vector<InnerIdType>
CompressedGraphDataCell::GetIds() const {
    Vector<InnerIdType> ids(allocator_);
    for (InnerIdType id = 0; id < static_cast<InnerIdType>(neighbor_sets_.size()); ++id) {
        if (neighbor_sets_[id] != nullptr) {
            ids.push_back(id);
        }
    }
    return ids;
}

void
CompressedGraphDataCell::PermuteEntries(const Vector<InnerIdType>& perm,
                                        const Vector<InnerIdType>& imap) {
    const uint64_t total_count = this->total_count_;
    if (perm.size() < total_count || imap.size() < total_count) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "compressed graph perm size is smaller than total count");
    }

    Vector<std::unique_ptr<EliasFanoEncoder>> new_neighbor_sets(allocator_);
    new_neighbor_sets.resize(neighbor_sets_.size());
    Vector<InnerIdType> neighbors(allocator_);
    Vector<InnerIdType> mapped(allocator_);
    for (InnerIdType new_id = 0; new_id < total_count; ++new_id) {
        const auto old_id = perm[new_id];
        neighbors.clear();
        mapped.clear();
        if (old_id < neighbor_sets_.size() && neighbor_sets_[old_id] != nullptr) {
            neighbor_sets_[old_id]->DecompressAll(neighbors);
            mapped.reserve(neighbors.size());
            for (const auto old_neighbor : neighbors) {
                if (old_neighbor < imap.size()) {
                    mapped.emplace_back(imap[old_neighbor]);
                }
            }
            std::sort(mapped.begin(), mapped.end());
            auto encoder = std::make_unique<EliasFanoEncoder>();
            encoder->Encode(mapped, this->max_capacity_, allocator_);
            new_neighbor_sets[new_id] = std::move(encoder);
        }
    }

    for (auto& encoder : neighbor_sets_) {
        if (encoder) {
            encoder->Clear(allocator_);
        }
    }
    neighbor_sets_.swap(new_neighbor_sets);
}

}  // namespace vsag
