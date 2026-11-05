
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

#include "sparse_graph_datacell.h"

#include <fmt/format.h>
#include <sstream>

#include "graph_interface_test.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "sparse_graph_datacell_parameter.h"
#include "unittest.h"
using namespace vsag;

void
TestSparseGraphDataCell(const GraphInterfaceParamPtr& param,
                        const IndexCommonParam& common_param,
                        bool test_delete) {
    auto count = GENERATE(1000, 2000);
    auto max_id = 10000;

    auto graph = GraphInterface::MakeInstance(param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(param, common_param);
    test.BasicTest(max_id, count, other, test_delete);
}

TEST_CASE("SparseGraphDataCell Basic Test", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32, 64);
    auto is_support_delete = GENERATE(true, false);

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->support_delete_ = is_support_delete;
    TestSparseGraphDataCell(graph_param, common_param, is_support_delete);
}

TEST_CASE("SparseGraphDataCell Remove Test", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32);
    auto is_support_delete = GENERATE(true);
    auto remove_flag_bit = GENERATE(4, 8);

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->support_delete_ = is_support_delete;
    graph_param->remove_flag_bit_ = remove_flag_bit;
    TestSparseGraphDataCell(graph_param, common_param, is_support_delete);
}

TEST_CASE("SparseGraphDataCell Merge Test", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32, 64);
    auto is_support_delete = GENERATE(true, false);
    int count = 1000;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->support_delete_ = is_support_delete;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(graph_param, common_param);
    test.MergeTest(other, count);
}

TEST_CASE("SparseGraphDataCell Reverse Edges", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 32;
    auto max_degree = 32;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    test.ReverseEdgeTest(100);
}

TEST_CASE("SparseGraphDataCell Move", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 32;
    auto max_degree = 32;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    test.MoveTest(100);
}

TEST_CASE("SparseGraphDataCell Move same id keeps neighbors", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);

    Vector<InnerIdType> empty(allocator.get());
    Vector<InnerIdType> neighbors(allocator.get());
    neighbors.emplace_back(1);
    neighbors.emplace_back(2);

    graph->InsertNeighborsById(0, neighbors);
    graph->InsertNeighborsById(1, empty);
    graph->InsertNeighborsById(2, empty);

    graph->Move(0, 0);

    Vector<InnerIdType> moved_neighbors(allocator.get());
    graph->GetNeighbors(0, moved_neighbors);
    REQUIRE(moved_neighbors.size() == 2);
    REQUIRE(moved_neighbors[0] == 1);
    REQUIRE(moved_neighbors[1] == 2);

    Vector<InnerIdType> incoming(allocator.get());
    graph->GetIncomingNeighbors(1, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 0);
}

TEST_CASE("SparseGraphDataCell decodes stored neighbors for reverse-edge updates",
          "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->support_delete_ = true;
    graph_param->remove_flag_bit_ = 4;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);

    Vector<InnerIdType> empty(allocator.get());
    Vector<InnerIdType> to_two(allocator.get());
    Vector<InnerIdType> to_three(allocator.get());
    to_two.emplace_back(2);
    to_three.emplace_back(3);

    graph->InsertNeighborsById(2, empty);
    graph->InsertNeighborsById(3, empty);
    graph->DeleteNeighborsById(2);
    graph->InsertNeighborsById(1, to_two);

    Vector<InnerIdType> incoming(allocator.get());
    graph->GetIncomingNeighbors(2, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 1);

    graph->InsertNeighborsById(1, to_three);

    graph->GetIncomingNeighbors(2, incoming);
    REQUIRE(incoming.empty());

    graph->GetIncomingNeighbors(3, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 1);
}

TEST_CASE("SparseGraphDataCell PermuteEntries Keeps Total Count After Move",
          "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->support_delete_ = true;
    graph_param->remove_flag_bit_ = 8;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);

    constexpr InnerIdType N = 16;
    for (InnerIdType id = 0; id < N; ++id) {
        Vector<InnerIdType> neighbors(allocator.get());
        neighbors.emplace_back((id + 1) % N);
        neighbors.emplace_back((id + 3) % N);
        graph->InsertNeighborsById(id, neighbors);
    }
    REQUIRE(graph->TotalCount() == N);
    REQUIRE(graph->CheckIdExists(N - 1));

    // same call as the HGraph force-remove path: the moved-from id is erased from
    // the neighbor map, while total_count_ must keep describing the N id slots
    graph->Move(N - 1, 2);
    REQUIRE_FALSE(graph->CheckIdExists(N - 1));
    REQUIRE(graph->TotalCount() == N);

    // permutation built like HGraph::build_load_permutation(): reversed order
    Vector<InnerIdType> perm(allocator.get());
    Vector<InnerIdType> imap(allocator.get());
    perm.resize(N);
    imap.resize(N);
    for (InnerIdType i = 0; i < N; ++i) {
        perm[i] = N - 1 - i;
        imap[N - 1 - i] = i;
    }

    uint64_t edge_count_before = 0;
    for (InnerIdType id = 0; id < N; ++id) {
        Vector<InnerIdType> neighbors(allocator.get());
        graph->GetNeighbors(id, neighbors);
        edge_count_before += neighbors.size();
    }

    graph->PermuteEntries(perm, imap);

    // the id space (and thus TotalCount) must survive the permutation
    REQUIRE(graph->TotalCount() == N);
    uint64_t edge_count_after = 0;
    for (InnerIdType id = 0; id < N; ++id) {
        Vector<InnerIdType> neighbors(allocator.get());
        graph->GetNeighbors(id, neighbors);
        edge_count_after += neighbors.size();
    }
    REQUIRE(edge_count_after == edge_count_before);
}

TEST_CASE("SparseGraphDataCell PermuteEntries Serialize Roundtrip Keeps Total Count",
          "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->support_delete_ = true;
    graph_param->remove_flag_bit_ = 8;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);

    constexpr InnerIdType N = 16;
    for (InnerIdType id = 0; id < N; ++id) {
        Vector<InnerIdType> neighbors(allocator.get());
        neighbors.emplace_back((id + 1) % N);
        graph->InsertNeighborsById(id, neighbors);
    }

    graph->Move(N - 1, 2);

    Vector<InnerIdType> perm(allocator.get());
    Vector<InnerIdType> imap(allocator.get());
    perm.resize(N);
    imap.resize(N);
    for (InnerIdType i = 0; i < N; ++i) {
        perm[i] = N - 1 - i;
        imap[N - 1 - i] = i;
    }
    graph->PermuteEntries(perm, imap);
    REQUIRE(graph->TotalCount() == N);

    std::stringstream ss;
    vsag::IOStreamWriter writer(ss);
    graph->Serialize(writer);

    auto loaded = GraphInterface::MakeInstance(graph_param, common_param);
    vsag::IOStreamReader reader(ss);
    loaded->Deserialize(reader);

    // Deserialize must restore the id-space size from the stream instead of
    // shrinking it to the neighbor-map size (the map lost one id to Move)
    REQUIRE(loaded->TotalCount() == N);
    // the hole survives the roundtrip: old id N-1 was erased by Move and maps
    // to new id 0 under the reversed permutation
    REQUIRE_FALSE(loaded->CheckIdExists(0));
    for (InnerIdType id = 1; id < N; ++id) {
        REQUIRE(loaded->CheckIdExists(id));
    }
}
