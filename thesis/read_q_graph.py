import torch

from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.data import NeighborSampler as OldNeighborSampler
from torch_geometric.nn import GATConv
from torch_geometric.loader.utils import get_input_nodes

data = torch.load(
    "./train_q_q.pth", weights_only=False
)

print(data)

print(data.edge_index.max())

loader = NeighborSampler(
    data,
    num_neighbors=[2],
    weight_attr="weight",
    # weight_attr="weight",
)

old_loader = OldNeighborSampler(
    data.edge_index,
    node_idx=None,
    sizes=[2],
    batch_size=3,
    return_e_id=False,
    shuffle=True,
    num_workers=1,
)

for _, sampled_qid, _ in old_loader:
    print(sampled_qid)
    break

raise Exception

for _ in range(10):
    qids = torch.tensor(
        [93627, 122536, 2458],  # 4828, 74986, 16071, 17061, 137207, 2030, 80832],
        dtype=torch.long,
    )

    _, old_n_id, old_edge_index = old_loader.sample(qids)

    # print("old n_id: ", old_n_id)
    # print("old edge_index: ", old_edge_index.edge_index)

    # 2458 имеет 3 соседей. При количестве соседей 10 достаются только 3, соотв там меньше соседей

    _, input_nodes, input_id = get_input_nodes(data, qids, qids)

    sample = loader.sample_from_nodes(
        NodeSamplerInput(input_id=input_id, node=input_nodes)
    )

    print(sample)

    edge_index = torch.stack([sample.row, sample.col])

    print("new n_id: ", sample.node)
    print("new edge_index: ", edge_index)


# for x in loader:
#    print(x)
# print(sampled_qid)
# print(sampled_index_tuple)

# sampled_qid, sampled_index = sampled_qid.cuda(), sampled_index_tuple[0].cuda()

# print(sampled_qid)
# print(sampled_index)

# print(batch.__dict__)
#   break
