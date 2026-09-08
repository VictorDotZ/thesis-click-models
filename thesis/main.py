import networkx as nx
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx

edge_index = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [4, 0]], dtype=torch.long)

x = torch.tensor([[-1], [1], [4], [8], [8]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
data["weight"] = torch.tensor([1, 4, 8, 8], dtype=torch.float)

sampler = NeighborLoader(data=data, num_neighbors=[1] * 2, weight_attr="weight")


# node_id = torch.tensor([0], dtype=torch.long)

# sampled_data = sampler.sample_from_nodes(node_id)
print(data)

G = to_networkx(data, to_undirected=False, edge_attrs=["weight"])
pos = nx.spring_layout(G, seed=7)
nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(u, v) for (u, v, d) in G.edges.data("weight")],
    width=3,
    connectionstyle="arc3, rad=-0.8",
)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    connectionstyle="arc3, rad=-0.8",
)
# nx.draw_shell(G, with_labels=True)
plt.savefig("path.png")

for sample in sampler:
    print(sample)

# Выводим результат
# print("Соседи узла 0:")
# print("Node IDs:", sampled_data.node_id)  # ID узлов в подграфе
# print("Edge indices:", sampled_data.edge_index)  # Ребра в подграфе
# print("Node features:", sampled_data.x)  # Признаки узлов в подграфе

# print(sampler)
