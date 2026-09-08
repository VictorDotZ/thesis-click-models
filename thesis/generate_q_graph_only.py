from pathlib import Path
from typing import List

from thesis.session.converter import Converter
from thesis.session.storage import Storage

from torch_geometric.loader import NeighborLoader

# infos_per_session: List | None = None

with open(
    "../GraphCM/data/VK/infos_per_session.list", "r", encoding="utf-8"
) as f_in:
    infos_per_session = eval(f_in.read())

converter = Converter()

train_val_idx = 340760
val_test_idx = 383355

train_sessions = converter.convert_batch(infos_per_session[:train_val_idx])
test_sessions = converter.convert_batch(infos_per_session)

train_storage = Storage(train_sessions)
test_storage = Storage(test_sessions)

Storage.save_Q_Q_graph(
    save_path=Path("./train_q_q.pth"), data=Storage.to_Q_Q_graph(train_storage)
)
Storage.save_Q_Q_graph(
    save_path=Path("./test_q_q.pth"), data=Storage.to_Q_Q_graph(test_storage)
)

# loader = NeighborLoader(
#     Storage.to_Q_Q_graph(test_storage),
#     num_neighbors=[10],
#     batch_size=12,
#     weight_attr="weight",
#     shuffle=True,
#     num_workers=12,
# )
#
# for t in loader:
#     print(t)
#     break
