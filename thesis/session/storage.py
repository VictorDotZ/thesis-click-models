from collections import defaultdict
from typing import Dict, List, Set, Tuple

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from thesis.session.session import Document, Query, Session


class Storage:
    def __init__(self, sessions: Set[Session]):
        if not self.__is_valid_sessions(sessions):
            raise ValueError(
                "Query and Document indices must be form a finite, "
                "contiguous sequence of non-negative integers "
                "starting at 0 (e.g., {0, 1, 2, ..., n})."
            )

        self._sessions = sessions

    def __is_valid_sessions(self, sessions: Set[Session]) -> bool:
        """
        Для того чтобы графы на тензорах нормально строились,
        индексы запросов и документов должны валидными
        """

        query_indices: Set[int] = set([session.query.q_id for session in sessions])

        document_indices: Set[int] = set(
            [document.doc_id for session in sessions for document in session.serp]
        )

        return self.__is_valid_indices(query_indices) and self.__is_valid_indices(
            document_indices
        )

    def __is_valid_indices(self, indices: Set[int]) -> bool:
        """
        Множество индексов валидно если:
            1. Существует минимальный элемент равный 0.
            2. Максимальный элемент + 1 равен количеству
                (<=> индексы линейно упорядоченны и между двумя соседними разность == 1)
        """

        if min(indices) != 0:
            return False

        if len(indices) != max(indices) + 1:
            return False

        return True

    @property
    def sessions(self) -> Set[Session]:
        return self._sessions

    @sessions.setter
    def sessions(self, value: Set[Session]):
        self._sessions = value

    def _get_clicked_documents_by_query(self) -> Dict[Query, Dict[Document, int]]:
        """
        Генерирует списки кликнутых документов по каждому запросу.
        Кроме клика в документ собирает статистику -- сколько раз клик был.
        Каждое появление документа среди кликнутых -- инкремент количества кликов.
        """
        clicked_documents_by_query: Dict[Query, Dict[Document, int]] = {}

        for session in self._sessions:
            if session.query not in clicked_documents_by_query:
                clicked_documents_by_query[session.query] = defaultdict(int)

            for document in session.clicked_documents:
                clicked_documents_by_query[session.query][document] += 1

        return clicked_documents_by_query

    def _get_queries_by_clicked_documents(self) -> Dict[Document, Dict[Query, int]]:
        """
        Генерирует списки запросов, по которым документ был кликнут.
        Кроме клика в документ собирает статистику -- сколько раз клик был по каждому запросу.
        Каждое появление запроса -- инкремент количества кликов.

        Пусть у нас есть три запроса q_1, q_2, q_3:
            * между q_1 и q_2 общий клик в d_1
            * между q_2 и q_3 общий клик в d_2

        Тогда в итоговом словаре для d_1 будут q_1 и q_2
        с количеством кликов в d_1 по каждому из запросов в отдельности.
        Аналогично для d_2.

        Итого пусть в d_1 так кликали 3 раза в q_2,
        а в d_2 так кликали 5 раз в q_3.
        """
        queries_by_clicked_documents: Dict[Document, Dict[Query, int]] = {}

        for session in self._sessions:
            for document in session.clicked_documents:
                if document not in queries_by_clicked_documents:
                    queries_by_clicked_documents[document] = defaultdict(int)

                queries_by_clicked_documents[document][session.query] += 1

        return queries_by_clicked_documents

    def get_queries_graph_as_adjacency_list(self) -> Dict[Query, Dict[Query, int]]:
        """
        Вершины графа -- запросы.
        Две вершины являются смежными, если по ним кликнули в один и тот же документ
        Возваращает граф в виде списка смежности.

        Хранит веса рёбер:
            * result[q_1][q_3] -- вес направленного ребра e_{1,3}
            * result[q_3][q_1] -- вес e_{3,1}
            вес e_{3,1}, вообще говоря, может быть не равен весу e_{1,3}
        """

        queries_by_clicked_documents = self._get_queries_by_clicked_documents()

        adjacency_list: Dict[Query, Dict[Query, int]] = {}

        # Для каждого документа
        for document in queries_by_clicked_documents:
            # Получаем запросы, по которым в него был клик
            # значения формировались так
            adjacency_queries: Dict[Query, int] = queries_by_clicked_documents[document]

            # Для каждого запроса из списка смежных
            for current_node_query in adjacency_queries:
                # добавляем его в словарь смежности если такой вершины ещё не было
                if current_node_query not in adjacency_list:
                    adjacency_list[current_node_query] = defaultdict(int)

                # добавляем все смежные запросы, кроме текущего, чтобы не было петель (self-loop)
                # TODO: Вообще говоря петли нужны, по идее их можно было бы сразу добавить тут
                for (
                    adjacency_query,
                    adjacency_query_weight,
                ) in adjacency_queries.items():
                    if adjacency_query != current_node_query:
                        adjacency_list[current_node_query][
                            adjacency_query
                        ] += adjacency_query_weight

        return adjacency_list

    def get_train_validation_test_split(
        self, proportion: Tuple[float, float, float]
    ) -> Tuple["Storage", "Storage", "Storage"]:
        """
        Делит текущее хранилище на Train, Validation и Test так, что:
            1. Количество сессий в каждом сплите пропорционально переданным параметрам
            2. Запросы также разделены пропорционально, если это возможно.
                Если сессия по запросу одна (или 2, т.е. недостаточно чтобы быть в каждом сплите),
                то такие сессии находятся преимущественно в Test и Validation (Test в приоритете)
        """
        raise NotImplementedError()

    @staticmethod
    def to_Q_Q_graph(storage: "Storage") -> Data:
        """
        Возвращает граф запросов (Query-Query) в формате torch_geometric
        """

        queries_graph_as_adjacency_list = storage.get_queries_graph_as_adjacency_list()

        edges: List[List[Query]] = [
            [source_query, destination_query]
            for source_query in queries_graph_as_adjacency_list
            for destination_query in queries_graph_as_adjacency_list[source_query]
        ]

        nodes: List[List[Query]] = [
            [source_query] for source_query in queries_graph_as_adjacency_list
        ]

        edges_indices: List[List[int]] = [
            [source_query.q_id, target_query.q_id]
            for [source_query, target_query] in edges
        ]

        nodes_indices: List[List[int]] = [[source_node.q_id] for [source_node] in nodes]

        edge_weigts: List[int] = [
            queries_graph_as_adjacency_list[source_query][target_query]
            for [source_query, target_query] in edges
        ]

        data = Data(
            x=torch.tensor(nodes_indices, dtype=torch.long),
            edge_index=torch.tensor(edges_indices, dtype=torch.float).t().contiguous(),
        )

        data["weight"] = torch.tensor(edge_weigts, dtype=torch.float)

        return data

    @staticmethod
    def save_Q_Q_graph(data: Data, save_path: Path) -> None:
        torch.save(data, save_path)

    @staticmethod
    def load_Q_Q_graph(load_path: Path) -> Data:
        return torch.load(load_path)

    @staticmethod
    def plot_graph(data: Data, save_path: Path) -> None:
        """
        Вспомагательный метод, позволяющий нарисовать граф.
        """
        # TODO: должен быть не в Storage классе
        G = to_networkx(data, to_undirected=False, edge_attrs=["weight"])
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=400)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for (u, v, d) in G.edges.data("weight")],
            width=2,
            connectionstyle="arc3, rad=-0.5",
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, connectionstyle="arc3, rad=-0.5"
        )
        plt.savefig(save_path)
