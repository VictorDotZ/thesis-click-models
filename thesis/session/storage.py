from collections import defaultdict
from typing import Dict, Set

from torch_geometric.data import Data

from thesis.session.session import Document, Query, Session


class Storage:
    def __init__(self, sessions: Set[Session]):
        self._sessions = sessions

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

    @staticmethod
    def to_graph(storage: "Storage") -> Data:
        """
        Возвращает граф в формате torch_geometric
        """

        return Data()
