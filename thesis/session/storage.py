from typing import Set, Dict

from torch_geometric.data import Data
from .session import Query, Document, Session


class Storage:
    def __init__(self, sessions: Set[Session]):
        self._sessions = sessions

    @property
    def sessions(self) -> Set[Session]:
        return self._sessions

    @sessions.setter
    def sessions(self, value: Set[Session]):
        self._sessions = value

    # Метод, который генерирует список кликнутых документов по запросу среди всех сессий
    def _get_clicked_documents_by_query(self) -> Dict[Query, Set[Document]]:
        clicked_documents_by_query: Dict[Query, Set[Document]] = {}

        for session in self._sessions:
            if session.query.q_id not in clicked_documents_by_query:
                clicked_documents_by_query[session.query] = set()

            for document in session.clicked_documents:
                clicked_documents_by_query[session.query].add(document)

        return clicked_documents_by_query

    # Метод, который генерирует список запросов, по которым документ был кликнут
    def _get_queries_by_clicked_documents(self) -> Dict[Document, Set[Query]]:
        queries_by_clicked_documents: Dict[Document, Set[Query]] = {}

        for session in self._sessions:
            for document in session.clicked_documents:
                if document not in queries_by_clicked_documents:
                    queries_by_clicked_documents[document] = set()

                queries_by_clicked_documents[document].add(session.query)

        return queries_by_clicked_documents

    # Метод, который генерирует граф запросов в виде списка смежности
    def get_queries_graph_as_adjacency_list(self) -> Dict[Query, Set[Query]]:
        """
        Вершины графа -- запросы
        Две вершины являются смежными, если по ним кликнули в один и тот же документ
        Возваращает граф в виде списка смежности
        """

        queries_by_clicked_documents = self._get_queries_by_clicked_documents()

        adjacency_list: Dict[Query, Set[Query]] = {}

        # Для каждого документа
        for document in queries_by_clicked_documents:
            # Получаем запросы, по которым в него был клик
            adjacency_queries = queries_by_clicked_documents[document]

            # Для каждого запроса из списка смежных
            for query in adjacency_queries:
                # добавляем его в словарь смежности если такой вершины ещё не было
                if query not in adjacency_list:
                    adjacency_list[query] = set()

                # добавляем все смежные запросы, кроме текущего, чтобы не было петель (self-loop)
                # TODO: Вообще говоря петли нужны, по идее их можно было бы сразу добавить тут
                adjacency_list[query].update(adjacency_queries - {query})
                # adjacency_list[query].update(adjacency_queries)

        return adjacency_list

    @staticmethod
    def to_graph(storage: "Storage") -> Data:
        """
        Возвращает граф в формате torch_geometric
        """

        return Data()
