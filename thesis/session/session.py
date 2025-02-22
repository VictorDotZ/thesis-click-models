from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    url: str
    doc_id: int

    def __hash__(self):
        return hash(self.doc_id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.doc_id == other.doc_id
        return False


@dataclass
class Query:
    text: str
    q_id: int

    def __hash__(self):
        return hash(self.q_id)

    def __eq__(self, other):
        if isinstance(other, Query):
            return self.q_id == other.q_id
        return False


class Session:
    def __init__(
        self,
        session_id: int,
        query: Query,
        clicked_documents: List[Document],
        serp: List[Document],
    ):
        self.session_id = session_id
        self.query = query
        self.clicked_documents = clicked_documents
        self.serp = serp
