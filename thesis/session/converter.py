from typing import Dict, Iterable, List, Set

from thesis.session.session import Document, Query, Session


class Converter:
    def __init__(self):
        pass

    def convert(self, session: Dict) -> Session:
        q_id = session["qids"][0]
        query = Query(q_id=q_id, text=str(q_id))

        uids = session["uidsS"][0]
        clicks = session["clicksS"][0]

        serp: List[Document] = list(
            map(lambda doc_id: Document(doc_id=doc_id, url=str(doc_id)), uids)
        )
        clicked_documents: List[Document] = []

        for i, is_clicked in enumerate(clicks):
            if is_clicked:
                clicked_documents.append(serp[i])

        return Session(
            session_id=session["sid"],
            query=query,
            serp=serp,
            clicked_documents=clicked_documents,
        )

    def convert_batch(self, sessions: Iterable[Dict]) -> Set[Session]:
        return set(map(self.convert, sessions))
