from thesis.session.storage import Storage
from thesis.session.session import Query, Document, Session


def test_queries_graph():
    d_1 = Document(url="url1", doc_id=1)
    d_2 = Document(url="url2", doc_id=2)
    d_3 = Document(url="url3", doc_id=3)
    d_4 = Document(url="url4", doc_id=4)

    q_1 = Query(text="q1", q_id=1)
    q_2 = Query(text="q2", q_id=2)
    q_3 = Query(text="q3", q_id=3)

    s_1 = Session(
        session_id=1,
        query=q_1,
        serp=[d_1, d_2, d_3],
        clicked_documents=[d_1, d_3],
    )
    s_2 = Session(session_id=2, query=q_2, serp=[d_3, d_4], clicked_documents=[d_3])

    s_3 = Session(
        session_id=3,
        query=q_3,
        serp=[d_1, d_2, d_3, d_4],
        clicked_documents=[d_2, d_1],
    )

    storage = Storage([s_1, s_2, s_3])

    adjacency_list = storage.get_queries_graph_as_adjacency_list()

    assert q_2 in adjacency_list[q_1]
    assert q_1 in adjacency_list[q_2]

    assert q_3 in adjacency_list[q_1]
    assert q_2 not in adjacency_list[q_3]
    assert q_1 in adjacency_list[q_3]

    assert q_1 not in adjacency_list[q_1], "self loop exists"
