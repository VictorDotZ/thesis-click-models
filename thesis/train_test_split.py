from dataclasses import dataclass
from typing import List


@dataclass
class Session:
    session_id: int
    query_id: int
    clicked_documents: List[int]
    documents: List[int]


sessions_by_query_id: dict[int, List[Session]] = {}

with open(
    "./data/iphone-20240201", "r", encoding="utf-8"
) as f_in:
    for line in f_in:
        splits = line.strip().split("\t")
        session_id = int(splits[0])
        query_id = int(splits[1])
        clicked_documents = list(map(int, splits[2].split(",")))
        documents = list(map(int, splits[3].split(",")))

        session = Session(
            session_id=session_id,
            query_id=query_id,
            clicked_documents=clicked_documents,
            documents=documents,
        )

        if query_id not in sessions_by_query_id:
            sessions_by_query_id[query_id] = []

        sessions_by_query_id[query_id].append(session)


def is_splittable(query_id: int) -> bool:
    documents: set[int] = set()
    # 1. Retrieve all the sessions related to a given query
    for session in sessions_by_query_id[query_id]:
        for document in session.documents:
            documents.add(document)

    # 2. Consider an url that appeared both in position 1 and some other positions;
    # Проверять будем так:
    # Берем какой-то документ
    found_first_doc_pos: bool = False
    found_not_first_doc_pos: bool = False

    for document in documents:
        # Затем ищем по сессиям, он должен встретиться как минимум один раз на первой позиции
        # и как минимум ещё один раз не на первой
        for session in sessions_by_query_id[query_id]:
            if document not in session.documents:
                continue
            pos = session.documents.index(document)
            if not found_first_doc_pos and pos == 1:
                found_first_doc_pos = True
            if not found_not_first_doc_pos and pos > 1:
                found_not_first_doc_pos = True

        # Документ найден по сессиям в нужных качествах
        if found_not_first_doc_pos and found_first_doc_pos:
            return True
        else:
            found_not_first_doc_pos = False
            found_first_doc_pos = False

    return found_not_first_doc_pos and found_first_doc_pos


splitable_queries: int = 0

for query_id in sessions_by_query_id:
    if is_splittable(query_id):
        splitable_queries += 1


print(f"total queries: {len(sessions_by_query_id)}")
print(f"splitable queries: {splitable_queries}")
