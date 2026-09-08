from argparse import ArgumentParser

from thesis.session.PyClickSessionStorage import PyClickSessionStorage

# from pyclick.click_models.VSDBN import VSDBN

from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List

from itertools import chain


import math

from pyclick.click_models.CTR import GCTR
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Param import ParamMLE
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer
from pyclick.click_models.Evaluation import Perplexity

__author__ = "Victor Zenin"


@dataclass
class Ratio:
    numerator: float = 0.0
    denominator: float = 0.01


@dataclass
class ViewInfo:
    stop_view_time: List[float] = field(default_factory=list)
    view_time: List[float] = field(default_factory=list)


class VSDBN(ClickModel):
    def __init__(self):
        self._attr = defaultdict(lambda: defaultdict(Ratio))
        self._lambda = defaultdict(lambda: defaultdict(Ratio))
        self._time = defaultdict(lambda: defaultdict(ViewInfo))

        self._beta = defaultdict(lambda: defaultdict(float))

        self.max_steps = 50

        self._eps = 10**-6

    def f(self, query, document, beta_u):
        all_time = list(chain(self._time[query].values()))

        # print(list(all_time))

        all_view_time = list(chain(*map(lambda x: x.view_time, all_time)))

        # print(list(map(lambda x: x.view_time, all_time)))
        # print(list(all_view_time))

        all_stop_view_time = list(chain(*map(lambda x: x.stop_view_time, all_time)))

        # print(f"all_time: {list(all_time)}")
        # print(list(all_stop_view_time))

        total_time = sum(all_view_time)

        tmp = self._eps

        for T_i in all_stop_view_time:
            print(f"beta_n: {beta_u}, T_i: {T_i}")
            tmp += (T_i * math.pow(math.e, -beta_u * T_i)) / (
                1.0 - math.pow(math.e, -beta_u * T_i) + self._eps
            )

        return tmp - total_time

    def H(self, query, document, beta_u):
        # print(f"beta_u: {beta_u}")
        tmp = self._eps
        # print(self._time[query][document].stop_view_time)
        # print(self._time[query][document].view_time)

        all_time = list(chain(self._time[query].values()))
        all_stop_view_time = list(chain(*map(lambda x: x.stop_view_time, all_time)))

        for T_i in all_stop_view_time:
            # print(f"T_i: {T_i}")

            tmp -= (T_i * T_i * math.pow(math.e, -beta_u * T_i)) / (
                math.pow(1.0 - math.pow(math.e, -beta_u * T_i), 2.0) + self._eps
            )

        return tmp

    def optimize(self, query, document):
        # if len(self._time[query][document].stop_view_time) == 0:
        # return 0.5

        # if len(self._time[query][document].view_time) == 0:
        # return 0.5

        # return 0.5

        beta_n = 1.0 / (
            sum(
                chain(
                    *map(lambda x: x.stop_view_time, chain(self._time[query].values()))
                )
            )
            + self._eps
        )

        # beta_n = 1.0
        step = 0

        # p = len(self._time[query][document].stop_view_time) > 1000
        # if p:
        # print(query, document)

        # print(f"last_time: {self._time[query][document].stop_view_time}")
        # print(f"all_time: {self._time[query][document].view_time}")

        while True:
            beta_n_plus_one = beta_n - self.f(query, document, beta_n) / self.H(
                query, document, beta_n
            )
            step += 1

            if query == "156105" and document == "563947":
                print(f"beta_n: {beta_n}")
                print(f"f(beta_n): {self.f(query, document, beta_n)}")
                print(f"f'(beta_n): {self.H(query, document, beta_n)}")
                print(f"beta_n+1: {beta_n_plus_one}")

            if beta_n_plus_one < 0.0:
                beta_n_plus_one = self._eps

            if math.fabs(beta_n_plus_one - beta_n) < self._eps or step > self.max_steps:
                # if query == "116575" and document == "414313":
                # print(beta_n_plus_one, beta_n, step)
                if step > self.max_steps:
                    print(beta_n_plus_one, beta_n)
                # print(f"f(beta_n): {self.f(query, document, beta_n)}")

                return beta_n_plus_one

            beta_n = beta_n_plus_one

        return beta_n

    def train(self, search_sessions):
        for session in search_sessions:
            query = session.query
            results = session.web_results
            last_click_rank = session.get_last_click_rank()

            if len(results) == last_click_rank:
                continue

            # print(query)
            # print(session)
            # print(last_click_rank)
            for rank in range(last_click_rank + 1):
                document = results[rank]
                doc_id = document.id

                # Вычисляем привлекательность a_u
                self._attr[query][doc_id].denominator += 1

                if document.click:
                    self._attr[query][doc_id].numerator += 1

                # вычисляем lambda_u

                if document.view_time > 0.0:
                    self._lambda[query][doc_id].numerator += 1
                    self._lambda[query][doc_id].denominator += document.view_time

                # собираем статистику для beta_u

                if (
                    document.click
                    and document.view_time > 0.0
                    and rank != last_click_rank
                ):
                    self._time[query][doc_id].view_time.append(document.view_time)

                # if document.click

                if (
                    document.click
                    and rank == last_click_rank
                    and document.view_time > 0.0
                ):
                    self._time[query][doc_id].stop_view_time.append(document.view_time)

        for query in self._time:
            for document in self._time[query]:
                self._beta[query][document] = self.optimize(query, document)

    def get_conditional_click_probs(self, search_session):
        return
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()
            sat = session_params[rank][self.param_names.sat].value()

            if result.click:
                click_prob = attr * exam
                exam = 1 - sat
            else:
                click_prob = 1 - attr * exam
                exam *= (1 - attr) / click_prob

            click_probs.append(click_prob)

        return click_probs

    def get_full_click_probs(self, search_session):
        query = search_session.query
        results = search_session.web_results

        exam = 1

        click_probs = []

        for j, result in enumerate(results):
            doc_id = result.id

            attr = (
                self._attr[query][doc_id].numerator
                / self._attr[query][doc_id].denominator
            )

            lambda_ = (
                self._lambda[query][doc_id].numerator
                / self._lambda[query][doc_id].denominator
            )

            beta = self._beta[query][doc_id]

            sat = beta / (lambda_ + beta + self._eps)
            # sat = 1 - math.pow(math.e, -beta * (1 / (lambda_ + self._eps)))

            # print(sat)

            # print(f"attr: {attr}")
            # print(f"exam: {}")
            click_probs.append(attr * exam)
            exam *= (1 - sat) * attr + (1 - attr)

        return click_probs

    def predict_relevance(self, query, doc_id):
        attr = (
            self._attr[query][doc_id].numerator / self._attr[query][doc_id].denominator
        )

        lambda_ = (
            self._lambda[query][doc_id].numerator
            / self._lambda[query][doc_id].denominator
        )

        beta = self._beta[query][doc_id]

        sat = beta / (lambda_ + beta + self._eps)
        # sat = 1 - math.pow(math.e, -beta * (1 / (lambda_ + self._eps)))

        return attr * sat


if __name__ == "__main__":
    parser = ArgumentParser("model")

    parser.add_argument(
        "--sessions", type=str, default=None, help="path to sessions in .txt format"
    )

    args = parser.parse_args()

    storage = PyClickSessionStorage(args.sessions)

    print(str(storage.get_train_sessions()[0]))

    model = VSDBN()

    m2 = SDBN()

    m2.train(storage.get_train_sessions())

    model.train(storage.get_train_sessions())

    perplexity = Perplexity()

    ppl = perplexity.evaluate(model, storage.get_test_sessions())[0]

    print(ppl)

    ppl = perplexity.evaluate(m2, storage.get_test_sessions())[0]

    print(ppl)

    m3 = GCTR()
    m3.train(storage.get_train_sessions())
    ppl = perplexity.evaluate(m3, storage.get_test_sessions())[0]
    print(ppl)

    max_total_view_time = 0
    q_max = None
    d_max = None

    @dataclass
    class DocStat:
        avg_view: float = 0.0
        click_rel: float = 0.0
        view_rel: float = 0.0

    q_d_v = defaultdict(lambda: defaultdict(DocStat))

    for query in model._time:
        for document in model._time[query]:
            time = model._time[query][document]

            if query == "131785" and document == "498623":
                print(time)

            q_d_v[query][document].avg_view = (
                sum(time.view_time + time.stop_view_time)
            ) / len(time.view_time + time.stop_view_time)
            q_d_v[query][document].click_rel = m3.predict_relevance(query, document)
            q_d_v[query][document].view_rel = model.predict_relevance(query, document)

            total_view_time = sum(time.view_time)
            if total_view_time > max_total_view_time:
                max_total_view_time = total_view_time
                q_max = query
                d_max = document

    print(q_max, d_max)
    print(max_total_view_time)
    print(m2.predict_relevance(q_max, d_max))
    print(model.predict_relevance(q_max, d_max))

    @dataclass
    class QueryStat:
        click_pos_1_avg_view: float = 0.0
        view_pos_1_avg_view: float = 0.0

    q_t = defaultdict(QueryStat)

    for query in q_d_v:
        serp = []

        for document in q_d_v[query]:
            serp.append(q_d_v[query][document])
            if query == "131785":
                print(document)
                print(q_d_v[query][document])

        click_sort = sorted(serp, key=lambda ds: ds.click_rel, reverse=True)
        views_sort = sorted(serp, key=lambda ds: ds.view_rel, reverse=True)

        q_t[query].click_pos_1_avg_view = click_sort[0].avg_view
        q_t[query].view_pos_1_avg_view = views_sort[0].avg_view

    click_pos_1_avg_view = sum(
        map(lambda x: x.click_pos_1_avg_view, q_t.values())
    ) / len(q_t)
    view_pos_1_avg_view = sum(map(lambda x: x.view_pos_1_avg_view, q_t.values())) / len(
        q_t
    )

    # for query in q_t:
    #     stat = q_t[query]
    #
    #     if stat.view_pos_1_avg_view < stat.click_pos_1_avg_view:
    #         print(query)
    #         print(q_t[query])

    print("avg view:")
    print(click_pos_1_avg_view)
    print(view_pos_1_avg_view)

    # query = "55816"
    query = "131785"
    for doc in ["498623", "385012", "498538", "498486"]:
        # doc = "744529"
        # doc = "756129"
        print(query, doc)
        print("vsdbn:")
        print(f"a_u: {model._attr[query][doc]}")
        print(f"lambda: {model._lambda[query][doc]}")
        print(f"beta: {model._beta[query][doc]}")
        print(
            f"sat: {model._beta[query][doc] / (model._beta[query][doc] + model._lambda[query][doc].numerator / (model._lambda[query][doc].denominator + model._eps) + model._eps)}"
        )
        print(f"rel: {model.predict_relevance(query, doc)}")

        print("sdbn:")
        print(f"a_u: {m2.params[m2.param_names.attr].get(query, doc).value()}")
        print(f"s_u: {m2.params[m2.param_names.sat].get(query, doc).value()}")
        # print(f"beta: {m2._beta[query][doc]}")
        print(f"rel: {m2.predict_relevance(query, doc)}")
