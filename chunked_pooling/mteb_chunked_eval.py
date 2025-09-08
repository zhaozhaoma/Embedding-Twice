import logging
from typing import Any, Optional

import numpy as np
import torch
from mteb.abstasks import AbsTask
from mteb.evaluation.evaluators import RetrievalEvaluator
from mteb import ScoresDict
from mteb.tasks import Retrieval
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from chunked_pooling import chunked_pooling
from chonkie import TokenChunker, SentenceChunker
logger = logging.getLogger(__name__)




class AbsTaskChunkedRetrieval(AbsTask):
    def __init__(
        self,
        chunking_strategy: str = None,
        embedding_mode: str = None,
        tokenizer: Optional[Any] = None,
        prune_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        n_sentences: Optional[int] = None,
        embedding_model_name: Optional[str] = None,  # for semantic chunking
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            self.retrieval_task = getattr(
                Retrieval,
                self.metadata_dict['dataset'].get('name', None)
                or self.metadata_dict.get('name'),
            )()
        except:
            logger.warning('Could not initialize retrieval_task')
        self.chunking_strategy = chunking_strategy
        self.embedding_mode = embedding_mode
        self.tokenizer = tokenizer
        self.prune_size = prune_size
        self.chunking_args = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'n_sentences': n_sentences,
            'embedding_model_name': embedding_model_name,
        }

    def evaluate(
        self, model, split: str = "test", encode_kwargs: dict[str, Any] = {}, **kwargs
    ) -> dict[str, ScoresDict]:
        scores: dict[str, ScoresDict] = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )

            scores[hf_subset] = self._evaluate_monolingual(
                model,
                corpus,
                queries,
                relevant_docs,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )

        return scores

    def _evaluate_monolingual(
        self,
        model,
        corpus,
        queries,
        relevant_docs,
        encode_kwargs=None,
        **kwargs,
    ):
        query_ids = list(queries.keys())
        corpus_ids = list(corpus.keys())
        if self.embedding_mode == "truncation":
            corpus_embs = model.encode_corpus(corpus_ids, corpus, encode_kwargs['batch_size'])
            query_embs = model.encode_queries(query_ids, queries, encode_kwargs['batch_size'])
            similarity_matrix = np.dot(query_embs.cpu(), corpus_embs.T.cpu())

        elif self.embedding_mode == "chunk_avg":
            corpus_embs = model.encode_corpus_chunk_avg(corpus_ids, corpus, encode_kwargs['batch_size'])
            query_embs = model.encode_queries(query_ids, queries, encode_kwargs['batch_size'])
            similarity_matrix = np.dot(query_embs.cpu(), corpus_embs.T.cpu())

        elif self.embedding_mode == "chunk_weighted":
            query_embs = model.encode_queries(query_ids, queries, encode_kwargs['batch_size'])
            corpus_embs = model.encode_corpus_chunk_weight(corpus_ids, corpus, query_embs, encode_kwargs['batch_size'], encode_kwargs['tau'])
            corpus_embs = corpus_embs.permute(1, 0, 2)  # [num_query, num_docs, embedding_dim]
            similarity_matrix = torch.einsum('qnd,qd->qn', corpus_embs.cpu(), query_embs.cpu())
            similarity_matrix = similarity_matrix.cpu().to(torch.float32).numpy()

        elif self.embedding_mode == "chunk_twice_avg":
            corpus_embs = model.encode_corpus_chunk_twice_ppr(corpus_ids, corpus, encode_kwargs['batch_size'], encode_kwargs['replaced_layers'], encode_kwargs['top_k_context'])
            query_embs = model.encode_queries(query_ids, queries, encode_kwargs['batch_size'])
            similarity_matrix = np.dot(query_embs.cpu(), corpus_embs.T.cpu())

        elif self.embedding_mode == "chunk_twice_weighted":
            query_embs = model.encode_queries(query_ids, queries, encode_kwargs['batch_size'])
            corpus_embs = model.encode_corpus_chunk_twice_weight(corpus_ids, corpus, query_embs, encode_kwargs['batch_size'], encode_kwargs['tau'], encode_kwargs['replaced_layers'])
            corpus_embs = corpus_embs.permute(1, 0, 2)  # [num_query, num_docs, embedding_dim]
            similarity_matrix = torch.einsum('qnd,qd->qn', corpus_embs.cpu(), query_embs.cpu())
            similarity_matrix = similarity_matrix.cpu().to(torch.float32).numpy()

        else:
            raise NotImplementedError(f"This embedding mode is not supported: {self.embedding_mode}")

        k_values = 10

        doc_results = self.get_doc_results(query_ids, corpus_ids, similarity_matrix, k_values)
        ndcg, _map, recall, precision, _ = RetrievalEvaluator.evaluate(
            relevant_docs,
            doc_results,
            [k_values],
            ignore_identical_ids=kwargs.get('ignore_identical_ids', True),
        )
        mrr, _ = RetrievalEvaluator.evaluate_custom(
            relevant_docs,
            doc_results,
            [k_values],
            'mrr',
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        self._add_main_score(scores)
        return scores

    def _truncate_documents(self, corpus, max_length):
        for k, v in corpus.items():
            title_tokens = 0
            if 'title' in v:
                tokens = self.tokenizer(
                    v['title'] + ' ',
                    return_offsets_mapping=True,
                    max_length=max_length,
                )
                title_tokens = len(tokens.input_ids)
            tokens = self.tokenizer(
                v['text'],
                return_offsets_mapping=True,
                max_length=max_length - title_tokens,
            )
            last_token_span = tokens.offset_mapping[-2]
            v['text'] = v['text'][: last_token_span[1]]
        return corpus

    @staticmethod
    def get_doc_results(query_ids, corpus_ids, similarity_matrix, k_values):
        doc_results = dict()
        for i, query_id in enumerate(query_ids):
            query_results = dict()
            for idx, score in enumerate(similarity_matrix[i]):
                doc_id = corpus_ids[idx]
                query_results[doc_id] = float(score)
            # Sort results by score and only keep the top k scores
            sorted_query_results = dict(
                sorted(query_results.items(), key=lambda item: item[1], reverse=True)[:k_values]
            )
            doc_results[query_id] = sorted_query_results
        return doc_results

    def _calculate_k_values(self, max_chunks):
        k_values = [1, 3, 5, 10, 20]
        n = 2
        while 10 ** n < 100 * max_chunks:
            k_values.append(10 ** n)
            n += 1
        return k_values

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]
        print(f"main_score: {self.metadata.main_score} {scores[self.metadata.main_score]}")

    @staticmethod
    def _prune(queries, corpus, relevant_docs, prune_size):
        new_queries = {'test': {}}
        new_corpus = {'test': {}}
        new_relevant_docs = {'test': {}}
        for i, key in enumerate(relevant_docs['test']):
            if i >= prune_size:
                break
            new_relevant_docs['test'][key] = relevant_docs['test'][key]
            for x in relevant_docs['test'][key]:
                new_corpus['test'][x] = corpus['test'][x]
            new_queries['test'][key] = queries['test'][key]
        return new_queries, new_corpus, new_relevant_docs

    def _calculate_metrics_from_split(*args, **kwargs):
        pass

    def _evaluate_subset(*args, **kwargs):
        pass
