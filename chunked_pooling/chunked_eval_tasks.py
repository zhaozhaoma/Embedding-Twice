import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata
from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval


class LEMBNarrativeQARetrievalChunked(AbsTaskChunkedRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBNarrativeQARetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "LEMBNarrativeQARetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("narrativeqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1000-01-01", "2017-12-31"),
        form=["written"],
        domains=["Fiction", "Non-fiction"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @article{kocisky-etal-2018-narrativeqa,
            title = "The {N}arrative{QA} Reading Comprehension Challenge",
            author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
            Schwarz, Jonathan  and
            Blunsom, Phil  and
            Dyer, Chris  and
            Hermann, Karl Moritz  and
            Melis, G{\'a}bor  and
            Grefenstette, Edward",
            editor = "Lee, Lillian  and
            Johnson, Mark  and
            Toutanova, Kristina  and
            Roark, Brian",
            journal = "Transactions of the Association for Computational Linguistics",
            volume = "6",
            year = "2018",
            address = "Cambridge, MA",
            publisher = "MIT Press",
            url = "https://aclanthology.org/Q18-1023",
            doi = "10.1162/tacl_a_00023",
            pages = "317--328",
            abstract = "",
        }
        """,
        n_samples={_EVAL_SPLIT: 10804},
        avg_character_length={_EVAL_SPLIT: 326399.3},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = 'narrativeqa'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]  # dict_keys(['qid', 'text'])
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]  # dict_keys(['doc_id', 'text'])
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]  # dict_keys(['qid', 'doc_id'])
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True


class LEMBWikimQARetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBWikimQARetrieval.py
    """

    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBWikimQARetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "10039a580487dacecf79db69166e17ace3ede392",
            "name": "LEMBWikimQARetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("2wikimqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=None,
        socioeconomic_status=None,
        n_samples=None,
        avg_character_length=None,
        form=None,
        text_creation=None,
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{ho2020constructing,
                title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
                author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
                booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
                pages={6609--6625},
                year={2020}
            }
        """,
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: 500},
            "avg_character_length": {
                "test": {
                    "average_document_length": 37445.60333333333,
                    "average_query_length": 67.57,
                    "num_documents": 300,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = '2wikimqa'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True


class LEMBSummScreenFDRetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBWikimQARetrieval.py
    """

    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBSummScreenFDRetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "10039a580487dacecf79db69166e17ace3ede392",
            "name": "LEMBSummScreenFDRetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("summ_screen_fd subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=None,
        socioeconomic_status=None,
        n_samples=None,
        avg_character_length=None,
        form=None,
        text_creation=None,
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{ho2020constructing,
                title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
                author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
                booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
                pages={6609--6625},
                year={2020}
            }
        """,
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: 500},
            "avg_character_length": {
                "test": {
                    "average_document_length": 30854.327,
                    "average_query_length": 591.49,
                    "num_documents": 300,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = 'summ_screen_fd'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True


class LEMBQMSumRetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBWikimQARetrieval.py
    """

    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBQMSumRetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "10039a580487dacecf79db69166e17ace3ede392",
            "name": "LEMBQMSumRetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("qmsum subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=None,
        socioeconomic_status=None,
        n_samples=None,
        avg_character_length=None,
        form=None,
        text_creation=None,
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{ho2020constructing,
                title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
                author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
                booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
                pages={6609--6625},
                year={2020}
            }
        """,
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: 500},
            "avg_character_length": {
                "test": {
                    "average_document_length": 53335.817,
                    "average_query_length": 433.50,
                    "num_documents": 300,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = 'qmsum'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
