import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn.functional as F
from torch import Tensor
from chonkie import TokenChunker, SentenceChunker
from tqdm import tqdm
from collections import defaultdict
from typing import List
import math
import time


class EmbeddingsWrapper(nn.Module):
    def __init__(
        self, model_name, embedding_mode, chunking_strategy, chunk_size, chunk_overlap, is_normalize=True, pooling_method='mean',
            query_instruction=None, document_instruction=None, twice_ratio=0.75, **model_kwargs
    ):
        super().__init__()
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        if model_name in ["Qwen/Qwen3-Embedding-0.6B"]:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = config.hidden_size
        self._get_encoder_layers()
        self.num_layers = len(self.layers)
        self.is_normalize = is_normalize
        self.pooling_method = pooling_method
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unk_token_id = self._tokenizer.unk_token_id
        self.unk_token = self._tokenizer.unk_token
        self.twice_ratio = twice_ratio  # Add twice_ratio parameter
        self.replaced_layers = int(self.num_layers * self.twice_ratio)  # Calculate replaced_layers based on ratio
        
        # Add metrics collector reference
        self._metrics_collector = None
        self._chunk_statistics = {}

        total_params = sum(p.numel() for p in self._model.parameters())
        self.model_card_data = {
            "model_name": model_name,
            "revision": config._name_or_path if hasattr(config, "_name_or_path") else "main",
            "release_date": None,
            "languages": None,
            "n_parameters": total_params,
            "memory_usage": None,
            "max_tokens": self._tokenizer.model_max_length,
            "embed_dim": self.hidden_size,
            "license": config.license if hasattr(config, "license") else None,
            "open_source": True,
            "similarity_fn_name": None,
            "framework": None,
            "loader": "HuggingFace"
        }
        print("=================================")
        print(f"Model: {model_name}")
        print(f"Total number of model's parameters: {total_params}")
        print(f"Maximum input length of model: {config.max_position_embeddings}")
        print(f"Maximum input length of tokenizer: {self._tokenizer.model_max_length}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Twice ratio: {self.twice_ratio}")
        print(f"Replaced layers: {self.replaced_layers}")
        print(f"Embedding strategy: {embedding_mode}")
        print("=================================")
        self.max_length = min(config.max_position_embeddings, self._tokenizer.model_max_length)

        if embedding_mode in ["chunk_twice_avg", "chunk_twice_weighted"]:
            self._setup_embedding_layers()
            self.chunk_token = "[CHUNK]"
            self._tokenizer.add_special_tokens({'additional_special_tokens': [self.chunk_token]})
            self.chunk_token_id = self._tokenizer.get_vocab()[self.chunk_token]
            self.pad_token_id = self._tokenizer.pad_token_id

        self.num_instruction_tokens = len(self._tokenizer(self.document_instruction, add_special_tokens=False)["input_ids"]) if self.document_instruction else 0

    def _get_encoder_layers(self):
        if hasattr(self._model, 'encoder') and hasattr(self._model.encoder, 'layer'):
            self.layers = self._model.encoder.layer
        elif hasattr(self._model, 'layers'):
            self.layers = self._model.layers
        else:
            raise AttributeError("Could not find encoder layers in model")

    def _setup_embedding_layers(self):
        """Flexibly setup embedding layers based on model architecture."""
        if hasattr(self._model, 'get_input_embeddings'):
            self.word_embeddings = self._model.get_input_embeddings()
        else:
            raise AttributeError("Could not find word embeddings layer in model")

        if hasattr(self._model, 'embeddings'):
            self.embeddings = self._model.embeddings
        elif hasattr(self._model, 'roberta') and hasattr(self._model.roberta, 'embeddings'):
            self.embeddings = self._model.roberta.embeddings
        else:
            raise AttributeError("Could not find embeddings layer in model")

    def apply_document_chunking(self, corpus_ids, corpus):
        doc_chunk_idx, chunked_corpus = [], []

        base_chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        max_chunks = 0
        chunks_per_doc = []
        
        for didx in corpus_ids:
            v = corpus[didx]
            title = v.get('title', '')
            body = v['text']

            title_tokens = self._tokenizer.tokenize(title, add_special_tokens = False) if title else []
            title_token_count = len(title_tokens)

            effective_chunk_size = max(1, base_chunk_size - title_token_count - self.num_instruction_tokens)

            if self.chunking_strategy == "token":
                chunker = TokenChunker(
                    self._tokenizer,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=chunk_overlap
                )
            elif self.chunking_strategy == "sentence":
                chunker = SentenceChunker(
                    self._tokenizer,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                raise NotImplementedError(f"Don't support chunking strategy: {self.chunking_strategy}")

            body_chunks = chunker(body)
            num_chunks = len(body_chunks)
            chunks_per_doc.append(num_chunks)
            max_chunks = max(max_chunks, num_chunks)
            
            for cid, chunk in enumerate(body_chunks):
                chunk_text = f"{title} {chunk.text}".strip()
                if self.document_instruction:
                    chunk_text = f"{self.document_instruction} {chunk_text}"

                doc_chunk_idx.append((str(didx), str(cid)))
                chunked_corpus.append(chunk_text.strip())
        
        # Store chunk statistics
        self._chunk_statistics = {
            'chunks_per_doc': chunks_per_doc,
            'total_chunks': len(chunked_corpus),
            'max_chunks': max_chunks
        }
        
        # If metrics collector is attached, record the stats
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            self._metrics_collector.record_chunking_stats(chunks_per_doc, len(chunked_corpus))

        return doc_chunk_idx, chunked_corpus, max_chunks

    def encode_corpus(
        self,
        corpus_ids,
        corpus,
        batch_size=32,
        *args,
        **kwargs,
    ):
        start_time = time.time()
        corpus_texts = []
        for k in corpus_ids:
            corpus_text = corpus[k]["text"] if 'title' not in corpus[k] else f"{corpus[k]['title']} {corpus[k]['text']}"
            if self.document_instruction:
                corpus_text = f"{self.document_instruction} {corpus_text}"
            corpus_texts.append(corpus_text.strip())

        corpus_embeddings = []
        total_tokens = 0  # Track total tokens
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(corpus_texts), batch_size)):
                end_index = min(start_index + batch_size, len(corpus_texts))
                batch_corpus_texts = corpus_texts[start_index:end_index]
                encoded_input = self._tokenizer(batch_corpus_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt', add_special_tokens=True).to(self._model.device)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens += batch_tokens
                
                batch_model_output = self._model(**encoded_input)
                batch_embeddings = self.pool_embeddings(batch_model_output.last_hidden_state, encoded_input['attention_mask'])
                if self.is_normalize:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                corpus_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
        
        # Record tokens processed if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            self._metrics_collector.record_tokens_processed(total_tokens)
            self._metrics_collector.encoder_passes = 1  # Single pass for truncation
            
        return corpus_embeddings

    def encode_queries(
            self,
            query_ids,
            queries,
            batch_size=32,
            *args,
            **kwargs,
    ):
        query_texts = [f"{self.query_instruction} {queries[k]}".strip() if self.query_instruction else queries[k] for k in query_ids ]
        query_embeddings = []
        total_tokens = 0  # Track total tokens
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(query_texts), batch_size)):
                end_index = min(start_index + batch_size, len(query_texts))
                batch_queries = query_texts[start_index:end_index]
                encoded_input = self._tokenizer(batch_queries, padding=True, truncation=True, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens += batch_tokens
                
                batch_model_output = self._model(**encoded_input)
                batch_embeddings = self.pool_embeddings(batch_model_output.last_hidden_state, encoded_input['attention_mask'])

                if self.is_normalize:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                query_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        query_embeddings = torch.cat(query_embeddings, dim=0)
        
        # Record tokens processed if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            self._metrics_collector.record_tokens_processed(total_tokens)
            
        return query_embeddings

    def encode_corpus_chunk_avg(
        self,
        corpus_ids,
        corpus,
        batch_size=32,
        *args,
        **kwargs,
    ):
        start_time = time.time()
        doc_chunk_idx, corpus_chunks, max_chunks = self.apply_document_chunking(corpus_ids, corpus)
        print("======================")
        print(f"Number of corpus: {len(corpus)}")
        print(f"Max chunks per corpus: {max_chunks}")
        print(f"Total number of chunks: {len(corpus_chunks)}")

        chunks_embeddings = []
        total_tokens = 0  # Track total tokens
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(corpus_chunks), batch_size)):
                end_index = min(start_index + batch_size, len(corpus_chunks))
                batch_corpus_texts = corpus_chunks[start_index:end_index]
                encoded_input = self._tokenizer(batch_corpus_texts, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens += batch_tokens
                
                batch_model_output = self._model(**encoded_input)
                batch_embeddings = self.pool_embeddings(batch_model_output.last_hidden_state,
                                                        encoded_input['attention_mask'])
                chunks_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        chunks_embeddings = torch.cat(chunks_embeddings, dim=0)
        if self.is_normalize:
            chunks_embeddings = F.normalize(chunks_embeddings, p=2, dim=-1)

        documents, docs_emb = self.get_doc_embeddings(doc_chunk_idx, corpus_chunks, chunks_embeddings)
        corpus_embeddings = []
        with torch.no_grad():
            for corpus_id in corpus_ids:
                corpus_embeddings.append(docs_emb[corpus_id].mean(dim=0, keepdim=True))

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
        if self.is_normalize:
            corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=-1)
        
        # Record tokens processed if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            self._metrics_collector.record_tokens_processed(total_tokens)
            self._metrics_collector.encoder_passes = 1

        return corpus_embeddings

    def encode_corpus_chunk_weight(
        self,
        corpus_ids,
        corpus,
        query_embs,
        batch_size=32,
        tau=0.1,
        *args,
        **kwargs,
    ):
        start_time = time.time()
        doc_chunk_idx, corpus_chunks, max_chunks = self.apply_document_chunking(corpus_ids, corpus)
        print("======================")
        print(f"Number of corpus: {len(corpus)}")
        print(f"Max chunks per corpus: {max_chunks}")
        print(f"Total number of chunks: {len(corpus_chunks)}")

        chunks_embeddings = []
        total_tokens = 0  # Track total tokens
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(corpus_chunks), batch_size)):
                end_index = min(start_index + batch_size, len(corpus_chunks))
                batch_corpus_texts = corpus_chunks[start_index:end_index]
                encoded_input = self._tokenizer(batch_corpus_texts, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens += batch_tokens
                
                batch_model_output = self._model(**encoded_input)
                batch_embeddings = self.pool_embeddings(batch_model_output.last_hidden_state,
                                                        encoded_input['attention_mask'])
                chunks_embeddings.append(batch_embeddings)
                torch.cuda.empty_cache()

        chunks_embeddings = torch.cat(chunks_embeddings, dim=0)
        if self.is_normalize:
            chunks_embeddings = F.normalize(chunks_embeddings, p=2, dim=-1)

        documents, docs_emb = self.get_doc_embeddings(doc_chunk_idx, corpus_chunks, chunks_embeddings)
        corpus_embeddings = []
        with torch.no_grad():
            for corpus_id in corpus_ids:
                chunk_embeddings = docs_emb[corpus_id]
                sim = torch.matmul(query_embs.to(self._model.device.type), chunk_embeddings.T) / tau
                weights = torch.softmax(sim, dim=1)
                doc_query_emb = torch.matmul(weights, chunk_embeddings)
                corpus_embeddings.append(doc_query_emb.unsqueeze(0))
                torch.cuda.empty_cache()

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
        if self.is_normalize:
            corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=-1)
        
        # Record tokens processed if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            self._metrics_collector.record_tokens_processed(total_tokens)
            self._metrics_collector.encoder_passes = 1

        return corpus_embeddings

    def encode_corpus_chunk_twice(
        self,
        corpus_ids,
        corpus,
        batch_size=32,
        replaced_layers=None,  # This can be overridden, otherwise use self.replaced_layers
        top_k_context=3,
        *args,
        **kwargs,
    ):
        start_time = time.time()
        
        if replaced_layers is None:
            replaced_layers = self.replaced_layers  # Use the ratio-based calculation
        
        top_k_context = int(top_k_context)
        print(f"Replaced layers: {replaced_layers}, top_k_context: {top_k_context}")
        
        # Chunking
        doc_chunk_idx, corpus_chunks, max_chunks = self.apply_document_chunking(corpus_ids, corpus)

        # First embedding
        chunks_embeddings = []
        total_tokens_first_pass = 0  # Track tokens in first pass
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(corpus_chunks), batch_size), desc="First pass encoding"):
                end_index = min(start_index + batch_size, len(corpus_chunks))
                batch_corpus_texts = corpus_chunks[start_index:end_index]
                encoded_input = self._tokenizer(batch_corpus_texts, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens_first_pass += batch_tokens
                
                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                batch_embeddings = [self.pool_embeddings(layer_hidden_states, encoded_input['attention_mask']) for
                                    layer_hidden_states in batch_model_output.hidden_states]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)
                chunks_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        chunks_embeddings = torch.cat(chunks_embeddings, dim=0)

        # Prepare for the second embedding
        documents, docs_emb = self.get_doc_embeddings(doc_chunk_idx, corpus_chunks, chunks_embeddings)

        # Second embedding
        docs_emb_second = []
        total_tokens_second_pass = 0  # Track tokens in second pass
        tau = 0.07
        beta = 1.0
        alpha = 0.60
        K_max = 40
        
        with torch.no_grad():
            for doc_id in tqdm(corpus_ids, desc="Second pass encoding"):
                try:
                    doc = documents[doc_id]
                    doc_emb = docs_emb[doc_id].to(self._model.device)

                    last_layer_emb = doc_emb[:, -1, :].cpu()
                    normalized_emb = F.normalize(last_layer_emb, p=2, dim=-1)
                    similarity_matrix = torch.mm(normalized_emb, normalized_emb.T)

                    chunks_padded = []
                    all_context_indices = []
                    num_chunks = len(doc)
                    max_num_context = 0

                    for id, chunk in enumerate(doc):
                        sim_row = similarity_matrix[id].clone()
                        sim_row[id] = float('-inf')
                        mask = torch.ones_like(sim_row, dtype=torch.bool)
                        mask[id] = False
                        vals = sim_row[mask]
                        mu = vals.mean()
                        std = vals.std(unbiased=False) + 1e-6
                        logits = (sim_row - mu) / std * beta
                        logits[id] = float('-inf')
                        prob = torch.softmax(logits, dim=-1)
                        prob_sorted, idx_sorted = prob.sort(descending=True)
                        cum_prob = prob_sorted.cumsum(0)
                        cut = (cum_prob < alpha).sum().item() + 1
                        ctx_idx = idx_sorted[:min(cut, K_max, top_k_context)]
                        context_indices_sorted = ctx_idx.sort().values.tolist()

                        max_num_context = max(max_num_context, len(context_indices_sorted))

                        left_context = []
                        right_context = []
                        for pos in context_indices_sorted:
                            if pos < id:
                                left_context.append(self.chunk_token)
                            else:
                                right_context.append(self.chunk_token)

                        modified_chunk = " ".join(left_context) + " " + chunk + " " + " ".join(right_context)
                        chunks_padded.append(modified_chunk.strip())
                        all_context_indices.append(context_indices_sorted)

                    # Tokenize sentences
                    encoded_input = self._tokenizer(chunks_padded, padding=True, truncation=True,
                                                    max_length=self.max_length, return_tensors='pt',
                                                    add_special_tokens=True).to(self._model.device.type)
                    
                    # Count tokens in second pass
                    batch_tokens = encoded_input['attention_mask'].sum().item()
                    total_tokens_second_pass += batch_tokens
                    
                    input_ids = encoded_input["input_ids"]
                    attention_mask = encoded_input["attention_mask"]
                    extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min

                    word_embeddings = self.word_embeddings_with_special_chunk_tokens(input_ids)
                    hidden_states = self.embeddings(inputs_embeds=word_embeddings)

                    max_chunk_tokens = min(max_num_context, num_chunks - 1)
                    chunk_token_indices = torch.full((num_chunks, max_chunk_tokens, 2), -1,
                                                     dtype=torch.long,
                                                     device=input_ids.device)
                    for chunk_idx in range(num_chunks):
                        positions = torch.nonzero(input_ids[chunk_idx] == self.chunk_token_id, as_tuple=False)
                        if len(positions) > max_chunk_tokens:
                            positions = positions[:max_chunk_tokens]

                        if len(positions) > 0:
                            chunk_token_indices[chunk_idx, :len(positions), 0] = chunk_idx
                            chunk_token_indices[chunk_idx, :len(positions), 1] = positions.squeeze(-1)
                    chunk_token_indices = chunk_token_indices[:, :, 1]
                    
                    for i, layer in enumerate(self.layers):
                        if i < replaced_layers:
                            replaced_embedding = doc_emb[:, i, :]
                            hidden_states = self.replace_chunk_embed(hidden_states, replaced_embedding, chunk_token_indices, all_context_indices)
                        hidden_states = layer(hidden_states, extended_attention_mask)[0]

                    chunk_embeddings = self.pool_embeddings(hidden_states, attention_mask)
                    if self.is_normalize:
                        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=-1)
                    doc_emb = chunk_embeddings.mean(dim=0, keepdim=True)
                    docs_emb_second.append(doc_emb)
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[CRASH ON] doc_id: {doc_id}, error: {str(e)}, number of chunks: {num_chunks}")
                    break

        docs_emb_second = torch.cat(docs_emb_second, dim=0)
        if self.is_normalize:
            docs_emb_second = F.normalize(docs_emb_second, p=2, dim=-1)
        
        # Record total tokens from both passes if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            total_tokens = total_tokens_first_pass + total_tokens_second_pass
            self._metrics_collector.record_tokens_processed(total_tokens)
            self._metrics_collector.encoder_passes = 2  # Two-pass encoding

        return docs_emb_second

    def encode_corpus_chunk_twice_ppr(
            self,
            corpus_ids,
            corpus,
            batch_size=32,
            replaced_layers=None,
            top_k_context=6,
            *args,
            **kwargs,
    ):
        start_time = time.time()
        
        if replaced_layers is None:
            replaced_layers = self.replaced_layers  # Use the ratio-based calculation
            
        def _build_transition(sim_matrix: torch.Tensor,
                              tau: float = 0.25,
                              top_m: int = 32,
                              symmetric: bool = True) -> torch.Tensor:
            S = sim_matrix.clone()
            n = S.size(0)
            S.fill_diagonal_(0.0)
            if tau is not None and tau > 0:
                S = (S - tau).clamp(min=0.0)

            if top_m is not None and 0 < top_m < (n - 1):
                vals, idx = torch.topk(S, k=min(top_m, n - 1), dim=1)
                mask = torch.zeros_like(S, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                S = torch.where(mask, S, torch.zeros_like(S))

            if symmetric:
                S = 0.5 * (S + S.T)

            row_sum = S.sum(dim=1, keepdim=True)
            zero_rows = (row_sum <= 1e-12).squeeze(-1)
            if zero_rows.any():
                S[zero_rows] = 1.0
                row_sum = S.sum(dim=1, keepdim=True)

            P = S / (row_sum + 1e-12)
            return P

        def _personalized_pagerank(P: torch.Tensor,
                                   seed_idx: int,
                                   alpha: float = 0.25,
                                   max_iter: int = 50,
                                   tol: float = 1e-6) -> torch.Tensor:
            n = P.size(0)
            dev = P.device
            e = torch.zeros(n, device=dev)
            e[seed_idx] = 1.0
            s = e.clone()
            Pt = P.T
            for _ in range(max_iter):
                s_new = (1 - alpha) * e + alpha * (Pt @ s)
                if torch.norm(s_new - s, p=1).item() < tol:
                    s = s_new
                    break
                s = s_new
            return s

        def _entropy_to_K(prob: torch.Tensor, K_min: int = 4, K_max: int = 40) -> int:
            n = prob.numel()
            if n <= 0:
                return K_min
            p = prob / (prob.sum() + 1e-12)
            H = -(p * (p + 1e-12).log()).sum()
            H_norm = (H / math.log(n)).item()
            K = int(round(K_min + H_norm * (K_max - K_min)))
            return max(K_min, min(K_max, K))

        ppr_alpha = 0.25
        tau_graph = 0.25
        top_m_edges = 32
        pos_decay = 0.0
        use_entropy_K = False
        K_min, K_max = 3, 30
        mix_with_cosine = True
        lam_mix = 0.6

        top_k_context = int(top_k_context)
        print(f"Replaced layers: {replaced_layers}, top_k_context: {top_k_context}")

        doc_chunk_idx, corpus_chunks, max_chunks = self.apply_document_chunking(corpus_ids, corpus)

        chunks_embeddings = []
        total_tokens_first_pass = 0  # Track tokens in first pass
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(corpus_chunks), batch_size), desc="First pass encoding (PPR)"):
                end_index = min(start_index + batch_size, len(corpus_chunks))
                batch_corpus_texts = corpus_chunks[start_index:end_index]
                encoded_input = self._tokenizer(
                    batch_corpus_texts, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors='pt',
                    add_special_tokens=True
                ).to(self._model.device.type)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens_first_pass += batch_tokens

                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                batch_embeddings = [
                    self.pool_embeddings(layer_hidden_states, encoded_input['attention_mask'])
                    for layer_hidden_states in batch_model_output.hidden_states
                ]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)
                chunks_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        chunks_embeddings = torch.cat(chunks_embeddings, dim=0)

        documents, docs_emb = self.get_doc_embeddings(doc_chunk_idx, corpus_chunks, chunks_embeddings)

        docs_emb_second = []
        total_tokens_second_pass = 0  # Track tokens in second pass

        with torch.no_grad():
            for doc_id in tqdm(corpus_ids, desc="Second pass encoding (PPR)"):
                try:
                    doc = documents[doc_id]
                    doc_emb = docs_emb[doc_id].to(self._model.device)

                    last_layer_emb = doc_emb[:, -1, :].cpu()
                    normalized_emb = F.normalize(last_layer_emb, p=2, dim=-1)
                    similarity_matrix = torch.mm(normalized_emb, normalized_emb.T)

                    n = len(doc)
                    if n <= 1:
                        pass

                    P = _build_transition(similarity_matrix, tau=tau_graph, top_m=top_m_edges, symmetric=True).to(
                        last_layer_emb.device)

                    chunks_padded = []
                    all_context_indices = []
                    max_num_context = 0

                    for i, chunk in enumerate(doc):
                        s = _personalized_pagerank(P, seed_idx=i, alpha=ppr_alpha, max_iter=50, tol=1e-6)
                        s[i] = 0.0

                        if pos_decay and pos_decay > 0:
                            pos = torch.arange(n, device=s.device, dtype=torch.float32)
                            dist = (pos - i).abs()
                            s = s * torch.exp(-pos_decay * dist)

                        if mix_with_cosine:
                            cos_row = similarity_matrix[i].to(s.device)
                            cos_row[i] = 0.0
                            s_prob = s / (s.sum() + 1e-12)
                            cos_prob = (cos_row - cos_row.min()).clamp_min(0)
                            cos_prob = cos_prob / (cos_prob.sum() + 1e-12)
                            score = (1 - lam_mix) * cos_prob + lam_mix * s_prob
                        else:
                            score = s / (s.sum() + 1e-12)

                        if use_entropy_K or top_k_context <= 0:
                            K_i = _entropy_to_K(score, K_min=K_min, K_max=min(K_max, n - 1))
                        else:
                            K_i = min(top_k_context, n - 1, K_max)
                        
                        ctx_idx = torch.topk(score, k=K_i).indices
                        context_indices_sorted = ctx_idx.sort().values.tolist()
                        max_num_context = max(max_num_context, len(context_indices_sorted))
                        all_context_indices.append(context_indices_sorted)

                        left_context, right_context = [], []
                        for pos_j in context_indices_sorted:
                            if pos_j < i:
                                left_context.append(self.chunk_token)
                            else:
                                right_context.append(self.chunk_token)

                        modified_chunk = " ".join(left_context) + " " + chunk + " " + " ".join(right_context)
                        chunks_padded.append(modified_chunk.strip())

                    encoded_input = self._tokenizer(
                        chunks_padded, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors='pt',
                        add_special_tokens=True
                    ).to(self._model.device.type)
                    
                    # Count tokens in second pass
                    batch_tokens = encoded_input['attention_mask'].sum().item()
                    total_tokens_second_pass += batch_tokens
                    
                    input_ids = encoded_input["input_ids"]
                    attention_mask = encoded_input["attention_mask"]
                    extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min

                    word_embeddings = self.word_embeddings_with_special_chunk_tokens(input_ids)
                    hidden_states = self.embeddings(inputs_embeds=word_embeddings)

                    max_chunk_tokens = min(max_num_context, n - 1)
                    chunk_token_indices = torch.full((n, max_chunk_tokens, 2), -1, dtype=torch.long,
                                                     device=input_ids.device)
                    for chunk_idx in range(n):
                        positions = torch.nonzero(input_ids[chunk_idx] == self.chunk_token_id, as_tuple=False)
                        if len(positions) > max_chunk_tokens:
                            positions = positions[:max_chunk_tokens]
                        if len(positions) > 0:
                            chunk_token_indices[chunk_idx, :len(positions), 0] = chunk_idx
                            chunk_token_indices[chunk_idx, :len(positions), 1] = positions.squeeze(-1)
                    chunk_token_indices = chunk_token_indices[:, :, 1]

                    for li, layer in enumerate(self.layers):
                        if li < replaced_layers:
                            replaced_embedding = doc_emb[:, li, :]
                            hidden_states = self.replace_chunk_embed(hidden_states, replaced_embedding,
                                                                     chunk_token_indices, all_context_indices)
                        hidden_states = layer(hidden_states, extended_attention_mask)[0]

                    chunk_embeddings = self.pool_embeddings(hidden_states, attention_mask)
                    if self.is_normalize:
                        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=-1)

                    doc_vec = chunk_embeddings.mean(dim=0, keepdim=True)
                    docs_emb_second.append(doc_vec)
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[CRASH ON] doc_id: {doc_id}, error: {str(e)}, number of chunks: {len(doc)}")
                    break

        docs_emb_second = torch.cat(docs_emb_second, dim=0)
        if self.is_normalize:
            docs_emb_second = F.normalize(docs_emb_second, p=2, dim=-1)
        
        # Record total tokens from both passes if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            total_tokens = total_tokens_first_pass + total_tokens_second_pass
            self._metrics_collector.record_tokens_processed(total_tokens)
            self._metrics_collector.encoder_passes = 2  # Two-pass encoding
            
        return docs_emb_second

    def encode_corpus_chunk_twice_weight(
        self,
        corpus_ids,
        corpus,
        query_embs,
        batch_size=32,
        tau=0.1,
        replaced_layers=None,
        *args,
        **kwargs,
    ):
        start_time = time.time()
        
        if replaced_layers is None:
            replaced_layers = self.replaced_layers  # Use the ratio-based calculation
            
        doc_chunk_idx, corpus_chunks, max_chunks = self.apply_document_chunking(corpus_ids, corpus)
        print("======================")
        print(f"Number of corpus: {len(corpus)}")
        print(f"Max chunks per corpus: {max_chunks}")
        print(f"Total number of chunks: {len(corpus_chunks)}")

        chunks_embeddings = []
        total_tokens_first_pass = 0  # Track tokens in first pass
        
        with torch.no_grad():
            for start_index in tqdm(range(0, len(corpus_chunks), batch_size), desc="First pass encoding"):
                end_index = min(start_index + batch_size, len(corpus_chunks))
                batch_corpus_texts = corpus_chunks[start_index:end_index]
                encoded_input = self._tokenizer(batch_corpus_texts, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                
                # Count tokens in this batch
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens_first_pass += batch_tokens
                
                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                batch_embeddings = [self.pool_embeddings(layer_hidden_states, encoded_input['attention_mask']) for
                                    layer_hidden_states in batch_model_output.hidden_states]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)
                chunks_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        chunks_embeddings = torch.cat(chunks_embeddings, dim=0)

        documents, docs_emb = self.get_doc_embeddings(doc_chunk_idx, corpus_chunks, chunks_embeddings)

        docs_emb_second = []
        total_tokens_second_pass = 0  # Track tokens in second pass
        
        with torch.no_grad():
            for doc_id in tqdm(corpus_ids, desc="Second pass encoding"):
                doc = documents[doc_id]
                doc_emb = docs_emb[doc_id].to(self._model.device)
                chunks_padded = []
                num_chunks = len(doc)

                for id, chunk in enumerate(doc):
                    chunk = " ".join([self.chunk_token for _ in range(id)]) + " " + chunk + " " + " ".join(
                        [self.chunk_token for _ in range(id + 1, num_chunks)])
                    chunks_padded.append(chunk)

                encoded_input = self._tokenizer(chunks_padded, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                
                # Count tokens in second pass
                batch_tokens = encoded_input['attention_mask'].sum().item()
                total_tokens_second_pass += batch_tokens
                
                input_ids = encoded_input["input_ids"]
                attention_mask = encoded_input["attention_mask"]
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min

                word_embeddings = self.word_embeddings_with_special_chunk_tokens(input_ids)
                hidden_states = self.embeddings(inputs_embeds=word_embeddings)
                chunk_token_indices = torch.nonzero(input_ids == self.chunk_token_id, as_tuple=False).reshape(
                    (num_chunks, num_chunks - 1, 2))[:, :, 1]

                for i, layer in enumerate(self.layers):
                    if i < replaced_layers:
                        replaced_embedding = doc_emb[:, i, :]
                        hidden_states = self.replace_chunk_embed(hidden_states, replaced_embedding, chunk_token_indices)
                    hidden_states = layer(hidden_states, extended_attention_mask)[0]

                chunk_embeddings = self.pool_embeddings(hidden_states, attention_mask)
                if self.is_normalize:
                    chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=-1)

                sim = torch.matmul(query_embs.to(self._model.device.type), F.normalize(doc_emb[:, -1, :], p=2, dim=-1).T) / tau
                weights = torch.softmax(sim, dim=1)
                doc_query_emb = torch.matmul(weights, chunk_embeddings)
                docs_emb_second.append(doc_query_emb.unsqueeze(0))
                torch.cuda.empty_cache()

        docs_emb_second = torch.cat(docs_emb_second, dim=0)
        if self.is_normalize:
            docs_emb_second = F.normalize(docs_emb_second, p=2, dim=-1)
        
        # Record total tokens from both passes if metrics collector exists
        if hasattr(self, '_metrics_collector') and self._metrics_collector:
            total_tokens = total_tokens_first_pass + total_tokens_second_pass
            self._metrics_collector.record_tokens_processed(total_tokens)
            self._metrics_collector.encoder_passes = 2  # Two-pass encoding

        return docs_emb_second

    # Additional helper methods
    def word_embeddings_with_special_chunk_tokens(self, input_ids):
        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[(input_ids == self.chunk_token_id)] = self.unk_token_id
        raw_embeds = self.word_embeddings(input_ids_for_embedding)
        return raw_embeds

    def replace_embed(self, embedding, replaced_embedding, chunk_token_indices):
        bs = embedding.shape[0]
        cz = replaced_embedding.shape[0]
        for bidx in range(bs):
            for i in range(cz):
                embedding[bidx, chunk_token_indices[i], :] = replaced_embedding[i, :]
        return embedding

    def replace_chunk_embed(self, embedding, replaced_embedding, chunk_token_indices, all_context_indices=None, alpha=0.5):
        if all_context_indices is None:
            # Fallback for methods that don't use all_context_indices
            replaced = embedding.clone()
            cz = embedding.shape[0]
            for bidx in range(cz):
                replaced_embedding_ = torch.cat((replaced_embedding[:bidx, :], replaced_embedding[bidx+1:, :]), dim=0)
                for i in range(cz-1):
                    replaced[bidx, chunk_token_indices[bidx, i], :] = replaced_embedding_[i, :]
            return replaced
        
        replaced = embedding.clone()
        cz = embedding.shape[0]
        for bidx in range(cz):
            context_indices = all_context_indices[bidx]
            for i in range(len(context_indices)):
                if chunk_token_indices[bidx, i] != -1:
                    replaced[bidx, chunk_token_indices[bidx, i], :] = replaced_embedding[context_indices[i]]
        return replaced

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, token_embeddings, attention_mask):
        return token_embeddings[:, 0]

    def last_pooling(self, token_embeddings, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return token_embeddings[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = token_embeddings.shape[0]
            return token_embeddings[torch.arange(batch_size, device=token_embeddings.device), sequence_lengths]

    def pool_embeddings(self, token_embeddings, attention_mask):
        if self.pooling_method == 'mean':
            return self.mean_pooling(token_embeddings, attention_mask)
        elif self.pooling_method == 'last':
            return self.last_pooling(token_embeddings, attention_mask)
        elif self.pooling_method == 'cls':
            return self.cls_pooling(token_embeddings, attention_mask)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def get_doc_embeddings(self, corpus_chunk_ids, corpus, corpus_embs):
        corpus_dict = defaultdict(list)

        for idx, id_str in enumerate(corpus_chunk_ids):
            corpus_id_str, chunk_id_str = id_str
            chunk_id = int(chunk_id_str)
            corpus_dict[corpus_id_str].append((chunk_id, idx))

        sorted_corpus_embs = {}
        sorted_corpus = {}
        for corpus_id, chunk_list in corpus_dict.items():
            sorted_chunk_list = sorted(chunk_list, key=lambda x: x[0])
            indices = [idx for chunk_id, idx in sorted_chunk_list]
            sorted_corpus_embs[corpus_id] = corpus_embs[indices]
            sorted_corpus[corpus_id] = [corpus[i] for i in indices]

        return sorted_corpus, sorted_corpus_embs

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return False


MODEL_WRAPPERS = {
    # Small models
    "sentence-transformers/all-MiniLM-L6-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "sentence-transformers/all-MiniLM-L12-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "sentence-transformers/paraphrase-MiniLM-L6-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "sentence-transformers/paraphrase-MiniLM-L12-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    
    # MPNet models
    "sentence-transformers/all-mpnet-base-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "sentence-transformers/all-distilroberta-v1": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    
    # BGE models
    "BAAI/bge-small-en-v1.5": {
        "is_normalize": True,
        "pooling_method": "cls",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "Represent this sentence for searching relevant passages:"
    },
    "BAAI/bge-base-en-v1.5": {
        "is_normalize": True,
        "pooling_method": "cls",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "Represent this sentence for searching relevant passages:"
    },
    "BAAI/bge-large-en-v1.5": {
        "is_normalize": True,
        "pooling_method": "cls",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "Represent this sentence for searching relevant passages:"
    },
    "BAAI/bge-m3": {
        "is_normalize": True,
        "pooling_method": "cls",
        "Wrapper": EmbeddingsWrapper
    },
    
    # E5 models
    "intfloat/e5-small": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "query:",
        "document_instruction": "passage:"
    },
    "intfloat/e5-base": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "query:",
        "document_instruction": "passage:"
    },
    "intfloat/e5-base-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "query:",
        "document_instruction": "passage:"
    },
    "intfloat/e5-large": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "query:",
        "document_instruction": "passage:"
    },
    "intfloat/e5-large-v2": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "query:",
        "document_instruction": "passage:"
    },
    
    # Multilingual models
    "intfloat/multilingual-e5-small": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "intfloat/multilingual-e5-base": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "intfloat/multilingual-e5-large": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "intfloat/multilingual-e5-large-instruct": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    },
    
    # GTE models
    "thenlper/gte-small": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "thenlper/gte-base": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "thenlper/gte-large": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    
    # Multi-QA models
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
        "is_normalize": False,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    
    # Contrastive models
    "facebook/contriever": {
        "is_normalize": False,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "facebook/contriever-msmarco": {
        "is_normalize": False,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    
    # Jina models
    "jinaai/jina-embeddings-v2-small-en": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    "jinaai/jina-embeddings-v2-base-en": {
        "is_normalize": True,
        "pooling_method": "mean",
        "Wrapper": EmbeddingsWrapper
    },
    
    # Qwen models
    "Qwen/Qwen3-Embedding-0.6B": {
        "is_normalize": True,
        "pooling_method": "last",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    },
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "is_normalize": True,
        "pooling_method": "last",
        "Wrapper": EmbeddingsWrapper,
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    },
}


def load_model(model_name, embedding_mode, chunking_args, twice_ratio=0.75):
    """Load model with twice_ratio support"""
    if model_name in MODEL_WRAPPERS:
        is_normalize = MODEL_WRAPPERS[model_name]["is_normalize"]
        pooling_method = MODEL_WRAPPERS[model_name]["pooling_method"]
        query_instruction = MODEL_WRAPPERS[model_name].get("query_instruction", None)
        document_instruction = MODEL_WRAPPERS[model_name].get("document_instruction", None)
        wrapper = MODEL_WRAPPERS[model_name]["Wrapper"]
        chunking_strategy, chunk_size, chunk_overlap = chunking_args["chunking_strategy"], chunking_args["chunk_size"], chunking_args["chunk_overlap"]
        
        model = wrapper(model_name, embedding_mode, chunking_strategy, chunk_size, chunk_overlap, 
                       is_normalize=is_normalize, pooling_method=pooling_method,
                       query_instruction=query_instruction, document_instruction=document_instruction,
                       twice_ratio=twice_ratio)
        return model
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_WRAPPERS.keys())}")