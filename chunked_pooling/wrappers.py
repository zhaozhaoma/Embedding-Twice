import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn.functional as F
from torch import Tensor


def construct_document(doc):
    if isinstance(doc, str):
        return doc
    elif 'title' in doc:
        return f'{doc["title"]} {doc["text"].strip()}'
    else:
        return doc['text'].strip()


class JinaEmbeddingsV3Wrapper(nn.Module):
    def __init__(
        self, model_name, tasks=['retrieval.query', 'retrieval.passage'], **model_kwargs
    ):
        super().__init__()
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.emb_layer = self._model.get_input_embeddings()
        self.tasks = tasks

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        batch_size=32,
        *args,
        task: Optional[str] = None,
        **kwargs,
    ):
        sentence_embeddings, output_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = sentences[start_index:end_index]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(
                    self._model.device.type)
                task_id = self._model._adaptation_map[self.tasks[0]]
                adapter_mask = torch.full((len(batch_sentences),), task_id, dtype=torch.int32)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input, adapter_mask=adapter_mask)
                # Perform pooling
                batch_embeddings = self.mean_pooling(batch_model_output, encoded_input['attention_mask'])
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings)

                # Extract token embeddings up to the true token length
                attention_mask = encoded_input['attention_mask']  # Shape: [batch_size, seq_len]
                token_embeddings = batch_model_output[0]  # Shape: [batch_size, seq_len, hidden_size]
                for i in range(len(batch_sentences)):
                    # Get the attention mask for this instance
                    attn_mask = attention_mask[i]  # Shape: [seq_len]
                    # Get the token embeddings for this instance
                    tokens = token_embeddings[i]  # Shape: [seq_len, hidden_size]
                    # Get the true length (number of tokens without padding)
                    true_length = attn_mask.sum().item()
                    # Slice the embeddings up to the true length
                    tokens = tokens[:true_length, :]
                    # Append to the list
                    output_embeddings.append(tokens)

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        # output_embeddings = torch.cat(output_embeddings, dim=0)

        return sentence_embeddings, output_embeddings

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        sentence_embeddings, output_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = sentences[start_index:end_index]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(
                    self._model.device.type)
                task_id = self._model._adaptation_map[self.tasks[1]]
                adapter_mask = torch.full((len(batch_sentences),), task_id, dtype=torch.int32)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input, adapter_mask=adapter_mask)
                # Perform pooling
                batch_embeddings = self.mean_pooling(batch_model_output, encoded_input['attention_mask'])
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings)

                # Extract token embeddings up to the true token length
                attention_mask = encoded_input['attention_mask']  # Shape: [batch_size, seq_len]
                token_embeddings = batch_model_output[0]  # Shape: [batch_size, seq_len, hidden_size]
                for i in range(len(batch_sentences)):
                    # Get the attention mask for this instance
                    attn_mask = attention_mask[i]  # Shape: [seq_len]
                    # Get the token embeddings for this instance
                    tokens = token_embeddings[i]  # Shape: [seq_len, hidden_size]
                    # Get the true length (number of tokens without padding)
                    true_length = attn_mask.sum().item()
                    # Slice the embeddings up to the true length
                    tokens = tokens[:true_length, :]
                    # Append to the list
                    output_embeddings.append(tokens)

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        # output_embeddings = torch.cat(output_embeddings, dim=0)

        return sentence_embeddings, output_embeddings

    def encode_embedding(
        self,
        embeddings_list: List[torch.Tensor],
        batch_size=32,
        *args,
        **kwargs,
    ):
        sentence_embeddings, output_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(embeddings_list), batch_size):
                end_index = min(start_index + batch_size, len(embeddings_list))
                batch_embeddings_list = embeddings_list[start_index:end_index]
                batch_size_actual = len(batch_embeddings_list)

                # Find the maximum sequence length in the batch
                max_seq_length = max(emb.shape[0] for emb in batch_embeddings_list)
                embedding_dim = batch_embeddings_list[0].shape[1]

                # Initialize tensors for inputs_embeds and attention_mask
                batch_inputs_embeds = torch.zeros(
                    batch_size_actual,
                    max_seq_length,
                    embedding_dim,
                    device=self._model.device
                )
                attention_mask = torch.zeros(
                    batch_size_actual,
                    max_seq_length,
                    dtype=torch.long,
                    device=self._model.device
                )
                # Pad embeddings and create attention masks
                for i, emb in enumerate(batch_embeddings_list):
                    seq_len = emb.shape[0]
                    batch_inputs_embeds[i, :seq_len, :] = emb.to(self._model.device)
                    attention_mask[i, :seq_len] = 1  # 1 indicates valid tokens

                task_id = self._model._adaptation_map[self.tasks[0]]
                adapter_mask = torch.full((batch_size_actual,), task_id, dtype=torch.int32)
                # Forward pass through the model
                batch_model_output = self._model(
                    inputs_embeds=batch_inputs_embeds,
                    attention_mask=attention_mask,
                    adapter_mask=adapter_mask,
                    **kwargs
                )
                # Perform pooling
                batch_embeddings = self.mean_pooling(batch_model_output, attention_mask)
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings)

                # Extract token embeddings up to the true token length
                token_embeddings = batch_model_output[0]  # Shape: [batch_size, max_seq_length, hidden_size]
                for i in range(batch_size_actual):
                    seq_len = attention_mask[i].sum().item()
                    tokens = token_embeddings[i, :seq_len, :]
                    output_embeddings.append(tokens)


        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        # output_embeddings = torch.cat(output_embeddings, dim=0)

        return sentence_embeddings, output_embeddings

    def tokenize_emb(self, text):
        with torch.no_grad():
            encoded_input = self._tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self._model.device.type)
            output_embeddings = self.emb_layer(encoded_input["input_ids"])

        return output_embeddings.squeeze(0)

    def get_instructions(self):
        return [self._model._task_instructions[x] for x in self.tasks]

    def forward(self, *args, **kwargs):
        task_id = self._model._adaptation_map[self.tasks[1]]
        num_examples = kwargs['input_ids'].shape[0]
        adapter_mask = torch.full(
            (num_examples,), task_id, dtype=torch.int32, device=self._model.device
        )
        return self._model.forward(*args, adapter_mask=adapter_mask, **kwargs)

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return True


class NomicAIWrapper(nn.Module):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self._model = SentenceTransformer(
            model_name, trust_remote_code=True, **model_kwargs
        )
        self.instructions = ['search_query: ', 'search_document: ']

    def get_instructions(self):
        return self.instructions

    def forward(self, *args, **kwargs):
        model_output = self._model.forward(kwargs)
        base_model_output = BaseModelOutputWithPooling(
            last_hidden_state=model_output['token_embeddings'],
            pooler_output=model_output['sentence_embedding'],
            attentions=model_output['attention_mask'],
        )
        return base_model_output

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self._model.encode(
            [self.instructions[0] + s for s in sentences], *args, **kwargs
        )

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self._model.encode(
            [self.instructions[1] + construct_document(s) for s in sentences],
            *args,
            **kwargs,
        )

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return True


class EmbeddingsWrapper(nn.Module):
    def __init__(
        self, model_name, is_embeddings_twice, **model_kwargs
    ):
        super().__init__()
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.word_embeddings = self._model.get_input_embeddings()
        self.embeddings = self._model.embeddings
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = config.max_position_embeddings
        self.layers = self._model.encoder.layer
        self.num_layers = len(self.layers)
        self.hidden_size = config.hidden_size
        if is_embeddings_twice:
            self.chunk_token = "[CHUNK]"
            self._tokenizer.add_special_tokens({'additional_special_tokens': [self.chunk_token]})
            self.chunk_token_id = self._tokenizer.get_vocab()[self.chunk_token]


    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        batch_size=32,
        *args,
        **kwargs,
    ):
        sentence_embeddings = []
        output_embeddings = []
        # cls_embeddings, sep_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = sentences[start_index:end_index]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True).to(self._model.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input)
                # Perform pooling
                batch_embeddings = self.mean_pooling(batch_model_output.last_hidden_state, encoded_input['attention_mask'])
                # # Normalize embeddings
                # batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings.cpu())

                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        return sentence_embeddings

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        batch_size=32,
        *args,
        **kwargs,
    ):
        sentence_embeddings = []
        output_embeddings = []
        # cls_embeddings, sep_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = sentences[start_index:end_index]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True).to(self._model.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                # cls_embeddings.append(batch_model_output[0][:, 0, :])
                # for i, mask in enumerate(encoded_input['attention_mask']):
                #     valid_length = mask.sum().item()
                #     sep_embeddings.append(
                #         batch_model_output[0][i, valid_length - 1, :])

                # Perform pooling
                batch_embeddings = [self.mean_pooling(layer_hidden_states, encoded_input['attention_mask']) for layer_hidden_states in batch_model_output.hidden_states]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)  # [batch_size, num_layers, embedding_dim]
                # # Normalize embeddings
                # batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)

                sentence_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

        return sentence_embeddings

    def encode_second(
            self,
            corpus: Union[str, List[str]],
            embeddings_corpus,
            batch_size=32,
            *args,
            **kwargs,
    ):
        corpus_embeddings = []
        with torch.no_grad():
            for corpus_id, chunks in corpus.items():
                num_chunks = min(len(chunks), self.max_length - 2)
                embeddings_chunks = embeddings_corpus[corpus_id][:num_chunks].cuda()
                embeddings_chunks = embeddings_chunks[:, -2, :].unsqueeze(0)
                attention_mask = torch.ones(1, num_chunks).cuda()
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
                hidden_states = self.layers[-1](embeddings_chunks, extended_attention_mask)[0]

                # # embeddings_chunks = embeddings_corpus[corpus_id].cuda()
                # # num_chunks = len(chunks)
                # chunks_padded = " ".join([self.chunk_token for _ in range(num_chunks)])
                # # Tokenize sentences
                # encoded_input = self._tokenizer(chunks_padded, padding=True, truncation=True,
                #                                 max_length=self.max_length, return_tensors='pt',
                #                                 add_special_tokens=True).to(self._model.device.type)
                # input_ids = encoded_input["input_ids"]
                # attention_mask = encoded_input["attention_mask"]
                # extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
                # # max_seq_length = input_ids.shape[1]
                # word_embeddings = self.embedding_chunk_as_unknown(input_ids)
                # hidden_states = self.embeddings(inputs_embeds=word_embeddings)
                # # print(f"input_embeddings: {input_embeddings.shape}")
                # # hidden_states = input_embeddings
                # chunk_token_indices = torch.nonzero(input_ids == self.chunk_token_id, as_tuple=False).reshape((num_chunks, 2))[:, 1]
                #
                # for i, layer in enumerate(self.layers):
                #     if i in range(self.num_layers):
                #         replaced_embedding = embeddings_chunks[:, i, :]
                #         hidden_states = self.replace_embed(hidden_states, replaced_embedding, chunk_token_indices)
                #         hidden_states = layer(hidden_states, extended_attention_mask)[0]
                #     else:
                #         hidden_states = layer(hidden_states, extended_attention_mask)[0]

                chunk_embeddings = self.mean_pooling(hidden_states, attention_mask)
                # print(chunk_embeddings.device, embeddings_chunks.device)
                # # Normalize embeddings
                # chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

                corpus_embeddings.append(chunk_embeddings.cpu())
                torch.cuda.empty_cache()

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

        return corpus_embeddings.to(torch.float32)

    def embedding_chunk_as_unknown(self, input_ids):
        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[(input_ids == self.chunk_token_id)] = self._tokenizer.unk_token_id
        raw_embeds = self.word_embeddings(input_ids_for_embedding)

        return raw_embeds

    def replace_embed(self, embedding, replaced_embedding, chunk_token_indices):
        '''
        :param chunk_token_indices: [num_chunks, num_chunks-1]
        :param embedding: [num_chunks, max_length, embedding_size]
        :param replace_embedding:[num_chunks, embedding_size]
        :return:
        '''
        bs = embedding.shape[0]
        cz = replaced_embedding.shape[0]
        for bidx in range(bs):
            for i in range(cz):
                embedding[bidx, chunk_token_indices[i], :] = replaced_embedding[i, :]

        return embedding

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return False


class E5_EmbeddingsWrapper(EmbeddingsWrapper):
    def __init__(
        self, model_name, is_embeddings_twice, **model_kwargs
    ):
        super().__init__(model_name, is_embeddings_twice, **model_kwargs)

    def encode_corpus(
            self,
            sentences: Union[str, List[str]],
            batch_size=32,
            *args,
            **kwargs,
    ):
        sentence_embeddings = []
        output_embeddings = []
        # cls_embeddings, sep_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = ["passage: " + sentence for sentence in sentences[start_index:end_index]]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                # Perform pooling
                batch_embeddings = [self.mean_pooling(layer_hidden_states, encoded_input['attention_mask']) for
                                    layer_hidden_states in batch_model_output.hidden_states]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)  # [batch_size, num_layers, embedding_dim]
                sentence_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

        return sentence_embeddings

    def encode_queries(
            self,
            sentences: Union[str, List[str]],
            batch_size=32,
            *args,
            **kwargs,
    ):
        sentence_embeddings = []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = ["query: " + sentence for sentence in sentences[start_index:end_index]]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input)
                # Perform pooling
                batch_embeddings = self.mean_pooling(batch_model_output.last_hidden_state,
                                                     encoded_input['attention_mask'])
                # # Normalize embeddings
                # batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings.cpu())

                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        return sentence_embeddings

    def encode_second(
            self,
            corpus: Union[str, List[str]],
            embeddings_corpus,
            batch_size=32,
            *args,
            **kwargs,
    ):
        corpus_embeddings = []
        instruction = "passage: "
        with torch.no_grad():
            for corpus_id, chunks in corpus.items():
                num_instruction_tokens = len(self._tokenizer(instruction, add_special_tokens=False)["input_ids"])
                num_chunks = min(len(chunks), self.max_length - 2 - num_instruction_tokens)
                embeddings_chunks = embeddings_corpus[corpus_id][:num_chunks].cuda()
                # chunks = corpus[corpus_id]
                # Pad special chunk id
                chunks_padded = instruction + " ".join([self.chunk_token for _ in range(num_chunks)])
                # Tokenize sentences
                encoded_input = self._tokenizer(chunks_padded, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                input_ids = encoded_input["input_ids"]
                attention_mask = encoded_input["attention_mask"]
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
                # max_seq_length = input_ids.shape[1]
                word_embeddings = self.embedding_chunk_as_unknown(input_ids)
                hidden_states = self.embeddings(inputs_embeds=word_embeddings)
                # print(f"input_embeddings: {input_embeddings.shape}")
                # hidden_states = input_embeddings
                chunk_token_indices = torch.nonzero(input_ids == self.chunk_token_id, as_tuple=False).reshape((num_chunks, 2))[:, 1]

                for i, layer in enumerate(self.layers):
                    if i in range(self.num_layers):
                        replaced_embedding = embeddings_chunks[:, i, :]
                        hidden_states = self.replace_embed(hidden_states, replaced_embedding, chunk_token_indices)
                        hidden_states = layer(hidden_states, extended_attention_mask)[0]
                    else:
                        hidden_states = layer(hidden_states, extended_attention_mask)[0]

                chunk_embeddings = self.mean_pooling(hidden_states, attention_mask)

                corpus_embeddings.append(chunk_embeddings.cpu())
                torch.cuda.empty_cache()

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

        return corpus_embeddings.to(torch.float32)


class BGE_EmbeddingsWrapper(EmbeddingsWrapper):
    def __init__(
        self, model_name, is_embeddings_twice, **model_kwargs
    ):
        super().__init__(model_name, is_embeddings_twice, **model_kwargs)

    def encode_corpus(
            self,
            sentences: Union[str, List[str]],
            batch_size=32,
            *args,
            **kwargs,
    ):
        sentence_embeddings = []
        output_embeddings = []
        # cls_embeddings, sep_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = ["passage: " + sentence for sentence in sentences[start_index:end_index]]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                # Perform pooling
                batch_embeddings = [self.cls_pooling(layer_hidden_states, encoded_input['attention_mask']) for
                                    layer_hidden_states in batch_model_output.hidden_states]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)  # [batch_size, num_layers, embedding_dim]
                sentence_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

        return sentence_embeddings

    def encode_queries(
            self,
            sentences: Union[str, List[str]],
            batch_size=32,
            *args,
            **kwargs,
    ):
        sentence_embeddings = []
        instruction = "Represent this sentence for searching relevant passages: "
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = [instruction + sentence for sentence in sentences[start_index:end_index]]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt',
                                                add_special_tokens=True).to(self._model.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input)
                # Perform pooling
                batch_embeddings = self.cls_pooling(batch_model_output.last_hidden_state,
                                                     encoded_input['attention_mask'])
                # # Normalize embeddings
                # batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings.cpu())

                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        return sentence_embeddings

    # def encode_second(
    #         self,
    #         corpus: Union[str, List[str]],
    #         embeddings_corpus,
    #         batch_size=32,
    #         *args,
    #         **kwargs,
    # ):
    #     corpus_embeddings = []
    #     with torch.no_grad():
    #         for corpus_id, chunks in corpus.items():
    #             num_chunks = min(len(chunks), self.max_length - 2)
    #             embeddings_chunks = embeddings_corpus[corpus_id][:num_chunks].cuda()
    #             # chunks = corpus[corpus_id]
    #             # Pad special chunk id
    #             chunks_padded = " ".join([self.chunk_token for _ in range(num_chunks)])
    #             # Tokenize sentences
    #             encoded_input = self._tokenizer(chunks_padded, padding=True, truncation=True,
    #                                             max_length=self.max_length, return_tensors='pt',
    #                                             add_special_tokens=True).to(self._model.device.type)
    #             input_ids = encoded_input["input_ids"]
    #             attention_mask = encoded_input["attention_mask"]
    #             extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
    #             extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
    #             # max_seq_length = input_ids.shape[1]
    #             word_embeddings = self.embedding_chunk_as_unknown(input_ids)
    #             hidden_states = self.embeddings(inputs_embeds=word_embeddings)
    #             # print(f"input_embeddings: {input_embeddings.shape}")
    #             # hidden_states = input_embeddings
    #             chunk_token_indices = torch.nonzero(input_ids == self.chunk_token_id, as_tuple=False).reshape((num_chunks, 2))[:, 1]
    #
    #             for i, layer in enumerate(self.layers):
    #                 if i in range(self.num_layers):
    #                     replaced_embedding = embeddings_chunks[:, i, :]
    #                     hidden_states = self.replace_embed(hidden_states, replaced_embedding, chunk_token_indices)
    #                     hidden_states = layer(hidden_states, extended_attention_mask)[0]
    #                 else:
    #                     hidden_states = layer(hidden_states, extended_attention_mask)[0]
    #
    #             chunk_embeddings = self.cls_pooling(hidden_states, attention_mask)
    #
    #             corpus_embeddings.append(chunk_embeddings.cpu())
    #             torch.cuda.empty_cache()
    #
    #     corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    #
    #     return corpus_embeddings.to(torch.float32)

    def encode_second(
            self,
            corpus: Union[str, List[str]],
            embeddings_corpus,
            batch_size=32,
            *args,
            **kwargs,
    ):
        corpus_embeddings = []
        with torch.no_grad():
            for corpus_id, chunks in corpus.items():
                num_chunks = min(len(chunks), self.max_length - 2)
                embeddings_chunks = embeddings_corpus[corpus_id][:num_chunks].cuda()
                embeddings_chunks = embeddings_chunks[:, -2, :].unsqueeze(0)
                attention_mask = torch.ones(1, num_chunks).cuda()
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
                hidden_states = self.layers[-1](embeddings_chunks, extended_attention_mask)[0]

                chunk_embeddings = self.cls_pooling(hidden_states, attention_mask)

                corpus_embeddings.append(chunk_embeddings.cpu())
                torch.cuda.empty_cache()

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

        return corpus_embeddings.to(torch.float32)

    def cls_pooling(self, model_output, attention_mask):
        return model_output[:, 0]


class MistralEmbeddingsWrapper(nn.Module):
    def __init__(
        self, model_name, is_embeddings_twice, **model_kwargs
    ):
        super().__init__()
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            # device_map='auto',
            **model_kwargs
        )
        # print(print(dir(self._model)))
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.word_embeddings = self._model.get_input_embeddings()
        self.rotary_emb = self._model.rotary_emb
        self.update_causal_mask = self._model._update_causal_mask
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = config
        self.max_length = 4096
        self.layers = self._model.layers
        self.num_layers = len(self.layers)
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self._model = torch.nn.DataParallel(self._model)
        if is_embeddings_twice:
            self.chunk_token = "[CHUNK]"
            self._tokenizer.add_special_tokens({'additional_special_tokens': [self.chunk_token]})
            self.chunk_token_id = self._tokenizer.get_vocab()[self.chunk_token]


    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        batch_size=4,
        *args,
        **kwargs,
    ):
        sentence_embeddings = []
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = [self.get_detailed_instruct(task, query) for query in sentences[start_index:end_index]]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt', add_special_tokens=True).to(self.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input)
                # Perform pooling
                batch_embeddings = self.last_token_pool(batch_model_output.last_hidden_state, encoded_input['attention_mask'])
                # # Normalize embeddings
                # batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                sentence_embeddings.append(batch_embeddings.cpu())

                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        return sentence_embeddings

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        batch_size=4,
        *args,
        **kwargs,
    ):
        sentence_embeddings = []
        output_embeddings = []
        # cls_embeddings, sep_embeddings = [], []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                end_index = min(start_index + batch_size, len(sentences))
                batch_sentences = sentences[start_index:end_index]
                # Tokenize sentences
                encoded_input = self._tokenizer(batch_sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt', add_special_tokens=True).to(self.device.type)
                # Compute token embeddings
                batch_model_output = self._model(**encoded_input, return_dict=True, output_hidden_states=True)
                # Perform pooling
                batch_embeddings = [self.last_token_pool(layer_hidden_states, encoded_input['attention_mask']) for layer_hidden_states in batch_model_output.hidden_states]
                batch_embeddings = torch.stack(batch_embeddings).transpose(1, 0)  # [batch_size, num_layers, embedding_dim]
                # # Normalize embeddings
                # batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                sentence_embeddings.append(batch_embeddings.cpu())
                torch.cuda.empty_cache()

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

        return sentence_embeddings

    def encode_second(
            self,
            corpus: Union[str, List[str]],
            embeddings_corpus,
            *args,
            **kwargs,
    ):
        corpus_embeddings = []
        with torch.no_grad():
            for corpus_id, chunks in corpus.items():
                num_chunks = min(len(chunks), self.max_length - 2)
                embeddings_chunks = embeddings_corpus[corpus_id][:num_chunks].cuda()

                chunks_padded = " ".join([self.chunk_token for _ in range(num_chunks)])
                # Tokenize sentences
                encoded_input = self._tokenizer(chunks_padded, padding=True, truncation=True,
                                                max_length=self.max_length, return_tensors='pt',
                                                add_special_tokens=True).to(self.device.type)
                input_ids = encoded_input["input_ids"]
                attention_mask = encoded_input["attention_mask"]
                # max_seq_length = input_ids.shape[1]
                inputs_embeds = self.embedding_chunk_as_unknown(input_ids)
                past_seen_tokens = 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
                position_ids = cache_position.unsqueeze(0)
                output_attentions = self.config.output_attentions
                causal_mask = self.update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, None, output_attentions
                )

                hidden_states = inputs_embeds
                # create position embeddings to be shared across the decoder layers
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                chunk_token_indices = torch.nonzero(input_ids == self.chunk_token_id, as_tuple=False).reshape((num_chunks, 2))[:, 1]

                for i, layer in enumerate(self.layers):
                    if i in range(self.num_layers):
                        replaced_embedding = embeddings_chunks[:, i, :]
                        hidden_states = self.replace_embed(hidden_states, replaced_embedding, chunk_token_indices)
                        hidden_states = layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=output_attentions,
                            use_cache=self.config.use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )[0]

                    else:
                        hidden_states = layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=output_attentions,
                            use_cache=self.config.use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )[0]

                chunk_embeddings = self.last_token_pool(hidden_states, attention_mask)
                # print(chunk_embeddings.device, embeddings_chunks.device)
                # # Normalize embeddings
                # chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

                corpus_embeddings.append(chunk_embeddings.cpu())
                torch.cuda.empty_cache()

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

        return corpus_embeddings.to(torch.float32)

    def embedding_chunk_as_unknown(self, input_ids):
        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[(input_ids == self.chunk_token_id)] = self._tokenizer.unk_token_id
        raw_embeds = self.word_embeddings(input_ids_for_embedding)

        return raw_embeds

    def replace_embed(self, embedding, replaced_embedding, chunk_token_indices):
        '''
        :param chunk_token_indices: [num_chunks, num_chunks-1]
        :param embedding: [num_chunks, max_length, embedding_size]
        :param replace_embedding:[num_chunks, embedding_size]
        :return:
        '''
        bs = embedding.shape[0]
        cz = replaced_embedding.shape[0]
        for bidx in range(bs):
            for i in range(cz):
                embedding[bidx, chunk_token_indices[i], :] = replaced_embedding[i, :]

        return embedding

    def last_token_pool(self,
                        last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    @property
    def device(self):
        return self._model.module.device

    @staticmethod
    def has_instructions():
        return False


MODEL_WRAPPERS = {
    'sentence-transformers/all-MiniLM-L6-v2': EmbeddingsWrapper,
    'facebook/contriever': EmbeddingsWrapper,
    'thenlper/gte-base': EmbeddingsWrapper,
    'thenlper/gte-large': EmbeddingsWrapper,
    'intfloat/e5-base': E5_EmbeddingsWrapper,
    'intfloat/e5-large': E5_EmbeddingsWrapper,
    'BAAI/bge-base-en-v1.5': BGE_EmbeddingsWrapper,
    'BAAI/bge-large-en-v1.5': BGE_EmbeddingsWrapper,
    'intfloat/e5-mistral-7b-instruct': MistralEmbeddingsWrapper
}


def remove_unsupported_kwargs(original_encode):
    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop('prompt_name', None)
        kwargs.pop('request_qid', None)
        return original_encode(self, *args, **kwargs)

    return wrapper


def load_model(model_name, is_embeddings_twice, model_weights=None, **model_kwargs):
    if model_name in MODEL_WRAPPERS:
        model = MODEL_WRAPPERS[model_name](model_name, is_embeddings_twice, **model_kwargs)
        if hasattr(MODEL_WRAPPERS[model_name], 'has_instructions'):
            has_instructions = MODEL_WRAPPERS[model_name].has_instructions()
        else:
            has_instructions = False
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        has_instructions = False

    if model_weights and os.path.exists(model_weights):
        model._model.load_state_dict(torch.load(model_weights, device=model.device))

    return model, has_instructions
