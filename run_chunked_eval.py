import click
import torch
import torch.cuda
import time
import csv
import os
import gc
import json
import glob
import psutil
import numpy as np
from datetime import datetime
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer, AutoConfig
from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.models import load_model


# List of models to test - expanded to 20+ models
MODELS_TO_TEST = [
    # Small models (< 100M params)
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'sentence-transformers/paraphrase-MiniLM-L12-v2',
    
    # Base models (100-400M params)
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-distilroberta-v1',
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'BAAI/bge-large-en-v1.5',
    'intfloat/e5-small',
    'intfloat/e5-base',
    'intfloat/e5-base-v2',
    'intfloat/e5-large',
    'intfloat/e5-large-v2',
    'thenlper/gte-small',
    'thenlper/gte-base',
    'thenlper/gte-large',
    
    # Multilingual and specialized models
    'intfloat/multilingual-e5-base',
    'intfloat/multilingual-e5-small',
    'intfloat/multilingual-e5-large',
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    
    # Contrastive models
    'facebook/contriever',
    'facebook/contriever-msmarco',
    
    # Newer models
    'jinaai/jina-embeddings-v2-small-en',
    'jinaai/jina-embeddings-v2-base-en',
]


class MetricsCollector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.encode_start_time = None
        self.eval_start_time = None
        self.encode_time = 0
        self.eval_time = 0
        self.peak_gpu_memory = 0
        self.gpu_memory_samples = []
        self.encoder_passes = 0
        self.gflops = 0
        self.num_samples = 0
        self.text_lengths = []
        self.chunks_per_doc = []
        self.total_chunks = 0
        self.chunk_stats_collected = False
        self.total_tokens_processed = 0  # Track total tokens processed
        self.model_params = 0  # Track model parameter count
        
    def start_timing(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
    def start_encoding(self):
        self.encode_start_time = time.time()
        
    def end_encoding(self):
        if self.encode_start_time:
            self.encode_time = time.time() - self.encode_start_time
            
    def start_evaluation(self):
        self.eval_start_time = time.time()
        
    def end_evaluation(self):
        if self.eval_start_time:
            self.eval_time = time.time() - self.eval_start_time
            
    def record_gpu_memory(self):
        if torch.cuda.is_available():
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            current_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_samples.append(current_mem)
            
    def get_avg_gpu_memory(self):
        if self.gpu_memory_samples:
            return np.mean(self.gpu_memory_samples)
        return 0
    
    def record_tokens_processed(self, num_tokens):
        """Record the number of tokens processed during encoding"""
        self.total_tokens_processed += num_tokens
        
    def calculate_gflops(self, model_params_millions, encode_time_seconds):
        """
        Estimate GFLOPS based on model parameters and processing time
        For Transformer models, each token requires approximately 2*params FLOPs
        """
        if encode_time_seconds > 0 and self.total_tokens_processed > 0:
            # Convert parameters from millions to actual count
            model_params = model_params_millions * 1e6
            
            # Estimate FLOPs: 2 * parameters * tokens * encoder_passes
            # encoder_passes accounts for two-pass encoding scenarios
            total_flops = 2 * model_params * self.total_tokens_processed * max(1, self.encoder_passes)
            
            # Convert to GFLOPS (billion FLOPs per second)
            self.gflops = (total_flops / 1e9) / encode_time_seconds
        else:
            self.gflops = 0
            
        return self.gflops
        
    def record_text_stats(self, corpus, chunk_info=None):
        """Record text and chunking statistics"""
        for doc_id, doc in corpus.items():
            text = doc.get('text', '')
            self.text_lengths.append(len(text))
        self.num_samples = len(corpus)
        
        # If chunk_info is provided, use it
        if chunk_info:
            self.chunks_per_doc = chunk_info.get('chunks_per_doc', [])
            self.total_chunks = chunk_info.get('total_chunks', 0)
            self.chunk_stats_collected = True
            
    def record_chunking_stats(self, chunks_per_doc_list, total_chunks_count):
        """Specifically record chunking statistics"""
        self.chunks_per_doc = chunks_per_doc_list
        self.total_chunks = total_chunks_count
        self.chunk_stats_collected = True
        
    def estimate_chunk_stats(self, corpus, chunk_size, mode):
        """Estimate chunk statistics if not collected"""
        if not self.chunk_stats_collected and corpus:
            # Estimate based on average text length and chunk size
            # Assuming ~4 characters per token on average
            chars_per_token = 4
            estimated_chunks_list = []
            total_estimated_chunks = 0
            estimated_total_tokens = 0
            
            for doc_id, doc in corpus.items():
                text_len = len(doc.get('text', ''))
                estimated_tokens = text_len // chars_per_token
                estimated_chunks = max(1, estimated_tokens // chunk_size)
                estimated_chunks_list.append(estimated_chunks)
                total_estimated_chunks += estimated_chunks
                # Estimate tokens processed (capped by chunk_size * chunks)
                estimated_total_tokens += min(estimated_tokens, chunk_size * estimated_chunks)
            
            self.chunks_per_doc = estimated_chunks_list
            self.total_chunks = total_estimated_chunks
            self.total_tokens_processed = estimated_total_tokens
            
            # Set encoder passes based on mode
            if mode in ["chunk_twice_avg", "chunk_twice_weighted", "chunk_twice_ppr"]:
                self.encoder_passes = 2  # Two-pass encoding
            elif mode in ["chunk_avg", "chunk_weighted"]:
                self.encoder_passes = 1  # Single-pass encoding
            else:
                self.encoder_passes = 1  # Truncation mode
        
    def get_metrics_dict(self):
        return {
            'time_encode_s': round(self.encode_time, 2),
            'time_eval_s': round(self.eval_time, 2),
            'time_total_s': round(self.encode_time + self.eval_time, 2),
            'peak_gpu_mem_mb': round(self.peak_gpu_memory, 2),
            'avg_gpu_mem_mb': round(self.get_avg_gpu_memory(), 2),
            'encoder_passes': self.encoder_passes,
            'gflops_total': round(self.gflops, 2),
            'num_samples': self.num_samples,
            'avg_text_length': round(np.mean(self.text_lengths) if self.text_lengths else 0, 2),
            'avg_chunks_per_doc': round(np.mean(self.chunks_per_doc) if self.chunks_per_doc else 0, 2),
            'total_chunks': self.total_chunks,
            'total_tokens_processed': self.total_tokens_processed,
        }


def extract_scores_from_json(output_folder, task_name, model_name):
    """Extract scores from MTEB's JSON output files"""
    # Clean model name for file matching
    model_name_clean = model_name.replace('/', '_')
    
    # Try different patterns to find the JSON file
    patterns = [
        os.path.join(output_folder, f"*{task_name}*.json"),
        os.path.join(output_folder, model_name_clean, f"*{task_name}*.json"),
        os.path.join(output_folder, model_name_clean, "*.json"),
        os.path.join(output_folder, "**", f"*{task_name}*.json"),
        os.path.join(output_folder, "**", "*.json"),
    ]
    
    json_files = []
    for pattern in patterns:
        json_files.extend(glob.glob(pattern, recursive=True))
    
    if not json_files:
        print(f"No JSON files found for task {task_name}")
        return {}
    
    # Get the most recent file
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"Reading scores from: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Navigate through different possible structures
        scores = {}
        
        # Check if it's a list (some MTEB versions return list)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        
        # Try to extract scores
        if 'scores' in data:
            scores_data = data['scores']
            # Check for different split names
            for split in ['test', 'default', 'validation', 'dev']:
                if split in scores_data:
                    if isinstance(scores_data[split], list) and len(scores_data[split]) > 0:
                        scores = scores_data[split][0] if isinstance(scores_data[split][0], dict) else {}
                    elif isinstance(scores_data[split], dict):
                        scores = scores_data[split]
                    break
        
        # Filter to only include numeric metrics
        metric_scores = {}
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                metric_scores[key] = value
        
        # Also try to get the main_score if it's stored elsewhere
        if 'main_score' in data and 'main_score' not in metric_scores:
            metric_scores['main_score'] = data['main_score']
        
        return metric_scores
    
    except Exception as e:
        print(f"Error reading JSON file {latest_file}: {e}")
        return {}


def save_results_to_csv(results, model_name, model_info, method_name, task_name, chunking_args, twice_ratio, top_k_context):
    """Save evaluation results to CSV file"""
    # Create results directory if it doesn't exist
    os.makedirs("/home/jialehan/ZZMA/embedding-twice-longdoc-k/csv_results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"/home/jialehan/ZZMA/embedding-twice-longdoc-k/csv_results/results_{timestamp}.csv"
    
    # Check if file exists to write headers
    write_header = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            # Model info
            'model_name', 'embedding_mode', 'task_name', 'timestamp',
            'model_params_m', 'hidden_dim', 'num_layers', 'max_length',
            
            # Chunking parameters
            'chunk_size', 'chunk_overlap', 'chunking_strategy',
            'twice_ratio', 'top_k_context', 'replaced_layers',
            
            # Performance metrics
            'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10',
            'map_at_1', 'map_at_3', 'map_at_5', 'map_at_10',
            'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
            'precision_at_1', 'precision_at_3', 'precision_at_5', 'precision_at_10',
            'mrr_at_1', 'mrr_at_3', 'mrr_at_5', 'mrr_at_10',
            'main_score',
            
            # Efficiency metrics
            'time_encode_s', 'time_eval_s', 'time_total_s',
            'peak_gpu_mem_mb', 'avg_gpu_mem_mb',
            'encoder_passes', 'gflops_total',
            'num_samples', 'avg_text_length', 'avg_chunks_per_doc', 'total_chunks',
            'total_tokens_processed',
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
            
        # Flatten results dictionary
        row = {
            'model_name': model_name,
            'embedding_mode': method_name,
            'task_name': task_name,
            'timestamp': timestamp,
            'model_params_m': model_info.get('model_params_m', 0),
            'hidden_dim': model_info.get('hidden_dim', 0),
            'num_layers': model_info.get('num_layers', 0),
            'max_length': model_info.get('max_length', 0),
            'chunk_size': chunking_args.get('chunk_size', 0),
            'chunk_overlap': chunking_args.get('chunk_overlap', 0),
            'chunking_strategy': chunking_args.get('chunking_strategy', ''),
            'twice_ratio': twice_ratio,
            'top_k_context': top_k_context,
            'replaced_layers': model_info.get('replaced_layers', 0),
        }
        
        # Add all metrics from results
        row.update(results)
        
        writer.writerow(row)
    
    print(f"✓ Results saved to {csv_filename}")
    return csv_filename


# Custom MTEB wrapper to capture timing
class MTEBWithTiming(MTEB):
    def __init__(self, *args, metrics_collector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
        
    def run(self, model, *args, **kwargs):
        if self.metrics_collector:
            self.metrics_collector.start_encoding()
        
        result = super().run(model, *args, **kwargs)
        
        if self.metrics_collector:
            self.metrics_collector.end_encoding()
            # The evaluation time is everything after encoding
            self.metrics_collector.eval_time = time.time() - self.metrics_collector.start_time - self.metrics_collector.encode_time
        
        return result


@click.command()
@click.option('--model-name', default=None, help='Single model to test, or use --test-all')
@click.option('--test-all', is_flag=True, help='Test all models in the predefined list')
@click.option('--model-weights', default=None, help='Path to model weights for finetuning')
@click.option('--chunking-strategy', default='token', help='Chunking strategy')
@click.option('--task-name', 'task_names', multiple=True, 
              default=("LEMBWikimQARetrievalChunked",), show_default=True)
@click.option('--eval-split', default='test', help='Evaluation split')
@click.option('--batch-size', default=1, help='Batch size')
@click.option('--tau', default=0.1, help='Temperature for weighted methods')
@click.option('--twice-ratio', default=0.75, help='Ratio of layers to use twice technique (0-1)')
@click.option('--top-k-context', default=30, type=int, help='Number of surrounding chunks for twice methods')
@click.option('--embedding-mode', type=click.Choice(
    ["truncation", "chunk_avg", "chunk_weighted", "chunk_twice_avg", "chunk_twice_weighted"],
    case_sensitive=False), default="chunk_twice_avg", help='Embedding mode to use')
@click.option('--test-all-modes', is_flag=True, help='Test all embedding modes')
@click.option('--chunk-size', default=256, type=int, help='Number of tokens per chunk')
@click.option('--chunk-overlap', default=0, type=int, help='Chunk overlap')
@click.option('--n-sentences', default=5, type=int, help='Number of sentences per chunk')
def main(model_name, test_all, model_weights, chunking_strategy, task_names, eval_split,
         batch_size, tau, twice_ratio, top_k_context, embedding_mode,
         test_all_modes, chunk_size, chunk_overlap, n_sentences):
    
    # Determine which models to test
    if test_all:
        models_to_test = MODELS_TO_TEST
    elif model_name:
        models_to_test = [model_name]
    else:
        models_to_test = ['intfloat/e5-large-v2']  # Default model
    
    # Determine which modes to test
    if test_all_modes:
        modes_to_test = ["truncation", "chunk_avg", "chunk_weighted", "chunk_twice_avg", "chunk_twice_weighted"]
    else:
        modes_to_test = [embedding_mode]
    
    # Initialize tasks
    task_list = []
    for name in task_names:
        try:
            task_cls = globals()[name]
        except KeyError:
            raise ValueError(f"Unknown task name: {name}")
        task_list.append(task_cls)
    
    # Test each model
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}\n")
        
        try:
            # Test each embedding mode
            for mode in modes_to_test:
                print(f"\n{'-'*60}")
                print(f"Testing mode: {mode}")
                print(f"{'-'*60}\n")
                
                metrics_collector = MetricsCollector()
                metrics_collector.start_timing()
                
                # Prepare chunking arguments
                chunking_args = {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'n_sentences': n_sentences,
                    'chunking_strategy': chunking_strategy,
                    'embedding_model_name': model_name,
                }
                
                # Load model
                try:
                    # Patch the model to capture chunk statistics
                    model = load_model(model_name, mode, chunking_args, twice_ratio=twice_ratio)
                    model._metrics_collector = metrics_collector  # Attach metrics collector
                    
                    # Calculate replaced_layers based on twice_ratio
                    if hasattr(model, 'num_layers'):
                        replaced_layers = int(model.num_layers * twice_ratio)
                    else:
                        replaced_layers = 18  # Default fallback
                    
                    # Get model info
                    model_info = {
                        'model_params_m': round(sum(p.numel() for p in model._model.parameters()) / 1e6, 2),
                        'hidden_dim': model.hidden_size if hasattr(model, 'hidden_size') else 0,
                        'num_layers': model.num_layers if hasattr(model, 'num_layers') else 0,
                        'max_length': model.max_length if hasattr(model, 'max_length') else 0,
                        'replaced_layers': replaced_layers,
                    }
                    
                except Exception as e:
                    print(f"Failed to load model {model_name} with mode {mode}: {e}")
                    continue
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                if torch.cuda.is_available():
                    model = model.cuda()
                
                model.eval()
                
                # Create tasks
                tasks = [
                    cls(
                        embedding_mode=mode,
                        tokenizer=tokenizer,
                        prune_size=None,
                        **chunking_args,
                    )
                    for cls in task_list
                ]
                
                # Run evaluation with custom wrapper
                evaluation = MTEBWithTiming(
                    tasks=tasks,
                    embedding_mode=mode,
                    tokenizer=tokenizer,
                    prune_size=None,
                    metrics_collector=metrics_collector,
                    **chunking_args,
                )
                
                # Record initial GPU memory
                metrics_collector.record_gpu_memory()
                
                # Run evaluation
                print("Running MTEB evaluation...")
                output_folder = f"results/{model_name.replace('/', '_')}"
                
                _ = evaluation.run(
                    model,
                    output_folder=output_folder,
                    eval_splits=[eval_split],
                    overwrite_results=True,
                    encode_kwargs={
                        'batch_size': batch_size,
                        'tau': tau,
                        'replaced_layers': replaced_layers,
                        'top_k_context': top_k_context
                    }
                )
                
                # Record final GPU memory
                metrics_collector.record_gpu_memory()
                
                # Extract corpus info for metrics
                for task in tasks:
                    if hasattr(task, 'corpus') and task.corpus:
                        corpus = task.corpus.get(eval_split, {})
                        metrics_collector.record_text_stats(corpus)
                        
                        # Estimate chunk stats if not already collected
                        metrics_collector.estimate_chunk_stats(corpus, chunk_size, mode)
                
                # Calculate GFLOPS before getting metrics dict
                metrics_collector.calculate_gflops(
                    model_info['model_params_m'],
                    metrics_collector.encode_time
                )
                
                # Get efficiency metrics
                efficiency_metrics = metrics_collector.get_metrics_dict()
                
                # Process results for each task
                for task_idx, task in enumerate(tasks):
                    task_name = task_names[task_idx]
                    
                    # Extract scores from JSON files
                    print(f"\nExtracting scores for {task_name}...")
                    perf_metrics = extract_scores_from_json(output_folder, task_name, model_name)
                    
                    if perf_metrics:
                        # Print metrics
                        print(f"\nPerformance metrics for {task_name}:")
                        for key, value in perf_metrics.items():
                            if isinstance(value, (float, int)):
                                print(f"  {key}: {value:.5f}")
                        
                        print(f"\nEfficiency metrics:")
                        for key, value in efficiency_metrics.items():
                            print(f"  {key}: {value}")
                        
                        # Combine performance and efficiency metrics
                        all_metrics = {**perf_metrics, **efficiency_metrics}
                        
                        # Save to CSV
                        csv_file = save_results_to_csv(
                            all_metrics,
                            model_name,
                            model_info,
                            mode,
                            task_name,
                            chunking_args,
                            twice_ratio,
                            top_k_context
                        )
                    else:
                        print(f"Warning: Could not extract scores for {task_name}")
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("All evaluations complete!")
    print("Check /home/jialehan/ZZMA/embedding-twice-longdoc-k/csv_results/ folder for saved results")
    print("="*80)


if __name__ == '__main__':
    main()