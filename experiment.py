import os
import sys
import json
import subprocess
import time
import pandas as pd
import glob
from datetime import datetime
from typing import List, Dict, Tuple
import traceback
import signal
import atexit

BASE_DIR = "/home/jialehan/ZZMA/embedding-twice-longdoc-k"

os.makedirs(BASE_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(BASE_DIR, "test_runner_checkpoint.json")
CONSOLIDATED_RESULTS_DIR = os.path.join(BASE_DIR, "consolidated_results")
FAILED_TESTS_FILE = os.path.join(BASE_DIR, "failed_tests.json")
LOG_FILE = os.path.join(BASE_DIR, "test_runner.log")
CSV_RESULTS_DIR = os.path.join(BASE_DIR, "csv_results")
RESUME_INFO_FILE = os.path.join(BASE_DIR, "resume_info.txt")

TEST_CONFIG = {
    "models": [
        'jinaai/jina-embeddings-v2-small-en',
        'jinaai/jina-embeddings-v2-base-en',
        'intfloat/multilingual-e5-base',
        'intfloat/multilingual-e5-small',
        'intfloat/multilingual-e5-large',
        'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
        'sentence-transformers/multi-qa-mpnet-base-dot-v1',
        'sentence-transformers/all-MiniLM-L6-v2',
        #'sentence-transformers/all-MiniLM-L12-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        #'sentence-transformers/paraphrase-MiniLM-L12-v2',
        # Base models
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
        'facebook/contriever',
        'facebook/contriever-msmarco',
    ],
    
    "embedding_modes": [
        "truncation",
        "chunk_avg",
        "chunk_weighted",
        "chunk_twice_avg",
        "chunk_twice_weighted"
    ],
    
    "tasks": [
        "LEMBNarrativeQARetrievalChunked",
        "LEMBWikimQARetrievalChunked",
        "LEMBSummScreenFDRetrievalChunked",
        "LEMBQMSumRetrievalChunked"
    ],
    
    #"chunk_sizes": [256, 512],
    "chunk_sizes": [256],
    #"twice_ratios": [0.5, 0.75, 1.0],
    "twice_ratios": [0.75],
    #"top_k_contexts": [3, 10, 30],
    "top_k_contexts": [30],
    
    # Fixed parameters
    "batch_size": 1,
    "tau": 0.1,
    "chunk_overlap": 0,
    "chunking_strategy": "token"
}


class TestRunner:
    def __init__(self):
        self.base_dir = BASE_DIR 
        self.ensure_directories()
        self.checkpoint = self.load_checkpoint()
        self.failed_tests = self.checkpoint.get("failed", [])
        self.completed_tests = self.checkpoint.get("completed", [])
        self.setup_logging()
        self.setup_signal_handlers()
        self.check_resume_status()
        
    def ensure_directories(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(CONSOLIDATED_RESULTS_DIR, exist_ok=True)
        os.makedirs(CSV_RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "results"), exist_ok=True)
        
    def check_resume_status(self):
        if self.completed_tests or self.failed_tests:
            self.log("="*60)
            self.log("RESUMING FROM PREVIOUS RUN")
            self.log(f"Previously completed: {len(self.completed_tests)} tests")
            self.log(f"Previously failed: {len(self.failed_tests)} tests")
            self.log("Will skip completed tests and retry failed ones")
            self.log("="*60 + "\n")
            with open(RESUME_INFO_FILE, 'w') as f:
                f.write(f"Resume Time: {datetime.now()}\n")
                f.write(f"Completed Tests: {len(self.completed_tests)}\n")
                f.write(f"Failed Tests: {len(self.failed_tests)}\n")
                f.write(f"Last checkpoint: {self.checkpoint.get('last_updated', 'Unknown')}\n")
        
    def setup_logging(self):
        self.log_file = open(LOG_FILE, 'a')
        self.log(f"\n{'='*80}")
        self.log(f"Test Runner Started: {datetime.now()}")
        self.log(f"Base Directory: {self.base_dir}")
        self.log(f"{'='*80}\n")
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + "\n")
        self.log_file.flush()
        
    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            self.log("\nReceived interrupt signal. Saving checkpoint...")
            self.save_checkpoint()
            self.log("Checkpoint saved successfully!")
            self.log("You can resume by running the script again.")
            self.report_summary()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.cleanup)
        
    def cleanup(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
            
    def load_checkpoint(self) -> Dict:
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint_data = json.load(f)
                    if not isinstance(checkpoint_data.get("completed", None), list):
                        checkpoint_data["completed"] = []
                    if not isinstance(checkpoint_data.get("failed", None), list):
                        checkpoint_data["failed"] = []
                    return checkpoint_data
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh...")
                return {"completed": [], "failed": []}
        return {"completed": [], "failed": []}
        
    def save_checkpoint(self):
        checkpoint_data = {
            "completed": self.completed_tests,
            "failed": self.failed_tests,
            "last_updated": datetime.now().isoformat(),
            "total_tests": len(self.generate_test_combinations()),
            "progress_percentage": (len(self.completed_tests) / len(self.generate_test_combinations())) * 100
        }
        
        if os.path.exists(CHECKPOINT_FILE):
            backup_file = CHECKPOINT_FILE + ".backup"
            os.rename(CHECKPOINT_FILE, backup_file)
            
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
    def generate_test_combinations(self) -> List[Dict]:
        test_combinations = []
        for model in TEST_CONFIG["models"]:
            for mode in TEST_CONFIG["embedding_modes"]:
                for task in TEST_CONFIG["tasks"]:
                    for chunk_size in TEST_CONFIG["chunk_sizes"]:
                        if "twice" not in mode:
                            test_combinations.append({
                                "model": model,
                                "mode": mode,
                                "task": task,
                                "chunk_size": chunk_size,
                                "twice_ratio": 0.75, 
                                "top_k_context": 30  
                            })
                        else:
                            for twice_ratio in TEST_CONFIG["twice_ratios"]:
                                for top_k in TEST_CONFIG["top_k_contexts"]:
                                    test_combinations.append({
                                        "model": model,
                                        "mode": mode,
                                        "task": task,
                                        "chunk_size": chunk_size,
                                        "twice_ratio": twice_ratio,
                                        "top_k_context": top_k
                                    })
        
        return test_combinations
        
    def get_test_id(self, test: Dict) -> str:
        return f"{test['model']}_{test['mode']}_{test['task']}_cs{test['chunk_size']}_tr{test['twice_ratio']}_tk{test['top_k_context']}"
        
    def run_single_test(self, test: Dict) -> bool:
        test_id = self.get_test_id(test)
        
        if test_id in self.completed_tests:
            self.log(f"Skipping completed test: {test_id}")
            return True
            
        self.log(f"Running test: {test_id}")
        script_path = os.path.join(self.base_dir, "run_chunked_eval.py")
        if not os.path.exists(script_path):
            self.log(f"Error: Script not found at {script_path}")
            return False
            
        cmd = [
            sys.executable, 
            script_path,
            "--model-name", test["model"],
            "--embedding-mode", test["mode"],
            "--task-name", test["task"],
            "--chunk-size", str(test["chunk_size"]),
            "--twice-ratio", str(test["twice_ratio"]),
            "--top-k-context", str(test["top_k_context"]),
            "--batch-size", str(TEST_CONFIG["batch_size"]),
            "--tau", str(TEST_CONFIG["tau"]),
            "--chunk-overlap", str(TEST_CONFIG["chunk_overlap"]),
            "--chunking-strategy", TEST_CONFIG["chunking_strategy"]
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per test
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                self.log(f"✓ Test completed successfully: {test_id}")
                self.completed_tests.append(test_id)
                # Remove from failed list if it was there (retry successful)
                self.failed_tests = [f for f in self.failed_tests if f.get("test_id") != test_id]
                self.save_checkpoint()
                return True
            else:
                self.log(f"✗ Test failed with return code {result.returncode}: {test_id}")
                self.log(f"  Error: {result.stderr[:500] if result.stderr else 'No error output'}")
                
                # Update or add to failed tests
                self.failed_tests = [f for f in self.failed_tests if f.get("test_id") != test_id]
                self.failed_tests.append({
                    "test_id": test_id,
                    "test": test,
                    "error": result.stderr[:1000] if result.stderr else "No error output",
                    "return_code": result.returncode,
                    "timestamp": datetime.now().isoformat()
                })
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ Test timed out: {test_id}")
            self.failed_tests = [f for f in self.failed_tests if f.get("test_id") != test_id]
            self.failed_tests.append({
                "test_id": test_id,
                "test": test,
                "error": "Test timed out after 1 hour",
                "return_code": -1,
                "timestamp": datetime.now().isoformat()
            })
            return False
            
        except Exception as e:
            self.log(f"✗ Unexpected error running test: {test_id}")
            self.log(f"  Error: {str(e)}")
            self.failed_tests = [f for f in self.failed_tests if f.get("test_id") != test_id]
            self.failed_tests.append({
                "test_id": test_id,
                "test": test,
                "error": str(e),
                "return_code": -2,
                "timestamp": datetime.now().isoformat()
            })
            return False
            
    def consolidate_results(self):
        self.log("\nConsolidating results...")
        os.makedirs(CONSOLIDATED_RESULTS_DIR, exist_ok=True)
        csv_pattern = os.path.join(CSV_RESULTS_DIR, "results_*.csv")
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            self.log(f"No CSV files found in {CSV_RESULTS_DIR}")
            return
        all_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                self.log(f"  Read {len(df)} rows from {os.path.basename(csv_file)}")
            except Exception as e:
                self.log(f"  Error reading {csv_file}: {e}")
                
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            key_columns = ['model_name', 'embedding_mode', 'task_name', 
                          'chunk_size', 'twice_ratio', 'top_k_context']
            initial_rows = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=key_columns, keep='last')
            removed_duplicates = initial_rows - len(combined_df)
            
            if removed_duplicates > 0:
                self.log(f"  Removed {removed_duplicates} duplicate rows")
            combined_df = combined_df.sort_values(by=['model_name', 'embedding_mode', 'task_name'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(CONSOLIDATED_RESULTS_DIR, f"consolidated_results_{timestamp}.csv")
            combined_df.to_csv(output_file, index=False)
            self.log(f"✓ Results consolidated to: {output_file}")
            self.log(f"  Total rows: {len(combined_df)}")
            self.create_summary_stats(combined_df, timestamp)
        else:
            self.log("No valid CSV data to consolidate")
            
    def create_summary_stats(self, df: pd.DataFrame, timestamp: str):
        summary_file = os.path.join(CONSOLIDATED_RESULTS_DIR, f"summary_stats_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total test results: {len(df)}\n")
            f.write(f"Unique models: {df['model_name'].nunique()}\n")
            f.write(f"Unique modes: {df['embedding_mode'].nunique()}\n")
            f.write(f"Unique tasks: {df['task_name'].nunique()}\n\n")
            f.write("BEST MODELS PER TASK (by main_score):\n")
            f.write("-"*40 + "\n")
            for task in df['task_name'].unique():
                task_df = df[df['task_name'] == task]
                if 'main_score' in task_df.columns and not task_df['main_score'].isna().all():
                    best_idx = task_df['main_score'].idxmax()
                    if pd.notna(best_idx):
                        best = task_df.loc[best_idx]
                        f.write(f"\n{task}:\n")
                        f.write(f"  Model: {best['model_name']}\n")
                        f.write(f"  Mode: {best['embedding_mode']}\n")
                        f.write(f"  Score: {best['main_score']:.5f}\n")
                        if 'chunk_size' in best:
                            f.write(f"  Chunk size: {best['chunk_size']}\n")
                        if 'twice_ratio' in best and 'twice' in best['embedding_mode']:
                            f.write(f"  Twice ratio: {best['twice_ratio']}\n")
                            f.write(f"  Top-k context: {best['top_k_context']}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("AVERAGE SCORES PER EMBEDDING MODE:\n")
            f.write("-"*40 + "\n")
            if 'main_score' in df.columns:
                mode_stats = df.groupby('embedding_mode')['main_score'].agg(['mean', 'std', 'count'])
                f.write(mode_stats.to_string())
            f.write("\n\n" + "="*80 + "\n")
            f.write("EFFICIENCY METRICS:\n")
            f.write("-"*40 + "\n")
            if 'time_total_s' in df.columns:
                f.write("\nFastest configurations (top 10):\n")
                fastest = df.nsmallest(10, 'time_total_s')[['model_name', 'embedding_mode', 'task_name', 'time_total_s']]
                f.write(fastest.to_string(index=False))
                
            if 'peak_gpu_mem_mb' in df.columns:
                f.write("\n\nLowest memory usage (top 10):\n")
                lowest_mem = df.nsmallest(10, 'peak_gpu_mem_mb')[['model_name', 'embedding_mode', 'peak_gpu_mem_mb']]
                f.write(lowest_mem.to_string(index=False))
                
        self.log(f"✓ Summary statistics saved to: {summary_file}")
        
    def report_summary(self):
        """Report final summary of the test run"""
        self.log("\n" + "="*80)
        self.log("TEST RUN SUMMARY")
        self.log("="*80)
        
        total_tests = len(self.generate_test_combinations())
        self.log(f"Total tests: {total_tests}")
        self.log(f"Completed: {len(self.completed_tests)}")
        self.log(f"Failed: {len(self.failed_tests)}")
        self.log(f"Remaining: {total_tests - len(self.completed_tests)}")
        self.log(f"Progress: {(len(self.completed_tests) / total_tests * 100):.1f}%")
        
        if self.failed_tests:
            self.log("\nFAILED TESTS:")
            self.log("-"*40)
            for failed in self.failed_tests[-10:]:  # Show last 10 failures
                self.log(f"• {failed['test_id']}")
                self.log(f"  Error: {failed['error'][:200]}")
                
            if len(self.failed_tests) > 10:
                self.log(f"  ... and {len(self.failed_tests) - 10} more")
            with open(FAILED_TESTS_FILE, 'w') as f:
                json.dump(self.failed_tests, f, indent=2)
            self.log(f"\nDetailed failure info saved to: {FAILED_TESTS_FILE}")
            
    def run_all_tests(self):
        test_combinations = self.generate_test_combinations()
        total_tests = len(test_combinations)
        
        self.log(f"Total test combinations to run: {total_tests}")
        self.log(f"Already completed: {len(self.completed_tests)}")
        self.log(f"Remaining: {total_tests - len(self.completed_tests)}")
        
        tests_run_this_session = 0
        
        for i, test in enumerate(test_combinations, 1):
            test_id = self.get_test_id(test)
            
            if test_id in self.completed_tests:
                continue
                
            self.log(f"\n[{i}/{total_tests}] Starting test {i} of {total_tests}")
            self.log(f"Tests run this session: {tests_run_this_session}")
            
            success = self.run_single_test(test)
            tests_run_this_session += 1
            
            if not success:
                self.log(f"Test failed, continuing with next test...")
                self.save_checkpoint()
                
            if tests_run_this_session % 10 == 0:
                self.log("Auto-saving checkpoint...")
                self.save_checkpoint()
            time.sleep(2)
            
        self.log(f"\n✓ All tests attempted")
        self.log(f"Tests run in this session: {tests_run_this_session}")
        
    def run(self):
        try:
            self.run_all_tests()
            self.consolidate_results()
            self.report_summary()
            
        except KeyboardInterrupt:
            self.log("\n✋ Interrupted by user")
            self.save_checkpoint()
            self.log("Progress saved. Run script again to resume.")
            self.report_summary()
            
        except Exception as e:
            self.log(f"\n✗ Fatal error: {e}")
            self.log(traceback.format_exc())
            self.save_checkpoint()
            self.report_summary()
            raise
            
        finally:
            self.cleanup()


def main():
    print("="*60)
    print("CHUNKED EVALUATION COMPREHENSIVE TEST RUNNER")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    
    if os.path.exists(CHECKPOINT_FILE):
        print("\n Found existing checkpoint - will RESUME from last position")
        print("   To start fresh, delete the checkpoint file:")
        print(f"   rm {CHECKPOINT_FILE}")
    else:
        print("\n No checkpoint found - starting fresh run")
    
    print("\nPress Ctrl+C anytime to pause (progress will be saved)")
    print("="*60)
    time.sleep(2)
    
    runner = TestRunner()
    runner.run()

if __name__ == "__main__":
    main()