"""
Analyze and classify performance metrics from Indic GLUE benchmark results.
Compares base vs mHC models across tasks, subsets, and languages.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceClassifier:
    """Classify performance levels based on accuracy thresholds."""
    
    # Performance thresholds (adjust as needed)
    THRESHOLDS = {
        'excellent': 0.80,      # >= 80%
        'good': 0.60,           # 60-79%
        'fair': 0.40,           # 40-59%
        'poor': 0.20,           # 20-39%
        'very_poor': 0.0,       # < 20%
    }
    
    @classmethod
    def classify(cls, accuracy: float) -> str:
        """Classify accuracy into performance categories."""
        for category, threshold in sorted(cls.THRESHOLDS.items(), key=lambda x: -x[1]):
            if accuracy >= threshold:
                return category
        return 'very_poor'
    
    @classmethod
    def get_color(cls, category: str) -> str:
        """Get color for performance category."""
        colors = {
            'excellent': '#2ecc71',    # green
            'good': '#3498db',         # blue
            'fair': '#f39c12',         # orange
            'poor': '#e74c3c',         # red
            'very_poor': '#c0392b',    # dark red
        }
        return colors.get(category, '#95a5a6')


class BenchmarkAnalyzer:
    """Analyze benchmark results and generate insights."""
    
    def __init__(self, benchmark_path: Path):
        self.benchmark_path = benchmark_path
        self.data = self._load_benchmark()
        self.base_results = self.data.get('base', {})
        self.mhc_results = self.data.get('mhc', {})
        
    def _load_benchmark(self) -> Dict[str, Any]:
        """Load benchmark JSON file."""
        with open(self.benchmark_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract key metrics from results."""
        return {
            'overall_accuracy': results.get('overall_accuracy', 0),
            'subset_accuracy': results.get('subset_accuracy', {}),
            'task_accuracy': results.get('task_accuracy', {}),
            'language_accuracy': results.get('language_accuracy', {}),
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary."""
        base_metrics = self._extract_metrics(self.base_results)
        mhc_metrics = self._extract_metrics(self.mhc_results)
        
        base_acc = base_metrics['overall_accuracy']
        mhc_acc = mhc_metrics['overall_accuracy']
        diff = mhc_acc - base_acc
        pct_change = (diff / base_acc * 100) if base_acc > 0 else 0
        
        return {
            'base': {
                'accuracy': base_acc,
                'classification': PerformanceClassifier.classify(base_acc),
            },
            'mhc': {
                'accuracy': mhc_acc,
                'classification': PerformanceClassifier.classify(mhc_acc),
            },
            'difference': diff,
            'percent_change': pct_change,
            'winner': 'mhc' if mhc_acc > base_acc else 'base' if base_acc > mhc_acc else 'tie',
        }
    
    def get_task_performance(self) -> pd.DataFrame:
        """Generate task-level performance comparison."""
        base_metrics = self._extract_metrics(self.base_results)
        mhc_metrics = self._extract_metrics(self.mhc_results)
        
        rows = []
        all_tasks = set(base_metrics['task_accuracy'].keys()) | set(mhc_metrics['task_accuracy'].keys())
        
        for task in sorted(all_tasks):
            base_acc = base_metrics['task_accuracy'].get(task, 0)
            mhc_acc = mhc_metrics['task_accuracy'].get(task, 0)
            diff = mhc_acc - base_acc
            
            rows.append({
                'task': task,
                'base_accuracy': base_acc,
                'base_classification': PerformanceClassifier.classify(base_acc),
                'mhc_accuracy': mhc_acc,
                'mhc_classification': PerformanceClassifier.classify(mhc_acc),
                'difference': diff,
                'winner': 'mhc' if mhc_acc > base_acc else 'base' if base_acc > mhc_acc else 'tie',
            })
        
        return pd.DataFrame(rows)
    
    def get_language_performance(self) -> pd.DataFrame:
        """Generate language-level performance comparison."""
        base_metrics = self._extract_metrics(self.base_results)
        mhc_metrics = self._extract_metrics(self.mhc_results)
        
        rows = []
        all_languages = set(base_metrics['language_accuracy'].keys()) | set(mhc_metrics['language_accuracy'].keys())
        
        for language in sorted(all_languages):
            base_acc = base_metrics['language_accuracy'].get(language, 0)
            mhc_acc = mhc_metrics['language_accuracy'].get(language, 0)
            diff = mhc_acc - base_acc
            
            rows.append({
                'language': language,
                'base_accuracy': base_acc,
                'base_classification': PerformanceClassifier.classify(base_acc),
                'mhc_accuracy': mhc_acc,
                'mhc_classification': PerformanceClassifier.classify(mhc_acc),
                'difference': diff,
                'winner': 'mhc' if mhc_acc > base_acc else 'base' if base_acc > mhc_acc else 'tie',
            })
        
        return pd.DataFrame(rows)
    
    def get_subset_performance(self) -> pd.DataFrame:
        """Generate subset-level performance comparison."""
        base_metrics = self._extract_metrics(self.base_results)
        mhc_metrics = self._extract_metrics(self.mhc_results)
        
        rows = []
        all_subsets = set(base_metrics['subset_accuracy'].keys()) | set(mhc_metrics['subset_accuracy'].keys())
        
        for subset in sorted(all_subsets):
            base_acc = base_metrics['subset_accuracy'].get(subset, 0)
            mhc_acc = mhc_metrics['subset_accuracy'].get(subset, 0)
            diff = mhc_acc - base_acc
            
            # Parse task and language from subset (e.g., "copa.en" -> task="copa", language="en")
            parts = subset.split('.')
            task = parts[0] if len(parts) > 0 else 'unknown'
            lang_code = parts[1] if len(parts) > 1 else 'unknown'
            
            lang_map = {'en': 'English', 'gu': 'Gujarati', 'hi': 'Hindi', 'mr': 'Marathi'}
            language = lang_map.get(lang_code, lang_code)
            
            rows.append({
                'subset': subset,
                'task': task,
                'language': language,
                'base_accuracy': base_acc,
                'base_classification': PerformanceClassifier.classify(base_acc),
                'mhc_accuracy': mhc_acc,
                'mhc_classification': PerformanceClassifier.classify(mhc_acc),
                'difference': diff,
                'winner': 'mhc' if mhc_acc > base_acc else 'base' if base_acc > mhc_acc else 'tie',
            })
        
        return pd.DataFrame(rows)
    
    def get_latency_comparison(self) -> Dict[str, Any]:
        """Compare inference latencies."""
        base_samples = self.base_results.get('samples', [])
        mhc_samples = self.mhc_results.get('samples', [])
        
        base_latencies = [s.get('latency_ms', 0) for s in base_samples]
        mhc_latencies = [s.get('latency_ms', 0) for s in mhc_samples]
        
        if not base_latencies or not mhc_latencies:
            return {'note': 'Latency data not available'}
        
        return {
            'base': {
                'mean_ms': statistics.mean(base_latencies),
                'median_ms': statistics.median(base_latencies),
                'stdev_ms': statistics.stdev(base_latencies) if len(base_latencies) > 1 else 0,
                'min_ms': min(base_latencies),
                'max_ms': max(base_latencies),
            },
            'mhc': {
                'mean_ms': statistics.mean(mhc_latencies),
                'median_ms': statistics.median(mhc_latencies),
                'stdev_ms': statistics.stdev(mhc_latencies) if len(mhc_latencies) > 1 else 0,
                'min_ms': min(mhc_latencies),
                'max_ms': max(mhc_latencies),
            },
        }
    
    def print_summary(self) -> None:
        """Print comprehensive summary to console."""
        print("\n" + "="*80)
        print("INDIC GLUE BENCHMARK ANALYSIS: BASE vs mHC".center(80))
        print("="*80 + "\n")
        
        # Overall summary
        print("OVERALL PERFORMANCE")
        print("-" * 80)
        summary = self.get_performance_summary()
        print(f"Base Model:   {summary['base']['accuracy']:.1%} ({summary['base']['classification']})")
        print(f"mHC Model:    {summary['mhc']['accuracy']:.1%} ({summary['mhc']['classification']})")
        print(f"Difference:   {summary['difference']:+.1%} ({summary['percent_change']:+.1f}%)")
        print(f"Winner:       {summary['winner'].upper()}")
        print()
        
        # Task-level performance
        print("TASK-LEVEL PERFORMANCE")
        print("-" * 80)
        task_df = self.get_task_performance()
        for _, row in task_df.iterrows():
            print(f"\n{row['task'].upper()}")
            print(f"  Base: {row['base_accuracy']:.1%} ({row['base_classification']})")
            print(f"  mHC:  {row['mhc_accuracy']:.1%} ({row['mhc_classification']})")
            print(f"  Diff: {row['difference']:+.1%} ({row['winner']})")
        print()
        
        # Language-level performance
        print("LANGUAGE-LEVEL PERFORMANCE")
        print("-" * 80)
        lang_df = self.get_language_performance()
        for _, row in lang_df.iterrows():
            print(f"\n{row['language']}")
            print(f"  Base: {row['base_accuracy']:.1%} ({row['base_classification']})")
            print(f"  mHC:  {row['mhc_accuracy']:.1%} ({row['mhc_classification']})")
            print(f"  Diff: {row['difference']:+.1%} ({row['winner']})")
        print()
        
        # Latency comparison
        print("LATENCY COMPARISON")
        print("-" * 80)
        latency = self.get_latency_comparison()
        if 'note' not in latency:
            print(f"Base Model (mean): {latency['base']['mean_ms']:.2f} ms")
            print(f"mHC Model (mean):  {latency['mhc']['mean_ms']:.2f} ms")
            print(f"Difference:        {latency['mhc']['mean_ms'] - latency['base']['mean_ms']:+.2f} ms")
        else:
            print(latency['note'])
        print()
        
        # Subset breakdown (top/bottom performers)
        print("SUBSET PERFORMANCE EXTREMES")
        print("-" * 80)
        subset_df = self.get_subset_performance()
        
        print("\nTOP PERFORMERS (Base)")
        top_base = subset_df.nlargest(3, 'base_accuracy')[['subset', 'base_accuracy', 'base_classification']]
        for _, row in top_base.iterrows():
            print(f"  {row['subset']}: {row['base_accuracy']:.1%} ({row['base_classification']})")
        
        print("\nTOP PERFORMERS (mHC)")
        top_mhc = subset_df.nlargest(3, 'mhc_accuracy')[['subset', 'mhc_accuracy', 'mhc_classification']]
        for _, row in top_mhc.iterrows():
            print(f"  {row['subset']}: {row['mhc_accuracy']:.1%} ({row['mhc_classification']})")
        
        print("\nLARGEST IMPROVEMENTS (mHC vs Base)")
        improvements = subset_df.nlargest(3, 'difference')[['subset', 'base_accuracy', 'mhc_accuracy', 'difference']]
        for _, row in improvements.iterrows():
            print(f"  {row['subset']}: {row['base_accuracy']:.1%} -> {row['mhc_accuracy']:.1%} ({row['difference']:+.1%})")
        
        print("\nLARGEST REGRESSIONS (mHC vs Base)")
        regressions = subset_df.nsmallest(3, 'difference')[['subset', 'base_accuracy', 'mhc_accuracy', 'difference']]
        for _, row in regressions.iterrows():
            print(f"  {row['subset']}: {row['base_accuracy']:.1%} -> {row['mhc_accuracy']:.1%} ({row['difference']:+.1%})")
        
        print("\n" + "="*80 + "\n")
    
    def save_detailed_csv(self, output_dir: Path) -> None:
        """Save detailed results to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task performance
        task_df = self.get_task_performance()
        task_df.to_csv(output_dir / 'task_performance.csv', index=False)
        
        # Language performance
        lang_df = self.get_language_performance()
        lang_df.to_csv(output_dir / 'language_performance.csv', index=False)
        
        # Subset performance
        subset_df = self.get_subset_performance()
        subset_df.to_csv(output_dir / 'subset_performance.csv', index=False)
        
        print(f"✓ Saved CSV files to {output_dir}")
    
    def plot_performance_comparison(self, output_dir: Path) -> None:
        """Generate and save visualization plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        
        # 1. Overall accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Task comparison
        task_df = self.get_task_performance()
        ax = axes[0, 0]
        x = np.arange(len(task_df))
        width = 0.35
        bars1 = ax.bar(x - width/2, task_df['base_accuracy'], width, label='Base', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, task_df['mhc_accuracy'], width, label='mHC', alpha=0.8, color='#2ecc71')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Task-Level Accuracy', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_df['task'])
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Language comparison
        lang_df = self.get_language_performance()
        ax = axes[0, 1]
        x = np.arange(len(lang_df))
        bars1 = ax.bar(x - width/2, lang_df['base_accuracy'], width, label='Base', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, lang_df['mhc_accuracy'], width, label='mHC', alpha=0.8, color='#2ecc71')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Language-Level Accuracy', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lang_df['language'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Difference heatmap
        subset_df = self.get_subset_performance()
        ax = axes[1, 0]
        diff_matrix = subset_df.pivot_table(
            values='difference',
            index='language',
            columns='task',
            aggfunc='mean'
        )
        sns.heatmap(diff_matrix, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Difference'})
        ax.set_title('Performance Difference (mHC - Base)', fontsize=12, fontweight='bold')
        
        # Overall summary
        ax = axes[1, 1]
        ax.axis('off')
        summary = self.get_performance_summary()
        
        summary_text = f"""
OVERALL PERFORMANCE SUMMARY

Base Model:
  • Accuracy: {summary['base']['accuracy']:.1%}
  • Classification: {summary['base']['classification'].upper()}

mHC Model:
  • Accuracy: {summary['mhc']['accuracy']:.1%}
  • Classification: {summary['mhc']['classification'].upper()}

Comparison:
  • Difference: {summary['difference']:+.1%}
  • Percent Change: {summary['percent_change']:+.1f}%
  • Winner: {summary['winner'].upper()}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: performance_comparison.png")
        plt.close()
        
        # 2. Subset-level accuracy heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        subset_accuracy = subset_df.pivot_table(
            values='mhc_accuracy',
            index='language',
            columns='task',
            aggfunc='mean'
        )
        sns.heatmap(subset_accuracy, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': 'Accuracy'})
        ax.set_title('mHC Model Accuracy by Language and Task', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'subset_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: subset_accuracy_heatmap.png")
        plt.close()
        
        # 3. Performance distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Base distribution
        ax = axes[0]
        base_accs = subset_df['base_accuracy'].values
        colors_base = [PerformanceClassifier.get_color(PerformanceClassifier.classify(acc)) for acc in base_accs]
        ax.scatter(range(len(base_accs)), base_accs, c=colors_base, s=100, alpha=0.7, edgecolors='black')
        ax.axhline(y=subset_df['base_accuracy'].mean(), color='blue', linestyle='--', label=f'Mean: {subset_df["base_accuracy"].mean():.1%}')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_xlabel('Subset Index', fontsize=11, fontweight='bold')
        ax.set_title('Base Model: Subset Accuracy Distribution', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # mHC distribution
        ax = axes[1]
        mhc_accs = subset_df['mhc_accuracy'].values
        colors_mhc = [PerformanceClassifier.get_color(PerformanceClassifier.classify(acc)) for acc in mhc_accs]
        ax.scatter(range(len(mhc_accs)), mhc_accs, c=colors_mhc, s=100, alpha=0.7, edgecolors='black')
        ax.axhline(y=subset_df['mhc_accuracy'].mean(), color='green', linestyle='--', label=f'Mean: {subset_df["mhc_accuracy"].mean():.1%}')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_xlabel('Subset Index', fontsize=11, fontweight='bold')
        ax.set_title('mHC Model: Subset Accuracy Distribution', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: accuracy_distribution.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and classify Indic GLUE benchmark performance'
    )
    parser.add_argument(
        '--benchmark',
        type=Path,
        default=Path('output/benchmark_indic_glue/indic_glue_benchmark_20260502_192931.json'),
        help='Path to benchmark JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/benchmark_indic_glue/analysis'),
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Skip saving CSV files'
    )
    
    args = parser.parse_args()
    
    # Check if benchmark file exists
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        return 1
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(args.benchmark)
    
    # Print summary
    analyzer.print_summary()
    
    # Save CSV files
    if not args.no_csv:
        analyzer.save_detailed_csv(args.output_dir)
    
    # Generate plots
    if not args.no_plots:
        analyzer.plot_performance_comparison(args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
