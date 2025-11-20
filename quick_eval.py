#!/usr/bin/env python3
"""
Quick evaluation script that just generates plots and summaries without interactive mode.
"""

from evaluation import ModelEvaluator

def quick_evaluation():
    """Run evaluation and generate plots only (no interactive mode)"""
    evaluator = ModelEvaluator()
    
    print("Running quick evaluation (plots only)...")
    
    # Evaluate all checkpoints
    results = evaluator.evaluate_all_checkpoints()
    
    if not results['lstm'] and not results['gpt']:
        print("No model checkpoints found for evaluation!")
        return
    
    # Generate all comparison plots
    print("\\nGenerating evaluation plots...")
    
    if results['lstm'] and results['gpt']:
        evaluator.plot_metric_comparison(results, 'accuracy', 'Accuracy')
        evaluator.plot_metric_comparison(results, 'f1_score', 'F1 Score')
        evaluator.plot_metric_comparison(results, 'precision', 'Precision')
        evaluator.plot_metric_comparison(results, 'recall', 'Recall')
        evaluator.plot_metric_comparison(results, 'loss', 'Loss')
        
        evaluator.plot_confusion_matrices(results)
        evaluator.plot_comprehensive_comparison(results)
    
    # Save results
    evaluator.save_evaluation_results(results)
    
    # Print summary
    print(f"\\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for model_type in ['lstm', 'gpt']:
        if results[model_type]:
            best_model = max(results[model_type], key=lambda x: x['f1_score'])
            print(f"{model_type.upper()} Best Model:")
            print(f"  Accuracy: {best_model['accuracy']:.3f}")
            print(f"  F1 Score: {best_model['f1_score']:.3f}")
            print(f"  Precision: {best_model['precision']:.3f}")
            print(f"  Recall: {best_model['recall']:.3f}")
            print()
    
    print(f"All evaluation plots saved to: {evaluator.evaluation_dir}")

if __name__ == "__main__":
    quick_evaluation()