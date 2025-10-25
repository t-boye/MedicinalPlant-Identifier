"""
Model evaluation utilities with bias detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from typing import List, Dict
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation with bias detection
    """

    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict:
        """
        Comprehensive evaluation of model predictions

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Detect bias: check if any class has significantly lower performance
        mean_accuracy = np.mean(class_accuracy)
        std_accuracy = np.std(class_accuracy)

        biased_classes = []
        for i, (cls_name, acc) in enumerate(zip(self.class_names, class_accuracy)):
            if acc < (mean_accuracy - std_accuracy):
                biased_classes.append({
                    'class': cls_name,
                    'accuracy': float(acc),
                    'difference_from_mean': float(mean_accuracy - acc)
                })

        evaluation = {
            'overall_accuracy': float(report['accuracy']),
            'macro_avg_precision': float(report['macro avg']['precision']),
            'macro_avg_recall': float(report['macro avg']['recall']),
            'macro_avg_f1': float(report['macro avg']['f1-score']),
            'weighted_avg_precision': float(report['weighted avg']['precision']),
            'weighted_avg_recall': float(report['weighted avg']['recall']),
            'weighted_avg_f1': float(report['weighted avg']['f1-score']),
            'per_class_metrics': {},
            'confusion_matrix': cm.tolist(),
            'bias_analysis': {
                'mean_class_accuracy': float(mean_accuracy),
                'std_class_accuracy': float(std_accuracy),
                'potentially_biased_classes': biased_classes,
                'bias_detected': len(biased_classes) > 0
            }
        }

        # Per-class metrics
        for cls_name in self.class_names:
            if cls_name in report:
                evaluation['per_class_metrics'][cls_name] = {
                    'precision': float(report[cls_name]['precision']),
                    'recall': float(report[cls_name]['recall']),
                    'f1-score': float(report[cls_name]['f1-score']),
                    'support': int(report[cls_name]['support'])
                }

        return evaluation

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str = None,
        figsize: tuple = (12, 10)
    ):
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            save_path: Path to save figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.close()

    def plot_class_performance(
        self,
        evaluation: Dict,
        save_path: str = None,
        figsize: tuple = (14, 6)
    ):
        """
        Plot per-class performance metrics

        Args:
            evaluation: Evaluation dictionary
            save_path: Path to save figure
            figsize: Figure size
        """
        per_class = evaluation['per_class_metrics']

        classes = list(per_class.keys())
        precision = [per_class[cls]['precision'] for cls in classes]
        recall = [per_class[cls]['recall'] for cls in classes]
        f1 = [per_class[cls]['f1-score'] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class performance plot saved to: {save_path}")

        plt.close()

    def generate_evaluation_report(
        self,
        evaluation: Dict,
        save_path: str = None
    ):
        """
        Generate and save evaluation report

        Args:
            evaluation: Evaluation dictionary
            save_path: Path to save report
        """
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append(f"  Accuracy: {evaluation['overall_accuracy']:.4f}")
        report.append(f"  Macro Avg Precision: {evaluation['macro_avg_precision']:.4f}")
        report.append(f"  Macro Avg Recall: {evaluation['macro_avg_recall']:.4f}")
        report.append(f"  Macro Avg F1-Score: {evaluation['macro_avg_f1']:.4f}")
        report.append("")

        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        for cls_name, metrics in evaluation['per_class_metrics'].items():
            report.append(f"\n  {cls_name}:")
            report.append(f"    Precision: {metrics['precision']:.4f}")
            report.append(f"    Recall: {metrics['recall']:.4f}")
            report.append(f"    F1-Score: {metrics['f1-score']:.4f}")
            report.append(f"    Support: {metrics['support']}")

        # Bias analysis
        report.append("\n" + "=" * 80)
        report.append("BIAS ANALYSIS:")
        report.append("=" * 80)

        bias = evaluation['bias_analysis']
        report.append(f"  Mean Class Accuracy: {bias['mean_class_accuracy']:.4f}")
        report.append(f"  Std Class Accuracy: {bias['std_class_accuracy']:.4f}")

        if bias['bias_detected']:
            report.append("\n  ⚠️ POTENTIAL BIAS DETECTED!")
            report.append("  Classes with significantly lower performance:")
            for biased_cls in bias['potentially_biased_classes']:
                report.append(f"    - {biased_cls['class']}: "
                            f"Accuracy={biased_cls['accuracy']:.4f} "
                            f"(Δ={biased_cls['difference_from_mean']:.4f})")
            report.append("\n  Recommendations:")
            report.append("    1. Collect more data for underperforming classes")
            report.append("    2. Apply stronger augmentation to minority classes")
            report.append("    3. Adjust class weights during training")
            report.append("    4. Review data quality for these classes")
        else:
            report.append("\n  ✓ No significant bias detected")
            report.append("  Model performance is balanced across classes")

        report.append("\n" + "=" * 80)

        report_text = "\n".join(report)
        print(report_text)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {save_path}")

        return report_text

    def save_evaluation_json(self, evaluation: Dict, filepath: str):
        """Save evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Evaluation results saved to: {filepath}")
