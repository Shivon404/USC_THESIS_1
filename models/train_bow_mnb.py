#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis using Bag of Words and Multinomial Naive Bayes

This script trains and evaluates a Multinomial Naive Bayes classifier for sentiment analysis
of Reddit posts about the University of San Carlos. The model classifies posts into three
sentiment categories: Positive, Neutral, or Negative.

Features:
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Detailed performance metrics and visualizations
- Model persistence and configuration logging
- Error handling and input validation

Author: [Your Name]
Date: [Current Date]
Version: 2.0
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentAnalyzer:
    """
    Enhanced Sentiment Analysis class for University of San Carlos Reddit data.
    
    This class handles the complete pipeline from data loading to model evaluation
    and persistence, with support for hyperparameter tuning and comprehensive reporting.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.label_names = ['Negative', 'Neutral', 'Positive']
        self.results = {}
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self._create_directories()
    
    def _setup_logging(self) -> None:
        """Configure logging for the analysis process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/sentiment_analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_directories(self) -> None:
        """Create necessary directories for outputs."""
        directories = ['results', 'models', 'plots']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def load_data(self, X_path: str = 'data/processed/X_bow.pkl', 
                  y_path: str = 'data/processed/y_encoded.pkl') -> Tuple[Any, Any]:
        """
        Load preprocessed features and labels.
        
        Args:
            X_path (str): Path to the feature matrix file
            y_path (str): Path to the labels file
            
        Returns:
            Tuple[Any, Any]: Feature matrix and labels
            
        Raises:
            FileNotFoundError: If input files don't exist
            Exception: For other loading errors
        """
        try:
            self.logger.info("📥 Loading BoW features and sentiment labels...")
            
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                raise FileNotFoundError(f"Input files not found: {X_path} or {y_path}")
            
            X = joblib.load(X_path)
            y = joblib.load(y_path)
            
            self.logger.info(f"✅ Data loaded successfully - Features: {X.shape}, Labels: {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def split_data(self, X: Any, y: Any, test_size: float = 0.2, 
                   stratify: bool = True) -> Tuple[Any, Any, Any, Any]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size (float): Proportion of test data
            stratify (bool): Whether to stratify the split
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        self.logger.info(f"🔀 Splitting data into train/test sets (test_size={test_size})...")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        self.logger.info(f"✅ Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def tune_hyperparameters(self, X_train: Any, y_train: Any) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dict: Best parameters found
        """
        self.logger.info("🔧 Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'fit_prior': [True, False],
            'class_prior': [None]  # Let the model learn class priors
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            MultinomialNB(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
        
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        self.logger.info(f"✅ Best parameters: {best_params}")
        self.logger.info(f"✅ Best CV accuracy: {best_score:.4f}")
        
        return best_params
    
    def train_model(self, X_train: Any, y_train: Any, 
                   tune_params: bool = True) -> MultinomialNB:
        """
        Train the Multinomial Naive Bayes model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_params (bool): Whether to perform hyperparameter tuning
            
        Returns:
            MultinomialNB: Trained model
        """
        self.logger.info("🤖 Training Multinomial Naive Bayes model...")
        
        if tune_params:
            best_params = self.tune_hyperparameters(X_train, y_train)
            self.model = MultinomialNB(**best_params)
        else:
            self.model = MultinomialNB(random_state=self.random_state)
        
        self.model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        self.results['cv_mean'] = cv_scores.mean()
        self.results['cv_std'] = cv_scores.std()
        
        self.logger.info(f"✅ Model trained successfully")
        self.logger.info(f"✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return self.model
    
    def evaluate_model(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("🔎 Evaluating model on test set...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # For multi-class ROC AUC, use ovr (one-vs-rest) strategy
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except ValueError:
            auc_score = None
            self.logger.warning("Could not calculate AUC score")
        
        # Store results
        self.results.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
        
        self.logger.info(f"✅ Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"✅ Test F1-Score: {f1:.4f}")
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive classification report.
        
        Returns:
            str: Formatted report
        """
        if not self.results:
            raise ValueError("Model must be evaluated before generating report")
        
        y_test = self.results['y_test']
        y_pred = self.results['y_pred']
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.label_names,
            digits=4
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Build comprehensive report
        report = f"""
=== ENHANCED SENTIMENT ANALYSIS REPORT ===
University of San Carlos Reddit Data Analysis
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL CONFIGURATION:
- Algorithm: Multinomial Naive Bayes
- Feature Extraction: Bag of Words (BoW)
- Target Classes: {', '.join(self.label_names)}

PERFORMANCE METRICS:
- Test Accuracy: {self.results['accuracy']:.4f}
- Weighted Precision: {self.results['precision']:.4f}
- Weighted Recall: {self.results['recall']:.4f}
- Weighted F1-Score: {self.results['f1_score']:.4f}
"""
        
        if self.results.get('auc_score'):
            report += f"- AUC Score (OvR): {self.results['auc_score']:.4f}\n"
        
        if 'cv_mean' in self.results:
            report += f"- Cross-Validation Accuracy: {self.results['cv_mean']:.4f} (±{self.results['cv_std']:.4f})\n"
        
        report += f"""
DETAILED CLASSIFICATION REPORT:
{class_report}

CONFUSION MATRIX:
{conf_matrix}

CLASS DISTRIBUTION IN TEST SET:
"""
        
        # Add class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            percentage = (count / len(y_test)) * 100
            report += f"- {self.label_names[class_idx]}: {count} samples ({percentage:.1f}%)\n"
        
        return report
    
    def create_visualizations(self) -> None:
        """Create and save visualization plots."""
        if not self.results:
            raise ValueError("Model must be evaluated before creating visualizations")
        
        self.logger.info("📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sentiment Analysis Results - University of San Carlos', fontsize=16, fontweight='bold')
        
        y_test = self.results['y_test']
        y_pred = self.results['y_pred']
        
        # 1. Confusion Matrix Heatmap
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, yticklabels=self.label_names, 
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Class Distribution
        unique, counts = np.unique(y_test, return_counts=True)
        class_names = [self.label_names[i] for i in unique]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        axes[0, 1].bar(class_names, counts, color=colors[:len(class_names)])
        axes[0, 1].set_title('Test Set Class Distribution')
        axes[0, 1].set_ylabel('Number of Samples')
        for i, count in enumerate(counts):
            axes[0, 1].text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # 3. Performance Metrics Bar Chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            self.results['accuracy'],
            self.results['precision'],
            self.results['recall'],
            self.results['f1_score']
        ]
        bars = axes[1, 0].bar(metrics, values, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        for bar, value in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Prediction Confidence Distribution
        y_pred_proba = self.results['y_pred_proba']
        max_probas = np.max(y_pred_proba, axis=1)
        axes[1, 1].hist(max_probas, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Prediction Confidence Distribution')
        axes[1, 1].set_xlabel('Maximum Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(max_probas), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(max_probas):.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('plots/sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("📊 Visualizations saved to plots/sentiment_analysis_results.png")
    
    def save_model_and_results(self, model_path: str = "models/bow_mnb_model.pkl",
                              report_path: str = "results/bow_mnb_report.txt") -> None:
        """
        Save the trained model and evaluation results.
        
        Args:
            model_path (str): Path to save the model
            report_path (str): Path to save the report
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.logger.info("💾 Saving model and results...")
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save report
        report = self.generate_report()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save model metadata
        metadata = {
            'model_type': 'MultinomialNB',
            'feature_extraction': 'Bag of Words',
            'label_names': self.label_names,
            'random_state': self.random_state,
            'training_date': datetime.now().isoformat(),
            'results': {k: v for k, v in self.results.items() 
                       if k not in ['y_test', 'y_pred', 'y_pred_proba']}
        }
        
        joblib.dump(metadata, model_path.replace('.pkl', '_metadata.pkl'))
        
        self.logger.info(f"✅ Model saved to: {model_path}")
        self.logger.info(f"✅ Report saved to: {report_path}")
        self.logger.info(f"✅ Metadata saved to: {model_path.replace('.pkl', '_metadata.pkl')}")


def main():
    """
    Main execution function for the sentiment analysis pipeline.
    """
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer(random_state=42)
        
        # Load data
        X, y = analyzer.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = analyzer.split_data(X, y)
        
        # Train model with hyperparameter tuning
        model = analyzer.train_model(X_train, y_train, tune_params=True)
        
        # Evaluate model
        results = analyzer.evaluate_model(X_test, y_test)
        
        # Generate visualizations
        analyzer.create_visualizations()
        
        # Save model and results
        analyzer.save_model_and_results()
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 SENTIMENT ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Test Accuracy: {results['accuracy']:.4f}")
        print(f"📊 F1-Score: {results['f1_score']:.4f}")
        print(f"📁 Results saved to: results/bow_mnb_report.txt")
        print(f"💾 Model saved to: models/bow_mnb_model.pkl")
        print(f"📈 Visualizations saved to: plots/sentiment_analysis_results.png")
        print("="*60)
        
    except Exception as e:
        logging.error(f"❌ Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()