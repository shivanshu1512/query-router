"""
Router Model — Learns to predict routing decisions from oracle labels.

This is a lightweight sklearn classifier that takes query features
and predicts whether to route to the small or large model.

The goal: approximate the oracle's routing decisions WITHOUT
needing to call both models + a judge at inference time.

Approach inspired by FrugalGPT (Chen, Zaharia, Stoica, 2023)
— arxiv.org/abs/2305.05176
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from query_router.schemas import OracleRecord, RoutingLabel


class QueryRouter:
    """
    A trained query router that predicts: should this query go to
    the small (cheap) model or the large (expensive) model?

    Features:
    - TF-IDF on the query text
    - Query length (chars)
    - Query word count
    - Presence of code-related keywords
    - Question complexity heuristics

    Model: GradientBoosting (works well on small datasets, handles
    imbalanced classes, interpretable feature importances).
    """

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.model: GradientBoostingClassifier | None = None
        self.metrics: dict[str, Any] = {}
        self._is_fitted = False

    @staticmethod
    def _extract_features(queries: list[str]) -> pd.DataFrame:
        """Extract hand-crafted features from query text."""
        features = []
        code_keywords = {
            "implement", "write", "code", "function", "class", "algorithm",
            "python", "javascript", "sql", "api", "debug", "fix", "error",
            "compile", "runtime", "test", "unittest",
        }
        complex_keywords = {
            "design", "architecture", "compare", "contrast", "explain",
            "analyze", "evaluate", "tradeoff", "distributed", "system",
            "scalable", "concurrent", "parallel", "optimize",
        }

        for q in queries:
            words = q.lower().split()
            features.append({
                "char_length": len(q),
                "word_count": len(words),
                "has_question_mark": int("?" in q),
                "has_code_keywords": int(bool(set(words) & code_keywords)),
                "has_complex_keywords": int(bool(set(words) & complex_keywords)),
                "num_code_keywords": len(set(words) & code_keywords),
                "num_complex_keywords": len(set(words) & complex_keywords),
                "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
                "has_multi_sentence": int(q.count(".") > 1 or q.count(",") > 1),
                "starts_with_write": int(q.lower().startswith("write")),
                "starts_with_design": int(q.lower().startswith("design")),
                "starts_with_explain": int(q.lower().startswith("explain")),
                "starts_with_what": int(q.lower().startswith("what")),
                "starts_with_name": int(q.lower().startswith("name")),
            })

        return pd.DataFrame(features)

    def train(
        self,
        records: list[OracleRecord],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Train the router on oracle-labeled data.

        Args:
            records: Oracle records with routing labels.
            test_size: Fraction held out for evaluation.
            random_state: For reproducibility.

        Returns:
            Dict with training metrics.
        """
        if len(records) < 20:
            raise ValueError(f"Need at least 20 records to train, got {len(records)}")

        # Prepare data
        queries = [r.query for r in records]
        labels = [1 if r.oracle_label == RoutingLabel.ROUTE_LARGE else 0 for r in records]

        # TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        tfidf_features = self.vectorizer.fit_transform(queries).toarray()

        # Hand-crafted features
        manual_features = self._extract_features(queries).values

        # Combine
        X = np.hstack([tfidf_features, manual_features])
        y = np.array(labels)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )

        # Train
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=random_state,
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)

        # Cross-validation on full data
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="f1")

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "pct_route_large_train": float(y_train.mean()),
            "pct_route_large_test": float(y_test.mean()),
        }

        return self.metrics

    def predict(self, query: str) -> RoutingLabel:
        """Predict routing decision for a single query."""
        if not self._is_fitted:
            raise RuntimeError("Router not trained. Call train() first.")

        proba = self.predict_proba(query)
        return RoutingLabel.ROUTE_LARGE if proba >= 0.5 else RoutingLabel.ROUTE_SMALL

    def predict_proba(self, query: str) -> float:
        """Return probability of routing to large model."""
        if not self._is_fitted:
            raise RuntimeError("Router not trained. Call train() first.")

        tfidf = self.vectorizer.transform([query]).toarray()
        manual = self._extract_features([query]).values
        X = np.hstack([tfidf, manual])

        return float(self.model.predict_proba(X)[0, 1])

    def predict_batch(self, queries: list[str]) -> list[RoutingLabel]:
        """Predict routing for a batch of queries."""
        return [self.predict(q) for q in queries]

    def save(self, path: str = "models/router.pkl") -> None:
        """Save the trained router to disk."""
        if not self._is_fitted:
            raise RuntimeError("Router not trained. Call train() first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "metrics": self.metrics,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        # Also save metrics as JSON for easy inspection
        metrics_path = save_path.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"[saved] Router model → {save_path}")
        print(f"[saved] Metrics → {metrics_path}")

    def load(self, path: str = "models/router.pkl") -> None:
        """Load a trained router from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.vectorizer = state["vectorizer"]
        self.model = state["model"]
        self.metrics = state["metrics"]
        self._is_fitted = True

        print(f"[loaded] Router from {path}")
        print(f"[loaded] Train accuracy: {self.metrics.get('accuracy', 'N/A')}")


def load_oracle_records(path: str) -> list[OracleRecord]:
    """Load oracle records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(OracleRecord.model_validate_json(line))
    print(f"[loaded] {len(records)} oracle records from {path}")
    return records


def train_router_from_file(
    oracle_results_path: str,
    save_path: str = "models/router.pkl",
) -> QueryRouter:
    """
    Convenience: load oracle results, train router, save model.

    Usage:
        python -m query_router.router --oracle-results data/results/oracle_results_*.jsonl
    """
    records = load_oracle_records(oracle_results_path)
    router = QueryRouter()
    metrics = router.train(records)

    print(f"\n{'='*60}")
    print("  ROUTER TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:   {metrics['accuracy']:.3f}")
    print(f"  F1 Score:   {metrics['f1_score']:.3f}")
    print(f"  CV F1:      {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
    print(f"  Train size: {metrics['train_size']}")
    print(f"  Test size:  {metrics['test_size']}")
    print(f"{'='*60}\n")

    router.save(save_path)
    return router


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the query router")
    parser.add_argument("--oracle-results", required=True, help="Path to oracle results JSONL")
    parser.add_argument("--save-path", default="models/router.pkl", help="Where to save the model")
    args = parser.parse_args()

    train_router_from_file(args.oracle_results, args.save_path)
