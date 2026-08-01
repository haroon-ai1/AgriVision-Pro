"""Train the classical baseline on the SAME splits the CNN uses.

Sharing ``artifacts/splits.json`` is the point: comparing a Random Forest
trained on one random split against a CNN trained on another tells you nothing.

Also fixed from v1: parallel feature extraction, a feature cache so you are not
re-extracting 50k images on every hyper-parameter tweak, stratified evaluation,
balanced class weights, and cross-validated scores instead of a single number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import extract_features, feature_names  # noqa: E402
from agrivision.data import load_or_create_splits  # noqa: E402


def extract_split(items, n_jobs: int, cache: Path | None):
    """Extract features for one split, caching the result to .npz."""
    if cache and cache.exists():
        blob = np.load(cache, allow_pickle=True)
        print(f"[baseline] Loaded cached features from {cache}")
        return blob["X"], blob["y"]

    t0 = time.time()
    print(f"[baseline] Extracting {len(items)} images on {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(extract_features)(path) for path, _ in items
    )

    X, y, failed = [], [], 0
    for (path, label), feat in zip(items, results):
        if feat is None:
            failed += 1
            continue
        X.append(feat)
        y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    print(f"[baseline] Done in {time.time() - t0:.1f}s. {len(X)} ok, {failed} failed.")

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
        print(f"[baseline] Cached to {cache}")
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical Random Forest baseline.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="artifacts/splits.json")
    parser.add_argument("--out-dir", default="artifacts/baseline")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits, class_names = load_or_create_splits(args.data_root, splits_path=args.splits)
    cache_dir = None if args.no_cache else out_dir / "cache"

    X_train, y_train = extract_split(
        splits["train"], args.n_jobs, None if args.no_cache else cache_dir / "train.npz"
    )
    X_test, y_test = extract_split(
        splits["test"], args.n_jobs, None if args.no_cache else cache_dir / "test.npz"
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",     # v1 ignored class imbalance entirely
        n_jobs=args.n_jobs,
        random_state=args.seed,
        min_samples_leaf=1,
        max_features="sqrt",
    )

    print("[baseline] 5-fold cross-validation on the training split...")
    cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=1)
    print(f"[baseline] CV macro-F1: {cv.mean():.4f} +/- {cv.std():.4f}")

    print("[baseline] Fitting final model...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"\n[baseline] Test accuracy: {accuracy * 100:.2f}%")
    print(f"[baseline] Test macro-F1: {macro_f1:.4f}\n")
    print(classification_report(y_test, preds, zero_division=0, digits=4))

    names = feature_names()
    order = np.argsort(clf.feature_importances_)[::-1][:15]
    print("Top 15 features by importance:")
    for rank, idx in enumerate(order, 1):
        label = names[idx] if idx < len(names) else f"feature_{idx}"
        print(f"  {rank:>2}. {label:<32} {clf.feature_importances_[idx]:.4f}")

    joblib.dump(
        {"model": clf, "class_names": list(class_names), "feature_names": names, "version": 2},
        out_dir / "baseline_rf.joblib",
    )
    (out_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "cv_macro_f1_mean": float(cv.mean()),
                "cv_macro_f1_std": float(cv.std()),
                "n_features": int(X_train.shape[1]),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            },
            indent=2,
        )
    )
    print(f"\n[baseline] Saved to {out_dir}")


if __name__ == "__main__":
    main()
