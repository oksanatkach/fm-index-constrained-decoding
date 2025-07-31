import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from pathlib import Path


# ─────── Debugging and Validation Functions ───────
def validate_transformer_output(transformer, sample_df, dep_vectorizer, onehot):
    """Validate that the transformer produces meaningful features"""
    print("\n" + "=" * 50)
    print("TRANSFORMER VALIDATION")
    print("=" * 50)

    # Transform a sample
    X = transformer.transform(sample_df)

    print(f"Sample data shape: {sample_df.shape}")
    print(f"Transformed features shape: {X.shape}")
    print(f"Feature matrix type: {type(X)}")

    # Check feature breakdown
    dep_features = dep_vectorizer.transform(sample_df['dep_path'].tolist())
    wh_features = onehot.transform(sample_df[['wh_type']])

    print(f"\nFeature breakdown:")
    print(f"  Dependency features: {dep_features.shape} (non-zero: {dep_features.nnz})")
    print(f"  WH-type features: {wh_features.shape} (non-zero: {wh_features.nnz})")
    print(f"  Length features: 1 column")
    print(f"  Num_words features: 1 column")
    print(f"  Total expected: {dep_features.shape[1] + wh_features.shape[1] + 2}")

    # Check for all-zero features
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X

    zero_cols = np.sum(X_dense, axis=0) == 0
    non_zero_cols = np.sum(zero_cols == False)
    zero_col_ratio = np.sum(zero_cols) / X.shape[1]

    print(f"  Non-zero columns: {non_zero_cols}/{X.shape[1]} ({(1 - zero_col_ratio) * 100:.1f}%)")

    if np.all(zero_cols):
        print("  ❌ ALL FEATURES ARE ZERO - PROBLEM!")
        return False
    elif zero_col_ratio > 0.95:  # More than 95% zero columns
        print(f"  ❌ ERROR: {np.sum(zero_cols)} columns are all zeros (>{zero_col_ratio * 100:.1f}%)")
        return False
    elif zero_col_ratio > 0.8:  # More than 80% zero columns
        print(f"  ⚠️  WARNING: {np.sum(zero_cols)} columns are all zeros ({zero_col_ratio * 100:.1f}%)")
        print("     This is normal with small vocabulary samples, but check larger batches")
    else:
        print("  ✅ Features look good!")

    # Show sample values
    print(f"\nSample feature values (first 3 rows, first 10 cols):")
    print(X_dense[:3, :10])

    # Average non-zero features per row
    non_zero_per_row = np.sum(X_dense != 0, axis=1)
    print(f"Average non-zero features per row: {np.mean(non_zero_per_row):.1f}")

    return True


def debug_vocabulary_building(dep_vectorizer, onehot, sample_data):
    """Debug the vocabulary building process"""
    print("\n" + "=" * 50)
    print("VOCABULARY DEBUG")
    print("=" * 50)

    print(f"Dependency vocabulary size: {len(dep_vectorizer.vocabulary_)}")
    print("Sample dependency terms:", list(dep_vectorizer.vocabulary_.keys())[:10])

    print(f"\nWH-type categories: {onehot.categories_[0]}")

    # Test individual transformers
    print(f"\nTesting individual transformers:")
    sample_dep = sample_data['dep_path'].tolist()[:3]
    sample_wh = sample_data[['wh_type']].iloc[:3]

    print(f"Sample dep_path: {sample_dep}")
    dep_result = dep_vectorizer.transform(sample_dep)
    print(f"Dep transform result shape: {dep_result.shape}, non-zero: {dep_result.nnz}")

    print(f"Sample wh_type: {sample_wh['wh_type'].tolist()}")
    wh_result = onehot.transform(sample_wh)
    print(f"WH transform result shape: {wh_result.shape}, non-zero: {wh_result.nnz}")

    return dep_result.nnz > 0 and wh_result.nnz > 0


# ─────── Partial Fit Extensions ───────
def partial_fit_count_vectorizer(self, data):
    """Partial fit for CountVectorizer to incrementally build vocabulary"""
    if hasattr(self, 'vocabulary_'):
        vocab = self.vocabulary_.copy()
    else:
        vocab = {}

    # Temporarily disable min_df/max_df for partial fitting
    original_min_df = self.min_df
    original_max_df = self.max_df
    self.min_df = 1
    self.max_df = 1.0

    try:
        # Fit on current batch
        self.fit(data)

        # Merge vocabularies
        merged_vocab = list(set(vocab.keys()).union(set(self.vocabulary_.keys())))
        self.vocabulary_ = {word: i for i, word in enumerate(merged_vocab)}
    finally:
        # Restore original parameters
        self.min_df = original_min_df
        self.max_df = original_max_df

    return self


def partial_fit_onehot_encoder(self, data):
    """Partial fit for OneHotEncoder to incrementally build categories"""
    if hasattr(self, 'categories_'):
        existing_cats = set(self.categories_[0])
    else:
        existing_cats = set()

    # Get new categories from current batch
    new_cats = set(data.iloc[:, 0].unique())

    # Merge categories
    all_cats = sorted(list(existing_cats.union(new_cats)))

    # Create a sample DataFrame with all categories to properly fit
    sample_df = pd.DataFrame({'wh_type': all_cats})

    # Use regular fit to ensure all internal attributes are set correctly
    self.fit(sample_df)

    return self


# Monkey patch the partial_fit methods
CountVectorizer.partial_fit = partial_fit_count_vectorizer
OneHotEncoder.partial_fit = partial_fit_onehot_encoder

WH_WORDS = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
nlp = spacy.load('en_core_web_sm')


# ─────── Feature Extraction ───────
def get_wh_type(q):
    q_lower = str(q).lower().strip()
    for wh in WH_WORDS:
        if q_lower.startswith(wh):
            return wh
    return 'other'


def spacy_features(q):
    doc = nlp(q)
    return " ".join([f"{token.dep_}_{token.pos_}" for token in doc])


def extract_features(df):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    df['dep_path'] = df['question'].apply(spacy_features)
    df['length'] = df['question'].str.len()
    df['num_words'] = df['question'].str.split().str.len()
    df['wh_type'] = df['question'].apply(get_wh_type)
    return df


# ─────── Batch Reader ───────
def question_batch_generator(tsv_path, chunksize=100000):
    for chunk in pd.read_csv(tsv_path, sep='\t', header=None, names=['id', 'title', 'text'], chunksize=chunksize):
        chunk['question'] = chunk['text'].str.extract(r'^(.*?)(?: Answer:|$)')[0].str.strip()
        yield chunk[['id', 'question']]


# ─────── Vectorizer Setup ───────
def build_incremental_vectorizers():
    """Build vectorizers that can be incrementally fitted"""
    # Use min_df=1 for partial fitting, we'll handle frequency filtering later
    dep_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_features=1000)
    onehot = OneHotEncoder(handle_unknown="ignore")

    return dep_vectorizer, onehot


class PreFittedColumnTransformer:
    """Custom transformer that works with already-fitted individual transformers"""

    def __init__(self, dep_vectorizer, onehot):
        self.dep_vectorizer = dep_vectorizer
        self.onehot = onehot
        self.length_transformer = FunctionTransformer(lambda x: np.array(x).reshape(-1, 1))
        self.num_words_transformer = FunctionTransformer(lambda x: np.array(x).reshape(-1, 1))

    def transform(self, df):
        # Transform each feature separately
        dep_features = self.dep_vectorizer.transform(df['dep_path'].tolist())
        wh_features = self.onehot.transform(df[['wh_type']])
        length_features = self.length_transformer.fit_transform(df[['length']])
        num_words_features = self.num_words_transformer.fit_transform(df[['num_words']])

        # Combine all features horizontally
        from scipy.sparse import hstack

        # Convert dense arrays to sparse for concatenation
        if hasattr(length_features, 'toarray'):
            length_sparse = length_features
        else:
            from scipy.sparse import csr_matrix
            length_sparse = csr_matrix(length_features)

        if hasattr(num_words_features, 'toarray'):
            num_words_sparse = num_words_features
        else:
            num_words_sparse = csr_matrix(num_words_features)

        # Stack all features
        combined = hstack([dep_features, wh_features, length_sparse, num_words_sparse])
        return combined


def build_transformer_from_fitted_vectorizers(dep_vectorizer, onehot):
    """Build custom transformer from already-fitted vectorizers"""
    return PreFittedColumnTransformer(dep_vectorizer, onehot)


# ─────── Cluster Optimization ───────
def find_optimal_clusters(tsv_path, k_range=range(2, 21), chunksize=100000):
    """
    Find optimal number of clusters using MiniBatchKMeans with elbow method and silhouette analysis
    """

    print("Pass 1: Building vocabulary incrementally...")

    # Initialize vectorizers
    dep_vectorizer, onehot = build_incremental_vectorizers()
    total_rows = 0

    # First pass: incrementally fit vectorizers
    for i, df in enumerate(question_batch_generator(tsv_path, chunksize=chunksize)):
        if i % 10 == 0:
            print(f"  Fitting batch {i}...")

        df = extract_features(df)

        # Partial fit vectorizers on this batch only
        dep_vectorizer.partial_fit(df['dep_path'].tolist())
        onehot.partial_fit(df[['wh_type']])

        total_rows += len(df)

        # Remove this break for full dataset processing
        if i == 0:
            break

    print(f"Vocabulary built from {total_rows:,} rows")
    print(f"Dependency vocabulary size: {len(dep_vectorizer.vocabulary_)}")
    print(f"WH-type categories: {len(onehot.categories_[0])}")

    # Build transformer with fitted vectorizers
    transformer = build_transformer_from_fitted_vectorizers(dep_vectorizer, onehot)

    # ✅ VALIDATION: Test the transformer with sample data
    print("\n" + "=" * 30 + " VALIDATION " + "=" * 30)

    # Get a small sample for testing
    # test_batch = None
    # for batch_df in question_batch_generator(tsv_path, chunksize=min(100, chunksize)):
    #     test_batch = extract_features(batch_df)
    #     break

    # if test_batch is not None:
    #     # Debug vocabulary
    #     vocab_ok = debug_vocabulary_building(dep_vectorizer, onehot, test_batch)
    #
    #     # Test transformer
    #     transform_ok = validate_transformer_output(transformer, test_batch, dep_vectorizer, onehot)
    #
    #     if not vocab_ok or not transform_ok:
    #         print("❌ VALIDATION FAILED - Stopping execution")
    #         return None, {}
    #     else:
    #         print("✅ VALIDATION PASSED - Proceeding with clustering")

    print("=" * 72)

    # Dictionary to store results for each k
    results = {}

    # Test each k value
    for k in k_range:
        print(f"\nPass 2: Testing k={k} clusters...")

        # Initialize MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=1000,
            max_iter=100,
            n_init=3
        )

        # Track metrics
        sample_silhouettes = []

        # Second pass: clustering with consistent feature space
        batch_count = 0
        for i, df in enumerate(question_batch_generator(tsv_path, chunksize=chunksize)):
            if i % 10 == 0:
                print(f"  Processing batch {i}...")

            df = extract_features(df)
            X = transformer.transform(df)

            # ✅ DEBUGGING: Check batch features periodically
            if i % 50 == 0:  # Every 50th batch
                if hasattr(X, 'toarray'):
                    X_sample = X.toarray()
                else:
                    X_sample = X

                non_zero_features = np.sum(X_sample != 0, axis=1)
                print(f"    Batch {i}: {X.shape} features, avg non-zero per row: {np.mean(non_zero_features):.1f}")

                if np.mean(non_zero_features) < 1:
                    print(f"    ⚠️  WARNING: Very few non-zero features in batch {i}")

            # Partial fit on this batch
            kmeans.partial_fit(X)

            # For silhouette score, sample some points periodically
            if i % 20 == 0 and X.shape[0] > 1:  # Sample every 20th batch
                sample_size = min(1000, X.shape[0])
                sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
                X_sample = X[sample_indices]

                # Get cluster labels for sample
                labels_sample = kmeans.predict(X_sample)

                # Calculate silhouette score if we have multiple clusters represented
                if len(np.unique(labels_sample)) > 1:
                    sil_score = silhouette_score(X_sample, labels_sample)
                    sample_silhouettes.append(sil_score)

            batch_count += 1
            # Remove this break for full dataset processing
            if i == 0:
                break

        # Store results
        results[k] = {
            'inertia': kmeans.inertia_,
            'avg_silhouette': np.mean(sample_silhouettes) if sample_silhouettes else 0,
            'total_samples': total_rows
        }

        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Avg Silhouette={results[k]['avg_silhouette']:.3f}")

    # Find optimal k using elbow method and silhouette scores
    inertias = [results[k]['inertia'] for k in k_range]
    silhouettes = [results[k]['avg_silhouette'] for k in k_range]

    # Elbow method: find point of maximum curvature
    def find_elbow(x, y):
        # Normalize the data
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Calculate distances from line connecting first and last points
        distances = []
        for i in range(len(x_norm)):
            # Distance from point to line
            d = abs((y_norm[-1] - y_norm[0]) * x_norm[i] - (x_norm[-1] - x_norm[0]) * y_norm[i] +
                    x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0]) / \
                np.sqrt((y_norm[-1] - y_norm[0]) ** 2 + (x_norm[-1] - x_norm[0]) ** 2)
            distances.append(d)

        return np.argmax(distances)

    elbow_idx = find_elbow(list(k_range), inertias)
    elbow_k = list(k_range)[elbow_idx]

    # Best silhouette score
    best_sil_idx = np.argmax(silhouettes)
    best_sil_k = list(k_range)[best_sil_idx]

    print(f"\n" + "=" * 50)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total samples processed: {total_rows:,}")
    print(f"\nElbow method suggests: k = {elbow_k}")
    print(f"Best silhouette score: k = {best_sil_k} (score: {silhouettes[best_sil_idx]:.3f})")

    # Print all results
    print(f"\nDetailed Results:")
    print("k\tInertia\t\tSilhouette")
    print("-" * 35)
    for k in k_range:
        print(f"{k}\t{results[k]['inertia']:.1f}\t\t{results[k]['avg_silhouette']:.3f}")

    # Recommendation
    if elbow_k == best_sil_k:
        recommended_k = elbow_k
        print(f"\n✓ RECOMMENDED: k = {recommended_k} (both methods agree)")
    else:
        recommended_k = best_sil_k if silhouettes[best_sil_idx] > 0.3 else elbow_k
        print(f"\n✓ RECOMMENDED: k = {recommended_k}")
        print(f"  (Elbow: {elbow_k}, Silhouette: {best_sil_k})")

    return recommended_k, results, transformer  # Return transformer too


def save_cluster_labels(tsv_path, k, transformer, chunksize=100000, output_file="cluster_labels.tsv"):
    """
    Run final clustering with optimal k and save cluster labels to file
    """
    print(f"\n" + "=" * 50)
    print(f"FINAL CLUSTERING WITH k={k}")
    print("=" * 50)

    # Initialize final clustering model
    final_kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1000,
        max_iter=200,  # More iterations for final model
        n_init=10  # More initializations for stability
    )

    # First pass: fit the model
    print("Pass 1: Training final clustering model...")
    total_samples = 0
    for i, df in enumerate(question_batch_generator(tsv_path, chunksize=chunksize)):
        if i % 10 == 0:
            print(f"  Training batch {i}...")

        df = extract_features(df)
        X = transformer.transform(df)
        final_kmeans.partial_fit(X)
        total_samples += len(df)

        # Remove this break for full dataset
        if i == 0:
            break

    print(f"Model trained on {total_samples:,} samples")
    print(f"Final inertia: {final_kmeans.inertia_:.2f}")

    # Second pass: predict labels and save
    print(f"Pass 2: Predicting labels and saving to '{output_file}'...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("id\tcluster_label\n")

        batch_count = 0
        total_saved = 0

        for i, df in enumerate(question_batch_generator(tsv_path, chunksize=chunksize)):
            if i % 10 == 0:
                print(f"  Predicting batch {i}...")

            df = extract_features(df)
            X = transformer.transform(df)

            # Predict cluster labels
            labels = final_kmeans.predict(X)

            # Save to file - only id and cluster_label
            for idx, (_, row) in enumerate(df.iterrows()):
                f.write(f"{row['id']}\t{labels[idx]}\n")

            total_saved += len(df)
            batch_count += 1

            # Remove this break for full dataset
            if i == 0:
                break

    print(f"✅ Saved {total_saved:,} labeled samples to '{output_file}'")

    # Show cluster distribution
    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)  # This will only show last batch
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples ({count / len(labels) * 100:.1f}% of last batch)")

    return output_file


# ─────── Usage ───────
if __name__ == "__main__":
    tsv_path = "data/PAQ/PAQ.tsv"  # Replace with your file path

    # Find optimal clusters (with validation)
    optimal_k, all_results, fitted_transformer = find_optimal_clusters(
        tsv_path,
        k_range=range(2, 16),  # Test k from 2 to 15
        # chunksize=100000,  # Process 100k rows at a time for full dataset
        chunksize=100  # Use smaller chunks for testing
    )

    if optimal_k is not None:
        print(f"\n✅ SUCCESS: Use {optimal_k} clusters for your 6M row dataset")

        # Save cluster labels to file
        print(f"\nSaving cluster labels...")
        output_file = save_cluster_labels(
            tsv_path,
            optimal_k,
            transformer=fitted_transformer,  # Use the fitted transformer
            # chunksize=100000,
            chunksize=100,
            output_file="PAQ_cluster_labels.tsv"
        )
        print(f"✅ Cluster labels saved to: {output_file}")

    else:
        print(f"\n❌ FAILED: Could not determine optimal clusters due to validation errors")
