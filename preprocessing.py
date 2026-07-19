"""
Inference-time preprocessing pipeline for the fraud detection app.

This mirrors, function-for-function, the point-in-time feature engineering
in fraud_detection.ipynb (Layers 1-3: temporal, graph, device-trust), then
reproduces the exact downstream steps used at training time (sequence model
scoring, risk-trend features, dual anomaly detection) using the artifacts
the notebook now exports to MODEL_DIR ("models/") via its export block at
the end of main().

Nothing here retrains anything - it only loads what was already fit during
training and applies it to new data, the same way proposed_model.predict()
does in the notebook.
"""

import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Constants - must match the values in fraud_detection.ipynb (cell "PARAMETERS")
# ---------------------------------------------------------------------------
BATCH_SIZE = 50_000
PAGERANK_EVERY_N_BATCHES = 3
ENABLE_EXPENSIVE_CENTRALITY = False
BETWEENNESS_SAMPLE_K = 200
DEVICE_RECENT_WINDOW = 20
BETA_PRIOR_ALPHA = 1
BETA_PRIOR_BETA = 27
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "DeviceInfo",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Validation / artifact loading
# ---------------------------------------------------------------------------
def validate_required_columns(df):
    """Returns a list of required raw columns that are missing from df."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def load_encoders(path):
    """Loads the {col: {category_string: index}} mapping exported by the notebook."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LAYER 1: temporal behavioral features (copied verbatim from the notebook)
# ---------------------------------------------------------------------------
def add_temporal_features(df):
    log("Building Layer 1: expanded temporal features (point-in-time)...")
    grouped = df.groupby("card1")

    df["time_since_last_txn"] = grouped["TransactionDT"].diff().fillna(-1)

    df["amt_rolling_mean_3"] = grouped["TransactionAmt"].transform(
        lambda s: s.shift().rolling(window=3, min_periods=1).mean()
    )
    df["amt_rolling_mean_3"] = df["amt_rolling_mean_3"].fillna(df["TransactionAmt"])
    df["amt_drift"] = df["TransactionAmt"] - df["amt_rolling_mean_3"]

    df["txn_count_so_far"] = grouped.cumcount()

    def velocity(sub):
        s = sub.set_index(pd.to_timedelta(sub["TransactionDT"], unit="s"))
        counts = s["TransactionDT"].shift(1).rolling("24h").count()
        return counts.values
    df["txn_velocity_24h"] = grouped.apply(
        lambda g: pd.Series(velocity(g), index=g.index)
    ).reset_index(level=0, drop=True).fillna(0)

    df["amt_volatility_5"] = grouped["TransactionAmt"].transform(
        lambda s: s.shift().rolling(window=5, min_periods=2).std()
    ).fillna(0)

    short_mean = grouped["TransactionAmt"].transform(
        lambda s: s.shift().rolling(window=3, min_periods=1).mean()
    )
    long_mean = grouped["TransactionAmt"].transform(
        lambda s: s.shift().rolling(window=10, min_periods=1).mean()
    )
    df["amt_trend"] = (short_mean - long_mean).fillna(0)

    df["amt_rolling_mean_10"] = long_mean.fillna(df["TransactionAmt"])

    def rolling_txn_count(sub, window):
        s = sub.set_index(pd.to_timedelta(sub["TransactionDT"], unit="s"))
        counts = s["TransactionDT"].shift(1).rolling(window).count()
        return counts.values

    df["txns_last_1h"] = grouped.apply(
        lambda g: pd.Series(rolling_txn_count(g, "1h"), index=g.index)
    ).reset_index(level=0, drop=True).fillna(0)
    df["txns_last_7d"] = grouped.apply(
        lambda g: pd.Series(rolling_txn_count(g, "7d"), index=g.index)
    ).reset_index(level=0, drop=True).fillna(0)

    mean_5 = grouped["TransactionAmt"].transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).mean()
    )
    mean_5_filled = mean_5.fillna(df["TransactionAmt"])
    df["amt_shock_ratio"] = df["TransactionAmt"] / (mean_5_filled + 1e-3)

    device_filled = df["DeviceInfo"].fillna("missing_device")
    device_switch_flag = df.groupby("card1").apply(
        lambda g: (device_filled.loc[g.index] != device_filled.loc[g.index].shift(1)).astype(int)
    ).reset_index(level=0, drop=True)
    df["device_switch_rate_5"] = device_switch_flag.groupby(df["card1"]).transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).mean()
    ).fillna(0)

    if "P_emaildomain" in df.columns:
        channel_filled = df["P_emaildomain"].fillna("missing_channel")
        channel_switch_flag = df.groupby("card1").apply(
            lambda g: (channel_filled.loc[g.index] != channel_filled.loc[g.index].shift(1)).astype(int)
        ).reset_index(level=0, drop=True)
        df["channel_switch_rate_5"] = channel_switch_flag.groupby(df["card1"]).transform(
            lambda s: s.shift().rolling(window=5, min_periods=1).mean()
        ).fillna(0)
    else:
        df["channel_switch_rate_5"] = 0.0

    # isFraud is unknown for the row(s) actually being scored right now (that's what
    # we're predicting), but is known for *past, already-resolved* rows in the same
    # upload. Only past rows feed this feature (via shift(1)/ffill), so a row never
    # sees its own label here - see run_feature_pipeline() below for how isFraud
    # is populated before this function runs.
    fraud_time = df["TransactionDT"].where(df["isFraud"] == 1)
    last_fraud_time = grouped.apply(
        lambda g: fraud_time.loc[g.index].shift(1).ffill()
    ).reset_index(level=0, drop=True)
    df["time_since_last_fraud"] = (df["TransactionDT"] - last_fraud_time).fillna(-1)

    return df


# ---------------------------------------------------------------------------
# LAYER 2: dynamic social graph features (copied verbatim from the notebook)
# ---------------------------------------------------------------------------
def add_dynamic_graph_features(df):
    log("Building Layer 2: dynamic social graph features (batched)...")

    missing_mask = df["DeviceInfo"].isna()
    df.loc[missing_mask, "DeviceInfo"] = "missing_" + df.loc[missing_mask].index.astype(str)
    df["_node_id"] = df.index

    extra_edge_keys = []
    for col in ["P_emaildomain", "R_emaildomain", "id_31"]:
        if col in df.columns:
            df[col] = df[col].fillna(f"missing_{col}")
            extra_edge_keys.append(col)

    for col in ["card4", "card6"]:
        if col in df.columns:
            df[col] = df[col].fillna(f"missing_{col}")
            extra_edge_keys.append(col)

    n = len(df)
    G = nx.Graph()

    graph_degree = np.zeros(n, dtype=np.float32)
    graph_component_size = np.ones(n, dtype=np.float32)
    graph_clustering = np.zeros(n, dtype=np.float32)
    graph_pagerank = np.zeros(n, dtype=np.float32)
    graph_community_size = np.ones(n, dtype=np.float32)
    graph_neighbor_fraud_rate = np.zeros(n, dtype=np.float32)
    graph_neighbor_fraud_count = np.zeros(n, dtype=np.float32)
    graph_betweenness = np.zeros(n, dtype=np.float32)
    graph_eigenvector = np.zeros(n, dtype=np.float32)
    graph_avg_neighbor_risk = np.zeros(n, dtype=np.float32)
    graph_devices_per_card = np.ones(n, dtype=np.float32)

    node_is_fraud = {}
    node_running_risk = {}
    card_fraud_count = {}
    card_txn_count = {}
    card_devices_seen = {}
    pagerank_cache = {}
    community_cache = {}
    betweenness_cache = {}
    eigenvector_cache = {}

    card_device_index = {}
    addr_device_index = {}
    extra_indexes = {col: {} for col in extra_edge_keys}

    num_batches = int(np.ceil(n / BATCH_SIZE))

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch = df.iloc[start:end]

        degree_dict, component_size, clustering_dict = {}, {}, {}
        if G.number_of_nodes() > 0:
            degree_dict = dict(G.degree())
            for comp in nx.connected_components(G):
                size = len(comp)
                for node in comp:
                    component_size[node] = size

            MAX_DEGREE_FOR_CLUSTERING = 500
            low_degree_nodes = [nd for nd, d in degree_dict.items() if d <= MAX_DEGREE_FOR_CLUSTERING]
            clustering_dict = nx.clustering(G, nodes=low_degree_nodes) if low_degree_nodes else {}

            if batch_idx % PAGERANK_EVERY_N_BATCHES == 0 and G.number_of_edges() > 0:
                try:
                    pagerank_cache = nx.pagerank(G, max_iter=50)
                except nx.PowerIterationFailedConvergence:
                    pass
                try:
                    communities = list(nx.algorithms.community.asyn_lpa_communities(G, seed=RANDOM_STATE))
                    community_cache = {}
                    for comm in communities:
                        size = len(comm)
                        for node in comm:
                            community_cache[node] = size
                except Exception:
                    pass

                if ENABLE_EXPENSIVE_CENTRALITY:
                    try:
                        k = min(BETWEENNESS_SAMPLE_K, G.number_of_nodes())
                        betweenness_cache = nx.betweenness_centrality(G, k=k, seed=RANDOM_STATE)
                    except Exception:
                        betweenness_cache = {}
                    try:
                        eigenvector_cache = nx.eigenvector_centrality(G, max_iter=100, tol=1e-4)
                    except (nx.PowerIterationFailedConvergence, Exception):
                        eigenvector_cache = {}

        for node, card, dev, addr in zip(
            batch["_node_id"].values, batch["card1"].values,
            batch["DeviceInfo"].values, batch["addr1"].values,
        ):
            neighbor_ids = set(card_device_index.get((card, dev), ())) | set(addr_device_index.get((addr, dev), ()))
            for col in extra_edge_keys:
                val = batch.at[node, col] if node in batch.index else None
                if val is not None:
                    neighbor_ids |= set(extra_indexes[col].get((val, dev), ()))

            graph_degree[node] = len(neighbor_ids)
            if neighbor_ids:
                comp_vals = [component_size.get(nb, 1) for nb in neighbor_ids]
                graph_component_size[node] = max(comp_vals) if comp_vals else 1

                clust_vals = [clustering_dict[nb] for nb in neighbor_ids if nb in clustering_dict]
                graph_clustering[node] = float(np.mean(clust_vals)) if clust_vals else 0.0

                pr_vals = [pagerank_cache[nb] for nb in neighbor_ids if nb in pagerank_cache]
                graph_pagerank[node] = float(np.mean(pr_vals)) if pr_vals else 0.0

                comm_vals = [community_cache.get(nb, 1) for nb in neighbor_ids]
                graph_community_size[node] = max(comm_vals) if comm_vals else 1

                fraud_flags = [node_is_fraud[nb] for nb in neighbor_ids if nb in node_is_fraud]
                graph_neighbor_fraud_rate[node] = float(np.mean(fraud_flags)) if fraud_flags else 0.0
                graph_neighbor_fraud_count[node] = float(np.sum(fraud_flags)) if fraud_flags else 0.0

                risk_vals = [node_running_risk[nb] for nb in neighbor_ids if nb in node_running_risk]
                graph_avg_neighbor_risk[node] = float(np.mean(risk_vals)) if risk_vals else 0.0

                if ENABLE_EXPENSIVE_CENTRALITY:
                    bt_vals = [betweenness_cache[nb] for nb in neighbor_ids if nb in betweenness_cache]
                    graph_betweenness[node] = float(np.mean(bt_vals)) if bt_vals else 0.0
                    ev_vals = [eigenvector_cache[nb] for nb in neighbor_ids if nb in eigenvector_cache]
                    graph_eigenvector[node] = float(np.mean(ev_vals)) if ev_vals else 0.0

            graph_devices_per_card[node] = len(card_devices_seen.get(card, set())) or 1

        batch_nodes = batch["_node_id"].values
        G.add_nodes_from(batch_nodes)
        for _, group in batch.groupby(["card1", "DeviceInfo"]):
            nodes = group["_node_id"].tolist()
            if len(nodes) > 1:
                anchor = nodes[0]
                for other in nodes[1:]:
                    G.add_edge(anchor, other)
        for _, group in batch.groupby(["addr1", "DeviceInfo"]):
            nodes = group["_node_id"].tolist()
            if len(nodes) > 1:
                anchor = nodes[0]
                for other in nodes[1:]:
                    G.add_edge(anchor, other)
        for col in extra_edge_keys:
            for _, group in batch.groupby([col, "DeviceInfo"]):
                nodes = group["_node_id"].tolist()
                if len(nodes) > 1:
                    anchor = nodes[0]
                    for other in nodes[1:]:
                        G.add_edge(anchor, other)

        for card, dev, node in zip(batch["card1"].values, batch["DeviceInfo"].values, batch_nodes):
            card_device_index.setdefault((card, dev), []).append(node)
        for addr, dev, node in zip(batch["addr1"].values, batch["DeviceInfo"].values, batch_nodes):
            addr_device_index.setdefault((addr, dev), []).append(node)
        for col in extra_edge_keys:
            for val, dev, node in zip(batch[col].values, batch["DeviceInfo"].values, batch_nodes):
                extra_indexes[col].setdefault((val, dev), []).append(node)

        for node, label in zip(batch_nodes, batch["isFraud"].values):
            node_is_fraud[node] = int(label)

        for card, node, label in zip(batch["card1"].values, batch_nodes, batch["isFraud"].values):
            prior_fraud = card_fraud_count.get(card, 0)
            prior_count = card_txn_count.get(card, 0)
            node_running_risk[node] = (prior_fraud + BETA_PRIOR_ALPHA) / (
                prior_count + BETA_PRIOR_ALPHA + BETA_PRIOR_BETA
            )
            card_fraud_count[card] = prior_fraud + int(label)
            card_txn_count[card] = prior_count + 1

        for card, dev in zip(batch["card1"].values, batch["DeviceInfo"].values):
            card_devices_seen.setdefault(card, set()).add(dev)

    df["graph_degree"] = graph_degree
    df["graph_component_size"] = graph_component_size
    df["graph_clustering"] = graph_clustering
    df["graph_pagerank"] = graph_pagerank
    df["graph_community_size"] = graph_community_size
    df["graph_neighbor_fraud_rate"] = graph_neighbor_fraud_rate
    df["graph_neighbor_fraud_count"] = graph_neighbor_fraud_count
    df["graph_betweenness"] = graph_betweenness
    df["graph_eigenvector"] = graph_eigenvector
    df["graph_avg_neighbor_risk"] = graph_avg_neighbor_risk
    df["graph_devices_per_card"] = graph_devices_per_card

    df = df.drop(columns=["_node_id"])
    return df


# ---------------------------------------------------------------------------
# LAYER 3: dynamic device trust score (copied verbatim from the notebook)
# ---------------------------------------------------------------------------
def add_device_trust_features(df):
    log("Building Layer 3: dynamic device trust score (point-in-time, single-pass)...")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)

    device_txn_count = np.zeros(n, dtype=np.float32)
    device_unique_cards = np.ones(n, dtype=np.float32)
    device_avg_gap = np.full(n, -1, dtype=np.float32)
    device_trust_score = np.ones(n, dtype=np.float32)
    device_recent_fraud_rate = np.zeros(n, dtype=np.float32)

    device_count = {}
    device_cards_seen = {}
    device_last_time = {}
    device_gap_sum = {}
    device_gap_count = {}
    device_fraud_sum = {}
    device_recent_labels = {}
    global_prior_fraud_rate = BETA_PRIOR_ALPHA / (BETA_PRIOR_ALPHA + BETA_PRIOR_BETA)

    devices = df["DeviceInfo"].values
    cards = df["card1"].values
    times = df["TransactionDT"].values
    labels = df["isFraud"].values

    for i in range(n):
        dev = devices[i]
        card = cards[i]
        t = times[i]

        count_so_far = device_count.get(dev, 0)
        device_txn_count[i] = count_so_far
        device_unique_cards[i] = len(device_cards_seen.get(dev, set())) if dev in device_cards_seen else 1

        gap_count = device_gap_count.get(dev, 0)
        if gap_count > 0:
            device_avg_gap[i] = device_gap_sum[dev] / gap_count

        fraud_count = device_fraud_sum.get(dev, 0)
        posterior_fraud_rate = (fraud_count + BETA_PRIOR_ALPHA) / (count_so_far + BETA_PRIOR_ALPHA + BETA_PRIOR_BETA)
        device_trust_score[i] = 1 - posterior_fraud_rate

        recent = device_recent_labels.get(dev)
        device_recent_fraud_rate[i] = float(np.mean(recent)) if recent else global_prior_fraud_rate

        device_count[dev] = count_so_far + 1
        device_cards_seen.setdefault(dev, set()).add(card)
        if dev in device_last_time:
            gap = t - device_last_time[dev]
            device_gap_sum[dev] = device_gap_sum.get(dev, 0) + gap
            device_gap_count[dev] = gap_count + 1
        device_last_time[dev] = t
        device_fraud_sum[dev] = fraud_count + int(labels[i])
        device_recent_labels.setdefault(dev, deque(maxlen=DEVICE_RECENT_WINDOW)).append(int(labels[i]))

    df["device_txn_count"] = device_txn_count
    df["device_unique_cards"] = device_unique_cards
    df["device_avg_gap"] = device_avg_gap
    df["device_trust_score"] = device_trust_score
    df["device_recent_fraud_rate"] = device_recent_fraud_rate

    card_grouped = df.groupby("card1")
    df["address_changed"] = (
        card_grouped["addr1"].transform(lambda s: (s != s.shift(1)).astype(int))
    ).fillna(0)
    df["device_changed"] = (
        card_grouped["DeviceInfo"].transform(lambda s: (s != s.shift(1)).astype(int))
    ).fillna(0)

    return df


# ---------------------------------------------------------------------------
# Sequence model support (copied verbatim from the notebook's build_sequences)
# ---------------------------------------------------------------------------
def _build_sequences(df, cols, seq_len):
    values = df[cols].values.astype(np.float32)
    card_ids = df["card1"].values
    n = len(df)
    num_features = len(cols)

    sequences = np.zeros((n, seq_len, num_features), dtype=np.float32)

    card_to_rows = {}
    for i, c in enumerate(card_ids):
        card_to_rows.setdefault(c, []).append(i)

    for card, rows in card_to_rows.items():
        history = []
        for i in rows:
            if history:
                past = history[-seq_len:]
                sequences[i, seq_len - len(past):, :] = np.array(past, dtype=np.float32)
            history.append(values[i])

    return sequences


def _add_risk_trend_features(X):
    grouped = X.groupby("card1")["sequence_model_prob"]

    X["previous_risk_mean_3"] = grouped.transform(
        lambda s: s.shift().rolling(window=3, min_periods=1).mean()
    ).fillna(0)
    X["previous_risk_max_5"] = grouped.transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).max()
    ).fillna(0)

    short_risk = grouped.transform(lambda s: s.shift().rolling(window=3, min_periods=1).mean())
    long_risk = grouped.transform(lambda s: s.shift().rolling(window=10, min_periods=1).mean())
    X["risk_change_rate"] = (short_risk - long_risk).fillna(0)

    return X


# ---------------------------------------------------------------------------
# Public pipeline entry points used by app.py
# ---------------------------------------------------------------------------
def run_feature_pipeline(raw_df):
    """
    Runs Layers 1-3 (temporal, graph, device-trust) on newly-uploaded
    transactions, exactly as done at training time.

    isFraud is not something the app ever receives for new transactions
    (that's the thing being predicted), so it's initialized to 0 for every
    row before feature-building. That's safe because every feature above
    that reads isFraud only ever looks at *other, earlier* rows via
    shift()/rolling() - a row's own isFraud value never contributes to its
    own features. If your CSV upload happens to include a real isFraud
    column for historical/resolved rows, it's used as-is (giving the
    behavioral/graph features real history to work from), and only rows
    where it's genuinely unknown should be left as 0.
    """
    df = raw_df.copy()
    if "isFraud" not in df.columns:
        df["isFraud"] = 0
    else:
        df["isFraud"] = df["isFraud"].fillna(0).astype(int)

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_dynamic_graph_features(df)
    df = add_device_trust_features(df)
    return df


def _encode_categoricals(X, encoders, cat_cols):
    for col in cat_cols:
        if col not in X.columns:
            X[col] = "missing"
        X[col] = X[col].fillna("missing").astype(str)
        mapping = encoders.get(col, {})
        # Any category never seen during training (including "missing" itself,
        # if it never appeared in training) gets one shared fallback bucket
        # rather than crashing or silently colliding with a real class.
        unseen_idx = len(mapping)
        X[col] = X[col].map(lambda v: mapping.get(v, unseen_idx)).astype(int)
    return X


def preprocess_for_inference(engineered_df, encoders, cat_cols, feature_list, model_dir="models"):
    """
    Takes the Layer 1-3 engineered dataframe and reproduces every remaining
    training-time step (categorical encoding, sequence model scoring,
    risk-trend features, dual anomaly detection) using the artifacts saved
    by the notebook, then returns a dataframe with exactly `feature_list`
    columns in the right order for the XGBoost booster.
    """
    df = engineered_df.copy()
    drop_cols = [c for c in ["isFraud", "TransactionID"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    X = _encode_categoricals(X, encoders, cat_cols)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(-999)

    # --- sequence model (LSTM) ---
    seq_cols_path = os.path.join(model_dir, "seq_feature_cols.json")
    seq_cfg_path = os.path.join(model_dir, "seq_config.json")
    lstm_path = os.path.join(model_dir, "lstm_sequence_model.keras")
    if os.path.exists(seq_cols_path) and os.path.exists(seq_cfg_path) and os.path.exists(lstm_path):
        with open(seq_cols_path) as f:
            seq_cols = json.load(f)
        with open(seq_cfg_path) as f:
            seq_cfg = json.load(f)
        import tensorflow as tf
        seq_model = tf.keras.models.load_model(lstm_path)
        embed_model = tf.keras.Model(
            inputs=seq_model.input, outputs=seq_model.get_layer("embedding").output
        )
        sequences = _build_sequences(df, seq_cols, seq_cfg["seq_len"])
        X["sequence_model_prob"] = seq_model.predict(sequences, batch_size=4096, verbose=0).flatten()
        embeds = embed_model.predict(sequences, batch_size=4096, verbose=0)
        for i in range(seq_cfg["embed_dim"]):
            X[f"seq_embed_{i}"] = embeds[:, i]
    else:
        X["sequence_model_prob"] = 0.0
        for col in feature_list:
            if col.startswith("seq_embed_"):
                X[col] = 0.0

    X = _add_risk_trend_features(X)

    # --- anomaly detection (Isolation Forest + autoencoder) ---
    anomaly_cols_path = os.path.join(model_dir, "anomaly_cols.json")
    anomaly_cols = None
    if os.path.exists(anomaly_cols_path):
        with open(anomaly_cols_path) as f:
            anomaly_cols = [c for c in json.load(f) if c in X.columns]

    iso_path = os.path.join(model_dir, "isolation_forest.pkl")
    if anomaly_cols and os.path.exists(iso_path):
        import joblib
        iso = joblib.load(iso_path)
        raw_scores = -iso.decision_function(X[anomaly_cols])
        rmin, rmax = raw_scores.min(), raw_scores.max()
        X["anomaly_score"] = (raw_scores - rmin) / (rmax - rmin + 1e-9)
    else:
        X["anomaly_score"] = 0.0

    ae_path = os.path.join(model_dir, "autoencoder_model.keras")
    scaler_path = os.path.join(model_dir, "autoencoder_scaler.pkl")
    if anomaly_cols and os.path.exists(ae_path) and os.path.exists(scaler_path):
        import joblib
        import tensorflow as tf
        autoencoder = tf.keras.models.load_model(ae_path)
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X[anomaly_cols])
        recon = autoencoder.predict(X_scaled, batch_size=4096, verbose=0)
        err = np.mean((X_scaled - recon) ** 2, axis=1)
        emin, emax = err.min(), err.max()
        X["autoencoder_score"] = (err - emin) / (emax - emin + 1e-9)
    else:
        X["autoencoder_score"] = 0.0

    # --- final alignment to the exact training-time feature list/order ---
    for col in feature_list:
        if col not in X.columns:
            X[col] = -999
    X = X[feature_list]
    return X
