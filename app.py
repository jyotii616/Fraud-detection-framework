import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

import preprocessing as prep

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model_xgboost.json")
ENCODERS_PATH = os.path.join(MODEL_DIR, "feature_encoders.json")
CAT_COLS_PATH = os.path.join(MODEL_DIR, "cat_cols.json")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
    div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    div[data-testid="stAlert"] { border-radius: 6px; }
    div[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
    div[data-testid="stExpander"] { border-radius: 6px; }
    div[data-testid="stFileUploaderDropzone"] { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, ENCODERS_PATH, CAT_COLS_PATH, FEATURE_LIST_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return None

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    encoders = prep.load_encoders(ENCODERS_PATH)
    with open(CAT_COLS_PATH) as f:
        cat_cols = json.load(f)
    with open(FEATURE_LIST_PATH) as f:
        feature_list = json.load(f)

    default_threshold = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            default_threshold = json.load(f)["threshold"]

    return {
        "booster": booster,
        "encoders": encoders,
        "cat_cols": cat_cols,
        "feature_list": feature_list,
        "default_threshold": default_threshold,
    }


artifacts = load_artifacts()

st.title("🛡️ Fraud Detection")
st.write("Upload transactions and get a fraud probability for each one.")

if artifacts is None:
    st.error(
        f"Model artifacts not found in '{MODEL_DIR}'. Run `train_pipeline.py` first - "
        "it produces model_xgboost.json, feature_encoders.json, cat_cols.json, "
        "feature_list.json, and threshold.json."
    )
    st.stop()

with st.container(border=True):
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Status", "Ready")
    stat_col2.metric("Features per transaction", f"{len(artifacts['feature_list']):,}")
    stat_col3.metric("Model type", "XGBoost")

st.divider()

st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Fraud probability threshold", min_value=0.0, max_value=1.0,
    value=float(artifacts["default_threshold"]), step=0.01,
)
st.sidebar.write(
    "Every upload goes through the same pipeline used during training: it looks at "
    "spending patterns over time, connections between cards and devices, and how "
    "trustworthy a device has been so far. A transaction is flagged as fraud if its "
    "score is above the threshold."
)

st.subheader("1. Choose how to upload")
mode = st.radio(
    "Input mode",
    [
        "Upload CSV (batch, recommended)",
        "Upload CSV (custom / other dataset schema)",
        "Manual entry (single transaction)",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


REQUIRED_MAP_FIELDS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "DeviceInfo"]
OPTIONAL_MAP_FIELDS = ["card4", "card6", "P_emaildomain", "R_emaildomain", "id_31", "isFraud"]

FIELD_HELP = {
    "TransactionID": "A unique ID per transaction. Leave this unmapped and one will be generated automatically.",
    "TransactionDT": "The time of the transaction. Either seconds elapsed since some reference point, "
                      "or an actual date/time column (check the box below if it's a real date).",
    "TransactionAmt": "The transaction amount.",
    "card1": "A card, account, or customer ID. Anything that repeats across the same person's "
              "transactions works. This is what the behavior and connection features are built around.",
    "addr1": "A billing or shipping address identifier: zip code, address ID, region code, or similar.",
    "DeviceInfo": "The device or browser used for the transaction.",
    "card4": "Card network, e.g. visa or mastercard. Optional.",
    "card6": "Card type, e.g. debit or credit. Optional.",
    "P_emaildomain": "Purchaser's email domain. Optional.",
    "R_emaildomain": "Recipient's email domain. Optional.",
    "id_31": "Browser or user-agent string. Optional.",
    "isFraud": "Known fraud outcome for past transactions, if you have it. Optional, but it gives the "
               "behavior and connection features real history to learn from.",
}

_FIELD_ALIASES = {
    "TransactionID": ["transactionid", "txnid", "id", "transactionid", "rowid", "recordid"],
    "TransactionDT": ["transactiondt", "timestamp", "date", "datetime", "txntime", "time", "eventtime"],
    "TransactionAmt": ["transactionamt", "amount", "amt", "txnamount", "value", "price"],
    "card1": ["card1", "cardid", "cardnumber", "accountid", "customerid", "userid"],
    "addr1": ["addr1", "address", "zip", "zipcode", "postalcode", "region", "billingaddress"],
    "DeviceInfo": ["deviceinfo", "device", "useragent", "devicename"],
    "card4": ["card4", "cardnetwork", "network", "brand", "cardbrand"],
    "card6": ["card6", "cardtype", "type"],
    "P_emaildomain": ["pemaildomain", "email", "purchaseremail", "emaildomain"],
    "R_emaildomain": ["remaildomain", "recipientemail"],
    "id_31": ["id31", "browser", "browserinfo"],
    "isFraud": ["isfraud", "fraud", "label", "target", "class"],
}


def _auto_guess(field, columns):
    normalized = {c.lower().replace(" ", "").replace("-", "").replace("_", ""): c for c in columns}
    for alias in _FIELD_ALIASES.get(field, []):
        if alias in normalized:
            return normalized[alias]
    return None


def run_pipeline_and_score(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing_required = prep.validate_required_columns(raw_df)
    if missing_required:
        raise ValueError(
            f"Uploaded CSV is missing required columns: {missing_required}. "
            "These are needed to compute the behavioral/graph features the model relies on."
        )

    with st.spinner("Running feature engineering (temporal, graph, device-trust)..."):
        engineered_df = prep.run_feature_pipeline(raw_df)

    with st.spinner("Encoding and scoring..."):
        X = prep.preprocess_for_inference(
            engineered_df, artifacts["encoders"], artifacts["cat_cols"], artifacts["feature_list"]
        )
        dmatrix = xgb.DMatrix(X.values, feature_names=artifacts["feature_list"])
        proba = artifacts["booster"].predict(dmatrix)

    out = engineered_df.copy()
    out["fraud_probability"] = proba
    out["is_fraud"] = proba >= threshold
    return out


if mode == "Upload CSV (batch, recommended)":
    st.write(
        "Upload your transaction CSV in the original IEEE-CIS format. If device and identity "
        "details live in a separate file, add that too and it'll be matched up automatically "
        "using TransactionID. Uploading a full batch lets the model see each card's history, "
        "which is what makes the scores meaningfully different from one transaction to the next."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        transaction_file = st.file_uploader("Transactions CSV (required)", type=["csv"], key="transaction_csv")
    with col_b:
        identity_file = st.file_uploader("Identity CSV (optional)", type=["csv"], key="identity_csv")

    input_df = None
    if transaction_file is not None:
        transaction_df = pd.read_csv(transaction_file)
        st.write(f"Transactions loaded: {len(transaction_df):,} rows, {transaction_df.shape[1]} columns.")

        if identity_file is not None:
            identity_df = pd.read_csv(identity_file)
            st.write(f"Identity loaded: {len(identity_df):,} rows, {identity_df.shape[1]} columns.")
            if "TransactionID" not in identity_df.columns:
                st.error("The identity CSV doesn't have a TransactionID column, so it can't be matched to the transactions.")
                input_df = None
            else:
                input_df = transaction_df.merge(identity_df, on="TransactionID", how="left")
                new_cols = input_df.shape[1] - transaction_df.shape[1]
                st.write(
                    f"Matched on TransactionID: {len(input_df):,} rows, {input_df.shape[1]} columns "
                    f"({new_cols} added from the identity file)."
                )
        else:
            input_df = transaction_df
            if "DeviceInfo" not in input_df.columns:
                st.warning(
                    "No DeviceInfo column here, and no identity CSV was uploaded. "
                    "If your identity fields live in a separate file, add it above. "
                    "Otherwise, the device-trust features won't have much to work with."
                )

    st.write("")
    if input_df is not None and st.button("Run predictions", type="primary"):
            try:
                results = run_pipeline_and_score(input_df)
            except ValueError as e:
                st.error(str(e))
            else:
                st.success(f"Done. Flagged {int(results['is_fraud'].sum()):,} of {len(results):,} as fraud.")
                st.subheader("Results")
                st.dataframe(
                    results.sort_values("fraud_probability", ascending=False),
                    use_container_width=True,
                )
                st.download_button(
                    "Download results as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

elif mode == "Upload CSV (custom / other dataset schema)":
    st.write(
        "Have a transactions file that isn't in IEEE-CIS format? Upload it below and tell "
        "the app which of your columns line up with what the model expects. Anything you "
        "skip either gets ignored (optional fields) or filled in automatically (TransactionID)."
    )
    custom_file = st.file_uploader("Your transactions CSV", type=["csv"], key="custom_csv")

    if custom_file is not None:
        custom_raw_df = pd.read_csv(custom_file)
        st.write(f"Loaded {len(custom_raw_df):,} rows, {custom_raw_df.shape[1]} columns.")
        columns_available = ["-- Not available --"] + list(custom_raw_df.columns)

        st.subheader("Required fields")
        mapping = {}
        req_cols_ui = st.columns(2)
        for i, field in enumerate(REQUIRED_MAP_FIELDS):
            guess = _auto_guess(field, custom_raw_df.columns)
            default_idx = columns_available.index(guess) if guess in columns_available else 0
            with req_cols_ui[i % 2]:
                mapping[field] = st.selectbox(
                    field, columns_available, index=default_idx, help=FIELD_HELP[field], key=f"map_{field}"
                )

        dt_is_datetime = st.checkbox(
            "My TransactionDT column above is a real date/time (not seconds elapsed since some reference point)",
            key="dt_is_datetime",
        )

        with st.expander("Optional fields (mapping these can meaningfully improve accuracy)"):
            opt_cols_ui = st.columns(2)
            for i, field in enumerate(OPTIONAL_MAP_FIELDS):
                guess = _auto_guess(field, custom_raw_df.columns)
                default_idx = columns_available.index(guess) if guess in columns_available else 0
                with opt_cols_ui[i % 2]:
                    mapping[field] = st.selectbox(
                        field, columns_available, index=default_idx, help=FIELD_HELP[field], key=f"map_{field}"
                    )

        still_missing = [
            f for f in REQUIRED_MAP_FIELDS
            if mapping[f] == "-- Not available --" and f != "TransactionID"
        ]
        if still_missing:
            st.warning(
                f"A few required fields still need mapping: {still_missing}. "
                "Everything except TransactionID has to be mapped before predictions can run."
            )

        st.write("")
        if not still_missing and st.button("Build mapped dataset & run predictions", type="primary"):
            mapped_df = pd.DataFrame(index=custom_raw_df.index)

            if mapping["TransactionID"] == "-- Not available --":
                mapped_df["TransactionID"] = np.arange(len(custom_raw_df))
            else:
                mapped_df["TransactionID"] = custom_raw_df[mapping["TransactionID"]]

            if dt_is_datetime:
                dt_parsed = pd.to_datetime(custom_raw_df[mapping["TransactionDT"]], errors="coerce")
                mapped_df["TransactionDT"] = (dt_parsed - dt_parsed.min()).dt.total_seconds()
            else:
                mapped_df["TransactionDT"] = pd.to_numeric(custom_raw_df[mapping["TransactionDT"]], errors="coerce")

            mapped_df["TransactionAmt"] = pd.to_numeric(custom_raw_df[mapping["TransactionAmt"]], errors="coerce")
            mapped_df["card1"] = custom_raw_df[mapping["card1"]]
            mapped_df["addr1"] = custom_raw_df[mapping["addr1"]]
            mapped_df["DeviceInfo"] = custom_raw_df[mapping["DeviceInfo"]]

            for field in OPTIONAL_MAP_FIELDS:
                if mapping[field] != "-- Not available --":
                    mapped_df[field] = custom_raw_df[mapping[field]]

            bad_dt = int(mapped_df["TransactionDT"].isna().sum())
            bad_amt = int(mapped_df["TransactionAmt"].isna().sum())
            if bad_dt or bad_amt:
                st.warning(
                    f"{bad_dt} row(s) had a time value that couldn't be read, and {bad_amt} row(s) "
                    "had an amount that couldn't be read. Those rows were dropped before scoring."
                )
                keep_mask = mapped_df["TransactionDT"].notna() & mapped_df["TransactionAmt"].notna()
                mapped_df = mapped_df.loc[keep_mask].reset_index(drop=True)
                custom_raw_df = custom_raw_df.loc[keep_mask].reset_index(drop=True)

            try:
                results = run_pipeline_and_score(mapped_df)
            except ValueError as e:
                st.error(str(e))
            else:
                display_df = pd.concat(
                    [
                        custom_raw_df.reset_index(drop=True),
                        results[["fraud_probability", "is_fraud"]].reset_index(drop=True),
                    ],
                    axis=1,
                )
                st.success(f"Done. Flagged {int(results['is_fraud'].sum()):,} of {len(results):,} as fraud.")
                st.subheader("Results")
                st.dataframe(
                    display_df.sort_values("fraud_probability", ascending=False),
                    use_container_width=True,
                )
                st.download_button(
                    "Download results as CSV",
                    data=display_df.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions_custom.csv",
                    mime="text/csv",
                )

else:
    st.info(
        "A single transaction with no history behind it doesn't give the model much to work "
        "with. Things like spending patterns and device trust are all based on past behavior, "
        "so a one-off entry gets treated like a brand-new card and device. For results that "
        "actually mean something, use the CSV upload instead."
    )
    st.subheader("2. Transaction details")
    with st.form("manual_entry"):
        cols = st.columns(3)
        values = {}
        values["TransactionID"] = 0
        with cols[0]:
            st.markdown("**Timing & amount**")
            values["TransactionDT"] = st.number_input("TransactionDT", value=86400.0, format="%.0f")
            values["TransactionAmt"] = st.number_input("TransactionAmt", value=100.0, format="%.2f")
        with cols[1]:
            st.markdown("**Card & address**")
            values["card1"] = st.number_input("card1", value=1000, format="%d")
            values["addr1"] = st.number_input("addr1", value=100, format="%d")
        with cols[2]:
            st.markdown("**Device & email**")
            values["DeviceInfo"] = st.text_input("DeviceInfo", value="Windows")
            values["P_emaildomain"] = st.text_input("P_emaildomain", value="gmail.com")
        st.write("")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row_df = pd.DataFrame([values])
        try:
            result = run_pipeline_and_score(row_df).iloc[0]
        except ValueError as e:
            st.error(str(e))
        else:
            prob = result["fraud_probability"]
            with st.container(border=True):
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric("Fraud probability", f"{prob:.2%}")
                with res_col2:
                    if result["is_fraud"]:
                        st.error(f"⚠️ Flagged as FRAUD (threshold = {threshold:.2f})")
                    else:
                        st.success(f"✅ Not flagged as fraud (threshold = {threshold:.2f})")
                st.progress(min(max(prob, 0.0), 1.0))