"""FraudShield dashboard.   Run:  streamlit run app.py   (after run_experiments.py)"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from sklearn.metrics import average_precision_score, brier_score_loss

from fraudshield.costs import APPROVE, BLOCK, DEFER, NAMES, CostModel
from fraudshield.data import add_features, validate
from fraudshield.decision import decide, evaluate_decisions, tune_thresholds
from fraudshield.database import (
    init_db, save_transaction, get_transaction_history,
    get_review_queue, complete_review,
)

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")

# Create the local SQLite database and tables if they do not exist.
init_db()


@st.cache_resource
def load_artifacts():
    art = joblib.load("artifacts/fraudshield.joblib")

    ev = np.load("artifacts/eval.npz")
    sample = pd.read_csv("artifacts/test_sample.csv")

    model = art["model"]

    if hasattr(model, "named_steps"):
        # Logistic Regression inside a preprocessing pipeline
        scaler = model.named_steps.get("scaler")
        classifier = model.steps[-1][1]

        if scaler is not None:
            background = scaler.transform(
                add_features(sample.head(100))
            )
        else:
            background = add_features(sample.head(100))

        explainer = shap.LinearExplainer(
            classifier,
            background
        )

    else:
        # Tree-based models
        explainer = shap.TreeExplainer(model)

    return art, {k: ev[k] for k in ev.files}, sample, explainer


def score_df(art, df):
    X = add_features(df)
    raw = art["model"].predict_proba(X)[:, 1]
    return X, art["calibrator"].predict(raw)



def expected_cost_batch_decisions(p, amount, cost, review_budget):
    """Choose approve/block/defer using expected costs, capped by review capacity."""
    p = np.asarray(p, dtype=float)
    amount = np.asarray(amount, dtype=float)

    approve_cost = p * amount
    block_cost = (1.0 - p) * cost.false_block_cost
    defer_cost = (
        cost.review_cost
        + (1.0 - cost.reviewer_accuracy)
        * (p * amount + (1.0 - p) * cost.false_block_cost)
    )

    best_automatic_cost = np.minimum(approve_cost, block_cost)
    decisions = np.where(approve_cost <= block_cost, APPROVE, BLOCK).astype(int)

    # Only transactions for which review has lower expected cost are eligible.
    eligible = np.flatnonzero(defer_cost < best_automatic_cost)
    capacity = int(np.floor(len(p) * float(review_budget)))

    if capacity > 0 and len(eligible) > 0:
        benefit = best_automatic_cost[eligible] - defer_cost[eligible]
        chosen = eligible[np.argsort(benefit)[::-1][:capacity]]
        decisions[chosen] = DEFER

    return decisions

def shap_contrib(explainer, X, model):
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")

        if scaler is not None:
            X_input = scaler.transform(X)
        else:
            X_input = X

        values = explainer(X_input).values

    else:
        values = explainer(X).values

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return values

if not os.path.exists("artifacts/fraudshield.joblib"):
    st.error("No trained artifacts found. Run `python run_experiments.py --data data/creditcard.csv` first.")
    st.stop()

art, ev, sample, explainer = load_artifacts()
headline = json.load(open("results/headline.json")) if os.path.exists("results/headline.json") else {}

st.sidebar.title("🛡️ FraudShield")
st.sidebar.caption("Explainable fraud triage")

st.sidebar.markdown("**DASHBOARD**")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Score a transaction",
        "Batch triage",
        "Transaction History",
        "Human Review Queue",
        "Cost simulator",
        "Performance Analytics",
        "Reports",
    ],
    label_visibility="collapsed",
)
if headline.get("synthetic"):
    st.sidebar.warning("Artifacts were built from SYNTHETIC data (smoke test).")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("FraudShield")
    st.caption("Explainable, cost-aware fraud triage with uncertainty-based deferral to human review")
    st.markdown(
        "A calibrated fraud probability is turned into one of three actions: **approve** if it is below "
        "`t_low`, **block** if it is at or above `t_high`, and **defer to a human analyst** in between. "
        "Thresholds are chosen to minimise expected cost under a review-capacity budget, and every score "
        "comes with a SHAP explanation.")
    c = st.columns(4)
    sp = headline.get("split", {})
    c[0].metric("Clean transactions", f"{headline.get('rows_clean', 0):,}")
    c[1].metric("Quarantined by gate", f"{headline.get('rows_quarantined', 0):,}")
    c[2].metric("Test-set frauds", sp.get("test", {}).get("frauds", "-"))
    c[3].metric("Review budget", f"{art['review_budget']:.0%}")
    st.subheader("Learned decision thresholds")
    st.write(f"approve below **{art['t_low']:.4f}**, block at or above **{art['t_high']:.4f}**, defer in between "
             "(tuned on a held-out, time-ordered tuning split).")
    st.info("Costs used here are assumptions: " + ", ".join(f"{k}={v}" for k, v in art["cost"].items()))
    if os.path.exists("results/figures/pr_curves.png"):
        c1, c2 = st.columns(2)
        c1.image("results/figures/pr_curves.png")
        c2.image("results/figures/deferral_tradeoff.png")

# ------------------------------------------------------------------ Score one
elif page == "Score a transaction":
    st.title("Score a transaction")
    kind = st.radio("Pick an example from the held-out test sample",
                    ["Random", "A known fraud", "A known legitimate transaction"], horizontal=True)
    pool = sample if kind == "Random" else sample[sample.Class == (1 if "fraud" in kind else 0)]
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    if st.button("Draw another"):
        st.session_state.seed += 1
    row = pool.sample(1, random_state=st.session_state.seed + 7)
    X, p = score_df(art, row)
    d = int(decide(p, art["t_low"], art["t_high"])[0])

    # Save this demo transaction once per selected sample, not on every Streamlit rerun.
    save_key = f"saved_demo_{kind}_{st.session_state.seed}"
    if save_key not in st.session_state:
        saved_id = save_transaction(
            amount=float(row["Amount"].iloc[0]),
            fraud_probability=float(p[0]),
            decision=NAMES[d],
            transaction_data=row.drop(columns=["Class"], errors="ignore").iloc[0].to_dict(),
            source="held_out_test_demo",
            true_label=int(row["Class"].iloc[0]),
        )
        st.session_state[save_key] = saved_id
        st.success(f"Transaction saved to history (ID: {saved_id}).")
    else:
        st.caption(f"Saved transaction ID: {st.session_state[save_key]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud probability (calibrated)", f"{p[0]:.4f}")
    c2.metric("Decision", NAMES[d])
    c3.metric("Amount", f"{row['Amount'].iloc[0]:,.2f}")
    c4.metric("True label (for the demo)", "FRAUD" if row["Class"].iloc[0] == 1 else "legitimate")
    {APPROVE: st.success, BLOCK: st.error, DEFER: st.warning}[d](
        {APPROVE: "Auto-approved: probability below the lower threshold.",
         BLOCK: "Auto-blocked: probability above the upper threshold.",
         DEFER: "Sent to a human analyst: the model is not sure enough to decide automatically."}[d])
    sv = shap_contrib(explainer, X, art["model"])[0]
    top = pd.DataFrame({"feature": X.columns, "contribution": sv, "value": X.iloc[0].values})
    top = top.reindex(top.contribution.abs().sort_values(ascending=False).index).head(10)
    fig = px.bar(top[::-1], x="contribution", y="feature", orientation="h", color="contribution",
                 color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                 title="Top 10 SHAP contributions (log-odds of the raw model; red pushes toward fraud)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("V1-V28 are PCA-anonymised in the public dataset, so these show which components drove the "
               "score, not business reasons.")

# ------------------------------------------------------------------ Batch
elif page == "Batch triage":
    st.title("Batch triage")
    up = st.file_uploader("Upload a CSV with columns Time, Amount, V1..V28 (Class optional)", type="csv")
    use_demo = st.checkbox("Use a demo batch from the test sample instead", value=up is None)
    raw_df = pd.read_csv(up) if up is not None and not use_demo else sample.drop(columns=["Class"]).head(2000)
    try:
        clean, bad = validate(raw_df, with_label=False)
    except ValueError as e:
        st.error(str(e)); st.stop()
    X, p = score_df(art, clean)

    # Batch triage uses the expected-cost policy and respects the configured
    # maximum human-review capacity. No test labels are used for these decisions.
    batch_cost = CostModel(**art["cost"])
    batch_decisions = expected_cost_batch_decisions(
        p,
        clean["Amount"].to_numpy(),
        batch_cost,
        art["review_budget"],
    )
    clean = clean.assign(
        fraud_probability=p,
        decision=[NAMES[i] for i in batch_decisions],
    )

    st.caption("Scoring a batch does not save it automatically. Use the button below to store these results in SQLite.")
    if st.button("Save this batch to transaction history", type="primary"):
        saved_count = 0
        for _, transaction in clean.iterrows():
            transaction_data = transaction.drop(
                labels=["fraud_probability", "decision"], errors="ignore"
            ).to_dict()
            save_transaction(
                amount=float(transaction["Amount"]),
                fraud_probability=float(transaction["fraud_probability"]),
                decision=str(transaction["decision"]),
                transaction_data=transaction_data,
                source="batch_upload" if up is not None and not use_demo else "demo_batch",
                true_label=None,
            )
            saved_count += 1
        st.success(f"Saved {saved_count:,} transactions to the database.")
    c = st.columns(4)
    c[0].metric("Scored", len(clean)); c[1].metric("Auto-approved", int((clean.decision == "approve").sum()))
    c[2].metric("Auto-blocked", int((clean.decision == "block").sum()))
    c[3].metric("Review queue", int((clean.decision == "defer to review").sum()))
    st.subheader("Review queue (highest risk first)")
    q = clean[clean.decision == "defer to review"].sort_values("fraud_probability", ascending=False)
    st.dataframe(q[["Time", "Amount", "fraud_probability", "decision"]], use_container_width=True)
    st.download_button("Download all decisions", clean.to_csv(index=False).encode(), "fraudshield_decisions.csv")
    if len(bad):
        st.subheader(f"Quarantined by the validation gate ({len(bad)})")
        st.dataframe(bad[["reason"] + [c for c in bad.columns if c != "reason"]], use_container_width=True)

# ------------------------------------------------------------------ Transaction history
elif page == "Transaction History":
    st.title("Transaction History")
    st.caption("Previously saved predictions are stored locally in data/fraudshield.db.")

    history = get_transaction_history(limit=1000)
    if history.empty:
        st.info("No transactions have been saved yet. Score a transaction or save a batch first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved transactions", f"{len(history):,}")
        c2.metric("Pending reviews", int((history["review_status"] == "pending").sum()))
        c3.metric("Completed reviews", int((history["review_status"] == "completed").sum()))

        decision_filter = st.selectbox(
            "Filter by model decision",
            ["All", "approve", "block", "defer to review"],
        )
        view = history.copy()
        view["created_at"] = (
            pd.to_datetime(view["created_at"], utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.strftime("%d-%m-%Y %I:%M:%S %p")
        )
        if decision_filter != "All":
            view = view[view["decision"] == decision_filter]

        st.dataframe(
            view[[
                "id", "created_at", "source", "amount", "fraud_probability",
                "decision", "review_status", "reviewer_decision"
            ]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download transaction history",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_transaction_history.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------ Human review queue
elif page == "Human Review Queue":
    st.title("Human Review Queue")
    st.caption("Review deferred transactions and record a human decision.")

    queue = get_review_queue()
    if queue.empty:
        st.success("There are no transactions waiting for review.")
    else:
        st.metric("Pending transactions", len(queue))
        display_queue = queue[["id", "created_at", "amount", "fraud_probability"]].copy()
        display_queue["created_at"] = (
            pd.to_datetime(display_queue["created_at"], utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.strftime("%d-%m-%Y %I:%M:%S %p")
        )
        st.dataframe(display_queue, use_container_width=True, hide_index=True)

        selected_id = st.selectbox(
            "Choose a transaction to review",
            queue["id"].astype(int).tolist(),
            format_func=lambda value: f"Transaction #{value}",
        )
        selected = queue[queue["id"] == selected_id].iloc[0]
        st.subheader(f"Transaction #{selected_id}")
        st.write(f"**Amount:** {selected['amount']:,.2f}")
        st.write(f"**Calibrated fraud probability:** {selected['fraud_probability']:.4f}")

        try:
            transaction_details = json.loads(selected["transaction_data"])
            with st.expander("View transaction input features"):
                st.json(transaction_details)

            # Recreate the model input for this saved transaction and explain
            # the same calibrated prediction shown in the review queue.
            st.subheader("Why was this transaction flagged?")
            review_features = pd.DataFrame([transaction_details])
            review_X = add_features(review_features)
            review_shap_values = shap_contrib(
                explainer, review_X, art["model"]
            )[0]

            explanation = pd.DataFrame({
                "feature": review_X.columns,
                "contribution": review_shap_values,
                "value": review_X.iloc[0].to_numpy(),
            })
            explanation = (
                explanation.assign(
                    absolute_contribution=lambda frame: frame["contribution"].abs()
                )
                .sort_values("absolute_contribution", ascending=False)
                .head(10)
            )

            explanation_plot = px.bar(
                explanation.sort_values("contribution"),
                x="contribution",
                y="feature",
                orientation="h",
                color="contribution",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                hover_data={"value": True, "absolute_contribution": False},
                title="Top 10 SHAP feature contributions",
                labels={
                    "contribution": "SHAP contribution (raw-model log-odds)",
                    "feature": "Input feature",
                    "value": "Feature value",
                },
            )
            st.plotly_chart(explanation_plot, use_container_width=True)
            st.caption(
                "Positive contributions push the raw model score toward fraud; "
                "negative contributions push it toward legitimate. These are "
                "model feature contributions, not proof that a transaction is fraudulent. "
                "V1-V28 are anonymised PCA components in this dataset."
            )
        except (TypeError, json.JSONDecodeError, ValueError, KeyError) as error:
            st.warning(
                "The transaction details could not be converted into a SHAP explanation: "
                f"{error}"
            )
        except Exception as error:
            st.warning(f"Could not generate the SHAP explanation: {error}")

        with st.form("complete_review_form"):
            human_decision = st.radio(
                "Human review outcome",
                ["fraud", "legitimate"],
                horizontal=True,
            )
            review_notes = st.text_area("Review notes (optional)")
            submitted = st.form_submit_button("Save review decision", type="primary")

        if submitted:
            try:
                complete_review(
                    transaction_id=int(selected_id),
                    reviewer_decision=human_decision,
                    review_notes=review_notes,
                )
                st.success(f"Review for transaction #{selected_id} saved.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))

# ------------------------------------------------------------------ Simulator
elif page == "Cost simulator":
    st.title("Cost simulator")
    st.caption("Change the assumed costs and review capacity; thresholds are re-tuned on the tuning split and "
               "evaluated on the held-out test split.")
    c1, c2, c3, c4 = st.columns(4)
    fb = c1.slider("Cost of a false block", 1.0, 200.0, float(art["cost"]["false_block_cost"]))
    rc = c2.slider("Cost per human review", 0.0, 50.0, float(art["cost"]["review_cost"]))
    ra = c3.slider("Reviewer accuracy", 0.5, 1.0, float(art["cost"]["reviewer_accuracy"]))
    bud = c4.slider("Review budget (% of transactions)", 0.0, 10.0, 2.0, 0.25) / 100
    cm = CostModel(fb, rc, ra)
    t = tune_thresholds(ev["p_tune"], ev["y_tune"], ev["a_tune"], cm, bud, allow_defer=bud > 0)
    t0 = tune_thresholds(ev["p_tune"], ev["y_tune"], ev["a_tune"], cm, 0.0, allow_defer=False)
    m = evaluate_decisions(decide(ev["p_test"], t.t_low, t.t_high), ev["y_test"], ev["a_test"], cm)
    m0 = evaluate_decisions(decide(ev["p_test"], t0.t_low, t0.t_high), ev["y_test"], ev["a_test"], cm)
    plain = evaluate_decisions(np.where(ev["p_test"] >= 0.5, BLOCK, APPROVE), ev["y_test"], ev["a_test"], cm)
    c = st.columns(4)
    c[0].metric("t_low / t_high", f"{t.t_low:.3f} / {t.t_high:.3f}")
    c[1].metric("Total cost (with deferral)", f"{m['total_cost']:,.0f}", f"{m['total_cost'] - m0['total_cost']:,.0f} vs no deferral",
                delta_color="inverse")
    c[2].metric("Silent miss rate", f"{m['silent_miss_rate']:.1%}")
    c[3].metric("Share sent to review", f"{m['defer_rate']:.2%}")
    df_cmp = pd.DataFrame({"Plain threshold 0.5": plain, "Cost-tuned, no deferral": m0, "FraudShield (with deferral)": m}).T
    st.dataframe(df_cmp[["total_cost", "silent_miss_rate", "auto_block_precision", "defer_rate", "n_false_blocks"]]
                 .style.format({"total_cost": "{:,.0f}", "silent_miss_rate": "{:.1%}", "auto_block_precision": "{:.1%}",
                                "defer_rate": "{:.2%}"}), use_container_width=True)
    st.caption("Results are conditional on these assumed costs. Only the test split's results are shown.")

# ------------------------------------------------------------------ Performance Analytics
elif page == "Performance Analytics":
    st.title("Performance Analytics")
    st.caption(
        "Offline evaluation of the trained model on the held-out chronological test set. "
        "These metrics describe this dataset, not live production traffic."
    )

    p_test = np.asarray(ev["p_test"], dtype=float)
    y_test = np.asarray(ev["y_test"], dtype=int)
    a_test = np.asarray(ev["a_test"], dtype=float)

    # Evaluate the deployed threshold policy using the saved cost assumptions.
    analytics_cost = CostModel(**art["cost"])
    threshold_decisions = decide(p_test, art["t_low"], art["t_high"])
    threshold_metrics = evaluate_decisions(
        threshold_decisions, y_test, a_test, analytics_cost
    )
    pr_auc = float(average_precision_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))

    # Also show the expected-cost policy used by batch triage.
    expected_decisions = expected_cost_batch_decisions(
        p_test, a_test, analytics_cost, art["review_budget"]
    )
    expected_metrics = evaluate_decisions(
        expected_decisions, y_test, a_test, analytics_cost
    )

    st.subheader("Model performance")
    c = st.columns(4)
    c[0].metric("Test PR-AUC", f"{pr_auc:.4f}")
    c[1].metric("Brier score", f"{brier:.4f}")
    c[2].metric("Frauds in test set", f"{int(y_test.sum()):,}")
    c[3].metric("Test transactions", f"{len(y_test):,}")

    st.caption(
        "PR-AUC measures ranking quality under class imbalance. Brier score measures "
        "probability accuracy (lower is better)."
    )

    st.subheader("Decision policy performance")
    policy_table = pd.DataFrame(
        {
            "Threshold policy": threshold_metrics,
            "Expected-cost batch policy": expected_metrics,
        }
    ).T
    visible_metrics = [
        name for name in [
            "total_cost", "silent_miss_rate", "fraud_blocked_rate",
            "fraud_deferred_rate", "auto_block_precision", "defer_rate",
            "n_false_blocks",
        ]
        if name in policy_table.columns
    ]
    st.dataframe(policy_table[visible_metrics].round(4), use_container_width=True)

    left, right = st.columns(2)
    with left:
        decision_counts = pd.DataFrame({
            "Decision": ["Approve", "Block", "Defer to review"],
            "Transactions": [
                int(np.sum(expected_decisions == APPROVE)),
                int(np.sum(expected_decisions == BLOCK)),
                int(np.sum(expected_decisions == DEFER)),
            ],
        })
        decision_fig = px.pie(
            decision_counts,
            names="Decision",
            values="Transactions",
            title="Expected-cost policy: test-set decisions",
            hole=0.45,
        )
        st.plotly_chart(decision_fig, use_container_width=True)

    with right:
        probability_frame = pd.DataFrame({
            "Fraud probability": p_test,
            "Actual class": np.where(y_test == 1, "Fraud", "Legitimate"),
        })
        probability_fig = px.histogram(
            probability_frame,
            x="Fraud probability",
            color="Actual class",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            title="Predicted fraud probability distribution",
        )
        st.plotly_chart(probability_fig, use_container_width=True)

    if os.path.exists("results/figures/calibration.png"):
        st.subheader("Calibration")
        st.image("results/figures/calibration.png", use_container_width=True)

    if os.path.exists("results/summary.csv"):
        with st.expander("Research model comparison (full experiment results)"):
            summary = pd.read_csv("results/summary.csv", index_col=0)
            summary_columns = [
                col for col in [
                    "pr_auc_mean", "pr_auc_std", "total_cost_mean", "total_cost_std",
                    "silent_miss_rate_mean", "defer_rate_mean",
                ]
                if col in summary.columns
            ]
            if summary_columns:
                st.dataframe(summary[summary_columns].round(4), use_container_width=True)
            else:
                st.info("The experiment summary does not contain the expected metric columns.")

    st.info(
        "Cost results depend on the configured false-block cost, review cost and reviewer "
        "accuracy. These are modelling assumptions, not measured real-world financial savings."
    )


# ------------------------------------------------------------------ Reports
elif page == "Reports":
    st.title("Reports")
    st.caption(
        "Export transaction activity and a concise summary of the current FraudShield evaluation."
    )

    history = get_transaction_history(limit=100000)
    if history.empty:
        st.info("No saved transactions yet. Score a transaction or save a batch first.")
    else:
        history_export = history.copy()
        if "created_at" in history_export.columns:
            history_export["created_at"] = (
                pd.to_datetime(history_export["created_at"], utc=True)
                .dt.tz_convert("Asia/Kolkata")
                .dt.strftime("%d-%m-%Y %I:%M:%S %p")
            )

        total_saved = len(history_export)
        pending_count = int((history_export["review_status"] == "pending").sum())
        completed_count = int((history_export["review_status"] == "completed").sum())
        deferred_count = int((history_export["decision"] == "defer to review").sum())
        fraud_review_count = int((history_export["reviewer_decision"] == "fraud").sum())
        legitimate_review_count = int((history_export["reviewer_decision"] == "legitimate").sum())

        c = st.columns(4)
        c[0].metric("Saved transactions", f"{total_saved:,}")
        c[1].metric("Deferred", f"{deferred_count:,}")
        c[2].metric("Pending reviews", f"{pending_count:,}")
        c[3].metric("Completed reviews", f"{completed_count:,}")

        st.subheader("Human review outcomes")
        outcomes = pd.DataFrame({
            "Outcome": ["Marked fraud", "Marked legitimate", "Still pending"],
            "Transactions": [
                fraud_review_count,
                legitimate_review_count,
                pending_count,
            ],
        })
        outcome_fig = px.bar(
            outcomes,
            x="Outcome",
            y="Transactions",
            title="Recorded review outcomes",
            text="Transactions",
        )
        st.plotly_chart(outcome_fig, use_container_width=True)

        st.download_button(
            "Download transaction report (CSV)",
            data=history_export.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_transaction_report.csv",
            mime="text/csv",
            type="primary",
        )

    st.divider()
    st.subheader("Evaluation summary")
    st.caption("Evaluation metrics from the held-out test set.")
    p_test = np.asarray(ev["p_test"], dtype=float)
    y_test = np.asarray(ev["y_test"], dtype=int)
    a_test = np.asarray(ev["a_test"], dtype=float)
    report_cost = CostModel(**art["cost"])
    report_decisions = expected_cost_batch_decisions(
        p_test, a_test, report_cost, art["review_budget"]
    )
    report_metrics = evaluate_decisions(
        report_decisions, y_test, a_test, report_cost
    )
    report_lines = [
        "FraudShield - Evaluation Summary",
        "================================",
        f"Test transactions: {len(y_test):,}",
        f"Fraud cases in test set: {int(y_test.sum()):,}",
        f"Review budget: {art['review_budget']:.2%}",
        f"False-block cost assumption: {art['cost']['false_block_cost']}",
        f"Review cost assumption: {art['cost']['review_cost']}",
        f"Reviewer accuracy assumption: {art['cost']['reviewer_accuracy']:.2%}",
        "",
        "Expected-cost batch policy (held-out test set)",
        f"Total modelled cost: {report_metrics['total_cost']:.2f}",
        f"Silent miss rate: {report_metrics['silent_miss_rate']:.2%}",
        f"Fraud blocked rate: {report_metrics['fraud_blocked_rate']:.2%}",
        f"Fraud deferred rate: {report_metrics['fraud_deferred_rate']:.2%}",
        f"Deferral rate: {report_metrics['defer_rate']:.2%}",
        "",
        "These are offline test-set results under hypothetical cost assumptions,",
        "not measured real-world financial savings or a live-system guarantee.",
    ]
    st.download_button(
        "Download evaluation summary (TXT)",
        data="\n".join(report_lines),
        file_name="fraudshield_evaluation_summary.txt",
        mime="text/plain",
    )
