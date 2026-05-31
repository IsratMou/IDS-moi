import shap
import gradio as gr
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
import json
import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Paths
# ============================================================
ARTIFACTS_DIR = "."

# ============================================================
# Load metadata
# ============================================================
print("Loading metadata...")

with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "r") as f:
    feature_meta = json.load(f)

NUMERIC_COLS = feature_meta["numeric_columns"]
OHE_COLS = feature_meta["ohe_columns"]
N_FEATURES = feature_meta["n_features"]

with open(os.path.join(ARTIFACTS_DIR, "preprocessing_metadata.json"), "r") as f:
    prep_meta = json.load(f)

CLASS_NAMES = prep_meta["class_names"]
N_CLASSES = len(CLASS_NAMES)
NORMAL_CLASS_ID = CLASS_NAMES.index("normal")
MINORITY_CLASS_NAMES = prep_meta["minority_class_names"]

attack_ids_orig = sorted([i for i in range(N_CLASSES) if i != NORMAL_CLASS_ID])
ATTACK_REMAP_INV = {new_id: orig_id for new_id,
                    orig_id in enumerate(attack_ids_orig)}
ATTACK_CLASS_NAMES = [CLASS_NAMES[orig_id] for orig_id in attack_ids_orig]
N_ATTACK_CLASSES = len(ATTACK_CLASS_NAMES)

with open(os.path.join(ARTIFACTS_DIR, "hierarchical_results.json"), "r") as f:
    hier_results = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

# ============================================================
# Categorical groups
# ============================================================
CATEGORICAL_GROUPS = {}
KNOWN_CAT_PREFIXES = ["proto", "service", "conn_state",
                      "dns_AA", "dns_RD", "dns_RA", "dns_rejected"]

for ohe_col in OHE_COLS:
    for prefix in KNOWN_CAT_PREFIXES:
        if ohe_col.startswith(prefix + "_"):
            value = ohe_col[len(prefix)+1:]
            if prefix not in CATEGORICAL_GROUPS:
                CATEGORICAL_GROUPS[prefix] = []
            CATEGORICAL_GROUPS[prefix].append(value)
            break

NUMERIC_INDICES = np.arange(len(NUMERIC_COLS), dtype=np.int32)
CATEGORICAL_INDICES = np.arange(len(NUMERIC_COLS), N_FEATURES, dtype=np.int32)

# ============================================================
# Custom layers (MOI‑Lite v4 only)
# ============================================================


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        batch_size = tf.shape(x)[0]
        rank = len(x.shape)
        shape = [batch_size] + [1] * (rank - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        binary_mask = tf.floor(random_tensor)
        return (x / keep_prob) * binary_mask

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class IndexSlice(tf.keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = tf.constant(indices, dtype=tf.int32)

    def call(self, x):
        return tf.gather(x, self.indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices.numpy().tolist()})
        return config


# ============================================================
# Architecture definitions (matching training)
# ============================================================

# ----- DNN (SATF) -----
def build_dnn_satf(input_dim, n_classes, binary=False):
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.GaussianNoise(0.05, name="satf_noise")(inputs)
    hidden_dims = (256, 128, 64)
    dropout_rate = 0.3
    for i, units in enumerate(hidden_dims, start=1):
        x = tf.keras.layers.Dense(units, use_bias=False, name=f"dense_{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.layers.Activation("swish", name=f"act_{i}")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}")(x)
    if binary:
        outputs = tf.keras.layers.Dense(
            1, activation="sigmoid", name="output")(x)
    else:
        outputs = tf.keras.layers.Dense(
            n_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="dnn_binary_satf" if binary else "dnn_multiclass_satf")

# ----- CNN (SATF) -----


def build_cnn_satf(input_dim, n_classes, binary=False):
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.GaussianNoise(0.05, name="satf_noise")(inputs)
    x = tf.keras.layers.Reshape((input_dim, 1), name="reshape")(x)
    filters = (64, 128, 64)
    kernel_sizes = (3, 5, 3)
    dropout_rate = 0.3
    for i, (f, k) in enumerate(zip(filters, kernel_sizes), start=1):
        x = tf.keras.layers.Conv1D(
            f, k, padding="same", use_bias=False, name=f"conv_{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.layers.Activation("swish", name=f"act_{i}")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(64, use_bias=False, name="head_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Activation("swish", name="head_act")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_drop")(x)
    if binary:
        outputs = tf.keras.layers.Dense(
            1, activation="sigmoid", name="output")(x)
    else:
        outputs = tf.keras.layers.Dense(
            n_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_binary_satf" if binary else "cnn_multiclass_satf")

# ----- MOI‑Lite v4 (SATF) -----


def light_multiscale_block(x, filters, dilation_rates, l2_reg, name_prefix):
    branches = []
    for d in dilation_rates:
        branch = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=3, dilation_rate=d,
            padding="same", use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
            name=f"{name_prefix}_conv_d{d}",
        )(x)
        branch = tf.keras.layers.BatchNormalization(
            name=f"{name_prefix}_bn_d{d}")(branch)
        branch = tf.keras.layers.ReLU(
            max_value=6.0, name=f"{name_prefix}_relu6_d{d}")(branch)
        branches.append(branch)
    return tf.keras.layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")(branches)


def squeeze_attention(x, reduction, name_prefix):
    channels = x.shape[-1]
    reduced = max(channels // reduction, 4)
    gap = tf.keras.layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    fc1 = tf.keras.layers.Dense(
        reduced, use_bias=False, name=f"{name_prefix}_fc1")(gap)
    fc1 = tf.keras.layers.ReLU(max_value=6.0, name=f"{name_prefix}_relu6")(fc1)
    fc2 = tf.keras.layers.Dense(channels, activation="sigmoid", use_bias=False,
                                name=f"{name_prefix}_fc2")(fc1)
    scale = tf.keras.layers.Reshape(
        (1, channels), name=f"{name_prefix}_reshape")(fc2)
    return tf.keras.layers.Multiply(name=f"{name_prefix}_scale")([x, scale])


def gated_residual_v4(x, filters, drop_path_rate, l2_reg, name_prefix):
    h = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=3, padding="same", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name=f"{name_prefix}_conv"
    )(x)
    h = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(h)
    h = tf.keras.layers.ReLU(max_value=6.0, name=f"{name_prefix}_relu6")(h)

    gate = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=1, padding="same", activation="sigmoid",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name=f"{name_prefix}_gate"
    )(x)

    gated = tf.keras.layers.Multiply(name=f"{name_prefix}_gated")([h, gate])
    gated = DropPath(drop_prob=drop_path_rate,
                     name=f"{name_prefix}_droppath")(gated)

    if x.shape[-1] != filters:
        x = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False,
                                   name=f"{name_prefix}_proj")(x)
    return tf.keras.layers.Add(name=f"{name_prefix}_add")([gated, x])


def build_moi_lite_v4_satf(input_dim, n_classes, binary=False):
    use_satf = True
    satf_noise = 0.05
    num_dim = 48
    cat_dim = 32
    fused_dim = 64
    base_filters = 24
    dilation_rates = (1, 2, 4)
    drop_path_rate = 0.1
    l2_reg = 1e-5
    dropout_rate = 0.25

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs

    if use_satf:
        x = tf.keras.layers.GaussianNoise(satf_noise, name="satf_noise")(x)

    num_features = IndexSlice(NUMERIC_INDICES, name="num_slice")(x)
    cat_features = IndexSlice(CATEGORICAL_INDICES, name="cat_slice")(x)

    num_stream = tf.keras.layers.Dense(num_dim, use_bias=False,
                                       kernel_regularizer=tf.keras.regularizers.l2(
                                           l2_reg),
                                       name="num_dense")(num_features)
    num_stream = tf.keras.layers.BatchNormalization(name="num_bn")(num_stream)
    num_stream = tf.keras.layers.ReLU(
        max_value=6.0, name="num_relu6")(num_stream)
    num_stream = tf.keras.layers.Dropout(0.1, name="num_drop")(num_stream)

    cat_stream = tf.keras.layers.Dense(cat_dim, use_bias=False,
                                       kernel_regularizer=tf.keras.regularizers.l2(
                                           l2_reg),
                                       name="cat_dense")(cat_features)
    cat_stream = tf.keras.layers.BatchNormalization(name="cat_bn")(cat_stream)
    cat_stream = tf.keras.layers.ReLU(
        max_value=6.0, name="cat_relu6")(cat_stream)
    cat_stream = tf.keras.layers.Dropout(0.1, name="cat_drop")(cat_stream)

    fused = tf.keras.layers.Concatenate(
        name="stream_concat")([num_stream, cat_stream])
    fused = tf.keras.layers.Dense(fused_dim, use_bias=False,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      l2_reg),
                                  name="fused_dense")(fused)
    fused = tf.keras.layers.BatchNormalization(name="fused_bn")(fused)
    fused = tf.keras.layers.ReLU(max_value=6.0, name="fused_relu6")(fused)

    x = tf.keras.layers.Reshape((fused_dim, 1), name="reshape")(fused)

    x = light_multiscale_block(x, base_filters, dilation_rates, l2_reg, "ms")
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_ms")(x)

    x = squeeze_attention(x, reduction=4, name_prefix="sa")

    ms_out_filters = base_filters * len(dilation_rates)
    x = gated_residual_v4(x, ms_out_filters, drop_path_rate, l2_reg, "gr")
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_gr")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    x = tf.keras.layers.Dense(64, use_bias=False,
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  l2_reg),
                              name="head_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="head_relu6")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_drop")(x)

    if binary:
        outputs = tf.keras.layers.Dense(
            1, activation="sigmoid", name="output")(x)
        model_name = "moi_lite_v4_binary_satf"
    else:
        outputs = tf.keras.layers.Dense(
            n_classes, activation="softmax", name="output")(x)
        model_name = "moi_lite_v4_multiclass_satf"

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


# ============================================================
# Pre‑load all three pipelines
# ============================================================
print("Loading models...")

pipelines = {}

# ----- DNN -----
s1_dnn = build_dnn_satf(N_FEATURES, N_CLASSES, binary=True)
s1_dnn.load_weights(os.path.join(
    ARTIFACTS_DIR, "dnn_binary_satf_best.weights.h5"))
s2_dnn = build_dnn_satf(N_FEATURES, N_ATTACK_CLASSES, binary=False)
s2_dnn.load_weights(os.path.join(
    ARTIFACTS_DIR, "dnn_multiclass_satf_best.weights.h5"))
threshold_dnn = hier_results.get(
    "S1[dnn_binary_satf]+S2[dnn_multiclass_satf", {}).get("stage1_threshold", 0.5732)
pipelines["DNN"] = {"s1": s1_dnn, "s2": s2_dnn, "threshold": threshold_dnn}

# ----- CNN -----
s1_cnn = build_cnn_satf(N_FEATURES, N_CLASSES, binary=True)
s1_cnn.load_weights(os.path.join(
    ARTIFACTS_DIR, "cnn_binary_satf_best.weights.h5"))
s2_cnn = build_cnn_satf(N_FEATURES, N_ATTACK_CLASSES, binary=False)
s2_cnn.load_weights(os.path.join(
    ARTIFACTS_DIR, "cnn_multiclass_satf_best.weights.h5"))
threshold_cnn = hier_results.get(
    "S1[cnn_binary_satf]+S2[cnn_multiclass_satf", {}).get("stage1_threshold", 0.4911)
pipelines["CNN"] = {"s1": s1_cnn, "s2": s2_cnn, "threshold": threshold_cnn}

# ----- MOI-Lite v4 -----
s1_moi = build_moi_lite_v4_satf(N_FEATURES, N_CLASSES, binary=True)
s1_moi.load_weights(os.path.join(
    ARTIFACTS_DIR, "moi_lite_binary_satf_best.weights.h5"))
s2_moi = build_moi_lite_v4_satf(N_FEATURES, N_ATTACK_CLASSES, binary=False)
s2_moi.load_weights(os.path.join(
    ARTIFACTS_DIR, "moi_lite_multiclass_satf_best.weights.h5"))
threshold_moi = hier_results.get(
    "S1[moi_lite_binary_satf]+S2[moi_lite_multiclass_satf", {}).get("stage1_threshold", 0.5742)
pipelines["MOI-Lite v4"] = {"s1": s1_moi,
                            "s2": s2_moi, "threshold": threshold_moi}

print("✅ All models loaded successfully!")

# ============================================================
# Load test data and sample database
# ============================================================
print("Loading test data...")
X_test_scaled = np.load(os.path.join(ARTIFACTS_DIR, "X_test.npy"))
y_test = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))
df_test_eng = pd.read_parquet(os.path.join(
    ARTIFACTS_DIR, "test_engineered.parquet"))

SAMPLES_PER_CLASS = 5
SAMPLE_DATABASE = {}

for class_name in CLASS_NAMES:
    cls_id = CLASS_NAMES.index(class_name)
    cls_indices = np.where(y_test == cls_id)[0]
    n_to_take = min(SAMPLES_PER_CLASS, len(cls_indices))
    if n_to_take == 0:
        SAMPLE_DATABASE[class_name] = []
        continue
    selected_idx = cls_indices[:n_to_take]
    samples = []
    for idx in selected_idx:
        raw_row = df_test_eng.iloc[idx].to_dict()
        sample = {
            "test_index": int(idx),
            "true_class": class_name,
            "raw_features": {k: v for k, v in raw_row.items()
                             if k not in ["type", "label", "target"]},
        }
        samples.append(sample)
    SAMPLE_DATABASE[class_name] = samples

# ============================================================
# SHAP explainer (Stage‑2 only, using MOI‑Lite as example)
# ============================================================
print("Setting up SHAP explainer...")
X_train_scaled = np.load(os.path.join(ARTIFACTS_DIR, "X_train.npy"))
SHAP_BG_SIZE = 100
np.random.seed(SEED)
bg_indices = np.random.choice(
    len(X_train_scaled), size=SHAP_BG_SIZE, replace=False)
SHAP_BACKGROUND = X_train_scaled[bg_indices].astype(np.float32)
SHAP_EXPLAINER_S2 = shap.GradientExplainer(
    s2_moi, SHAP_BACKGROUND)  # use MOI for SHAP by default

# ============================================================
# Helper functions (common)
# ============================================================


def build_feature_vector(numeric_values, categorical_values):
    full_features = {}
    for col in NUMERIC_COLS:
        full_features[col] = float(numeric_values.get(col, 0.0))
    for ohe_col in OHE_COLS:
        full_features[ohe_col] = 0.0
    for cat_name, selected_value in categorical_values.items():
        ohe_col = f"{cat_name}_{selected_value}"
        if ohe_col in OHE_COLS:
            full_features[ohe_col] = 1.0
    feature_vector = np.array(
        [full_features[col] for col in (NUMERIC_COLS + OHE_COLS)],
        dtype=np.float32
    ).reshape(1, -1)
    return SCALER.transform(feature_vector).astype(np.float32)


def predict_hierarchical(feature_vector_scaled, pipeline_name):
    pipe = pipelines[pipeline_name]
    s1_model = pipe["s1"]
    s2_model = pipe["s2"]
    threshold = pipe["threshold"]

    s1_proba = float(s1_model.predict(
        feature_vector_scaled, verbose=0).reshape(-1)[0])
    is_attack = s1_proba >= threshold

    s2_proba_dist = s2_model.predict(
        feature_vector_scaled, verbose=0).reshape(-1)
    s2_class_local = int(np.argmax(s2_proba_dist))
    s2_class_orig = ATTACK_REMAP_INV[s2_class_local]
    s2_class_name = CLASS_NAMES[s2_class_orig]
    s2_confidence = float(s2_proba_dist[s2_class_local])

    if is_attack:
        final_class_id = s2_class_orig
        final_class_name = s2_class_name
        final_confidence = s2_confidence
    else:
        final_class_id = NORMAL_CLASS_ID
        final_class_name = "normal"
        final_confidence = 1.0 - s1_proba

    return {
        "stage1_attack_proba": s1_proba,
        "stage1_threshold": threshold,
        "stage1_decision": "ATTACK" if is_attack else "NORMAL",
        "stage2_attack_class": s2_class_name,
        "stage2_confidence": s2_confidence,
        "stage2_full_distribution": {
            ATTACK_CLASS_NAMES[i]: float(s2_proba_dist[i])
            for i in range(N_ATTACK_CLASSES)
        },
        "final_class_id": int(final_class_id),
        "final_class_name": final_class_name,
        "final_confidence": float(final_confidence),
    }


def get_sample_data(class_name, sample_id):
    if class_name not in SAMPLE_DATABASE:
        return None
    samples = SAMPLE_DATABASE[class_name]
    if sample_id >= len(samples):
        return None
    return samples[sample_id]


def split_sample_into_inputs(sample_dict):
    raw = sample_dict["raw_features"]
    numeric_dict = {col: float(raw.get(col, 0.0)) for col in NUMERIC_COLS}
    categorical_dict = {}
    for cat_name, possible_values in CATEGORICAL_GROUPS.items():
        if cat_name in raw:
            categorical_dict[cat_name] = str(raw[cat_name])
        else:
            categorical_dict[cat_name] = possible_values[0]
    return numeric_dict, categorical_dict


def compute_sample_shap(feature_vector_scaled, predicted_class_local):
    # SHAP uses MOI model; for other pipelines we still show the same SHAP as reference
    shap_values = SHAP_EXPLAINER_S2.shap_values(feature_vector_scaled)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    if shap_values.ndim == 3:
        sample_shap = shap_values[0, :, predicted_class_local]
    else:
        sample_shap = shap_values[0]
    all_features = NUMERIC_COLS + OHE_COLS
    feature_shap_pairs = list(zip(all_features, sample_shap))
    feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_shap_pairs


def format_prediction_html(prediction, true_class=None):
    final_name = prediction["final_class_name"]
    final_conf = prediction["final_confidence"]

    if final_name == "normal":
        color = "#2ecc71"
        emoji = "✅"
    else:
        color = "#e74c3c"
        emoji = "🚨"

    if true_class:
        match = "✓ CORRECT" if final_name == true_class else "✗ INCORRECT"
        match_color = "#27ae60" if final_name == true_class else "#c0392b"
    else:
        match = ""
        match_color = "#000"

    stage1_emoji = "🚨" if prediction["stage1_decision"] == "ATTACK" else "✅"
    s2_dist = prediction["stage2_full_distribution"]
    top5 = sorted(s2_dist.items(), key=lambda x: -x[1])[:5]
    top5_html = ""
    for cls, prob in top5:
        bar_width = int(prob * 200)
        top5_html += f"""
        <div style='margin: 4px 0;'>
            <span style='display:inline-block; width:100px;'>{cls}</span>
            <span style='display:inline-block; width:{bar_width}px; background:#3498db; 
                         height:18px; border-radius:3px; vertical-align:middle;'></span>
            <span style='margin-left:8px; font-weight:bold;'>{prob*100:.1f}%</span>
        </div>"""

    return f"""
    <div style='font-family: Arial; padding:15px;'>
        <div style='background:{color}; color:white; padding:20px; border-radius:8px;
                     text-align:center; margin-bottom:15px;'>
            <div style='font-size:32px;'>{emoji}</div>
            <div style='font-size:24px; font-weight:bold; margin-top:8px;'>{final_name.upper()}</div>
            <div style='font-size:18px; margin-top:8px;'>Confidence: {final_conf*100:.2f}%</div>
            {f"<div style='font-size:16px; margin-top:8px; color:{match_color}; background:white; padding:5px 10px; border-radius:4px; display:inline-block;'>{match} (true: {true_class})</div>" if true_class else ""}
        </div>
        <div style='background:#f8f9fa; padding:15px; border-radius:8px; margin-bottom:15px;'>
            <h3 style='margin-top:0; color:#2c3e50;'>📊 Pipeline Breakdown</h3>
            <div style='margin: 8px 0;'>
                <strong>{stage1_emoji} Stage-1 (Binary):</strong>
                P(attack) = <code>{prediction['stage1_attack_proba']:.4f}</code>
                (threshold = <code>{prediction['stage1_threshold']:.4f}</code>)
                → <span style='font-weight:bold; color:{color};'>{prediction['stage1_decision']}</span>
            </div>
            <div style='margin: 8px 0;'>
                <strong>🎯 Stage-2 (Attack Type):</strong>
                <code>{prediction['stage2_attack_class']}</code>
                (confidence: <code>{prediction['stage2_confidence']*100:.2f}%</code>)
            </div>
        </div>
        <div style='background:#f8f9fa; padding:15px; border-radius:8px;'>
            <h3 style='margin-top:0; color:#2c3e50;'>📈 Top 5 Attack Class Probabilities</h3>
            {top5_html}
        </div>
    </div>"""


def format_shap_html(shap_pairs, top_k=10):
    top_pairs = shap_pairs[:top_k]
    max_abs = max(abs(v) for _, v in top_pairs) if top_pairs else 1.0
    rows_html = ""
    for i, (feat, val) in enumerate(top_pairs, start=1):
        color = "#e74c3c" if val > 0 else "#3498db"
        sign = "+" if val > 0 else "−"
        bar_width = int(abs(val) / max_abs * 200)
        display_feat = feat if len(feat) <= 30 else feat[:27] + "..."
        rows_html += f"""
        <div style='margin: 6px 0; display:flex; align-items:center;'>
            <span style='display:inline-block; width:30px; color:#7f8c8d; font-size:13px;'>#{i}</span>
            <span style='display:inline-block; width:160px; font-family:monospace; font-size:13px; color:#2c3e50;'>{display_feat}</span>
            <span style='display:inline-block; width:{bar_width}px; background:{color}; height:18px; border-radius:3px;'></span>
            <span style='margin-left:8px; font-weight:bold; color:{color}; font-family:monospace;'>{sign}{abs(val):.4f}</span>
        </div>"""
    return f"""
    <div style='background:#f8f9fa; padding:15px; border-radius:8px;'>
        <h3 style='margin-top:0; color:#2c3e50;'>🔍 Top {top_k} Important Features (SHAP)</h3>
        <div style='font-size:12px; color:#7f8c8d; margin-bottom:10px;'>
            Red = supports prediction · Blue = opposes prediction
        </div>
        {rows_html}
    </div>"""


# ============================================================
# Gradio event handlers
# ============================================================
def load_sample_handler(class_name, sample_id):
    sample = get_sample_data(class_name, int(sample_id))
    if sample is None:
        numeric_outputs = [0.0] * len(NUMERIC_COLS)
        categorical_outputs = [vals[0] for vals in CATEGORICAL_GROUPS.values()]
        return tuple(numeric_outputs + categorical_outputs + ["⚠️ Sample not found", class_name])

    numeric_dict, categorical_dict = split_sample_into_inputs(sample)
    numeric_outputs = [numeric_dict[col] for col in NUMERIC_COLS]
    categorical_outputs = [categorical_dict[cat]
                           for cat in CATEGORICAL_GROUPS.keys()]
    status = f"✅ Loaded: {class_name} #{sample_id} (test_index={sample['test_index']}, true class: {sample['true_class']})"
    return tuple(numeric_outputs + categorical_outputs + [status, class_name])


def predict_handler(pipeline_name, *all_inputs):
    n_numeric = len(NUMERIC_COLS)
    n_categorical = len(CATEGORICAL_GROUPS)

    numeric_values_list = all_inputs[:n_numeric]
    categorical_values_list = all_inputs[n_numeric:n_numeric + n_categorical]
    true_class_for_compare = all_inputs[-1]

    numeric_dict = {col: float(val) for col, val in zip(
        NUMERIC_COLS, numeric_values_list)}
    categorical_dict = {cat: val for cat, val in zip(
        CATEGORICAL_GROUPS.keys(), categorical_values_list)}

    fv = build_feature_vector(numeric_dict, categorical_dict)
    pred = predict_hierarchical(fv, pipeline_name)

    final_class_orig = pred["final_class_id"]
    if final_class_orig != NORMAL_CLASS_ID:
        predicted_local = None
        for local, orig in ATTACK_REMAP_INV.items():
            if orig == final_class_orig:
                predicted_local = local
                break
    else:
        s2_dist = pred["stage2_full_distribution"]
        top_attack_name = max(s2_dist, key=s2_dist.get)
        predicted_local = ATTACK_CLASS_NAMES.index(top_attack_name)

    shap_pairs = compute_sample_shap(fv, predicted_local)
    pred_html = format_prediction_html(pred, true_class=true_class_for_compare)
    shap_html = format_shap_html(shap_pairs, top_k=10)
    return pred_html, shap_html


# ============================================================
# Build Gradio UI
# ============================================================
CLASS_CHOICES = list(SAMPLE_DATABASE.keys())
SAMPLE_ID_CHOICES = list(range(SAMPLES_PER_CLASS))
PIPELINE_CHOICES = ["DNN", "CNN", "MOI-Lite v4"]

with gr.Blocks(
    title="MOI-Lite + E-SATF — IoT IDS Demo",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 1400px !important; }"
) as demo:

    gr.HTML("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h1 style='margin:0;'>🛡️ MOI-Lite + E-SATF</h1>
        <h3 style='margin: 5px 0 0 0;'>Hierarchical IoT Intrusion Detection System Demo</h3>
        <p style='margin: 10px 0 0 0;'>
            Compare DNN, CNN, and MOI-Lite pipelines (all SATF variants)<br>
            Trained on TON-IoT · MOI-Lite: 37K params · 65 KB INT8 deployment-ready
        </p>
    </div>
    """)

    gr.Markdown("## 🧠 Step 0: Select Model Pipeline")
    pipeline_dropdown = gr.Dropdown(
        choices=PIPELINE_CHOICES, value="MOI-Lite v4", label="Pipeline")

    gr.Markdown("## 📋 Step 1: Load a Test Sample (Optional)")

    with gr.Row():
        with gr.Column(scale=2):
            class_dropdown = gr.Dropdown(
                choices=CLASS_CHOICES, value="backdoor", label="Sample Class")
        with gr.Column(scale=2):
            sample_id_dropdown = gr.Dropdown(
                choices=SAMPLE_ID_CHOICES, value=0, label="Sample ID")
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 Auto-fill from sample",
                                 variant="secondary", size="lg")

    sample_status = gr.Markdown("*No sample loaded yet*")
    true_class_state = gr.State(value="backdoor")

    gr.Markdown("## 🔢 Step 2: Numeric Features (24 fields)")

    numeric_input_components = []
    with gr.Row():
        cols = [gr.Column(), gr.Column(), gr.Column(), gr.Column()]
        for idx, col_name in enumerate(NUMERIC_COLS):
            with cols[idx % 4]:
                inp = gr.Number(label=col_name, value=0.0, precision=4)
                numeric_input_components.append(inp)

    gr.Markdown("## 🏷️ Step 3: Categorical Features (7 dropdowns)")

    categorical_input_components = []
    with gr.Row():
        for cat_name, cat_values in CATEGORICAL_GROUPS.items():
            inp = gr.Dropdown(choices=cat_values,
                              value=cat_values[0], label=cat_name)
            categorical_input_components.append(inp)

    gr.Markdown("---")
    predict_btn = gr.Button("🚀 Run Prediction", variant="primary", size="lg")

    gr.Markdown("## 🎯 Prediction Result")

    with gr.Row():
        with gr.Column(scale=1):
            prediction_output = gr.HTML(
                value="<i style='color:#7f8c8d;'>Click 'Run Prediction' to see results</i>")
        with gr.Column(scale=1):
            shap_output = gr.HTML(
                value="<i style='color:#7f8c8d;'>SHAP feature importance will appear here</i>")

    gr.HTML("""
    <div style='margin-top: 30px; padding: 15px; background: #ecf0f1; border-radius: 8px; font-size: 13px; color: #7f8c8d;'>
        <strong>About this Demo:</strong><br>
        Two-stage hierarchical pipeline: <strong>Stage 1</strong> (binary detector) + 
        <strong>Stage 2</strong> (multiclass classifier).<br>
        <strong>SHAP</strong> explanation is computed using the MOI‑Lite model as reference.<br>
        <em>Trained on TON-IoT · Stage-1 F1 ≥ 99.3% · Stage-2 Macro F1 ≥ 90% · 
        Statistically equivalent to DNN/CNN baselines (McNemar p > 0.05)</em>
    </div>
    """)

    load_btn.click(
        fn=load_sample_handler,
        inputs=[class_dropdown, sample_id_dropdown],
        outputs=numeric_input_components + categorical_input_components +
        [sample_status, true_class_state]
    )

    predict_btn.click(
        fn=predict_handler,
        inputs=[pipeline_dropdown] + numeric_input_components +
        categorical_input_components + [true_class_state],
        outputs=[prediction_output, shap_output]
    )

if __name__ == "__main__":
    demo.launch()
