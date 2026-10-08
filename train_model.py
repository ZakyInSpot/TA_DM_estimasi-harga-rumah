"""
Melatih model prediksi harga rumah Jabodetabek untuk aplikasi web.

Pemakaian:
    python train_model.py jabodetabek_house_price.csv

Hasil: model/house_model.joblib (dipakai oleh app.py)

Tahap cleaning dan feature engineering sama dengan notebook
analisis_harga_rumah_jabodetabek.ipynb. Fitur yang dipakai model aplikasi
dibatasi pada hal yang bisa diisi pengguna di formulir (tanpa lat/long dsb.).
"""
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "jabodetabek_house_price.csv"

# ---------------------------------------------------------------- cleaning
df = pd.read_csv(CSV_PATH)
n0 = len(df)
df["city"] = df["city"].str.strip()
for c in ["certificate", "property_condition", "furnishing"]:
    df[c] = df[c].str.strip().str.lower()

df = df[~(df["ads_id"].notna() & df.duplicated("ads_id"))]
df = df.dropna(subset=["land_size_m2", "building_size_m2"])
df = df[(df.bedrooms.isna() | (df.bedrooms <= 15)) & (df.bathrooms.isna() | (df.bathrooms <= 15))]
df = df[df.building_size_m2 >= 20]
df = df[df.price_in_rp >= 1e8]
df.loc[df.building_age > 100, "building_age"] = np.nan

df["ppm"] = df["price_in_rp"] / df["building_size_m2"]
lp = np.log(df["ppm"])
q1, q3 = lp.quantile([.25, .75]); iqr = q3 - q1
df = df[(lp >= q1 - 3 * iqr) & (lp <= q3 + 3 * iqr)].copy()

df["furnishing"] = df["furnishing"].replace({"baru": np.nan}).fillna("tidak diketahui")
df["property_condition"] = df["property_condition"].fillna("tidak diketahui")

def simplify_cert(x):
    if pd.isna(x): return "tidak diketahui"
    x = x.split(" - ")[0]
    return x if x in ("shm", "hgb") else "lainnya"
df["certificate_type"] = df["certificate"].apply(simplify_cert)
df["electricity_va"] = pd.to_numeric(df["electricity"].str.extract(r"(\d+)")[0], errors="coerce")

# --------------------------------------------------- feature engineering
fac = df["facilities"].fillna("").str.lower()
for name, pat in {"has_security": "keamanan", "has_cctv": "cctv", "has_pool": "kolam renang"}.items():
    df[name] = fac.str.contains(pat).astype(int)
df["log_land"] = np.log(df["land_size_m2"])
df["log_building"] = np.log(df["building_size_m2"])
df["log_price"] = np.log(df["price_in_rp"])
print(f"Data: {n0} -> {len(df)} baris setelah cleaning")

# ------------------------------------------------------------------ model
NUM = ["log_land", "log_building", "bedrooms", "bathrooms", "floors", "carports", "garages",
       "building_age", "electricity_va", "has_security", "has_cctv", "has_pool"]
CAT = ["city", "certificate_type", "property_condition", "furnishing"]

def make_pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="tidak diketahui")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))])

X, y = df[NUM + CAT], df["log_price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
pipe = make_pipe().fit(X_tr, y_tr)
pred = pipe.predict(X_te)
rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
metrics = {"rmse_log": rmse, "mae_log": float(mean_absolute_error(y_te, pred)),
           "r2": float(r2_score(y_te, pred)), "n_train": len(X_tr), "n_test": len(X_te)}
print("Evaluasi data uji:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})

# model akhir dilatih ulang pada seluruh data
pipe = make_pipe().fit(X, y)

# ------------------------------------------------------ segmentasi (K-Means)
K = 4
cl_cols = ["log_price", "log_building", "log_land", "bedrooms", "bathrooms"]
cl_med = df[cl_cols].median()
cl_scaler = StandardScaler().fit(df[cl_cols].fillna(cl_med))
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(
    cl_scaler.transform(df[cl_cols].fillna(cl_med)))
df["cluster"] = km.labels_
prof = df.groupby("cluster").agg(jumlah=("price_in_rp", "size"),
                                 median_harga=("price_in_rp", "median"),
                                 median_bangunan=("building_size_m2", "median"),
                                 median_tanah=("land_size_m2", "median"))
order = prof["median_harga"].sort_values().index.tolist()
tiers = ["Segmen 1: terjangkau", "Segmen 2: menengah", "Segmen 3: menengah atas", "Segmen 4: premium"]
segment_names = {int(c): tiers[i] for i, c in enumerate(order)}
segment_profile = {int(c): prof.loc[c].to_dict() for c in prof.index}

# ----------------------------------------------------- info untuk formulir
options = {
    "city": sorted(df["city"].unique().tolist()),
    "certificate_type": ["shm", "hgb", "lainnya", "tidak diketahui"],
    "property_condition": sorted(df["property_condition"].unique().tolist()),
    "furnishing": sorted(df["furnishing"].unique().tolist()),
}
defaults = {c: float(df[c].median()) for c in
            ["bedrooms", "bathrooms", "floors", "carports", "garages", "electricity_va", "building_age"]}
ranges = {"land_size_m2": [float(df.land_size_m2.min()), float(df.land_size_m2.max())],
          "building_size_m2": [float(df.building_size_m2.min()), float(df.building_size_m2.max())]}

import sklearn
joblib.dump({"pipeline": pipe, "NUM": NUM, "CAT": CAT, "metrics": metrics,
             "kmeans": km, "cl_scaler": cl_scaler, "cl_cols": cl_cols, "cl_med": cl_med.to_dict(),
             "segment_names": segment_names, "segment_profile": segment_profile,
             "options": options, "defaults": defaults, "ranges": ranges,
             "sklearn_version": sklearn.__version__},
            "model/house_model.joblib", compress=3)
print("Tersimpan: model/house_model.joblib | scikit-learn", sklearn.__version__)
print("Segmen:", {segment_names[c]: int(prof.loc[c, 'jumlah']) for c in prof.index})
