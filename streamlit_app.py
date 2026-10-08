"""Aplikasi web estimasi harga rumah Jabodetabek (Streamlit).

Jalankan lokal:  streamlit run streamlit_app.py
Model dibuat oleh train_model.py (model/house_model.joblib).
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Estimasi Harga Rumah Jabodetabek", page_icon="🏠", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load("model/house_model.joblib")


try:
    ART = load_artifacts()
except Exception as e:  # biasanya karena versi scikit-learn berbeda
    st.error("Model gagal dimuat. Biasanya penyebabnya beda versi scikit-learn. "
             "Jalankan ulang `python train_model.py jabodetabek_house_price.csv` di lingkungan yang sama.")
    st.exception(e)
    st.stop()

PIPE, OPT, DEF = ART["pipeline"], ART["options"], ART["defaults"]
RMSE, R2 = ART["metrics"]["rmse_log"], ART["metrics"]["r2"]
LAND_RANGE, BLD_RANGE = ART["ranges"]["land_size_m2"], ART["ranges"]["building_size_m2"]


def rupiah(x: float) -> str:
    """Format ke juta/miliar dengan gaya Indonesia (koma desimal, titik ribuan)."""
    if x >= 1e9:
        return f"Rp {x / 1e9:,.2f} miliar".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {x / 1e6:,.0f} juta".replace(",", ".")


def predict(inp: dict):
    row = pd.DataFrame([{
        "log_land": np.log(inp["land"]), "log_building": np.log(inp["building"]),
        "bedrooms": inp["bedrooms"], "bathrooms": inp["bathrooms"], "floors": inp["floors"],
        "carports": inp["carports"], "garages": inp["garages"],
        "building_age": np.nan if inp["age"] is None else inp["age"],
        "electricity_va": inp["electricity"],
        "has_security": int(inp["security"]), "has_cctv": int(inp["cctv"]), "has_pool": int(inp["pool"]),
        "city": inp["city"], "certificate_type": inp["certificate"],
        "property_condition": inp["condition"], "furnishing": inp["furnishing"],
    }])
    log_pred = float(PIPE.predict(row[ART["NUM"] + ART["CAT"]])[0])
    seg_row = pd.DataFrame([{"log_price": log_pred, "log_building": np.log(inp["building"]),
                             "log_land": np.log(inp["land"]), "bedrooms": inp["bedrooms"],
                             "bathrooms": inp["bathrooms"]}])
    seg_id = int(ART["kmeans"].predict(ART["cl_scaler"].transform(seg_row[ART["cl_cols"]]))[0])
    return (np.exp(log_pred), np.exp(log_pred - RMSE), np.exp(log_pred + RMSE),
            ART["segment_names"][seg_id], ART["segment_profile"][seg_id])


st.title("🏠 Estimasi Harga Rumah Jabodetabek")
st.write("Masukkan data rumah, lalu sistem memperkirakan harga dan segmen pasarnya.")

left, right = st.columns(2)
with left:
    city = st.selectbox("Kota", OPT["city"])
    land = st.number_input("Luas tanah (m²)", min_value=1.0, value=100.0, step=5.0)
    building = st.number_input("Luas bangunan (m²)", min_value=1.0, value=80.0, step=5.0)
    c1, c2, c3 = st.columns(3)
    bedrooms = c1.number_input("Kamar tidur", min_value=0, value=int(DEF["bedrooms"]), step=1)
    bathrooms = c2.number_input("Kamar mandi", min_value=0, value=int(DEF["bathrooms"]), step=1)
    floors = c3.number_input("Lantai", min_value=1, value=int(DEF["floors"]), step=1)
    c4, c5 = st.columns(2)
    carports = c4.number_input("Carport", min_value=0, value=int(DEF["carports"]), step=1)
    garages = c5.number_input("Garasi", min_value=0, value=int(DEF["garages"]), step=1)
with right:
    certificate = st.selectbox("Sertifikat", OPT["certificate_type"], index=0)
    condition = st.selectbox("Kondisi properti", OPT["property_condition"])
    furnishing = st.selectbox("Furnitur", OPT["furnishing"])
    age = st.number_input("Umur bangunan (tahun, boleh dikosongkan)", min_value=0.0, value=None, step=1.0)
    electricity = st.number_input("Daya listrik (VA)", min_value=0, value=int(DEF["electricity_va"]), step=100)
    security = st.checkbox("Keamanan 24 jam")
    cctv = st.checkbox("CCTV")
    pool = st.checkbox("Kolam renang")

if st.button("Estimasi harga", type="primary"):
    est, low, high, seg_name, prof = predict(dict(
        city=city, certificate=certificate, condition=condition, furnishing=furnishing,
        land=land, building=building, bedrooms=bedrooms, bathrooms=bathrooms, floors=floors,
        carports=carports, garages=garages, age=age, electricity=electricity,
        security=security, cctv=cctv, pool=pool))
    st.divider()
    st.metric("Estimasi harga", rupiah(est))
    st.write(f"Kisaran tipikal: {rupiah(low)} sampai {rupiah(high)}")
    st.write(f"**Segmen pasar:** {seg_name} (median harga segmen ini {rupiah(prof['median_harga'])}, "
             f"bangunan {prof['median_bangunan']:.0f} m², tanah {prof['median_tanah']:.0f} m²)")
    if not (LAND_RANGE[0] <= land <= LAND_RANGE[1]) or not (BLD_RANGE[0] <= building <= BLD_RANGE[1]):
        st.warning("Ukuran rumah di luar rentang data latih, estimasi kurang dapat diandalkan.")
    st.caption(f"Model: Random Forest, R² data uji {R2:.2f}. Harga di dataset adalah harga penawaran "
               "pada listing Rumah123.com, bukan harga transaksi, sehingga hasil ini hanya perkiraan awal.")
