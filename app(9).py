"""
app.py
------
Streamlit UI - Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt
bằng Google OR-Tools + Guided Local Search (GLS), dùng OSRM/OSM làm mạng
lưới đường thực tế.

PHẠM VI NGHIÊN CỨU: Tuyến Lê Văn Việt và khu vực lân cận, TP. Thủ Đức, TP.HCM.
"""

from __future__ import annotations

import io
import math
import time
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

try:
    from streamlit_geolocation import streamlit_geolocation
    HAS_GEOLOCATION = True
except ImportError:
    HAS_GEOLOCATION = False

from baseline import nearest_neighbor_baseline
from data_generator import DemoConfig, generate_demo_data, load_points_from_dataframe, DEPOT_LOCATION
from clustering import run_clustering_cvrp, split_time_budget
from dynamic_routing import DynamicRoutingEngine, GPSTrackerConfig, interpolate_along_route
from optimizer import OptimizeConfig, solve_cvrp, solve_cvrp_with_auto_scaling
from routing import DEFAULT_OSRM_BASE_URL, OSRMError, check_point_count_limit, get_osrm_matrices, get_osrm_route_geometry
from waste_streams import WASTE_STREAMS, build_stream_problem, run_stream_optimization, aggregate_kpi
from simulation import (
    FleetSimulator, RouteLeg, build_route_legs, format_hhmm,
    STATE_IDLE, STATE_MOVING, STATE_COLLECTING, STATE_FULL,
    STATE_RETURNING_TO_DEPOT, STATE_BROKEN, STATE_BLOCKED, STATE_COMPLETED,
)
from forecasting import (
    WASTE_TYPES,
    prepare_history_from_dataframe,
    train_and_forecast_uploaded_history,
    estimate_vehicles_needed,
    build_forecast_excel,
    build_template_history_excel,
)

st.set_page_config(page_title="Tối ưu tuyến thu gom rác - Lê Văn Việt", layout="wide")

# File dữ liệu 100 điểm: đặt cùng thư mục với app.py để app tự nhận, không cần upload lại.
DEFAULT_100_DATA_FILE = Path(__file__).with_name("du_lieu_ML_100_diem_365_ngay_bo_sung(1).xlsx")

# ============================================================================
# HEADER - Ghi rõ phạm vi & giả định nghiên cứu (bắt buộc theo yêu cầu đề tài)
# ============================================================================
st.title("Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt")

st.markdown(
    """
| Hạng mục | Nội dung |
|---|---|
| **Khu vực thử nghiệm** | Tuyến Lê Văn Việt – TP. Thủ Đức, TP.HCM (khu vực Quận 9 cũ) |
| **Routing engine** | OpenStreetMap + OSRM |
| **Optimization** | Google OR-Tools + Guided Local Search |
| **Real-time traffic** | Chưa xét |
| **Baseline** | Simulated baseline – Nearest Neighbor / Greedy |
"""
)
st.caption(
    "Đây là mô hình nghiên cứu/prototype phục vụ mục đích học thuật (Green Logistics), "
    "KHÔNG phải hệ thống điều hành xe thu gom rác thực tế."
)
st.divider()

# ============================================================================
# SIDEBAR - CẤU HÌNH
# ============================================================================
with st.sidebar:
    st.header("1. Dữ liệu")
    data_mode = st.radio(
        "Nguồn dữ liệu",
        ["Dữ liệu 100 điểm (Excel)", "Dữ liệu demo (Lê Văn Việt)", "Upload CSV/XLSX"],
        index=0,
    )

    if data_mode == "Dữ liệu 100 điểm (Excel)":
        st.caption(
            "100 điểm P01–P100 từ file du_lieu_ML_100_diem_365_ngay_bo_sung(1).xlsx "
            "+ 1 depot. Tọa độ lấy trực tiếp từ Excel."
        )
        if not DEFAULT_100_DATA_FILE.exists():
            st.error("Không tìm thấy file Excel 100 điểm cạnh app.py. Hãy đặt 2 file cùng thư mục.")
    elif data_mode == "Dữ liệu demo (Lê Văn Việt)":
        num_points = st.slider("Số điểm thu gom (demo, quanh Lê Văn Việt)", 20, 100, 25)
        seed = st.number_input("Random seed", value=42, step=1)
        use_tw_demo = st.checkbox("Sinh time window demo (VRPTW)", value=False)
        st.caption("Dữ liệu 100 điểm P01–P100 được dùng làm bộ dữ liệu chính.")
    else:
        uploaded_file = st.file_uploader("Upload file (CSV hoặc XLSX)", type=["csv", "xlsx"])
        st.caption(
            "Cột bắt buộc: node_id, latitude, longitude, waste_kg. "
            "Tuỳ chọn: service_time, time_window_start, time_window_end, is_depot."
        )

    st.header("2. Dự báo nhu cầu")
    forecast_method = st.selectbox("Mô hình dự báo", ["XGBoost", "Prophet"], index=0)
    forecast_horizon = st.slider("Số ngày dự báo", 1, 365, 7)
    st.caption("Upload lịch sử 365 ngày ở phần **🔮 Dự báo nhu cầu** bên dưới. Kết quả dự báo sẽ được truyền trực tiếp sang định tuyến.")

    st.header("3. Routing (OSRM)")
    osrm_base_url = st.text_input("OSRM base URL", value=DEFAULT_OSRM_BASE_URL)
    allow_fallback = st.checkbox(
        "Cho phép fallback Haversine nếu OSRM lỗi (KHÔNG khuyến nghị)", value=False
    )
    if allow_fallback:
        st.warning("Fallback mode – không sử dụng mạng lưới đường thực tế nếu được kích hoạt.")
        fallback_speed = st.slider("Tốc độ giả định cho fallback (km/h)", 10, 50, 25)
    else:
        fallback_speed = 25

    st.header("4. Xe & ràng buộc")
    num_vehicles = st.slider("Số xe tối đa (upper bound cho OR-Tools)", 1, 10, 3)
    vehicle_capacity_kg = st.number_input("Vehicle capacity (kg)", value=1000, step=50)
    max_route_hours = st.slider("Max route duration (giờ)", 1.0, 8.0, 4.0, step=0.5)

    st.header("5. Thuật toán tối ưu (OR-Tools)")
    use_gls = st.checkbox("Bật Guided Local Search (GLS)", value=True)
    first_solution_strategy = st.selectbox(
        "Chiến lược khởi tạo (initial solution)",
        ["PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC"],
    )
    time_limit_sec = st.slider("Thời gian chạy tối ưu (giây)", 5, 120, 20)

    st.header("6. Hệ số tiêu hao & phát thải (có thể chỉnh)")
    fuel_rate_l_per_km = st.number_input("Fuel rate (lít/km)", value=0.35, step=0.01, format="%.2f")
    emission_factor_kg_per_l = st.number_input(
        "Emission factor (kg CO2 / lít nhiên liệu)", value=2.68, step=0.01, format="%.2f"
    )

    run_btn = st.button("Chạy tối ưu", type="primary", use_container_width=True)

st.divider()

# ============================================================================
# LOAD DỮ LIỆU
# ============================================================================
def _load_100_point_case() -> pd.DataFrame | None:
    """Đọc 100 điểm P01–P100 từ Excel và thêm 1 depot cho routing."""
    if not DEFAULT_100_DATA_FILE.exists():
        return None

    raw = pd.read_excel(DEFAULT_100_DATA_FILE)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    latest = (
        raw.sort_values("date")
        .groupby("node_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest = latest.sort_values("node_id").reset_index(drop=True)
    latest["is_depot"] = False
    latest["service_time"] = 5.0
    latest["time_window_start"] = 0.0
    latest["time_window_end"] = 24.0 * 60.0

    # Depot cố định cho bộ dữ liệu 100 điểm.
    depot = {
        "node_id": "DEPOT",
        "latitude": float(DEPOT_LOCATION[0]),
        "longitude": float(DEPOT_LOCATION[1]),
        "waste_kg": 0.0,
        "is_depot": True,
        "service_time": 0.0,
        "time_window_start": 0.0,
        "time_window_end": 24.0 * 60.0,
    }
    depot_df = pd.DataFrame([depot])
    for col in latest.columns:
        if col not in depot_df.columns:
            depot_df[col] = None
    for col in depot_df.columns:
        if col not in latest.columns:
            latest[col] = None

    return pd.concat([depot_df[latest.columns], latest], ignore_index=True)


def _load_data() -> pd.DataFrame | None:
    if data_mode == "Dữ liệu 100 điểm (Excel)":
        return _load_100_point_case()

    if data_mode == "Dữ liệu demo (Lê Văn Việt)":
        cfg = DemoConfig(num_points=num_points, seed=int(seed), use_time_windows=use_tw_demo)
        return generate_demo_data(cfg)

    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    return load_points_from_dataframe(raw)


if "df_points" not in st.session_state:
    st.session_state.df_points = None
if "results" not in st.session_state:
    st.session_state.results = None

df_points = _load_data()
if df_points is not None:
    st.session_state.df_points = df_points

if st.session_state.df_points is None:
    st.info("Vui lòng upload dữ liệu hoặc dùng dữ liệu demo, sau đó nhấn **Chạy tối ưu**.")
    st.stop()

df_points = st.session_state.df_points

with st.expander("Xem dữ liệu điểm thu gom", expanded=False):
    st.dataframe(df_points, use_container_width=True)

# ============================================================================
# BƯỚC 2 — DỰ BÁO NHU CẦU TỪ FILE LỊCH SỬ 365 NGÀY
# ============================================================================
st.subheader("🔮 Bước 2 — Dự báo nhu cầu rác")
st.caption(
    "Upload dữ liệu lịch sử 365 ngày để huấn luyện XGBoost/Prophet. "
    "Sau khi dự báo, bảng demand theo từng điểm sẽ được dùng trực tiếp làm đầu vào cho CVRP/K-means++, "
    "không cần tải Excel dự báo xuống rồi upload lại."
)

history_file = st.file_uploader(
    "📂 Tải Excel dữ liệu quá khứ (khuyến nghị 365 ngày; có thể bỏ qua nếu đang dùng bộ 100 điểm)",
    type=["xlsx", "xls", "csv"],
    key="history_365_upload",
    help=(
        "Cột tối thiểu: date, node_id, waste_kg. Có thể dùng thêm num_households. "
        "Nếu có 3 loại rác, dùng waste_organic_kg, waste_recyclable_kg, waste_other_kg."
    ),
)

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "forecast_selected_day" not in st.session_state:
    st.session_state.forecast_selected_day = None

fc1, fc2, fc3 = st.columns([1, 1, 1])
with fc1:
    st.metric("Lịch sử yêu cầu", "365 ngày")
with fc2:
    st.metric("Mô hình", forecast_method)
with fc3:
    st.metric("Horizon", f"{forecast_horizon} ngày")

bundled_history_available = data_mode == "Dữ liệu 100 điểm (Excel)" and DEFAULT_100_DATA_FILE.exists()
if bundled_history_available:
    st.info(
        "Đang dùng tự động file 100 điểm × 365 ngày làm dữ liệu lịch sử. "
        "Không cần upload lại; nếu muốn dùng file khác, upload file mới ở ô phía trên."
    )

if history_file is None and not bundled_history_available:
    st.download_button(
        "📄 Tải file mẫu Excel lịch sử 365 ngày",
        data=build_template_history_excel(),
        file_name="template_lich_su_365_ngay.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if history_file is not None or bundled_history_available:
    if st.button("🧠 Huấn luyện & dự báo", type="primary", use_container_width=True):
        try:
            if history_file is not None:
                if history_file.name.lower().endswith(".csv"):
                    raw_history = pd.read_csv(history_file)
                else:
                    raw_history = pd.read_excel(history_file)
            else:
                raw_history = pd.read_excel(DEFAULT_100_DATA_FILE)

            history = prepare_history_from_dataframe(raw_history, df_points=df_points)
            if history["date"].nunique() < 30:
                st.warning("File có ít hơn 30 ngày dữ liệu. Với đề tài này nên dùng đủ khoảng 365 ngày để học xu hướng theo thời gian.")
            with st.spinner(f"Đang huấn luyện {forecast_method} trên dữ liệu lịch sử... "):
                forecast_result = train_and_forecast_uploaded_history(
                    history,
                    method=forecast_method.lower(),
                    forecast_days=int(forecast_horizon),
                )
            st.session_state.forecast_history = history
            st.session_state.forecast_result = forecast_result
            st.session_state.forecast_df = forecast_result["forecast_df"]
            st.session_state.forecast_selected_day = None
            st.success(
                f"Đã huấn luyện {forecast_method} trên {history['date'].nunique()} ngày × "
                f"{history['node_id'].nunique()} điểm. Dự báo {forecast_horizon} ngày đã sẵn sàng."
            )
        except Exception as exc:
            st.error(f"Không thể đọc/huấn luyện file lịch sử: {exc}")

if st.session_state.forecast_result is not None:
    fr = st.session_state.forecast_result
    history = st.session_state.forecast_history
    forecast_df = st.session_state.forecast_df.copy()

    st.markdown("### 📊 Kết quả dự báo theo khoảng thời gian")
    available_dates = pd.to_datetime(sorted(forecast_df["date"].unique())).normalize()
    min_forecast_date = available_dates.min().date()
    max_forecast_date = available_dates.max().date()

    # Chọn một khoảng ngày trong tương lai để xem tổng quan.
    default_end = min(min_forecast_date + pd.Timedelta(days=6), max_forecast_date)
    selected_range = st.date_input(
        "Chọn khoảng ngày dự báo",
        value=(min_forecast_date, default_end.date() if hasattr(default_end, "date") else default_end),
        min_value=min_forecast_date,
        max_value=max_forecast_date,
        key="selected_forecast_range",
    )
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        range_start, range_end = selected_range
    else:
        range_start = range_end = selected_range

    range_start = pd.Timestamp(range_start)
    range_end = pd.Timestamp(range_end)
    range_selected = forecast_df[
        (pd.to_datetime(forecast_df["date"]).dt.normalize() >= range_start)
        & (pd.to_datetime(forecast_df["date"]).dt.normalize() <= range_end)
    ].copy()

    if range_selected.empty:
        st.warning("Khoảng ngày chưa có dữ liệu dự báo.")
        selected_date = available_dates[0]
        selected = forecast_df[forecast_df["date"] == selected_date].copy()
    else:
        range_dates = pd.to_datetime(sorted(range_selected["date"].unique())).normalize()
        default_route_date = range_dates[0].date()
        selected_route_date = st.date_input(
            "Chọn ngày cụ thể dùng cho định tuyến",
            value=default_route_date,
            min_value=range_dates.min().date(),
            max_value=range_dates.max().date(),
            key="selected_route_date",
        )
        selected_date = pd.Timestamp(selected_route_date).normalize()
        selected = forecast_df[pd.to_datetime(forecast_df["date"]).dt.normalize() == selected_date].copy()

    total_predicted = float(selected["total_kg"].sum())
    range_total_predicted = float(range_selected["total_kg"].sum()) if not range_selected.empty else total_predicted
    vehicles_needed = estimate_vehicles_needed(
        {r["node_id"]: {"total_kg": r["total_kg"]} for _, r in selected.iterrows()},
        vehicle_capacity_kg,
    )[1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Khoảng dự báo", f"{range_start:%d/%m/%Y} → {range_end:%d/%m/%Y}")
    m2.metric("Ngày dùng định tuyến", selected_date.strftime("%d/%m/%Y"))
    m3.metric("Demand ngày chọn", f"{total_predicted:,.1f} kg")
    m4.metric("Demand cả khoảng", f"{range_total_predicted:,.1f} kg")

    display_cols = [c for c in [
        "date", "node_id", "organic", "recyclable", "other", "total_kg"
    ] if c in selected.columns]
    st.dataframe(
        selected[display_cols].rename(columns={
            "date": "Ngày", "node_id": "Điểm", "organic": "Hữu cơ (kg)",
            "recyclable": "Tái chế (kg)", "other": "Còn lại (kg)", "total_kg": "Tổng dự báo (kg)"
        }),
        use_container_width=True, hide_index=True,
    )

    if fr.get("validation_df") is not None:
        st.markdown("### Kiểm định mô hình")
        st.dataframe(fr["validation_df"], use_container_width=True, hide_index=True)

    export_bytes = build_forecast_excel(history, forecast_df, fr.get("validation_df"))
    st.download_button(
        "⬇️ Xuất Excel dữ liệu dự báo",
        data=export_bytes,
        file_name="du_bao_nhu_cau_365_ngay.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption(
        "Excel gồm: lịch sử đã chuẩn hoá, dự báo theo từng điểm/ngày và bảng đánh giá mô hình. "
        "Định tuyến sẽ lấy đúng ngày dự báo được chọn ở trên."
    )

else:
    st.info("Chưa có kết quả dự báo. Bạn có thể upload Excel 365 ngày và nhấn **Huấn luyện & dự báo**. Nếu chưa dùng dự báo, hệ thống vẫn có thể chạy định tuyến bằng waste_kg hiện tại.")

# ============================================================================
# CHẠY PIPELINE KHI NHẤN NÚT
# ============================================================================
if run_btn:
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    service_times_s = (df_points["service_time"] * 60).tolist()

    # Nếu đã có dự báo, demand của routing = đúng ngày dự báo được chọn.
    # Nếu chưa có, giữ nguyên waste_kg của file điểm như chế độ fallback.
    forecast_active = st.session_state.get("forecast_result") is not None and st.session_state.get("forecast_df") is not None
    if forecast_active:
        forecast_df_live = st.session_state.forecast_df
        forecast_date = pd.Timestamp(st.session_state.get("selected_route_date", sorted(forecast_df_live["date"].unique())[0])).normalize()
        demand_map = (
            forecast_df_live[forecast_df_live["date"] == forecast_date]
            .set_index("node_id")["total_kg"]
            .to_dict()
        )
        demands = [0.0 if bool(row["is_depot"]) else float(demand_map.get(row["node_id"], row["waste_kg"])) for _, row in df_points.iterrows()]
    else:
        forecast_date = None
        demands = df_points["waste_kg"].tolist()
    time_windows_s = list(
        zip(df_points["time_window_start"] * 60, df_points["time_window_end"] * 60)
    )
    use_tw = df_points["time_window_start"].sum() > 0 or (
        data_mode == "Dữ liệu demo (Lê Văn Việt)" and use_tw_demo
    )

    with st.spinner("Đang gọi OSRM để lấy ma trận khoảng cách/thời gian đường bộ..."):
        try:
            matrix_result = get_osrm_matrices(
                coords,
                base_url=osrm_base_url,
                allow_haversine_fallback=allow_fallback,
                fallback_avg_speed_kmh=fallback_speed,
            )
        except OSRMError as exc:
            st.error(str(exc))
            st.stop()

    if matrix_result.source == "HAVERSINE_FALLBACK":
        st.warning(matrix_result.warning)
    else:
        st.success("Đã lấy ma trận khoảng cách/thời gian từ OSRM (Routing: OpenStreetMap + OSRM).")

    dist_m = matrix_result.distance_matrix_m
    dur_s = matrix_result.duration_matrix_s

    depot_index = int(df_points.index[df_points["is_depot"]][0])

    # ---------------- BASELINE ----------------
    with st.spinner("Đang xây dựng baseline (Nearest Neighbor / Greedy)..."):
        try:
            baseline_routes = nearest_neighbor_baseline(
                dist_m, dur_s, demands, service_times_s,
                vehicle_capacity_kg=vehicle_capacity_kg,
                depot_index=depot_index,
                max_route_time_s=max_route_hours * 3600,
                max_vehicles=max(num_vehicles, 20),
            )
        except RuntimeError as exc:
            st.error(f"Baseline thất bại: {exc}")
            st.stop()

    # ---------------- OPTIMIZED (OR-TOOLS + GLS) ----------------
    with st.spinner("Đang chạy OR-Tools + Guided Local Search..."):
        opt_config = OptimizeConfig(
            num_vehicles=max(num_vehicles, len(baseline_routes)),
            vehicle_capacity_kg=vehicle_capacity_kg,
            depot_index=depot_index,
            use_gls=use_gls,
            first_solution_strategy=first_solution_strategy,
            time_limit_sec=time_limit_sec,
            max_route_time_s=max_route_hours * 3600,
            use_time_windows=use_tw,
            time_windows_s=time_windows_s,
        )
        optimized_routes, solved, msg = solve_cvrp(dist_m, dur_s, demands, service_times_s, opt_config)

    if not solved:
        st.error(msg)
        st.stop()

    st.session_state.results = {
        "df_points": df_points,
        "coords": coords,
        "matrix_result": matrix_result,
        "baseline_routes": baseline_routes,
        "optimized_routes": optimized_routes,
        "depot_index": depot_index,
        "osrm_base_url": osrm_base_url,
        "fuel_rate_l_per_km": fuel_rate_l_per_km,
        "emission_factor_kg_per_l": emission_factor_kg_per_l,
        "demands": demands,
        "service_times_s": service_times_s,
        "forecast_active": forecast_active,
        "forecast_date": forecast_date,
        "forecast_model": forecast_method if forecast_active else None,
    }

# ============================================================================
# HIỂN THỊ KẾT QUẢ
# ============================================================================
if st.session_state.results is None:
    st.stop()

res = st.session_state.results
df_points = res["df_points"]
coords = res["coords"]
node_ids_full = df_points["node_id"].tolist()
index_of_node = {nid: i for i, nid in enumerate(node_ids_full)}
depot_node_id = df_points.loc[df_points["is_depot"], "node_id"].iloc[0]
dist_m_full = res["matrix_result"].distance_matrix_m
dur_s_full = res["matrix_result"].duration_matrix_s
demands_full = res["demands"]
service_times_s_full = res["service_times_s"]
baseline_routes = res["baseline_routes"]
optimized_routes = res["optimized_routes"]
depot_index = res["depot_index"]
fuel_rate = res["fuel_rate_l_per_km"]
emission_factor = res["emission_factor_kg_per_l"]

if res.get("forecast_active"):
    st.success(
        f"🚛 Tuyến đang được tối ưu theo demand DỰ BÁO ngày "
        f"{pd.Timestamp(res['forecast_date']).strftime('%d/%m/%Y')} "
        f"bằng {res.get('forecast_model', 'model')} — không dùng waste_kg cố định."
    )

def _aggregate(routes) -> dict:
    total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0
    travel_time_min = sum(r.travel_time_s for r in routes) / 60.0
    service_time_min = sum(r.service_time_s for r in routes) / 60.0
    total_time_min = sum(r.total_route_time_s for r in routes) / 60.0
    num_vehicles_used = len(routes)
    total_waste_kg = sum(r.collected_waste_kg for r in routes)
    avg_util = (
        sum(r.capacity_utilization_pct for r in routes) / len(routes) if routes else 0.0
    )
    fuel_l = total_distance_km * fuel_rate
    co2_kg = fuel_l * emission_factor
    return {
        "Total distance (km)": round(total_distance_km, 2),
        "Travel time (min)": round(travel_time_min, 1),
        "Service time (min)": round(service_time_min, 1),
        "Total route time (min)": round(total_time_min, 1),
        "Number of vehicles": num_vehicles_used,
        "Total waste collected (kg)": round(total_waste_kg, 1),
        "Average capacity utilization (%)": round(avg_util, 1),
        "Estimated fuel consumption (L)": round(fuel_l, 2),
        "Estimated CO2 emissions (kg)": round(co2_kg, 2),
    }


baseline_kpi = _aggregate(baseline_routes)
optimized_kpi = _aggregate(optimized_routes)


def _reduction(base, opt):
    if base == 0:
        return 0.0
    return round((base - opt) / base * 100, 1)


kpi_rows = []
for key in baseline_kpi:
    row = {"Chỉ tiêu": key, "Baseline": baseline_kpi[key], "Optimized": optimized_kpi[key]}
    if key in (
        "Total distance (km)", "Total route time (min)",
        "Estimated CO2 emissions (kg)", "Estimated fuel consumption (L)",
    ):
        row["Reduction (%)"] = _reduction(baseline_kpi[key], optimized_kpi[key])
    else:
        row["Reduction (%)"] = "-"
    kpi_rows.append(row)

st.subheader("So sánh KPI: Baseline vs Optimized")
st.caption(
    "Baseline = tuyến cơ sở mô phỏng bằng heuristic Nearest Neighbor/Greedy "
    f"(nguồn khoảng cách/thời gian: {res['matrix_result'].source}). "
    "CO2 emissions là giá trị ước tính (Estimated CO2 emissions), không phải đo trực tiếp."
)
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

# ============================================================================
# BẢNG VẬN HÀNH DẠNG TABLE 4 — DAILY COLLECTION OPERATIONS
# ============================================================================
node_ids = df_points["node_id"].tolist()

def _daily_operation_row(routes, capacity, csl=0.80):
    total_points = sum(r.num_stops for r in routes)
    distance_km = sum(r.total_distance_m for r in routes) / 1000.0
    travel_min = sum(r.travel_time_s for r in routes) / 60.0
    total_route_min = sum(r.total_route_time_s for r in routes) / 60.0
    total_weight = sum(r.collected_waste_kg for r in routes)
    visits = len(routes)
    capacity_total = capacity * visits
    extra_weight = max(0.0, total_weight - csl * capacity_total)
    carrying_avg = (total_weight / capacity_total * 100) if capacity_total > 0 else 0.0
    # Tổng khối lượng nằm trên các xe có mức sử dụng tải trọng >100% và <115%.
    weight_100_115 = sum(
        float(r.collected_waste_kg)
        for r in routes
        if capacity > 0 and 1.0 < (float(r.collected_waste_kg) / capacity) < 1.15
    )
    return {
        "Tổng điểm thu gom": total_points,
        "Tổng quãng đường (km)": round(distance_km, 2),
        "Tổng thời gian di chuyển (phút)": round(travel_min, 1),
        "Tổng thời gian tuyến (phút)": round(total_route_min, 1),
        "Tổng khối lượng (kg)": round(total_weight, 1),
        "Khối lượng xe chở >100% và <115% tải trọng (kg)": round(weight_100_115, 1),
        "Số lần về Depot": visits,
        "% thời gian di chuyển": round((travel_min / total_route_min * 100) if total_route_min > 0 else 0.0, 1),
        "% tải trọng bình quân": round(carrying_avg, 1),
        "% tải trọng so với CSL 80%": round((carrying_avg / csl) if csl > 0 else 0.0, 1),
    }

base_ops = _daily_operation_row(baseline_routes, vehicle_capacity_kg)
prop_ops = _daily_operation_row(optimized_routes, vehicle_capacity_kg)
route_table4 = pd.DataFrame({
    "Chỉ tiêu": list(base_ops.keys()),
    "Baseline – Nearest Neighbor": list(base_ops.values()),
    "Proposed – OR-Tools + GLS": list(prop_ops.values()),
})

st.subheader("📋 Daily Collection Operations — cấu trúc theo Table 4")
st.caption(
    "Ngày vận hành: " + (pd.Timestamp(res["forecast_date"]).strftime("%d/%m/%Y") if res.get("forecast_date") is not None else "dữ liệu hiện tại") +
    ". CSL (Customer Service Level) được đặt ở 80% để theo dõi mức sử dụng tải xe."
)
st.dataframe(route_table4, use_container_width=True, hide_index=True)

# Chi tiết phân bổ xe để thấy xe nào nhận điểm nào
route_detail_rows = []
for r in optimized_routes:
    route_detail_rows.append({
        "Xe": f"Vehicle {r.vehicle_id}",
        "Số điểm": r.num_stops,
        "Điểm thu gom": " → ".join(node_ids[i] for i in r.node_sequence),
        "Khối lượng (kg)": round(r.collected_waste_kg, 1),
        "Tải trọng (%)": round(r.capacity_utilization_pct, 1),
        "Quãng đường (km)": round(r.total_distance_m / 1000.0, 2),
        "Thời gian tuyến (phút)": round(r.total_route_time_s / 60.0, 1),
    })
st.markdown("**Phân bổ điểm → xe**")
route_detail_df = pd.DataFrame(route_detail_rows)
st.dataframe(route_detail_df, use_container_width=True, hide_index=True)

# Xuất workbook kết quả định tuyến theo cấu trúc báo cáo
route_export = io.BytesIO()
with pd.ExcelWriter(route_export, engine="openpyxl") as writer:
    route_table4.to_excel(writer, index=False, sheet_name="Daily_Operations")
    route_detail_df.to_excel(writer, index=False, sheet_name="Vehicle_Allocation")
    pd.DataFrame(kpi_rows).to_excel(writer, index=False, sheet_name="KPI_Comparison")

st.download_button(
    "⬇️ Xuất Excel kết quả phân bổ xe & vận hành",
    data=route_export.getvalue(),
    file_name="ket_qua_phan_bo_xe_table4.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# ---------------- Chi tiết từng tuyến ----------------

def _route_label(route) -> str:
    names = [node_ids[i] for i in route.node_sequence]
    return " → ".join(names)


col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Baseline routes**")
    for r in baseline_routes:
        st.text(f"Vehicle {r.vehicle_id}: {_route_label(r)}")
        st.caption(
            f"Distance: {r.total_distance_m/1000:.2f} km | "
            f"Total time: {r.total_route_time_s/60:.1f} phút | "
            f"Waste: {r.collected_waste_kg:.1f} kg | "
            f"Utilization: {r.capacity_utilization_pct:.1f}%"
        )
with col_b:
    st.markdown("**Optimized routes (OR-Tools + GLS)**")
    for r in optimized_routes:
        st.text(f"Vehicle {r.vehicle_id}: {_route_label(r)}")
        st.caption(
            f"Distance: {r.total_distance_m/1000:.2f} km | "
            f"Total time: {r.total_route_time_s/60:.1f} phút | "
            f"Waste: {r.collected_waste_kg:.1f} kg | "
            f"Utilization: {r.capacity_utilization_pct:.1f}%"
        )

st.divider()

# ============================================================================
# BẢN ĐỒ (Folium + OSRM road geometry thực tế)
# ============================================================================
st.subheader("Bản đồ tuyến (road geometry thực tế từ OSRM)")

center_lat, center_lon = DEPOT_LOCATION
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

# Depot & các điểm thu gom
for idx, row in df_points.iterrows():
    if row["is_depot"]:
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup="DEPOT",
            icon=folium.Icon(color="black", icon="home"),
        ).add_to(fmap)
    else:
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=5,
            popup=f"{row['node_id']} - {row['waste_kg']:.0f} kg",
            color="#555555",
            fill=True,
            fill_opacity=0.8,
        ).add_to(fmap)


def _draw_routes(routes, color, label_prefix):
    for r in routes:
        ordered_coords = tuple(coords[i] for i in r.node_sequence)
        geometry = get_osrm_route_geometry(ordered_coords, base_url=res["osrm_base_url"])
        if not geometry:
            # OSRM route service không khả dụng cho tuyến này -> vẽ tạm bằng
            # đường nối các điểm (KHÔNG phải road geometry thực tế), có ghi chú.
            geometry = list(ordered_coords)
            dash = "5, 10"
        else:
            dash = None
        folium.PolyLine(
            geometry,
            color=color,
            weight=4,
            opacity=0.8,
            dash_array=dash,
            tooltip=f"{label_prefix} - Vehicle {r.vehicle_id} ({r.total_distance_m/1000:.2f} km)",
        ).add_to(fmap)


_draw_routes(baseline_routes, "#1f77b4", "Baseline")
_draw_routes(optimized_routes, "#d62728", "Optimized")

st.caption(
    "🔵 Xanh dương = Baseline (Nearest Neighbor/Greedy) · 🔴 Đỏ = Optimized (OR-Tools + GLS). "
    "Nét đứt (nếu có) nghĩa là OSRM Route Service không trả về được geometry cho đoạn đó."
)
st_folium(fmap, use_container_width=True, height=600, returned_objects=[])

st.divider()

# ============================================================================
# SO SÁNH: ĐƯỜNG ĐI NGẮN NHẤT vs CVRP TRỰC TIẾP vs MÔ HÌNH ĐỀ XUẤT
# (K-means++ Clustering + CVRP theo cụm) — bộ dữ liệu mô phỏng 100 điểm
# ============================================================================
st.subheader("🧩 So sánh: Đường đi ngắn nhất · CVRP trực tiếp · Mô hình đề xuất (K-means++ + CVRP)")
st.caption(
    "Mô hình đề xuất gồm 2 giai đoạn: (1) Phân cụm K-means (khởi tạo k-means++), số cụm k được "
    "xác định tự động theo tổng khối lượng rác và sức chứa xe (k = ⌈ΣTᵢ / Q⌉, có biên an toàn để "
    "còn dư địa cân bằng tải trọng); (2) Giải CVRP cho TỪNG cụm bằng OR-Tools (PATH_CHEAPEST_ARC + "
    "GUIDED_LOCAL_SEARCH). Bộ dữ liệu mô phỏng riêng cho phần này (mặc định 100 điểm dọc Lê Văn "
    "Việt) — độc lập với dữ liệu ở phần tối ưu chính phía trên."
)

with st.expander("⚙️ Cấu hình bộ dữ liệu 100 điểm & tham số so sánh", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        cmp_num_points = st.slider("Số điểm mô phỏng", 50, 100, 100, key="cmp_num_points")
        cmp_seed = st.number_input("Seed", value=42, step=1, key="cmp_seed")
    with c2:
        cmp_capacity = st.number_input("Sức chứa xe Q (kg)", value=1500, step=50, key="cmp_capacity")
        cmp_max_route_hours = st.slider("Max route duration (giờ)", 1.0, 10.0, 8.0, step=0.5, key="cmp_max_hours")
    with c3:
        cmp_time_budget = st.slider(
            "Ngân sách thời gian thuật toán (giây, dùng CHUNG cho PP2 & PP3 để so sánh công bằng)",
            10, 120, 20, key="cmp_time_budget",
        )
        cmp_use_gls = st.checkbox("Bật GLS", value=True, key="cmp_use_gls")

run_compare_btn = st.button("🧩 Sinh dữ liệu 100 điểm & chạy so sánh 3 phương pháp")

if run_compare_btn:
    cmp_df = generate_demo_data(DemoConfig(num_points=cmp_num_points, seed=int(cmp_seed)))
    cmp_coords = tuple(zip(cmp_df["latitude"], cmp_df["longitude"]))
    cmp_node_ids = cmp_df["node_id"].tolist()
    cmp_demands = cmp_df["waste_kg"].tolist()
    cmp_service_s = (cmp_df["service_time"] * 60).tolist()
    cmp_depot_idx = int(cmp_df.index[cmp_df["is_depot"]][0])

    warn = check_point_count_limit(cmp_num_points, osrm_base_url)
    if warn:
        st.warning(warn)

    cmp_matrix = None
    with st.spinner("Đang lấy ma trận khoảng cách/thời gian từ OSRM cho bộ 100 điểm..."):
        try:
            cmp_matrix = get_osrm_matrices(
                cmp_coords, base_url=osrm_base_url,
                allow_haversine_fallback=allow_fallback, fallback_avg_speed_kmh=fallback_speed,
            )
        except OSRMError as exc:
            # Chỉ dừng RIÊNG phần so sánh này - không dùng st.stop() ở đây vì
            # nó sẽ chặn luôn toàn bộ phần pipeline chính phía dưới trong lần
            # render này (section này được đặt sớm trong luồng script để có
            # thể chạy độc lập, không phụ thuộc pipeline chính).
            st.error(str(exc))

    if cmp_matrix is not None:
        if cmp_matrix.source == "HAVERSINE_FALLBACK":
            st.warning(cmp_matrix.warning)

        dist_cmp = cmp_matrix.distance_matrix_m
        dur_cmp = cmp_matrix.duration_matrix_s
        max_route_s = cmp_max_route_hours * 3600

        def _summarize(routes):
            return {
                "Số xe": len(routes),
                "Tổng quãng đường (km)": round(sum(r.total_distance_m for r in routes) / 1000, 2),
                "Tổng thời gian tuyến (phút)": round(sum(r.total_route_time_s for r in routes) / 60, 1),
            }

        with st.spinner("1/3 — Đang chạy Nearest Neighbor (đường đi ngắn nhất, không phân cụm)..."):
            t0 = time.time()
            nn_routes = nearest_neighbor_baseline(
                dist_cmp, dur_cmp, cmp_demands, cmp_service_s, cmp_capacity, cmp_depot_idx,
                max_route_s, max_vehicles=60,
            )
            t_nn = time.time() - t0

        with st.spinner("2/3 — Đang chạy CVRP trực tiếp (OR-Tools + GLS, không phân cụm)..."):
            t0 = time.time()
            direct_cfg = OptimizeConfig(
                num_vehicles=max(3, math.ceil(sum(cmp_demands) / cmp_capacity) + 2),
                vehicle_capacity_kg=cmp_capacity, depot_index=0, use_gls=cmp_use_gls,
                first_solution_strategy=first_solution_strategy, time_limit_sec=cmp_time_budget,
                max_route_time_s=max_route_s,
            )
            direct_routes, direct_solved, direct_msg, _n = solve_cvrp_with_auto_scaling(
                dist_cmp, dur_cmp, cmp_demands, cmp_service_s, direct_cfg, max_extra_vehicles=5,
            )
            t_direct = time.time() - t0

        with st.spinner("3/3 — Đang chạy Mô hình đề xuất (K-means++ phân cụm + CVRP theo cụm)..."):
            t0 = time.time()
            cluster_result = run_clustering_cvrp(
                list(cmp_coords), cmp_depot_idx, cmp_demands, cmp_service_s, dist_cmp, dur_cmp, cmp_node_ids,
                vehicle_capacity_kg=cmp_capacity, use_gls=cmp_use_gls,
                first_solution_strategy=first_solution_strategy, time_limit_sec=cmp_time_budget,
                max_route_time_s=max_route_s,
            )
            t_cluster = time.time() - t0

        st.session_state.cluster_compare = {
            "df": cmp_df, "coords": cmp_coords,
            "nn_routes": nn_routes, "t_nn": t_nn,
            "direct_routes": direct_routes if direct_solved else [], "direct_solved": direct_solved,
            "direct_msg": direct_msg, "t_direct": t_direct,
            "cluster_result": cluster_result, "t_cluster": t_cluster,
        }

if st.session_state.get("cluster_compare"):
    cc = st.session_state.cluster_compare
    cmp_df = cc["df"]

    def _summarize(routes):
        return {
            "Số xe": len(routes),
            "Tổng quãng đường (km)": round(sum(r.total_distance_m for r in routes) / 1000, 2),
            "Tổng thời gian tuyến (phút)": round(sum(r.total_route_time_s for r in routes) / 60, 1),
        }

    nn_kpi = _summarize(cc["nn_routes"])
    direct_kpi = _summarize(cc["direct_routes"]) if cc["direct_solved"] else None
    cluster_kpi = _summarize(cc["cluster_result"].routes)
    base_km = nn_kpi["Tổng quãng đường (km)"]

    rows = [{
        "Phương pháp": "1. Đường đi ngắn nhất (Nearest Neighbor, không phân cụm)",
        "Số cụm": "-", "Số xe": nn_kpi["Số xe"],
        "Quãng đường (km)": nn_kpi["Tổng quãng đường (km)"],
        "Giảm so với PP1 (%)": 0.0,
        "Runtime thuật toán (giây)": round(cc["t_nn"], 2),
    }]
    if direct_kpi:
        rows.append({
            "Phương pháp": "2. CVRP trực tiếp (OR-Tools + GLS, không phân cụm)",
            "Số cụm": "-", "Số xe": direct_kpi["Số xe"],
            "Quãng đường (km)": direct_kpi["Tổng quãng đường (km)"],
            "Giảm so với PP1 (%)": round((base_km - direct_kpi["Tổng quãng đường (km)"]) / base_km * 100, 1),
            "Runtime thuật toán (giây)": round(cc["t_direct"], 2),
        })
    else:
        rows.append({
            "Phương pháp": "2. CVRP trực tiếp (OR-Tools + GLS, không phân cụm)",
            "Số cụm": "-", "Số xe": "-", "Quãng đường (km)": "-",
            "Giảm so với PP1 (%)": "-", "Runtime thuật toán (giây)": round(cc["t_direct"], 2),
        })
    rows.append({
        "Phương pháp": "3. Mô hình đề xuất (K-means++ phân cụm + CVRP theo cụm)",
        "Số cụm": cc["cluster_result"].clustering.k, "Số xe": cluster_kpi["Số xe"],
        "Quãng đường (km)": cluster_kpi["Tổng quãng đường (km)"],
        "Giảm so với PP1 (%)": round((base_km - cluster_kpi["Tổng quãng đường (km)"]) / base_km * 100, 1),
        "Runtime thuật toán (giây)": round(cc["t_cluster"], 2),
    })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not cc["cluster_result"].clustering.balanced:
        st.warning(
            "Một số cụm sau cân bằng vẫn vượt nhẹ sức chứa xe (dữ liệu quá khít so với capacity). "
            "Thử tăng Q hoặc giảm số điểm để có dư địa cân bằng tốt hơn."
        )

    st.caption(
        "Lưu ý đọc kết quả: (1) Số xe của Mô hình đề xuất thường ≥ CVRP trực tiếp vì k được xác định "
        "trước theo công thức capacity (có biên an toàn), không phải kết quả tối ưu hoá số xe như "
        "CVRP trực tiếp. (2) Runtime của Mô hình đề xuất là TỔNG thời gian giải TUẦN TỰ từng cụm "
        "(mỗi cụm cần tối thiểu ~2s để OR-Tools khởi tạo) — nếu triển khai giải SONG SONG các cụm "
        "(parallel), runtime thực tế có thể giảm gần bằng runtime của cụm chậm nhất thay vì tổng cộng "
        "dồn lại. Đây là điểm cần nêu rõ trong phần thảo luận/hạn chế của đề tài."
    )

    with st.expander("🗺️ Bản đồ cụm & tuyến — Mô hình đề xuất (K-means++ + CVRP)", expanded=False):
        cmap = folium.Map(location=[DEPOT_LOCATION[0], DEPOT_LOCATION[1]], zoom_start=14, tiles="cartodbpositron")
        folium.Marker(DEPOT_LOCATION, popup="DEPOT", icon=folium.Icon(color="black", icon="home")).add_to(cmap)
        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
            "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
            "#808000", "#ffd8b1", "#000075", "#808080", "#000000", "#a9a9a9",
        ]
        clustering_labels = cc["cluster_result"].clustering.labels
        for idx, cluster_id in clustering_labels.items():
            row = cmp_df.iloc[idx]
            color = palette[cluster_id % len(palette)]
            folium.CircleMarker(
                [row["latitude"], row["longitude"]], radius=5, color=color, fill=True, fill_opacity=0.9,
                popup=f"{row['node_id']} - cụm {cluster_id} - {row['waste_kg']:.0f} kg",
            ).add_to(cmap)
        for r in cc["cluster_result"].routes:
            route_coords = [(cmp_df.iloc[i]["latitude"], cmp_df.iloc[i]["longitude"]) for i in r.node_sequence]
            folium.PolyLine(route_coords, color="#555555", weight=2, opacity=0.6).add_to(cmap)
        st_folium(cmap, use_container_width=True, height=550, returned_objects=[], key="cluster_map")
        st.caption(
            "Mỗi màu = 1 cụm (K-means++). Đường nối là tuyến CVRP trong cụm đó (vẽ đường thẳng nối "
            "điểm để xem nhanh cấu trúc cụm — không phải road geometry OSRM thật; xem road geometry "
            "thật ở bản đồ chính phía trên nếu cần)."
        )



# ============================================================================
# TỐI ƯU ĐA LUỒNG THEO ĐỀ ÁN PHÂN LOẠI RÁC TẠI NGUỒN
# ============================================================================
st.subheader("♻️ Tối ưu tuyến theo từng luồng rác (đề án phân loại rác tại nguồn)")
st.caption(
    "Theo Luật Bảo vệ môi trường 2020 và lộ trình TP.HCM: đa số điểm phân loại 2 nhóm "
    "(Tái chế / Còn lại); riêng nhóm phát sinh nhiều rác thực phẩm (chợ, nhà hàng, khách sạn, "
    "TTTM có dịch vụ ăn uống) đang thí điểm phân 3 nhóm (thêm Thực phẩm riêng). Mỗi luồng được "
    "tối ưu như MỘT bài toán CVRP độc lập bằng OR-Tools + GLS, tái sử dụng ma trận OSRM đã cache "
    "(không gọi lại OSRM), để so sánh công bằng baseline vs optimized trong từng luồng."
)

# ---- 📊 Dashboard thành phần rác (dựa trên dữ liệu điểm hiện tại) ----
st.markdown("#### 📊 Dashboard thành phần rác")

_total_recyclable = float(df_points["waste_recyclable_kg"].sum())
_total_food = float(df_points["waste_food_kg"].sum())
_total_other = float(df_points["waste_other_kg"].sum())
_total_all = _total_recyclable + _total_food + _total_other
_n_food_generators = int(df_points["is_major_food_generator"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tổng khối lượng rác", f"{_total_all:,.0f} kg")
m2.metric("Tái chế", f"{_total_recyclable:,.0f} kg", f"{_total_recyclable/_total_all*100:.1f}%" if _total_all else "0%")
m3.metric("Thực phẩm riêng", f"{_total_food:,.0f} kg", f"{_total_food/_total_all*100:.1f}%" if _total_all else "0%")
m4.metric("Điểm phát sinh nhiều thực phẩm", f"{_n_food_generators} điểm")

dash_col1, dash_col2 = st.columns([1, 1])
with dash_col1:
    _pie_df = pd.DataFrame({
        "Luồng": ["Tái chế", "Thực phẩm (nhóm phát sinh nhiều)", "Còn lại"],
        "Khối lượng (kg)": [_total_recyclable, _total_food, _total_other],
    })
    _pie_df = _pie_df[_pie_df["Khối lượng (kg)"] > 0]
    if not _pie_df.empty:
        fig_pie = px.pie(
            _pie_df, names="Luồng", values="Khối lượng (kg)",
            color="Luồng",
            color_discrete_map={
                "Tái chế": "#2ca02c", "Thực phẩm (nhóm phát sinh nhiều)": "#ff7f0e", "Còn lại": "#7f7f7f",
            },
            title="Tỷ trọng khối lượng theo luồng rác",
            hole=0.35,
        )
        fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
with dash_col2:
    _top_df = df_points[~df_points["is_depot"]].nlargest(10, "waste_kg")[["node_id", "waste_kg", "is_major_food_generator"]]
    _top_df = _top_df.rename(columns={
        "node_id": "Điểm", "waste_kg": "Khối lượng (kg)", "is_major_food_generator": "Phát sinh nhiều thực phẩm",
    })
    fig_bar = px.bar(
        _top_df.sort_values("Khối lượng (kg)"), x="Khối lượng (kg)", y="Điểm", orientation="h",
        color="Phát sinh nhiều thực phẩm",
        color_discrete_map={True: "#ff7f0e", False: "#4c72b0"},
        title="Top 10 điểm phát sinh khối lượng rác lớn nhất",
    )
    fig_bar.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320, showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)

with st.expander("📋 Bảng chi tiết khối lượng theo điểm", expanded=False):
    detail_cols = [
        "node_id", "waste_kg", "waste_recyclable_kg", "waste_food_kg", "waste_other_kg",
        "is_major_food_generator", "requires_small_vehicle",
    ]
    st.dataframe(
        df_points[~df_points["is_depot"]][detail_cols].rename(columns={
            "node_id": "Điểm", "waste_kg": "Tổng (kg)", "waste_recyclable_kg": "Tái chế (kg)",
            "waste_food_kg": "Thực phẩm (kg)", "waste_other_kg": "Còn lại (kg)",
            "is_major_food_generator": "Phát sinh nhiều thực phẩm", "requires_small_vehicle": "Cần xe nhỏ (hẻm)",
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()



with st.expander("⚙️ Cấu hình đội xe theo từng luồng rác", expanded=False):
    stream_vehicle_cfg = {}
    for key, meta in WASTE_STREAMS.items():
        c1, c2 = st.columns(2)
        with c1:
            nv = st.slider(f"Số xe – {meta['label']}", 1, 6, 2, key=f"stream_nv_{key}")
        with c2:
            cap = st.number_input(
                f"Capacity xe (kg) – {meta['label']}", value=800 if key != "recyclable" else 500,
                step=50, key=f"stream_cap_{key}",
            )
        stream_vehicle_cfg[key] = {"num_vehicles": nv, "vehicle_capacity_kg": cap}

run_streams_btn = st.button("♻️ Chạy tối ưu đa luồng theo rác đã phân loại")

if run_streams_btn:
    depot_idx_global = int(df_points.index[df_points["is_depot"]][0])
    stream_results = {}
    for key in WASTE_STREAMS:
        problem = build_stream_problem(
            df_points, dist_m_full, dur_s_full, service_times_s_full, key, depot_idx_global
        )
        if problem is None:
            stream_results[key] = None
            continue
        with st.spinner(f"Đang tối ưu luồng '{WASTE_STREAMS[key]['label']}'..."):
            stream_results[key] = run_stream_optimization(
                problem,
                num_vehicles=stream_vehicle_cfg[key]["num_vehicles"],
                vehicle_capacity_kg=stream_vehicle_cfg[key]["vehicle_capacity_kg"],
                use_gls=use_gls,
                first_solution_strategy=first_solution_strategy,
                time_limit_sec=min(time_limit_sec, 20),
                max_route_time_s=max_route_hours * 3600,
            )
    st.session_state.stream_results = stream_results

if st.session_state.get("stream_results"):
    stream_results = st.session_state.stream_results
    total_after_km = 0.0
    total_after_vehicles = 0
    stream_kpi_rows = []

    for key, meta in WASTE_STREAMS.items():
        result = stream_results.get(key)
        st.markdown(f"**Luồng: {meta['label']}**")
        if result is None:
            st.caption("Không có điểm nào phát sinh khối lượng cho luồng này trong dữ liệu hiện tại.")
            continue
        if not result.solved:
            st.error(f"Không tối ưu được luồng '{meta['label']}': {result.message}")
            continue

        kpi_base = aggregate_kpi(result.baseline_routes)
        kpi_opt = aggregate_kpi(result.optimized_routes)
        total_after_km += kpi_opt["Tổng quãng đường (km)"]
        total_after_vehicles += kpi_opt["Số xe"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Baseline (NN):", kpi_base)
        with col_b:
            st.write("Optimized (OR-Tools+GLS):", kpi_opt)

        for r in result.optimized_routes:
            names = [result.problem.node_ids[i] for i in r.node_sequence]
            st.text(f"  Vehicle {r.vehicle_id}: " + " → ".join(names))

        stream_kpi_rows.append({
            "Luồng": meta["label"],
            "Số xe (optimized)": kpi_opt["Số xe"],
            "Quãng đường (km)": kpi_opt["Tổng quãng đường (km)"],
            "Khối lượng (kg)": kpi_opt["Khối lượng thu gom (kg)"],
        })

    st.divider()
    st.markdown("**So sánh: TRƯỚC phân loại (1 luồng gộp) vs SAU phân loại (tổng các luồng riêng)**")
    before_km = sum(r.total_distance_m for r in optimized_routes) / 1000.0
    before_vehicles = len(optimized_routes)
    compare_df = pd.DataFrame([
        {"Kịch bản": "Trước phân loại (1 luồng gộp – Optimized)", "Tổng quãng đường (km)": round(before_km, 2), "Tổng số xe": before_vehicles},
        {"Kịch bản": "Sau phân loại (tổng các luồng riêng – Optimized)", "Tổng quãng đường (km)": round(total_after_km, 2), "Tổng số xe": total_after_vehicles},
    ])
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
    delta_km = total_after_km - before_km
    st.caption(
        f"Chênh lệch: {delta_km:+.2f} km, {total_after_vehicles - before_vehicles:+d} xe so với gộp chung 1 luồng. "
        "Việc tách luồng theo phân loại rác THƯỜNG làm tăng tổng quãng đường/số xe (mỗi luồng phải "
        "chạy tuyến riêng), nhưng đổi lại tách bạch được dòng rác tái chế (miễn phí thu gom theo quy "
        "định giá dịch vụ TP.HCM) và rác thực phẩm (giảm khối lượng chôn lấp, phù hợp lộ trình giảm "
        "chôn lấp còn 20% vào 2025) — đây là đánh đổi giữa hiệu quả vận tải và mục tiêu môi trường, "
        "nên đưa vào phần đánh giá của đề tài."
    )

st.divider()

# ============================================================================
# DYNAMIC ROUTING (GPS TỰ ĐỘNG) - KHÔNG cần tài xế bấm nút xác nhận
# ============================================================================
st.subheader("🛰️ Dynamic Routing – GPS tự động (Auto completion & Auto re-optimize)")
st.caption(
    "Xe được theo dõi qua GPS điện thoại. Khi xe vào bán kính điểm thu gom và đứng đủ lâu "
    "→ điểm tự động chuyển pending → completed. Khi xe vào bán kính DEPOT và đứng đủ lâu "
    "→ hệ thống tự xác nhận 'xe đã về DEPOT' và TỰ ĐỘNG tái tối ưu (OR-Tools + GLS) cho các "
    "điểm còn pending, KHÔNG cần tài xế bấm bất kỳ nút xác nhận nào."
)

vehicle_options = {f"Vehicle {r.vehicle_id} ({r.num_stops} điểm)": r for r in optimized_routes}

with st.expander("⚙️ Cấu hình Dynamic Routing", expanded=st.session_state.get("gps_enabled", False)):
    gps_enabled = st.checkbox("Bật theo dõi GPS tự động cho 1 xe", key="gps_enabled")
    chosen_label = st.selectbox("Chọn xe để theo dõi GPS", list(vehicle_options.keys()))
    gps_mode = st.radio(
        "Nguồn GPS",
        ["Mô phỏng GPS (demo/test, không cần thiết bị)", "GPS thực từ điện thoại (thử nghiệm)"],
    )
    col1, col2 = st.columns(2)
    with col1:
        geofence_radius_m = st.slider("Bán kính auto-completion & depot (m)", 30, 50, 40)
    with col2:
        dwell_seconds = st.slider("Thời gian lưu tối thiểu trong vùng (giây)", 5, 30, 10)

    if gps_mode.startswith("Mô phỏng"):
        sim_speed_kmh = st.slider("Tốc độ xe mô phỏng (km/h)", 5, 40, 20)
        sim_accel = st.slider("Tăng tốc mô phỏng (số giây mô phỏng / lần refresh)", 1, 20, 6)
    else:
        sim_speed_kmh, sim_accel = 20, 6
        if not HAS_GEOLOCATION:
            st.error(
                "Chưa cài được package streamlit-geolocation trong môi trường này. "
                "Hãy `pip install streamlit-geolocation` rồi chạy lại."
            )
        st.info(
            "Lưu ý: trình duyệt yêu cầu quyền định vị và (tuỳ thiết bị/trình duyệt) có thể cần "
            "tap lại nút định vị để cấp phép — đây là giới hạn bảo mật của trình duyệt, "
            "không phải giới hạn của logic auto-completion/auto re-optimize."
        )

    if st.button("🔄 Khởi tạo / Reset theo dõi GPS cho xe đã chọn"):
        chosen_route = vehicle_options[chosen_label]
        points_for_engine = [
            {
                "node_id": node_ids_full[i],
                "latitude": df_points.iloc[i]["latitude"],
                "longitude": df_points.iloc[i]["longitude"],
            }
            for i in chosen_route.node_sequence
            if not df_points.iloc[i]["is_depot"]
        ]
        engine = DynamicRoutingEngine(
            points_for_engine,
            depot_latlon=DEPOT_LOCATION,
            config=GPSTrackerConfig(
                completion_radius_m=geofence_radius_m,
                depot_radius_m=geofence_radius_m,
                dwell_seconds_required=dwell_seconds,
            ),
        )
        active_coords = tuple(coords[i] for i in chosen_route.node_sequence)
        active_geometry = get_osrm_route_geometry(active_coords, base_url=res["osrm_base_url"])
        if not active_geometry:
            active_geometry = list(active_coords)

        st.session_state.gps_engine = engine
        st.session_state.gps_active_route_nodes = list(chosen_route.node_sequence)
        st.session_state.gps_active_geometry = active_geometry
        st.session_state.gps_sim_time_s = 0.0
        st.session_state.gps_sim_progress_m = 0.0
        st.session_state.gps_event_log = ["Đã khởi tạo theo dõi GPS cho " + chosen_label]
        st.session_state.gps_tour_finished = False
        st.rerun()

if gps_enabled and "gps_engine" in st.session_state:
    engine: DynamicRoutingEngine = st.session_state.gps_engine

    # Tick tự động ~1.5s/lần để mô phỏng/đọc GPS liên tục (không cần bấm nút)
    st_autorefresh(interval=1500, key="gps_autorefresh_tick")

    def _handle_event(ev: dict):
        if ev["newly_completed"]:
            for nid in ev["newly_completed"]:
                st.session_state.gps_event_log.append(f"✅ Auto-completed: {nid}")
        if ev["depot_confirmed"]:
            st.session_state.gps_event_log.append("🏠 Auto depot detected: xe đã về DEPOT")
            pending_ids = engine.pending_node_ids()
            if not pending_ids:
                st.session_state.gps_tour_finished = True
                st.session_state.gps_event_log.append("🎉 Đã hoàn thành toàn bộ tuyến.")
                return
            # ---- AUTO RE-OPTIMIZATION: chỉ các điểm pending, DEPOT là điểm xuất phát ----
            sub_indices = [index_of_node[depot_node_id]] + [index_of_node[nid] for nid in pending_ids]
            sub_dist = [[dist_m_full[a][b] for b in sub_indices] for a in sub_indices]
            sub_dur = [[dur_s_full[a][b] for b in sub_indices] for a in sub_indices]
            sub_demands = [demands_full[i] for i in sub_indices]
            sub_service = [service_times_s_full[i] for i in sub_indices]

            reopt_cfg = OptimizeConfig(
                num_vehicles=1,  # cùng 1 xe vật lý tiếp tục hành trình
                vehicle_capacity_kg=vehicle_capacity_kg,
                depot_index=0,
                use_gls=use_gls,
                first_solution_strategy=first_solution_strategy,
                time_limit_sec=min(time_limit_sec, 15),
                max_route_time_s=max_route_hours * 3600,
            )
            new_routes, solved, msg, _n = solve_cvrp_with_auto_scaling(
                sub_dist, sub_dur, sub_demands, sub_service, reopt_cfg, max_extra_vehicles=0
            )
            if solved and new_routes:
                new_route_global_nodes = [sub_indices[i] for i in new_routes[0].node_sequence]
                st.session_state.gps_active_route_nodes = new_route_global_nodes
                new_active_coords = tuple(coords[i] for i in new_route_global_nodes)
                new_geom = get_osrm_route_geometry(new_active_coords, base_url=res["osrm_base_url"])
                st.session_state.gps_active_geometry = new_geom or list(new_active_coords)
                st.session_state.gps_sim_progress_m = 0.0
                engine.sync_pending_after_reoptimize(pending_ids)
                st.session_state.gps_event_log.append(
                    f"🔁 Auto re-optimized: DEPOT → {len(pending_ids)} điểm pending còn lại "
                    f"({new_routes[0].total_distance_m/1000:.2f} km)"
                )
            else:
                st.session_state.gps_event_log.append(f"⚠️ Re-optimize thất bại: {msg}")

    if not st.session_state.get("gps_tour_finished", False):
        if gps_mode.startswith("Mô phỏng"):
            speed_mps = sim_speed_kmh * 1000 / 3600
            geometry = st.session_state.gps_active_geometry
            for _ in range(int(sim_accel)):
                st.session_state.gps_sim_time_s += 1.0
                st.session_state.gps_sim_progress_m += speed_mps * 1.0
                lat, lon, finished = interpolate_along_route(geometry, st.session_state.gps_sim_progress_m)
                if lat is None:
                    break
                ev = engine.update_position(lat, lon, ts=st.session_state.gps_sim_time_s)
                _handle_event(ev)
                if st.session_state.get("gps_tour_finished", False):
                    break
        else:
            if HAS_GEOLOCATION:
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                    ev = engine.update_position(loc["latitude"], loc["longitude"], ts=time.time())
                    _handle_event(ev)

    # ---- Hiển thị trạng thái ----
    col_map, col_status = st.columns([2, 1])
    with col_status:
        st.markdown("**Trạng thái điểm thu gom**")
        status_rows = [
            {"node_id": nid, "status": s.status}
            for nid, s in engine.point_status.items()
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True, height=250)
        st.metric("Điểm còn pending", len(engine.pending_node_ids()))
        if st.session_state.get("gps_tour_finished"):
            st.success("Xe đã hoàn thành toàn bộ tuyến (tất cả điểm completed).")
        with st.expander("Nhật ký sự kiện", expanded=True):
            for line in st.session_state.gps_event_log[-15:][::-1]:
                st.text(line)

    with col_map:
        gmap = folium.Map(location=[DEPOT_LOCATION[0], DEPOT_LOCATION[1]], zoom_start=15, tiles="cartodbpositron")
        folium.Marker(DEPOT_LOCATION, popup="DEPOT", icon=folium.Icon(color="black", icon="home")).add_to(gmap)
        for nid, s in engine.point_status.items():
            plat, plon = engine.point_coords[nid]
            color = "#2ca02c" if s.status == "completed" else "#ff7f0e"
            folium.CircleMarker(
                [plat, plon], radius=6, color=color, fill=True, fill_opacity=0.9,
                popup=f"{nid} - {s.status}",
            ).add_to(gmap)
        folium.PolyLine(
            st.session_state.gps_active_geometry, color="#9467bd", weight=4, opacity=0.7,
            tooltip="Tuyến đang chạy (auto re-optimize khi về Depot)",
        ).add_to(gmap)
        if engine.current_position:
            folium.Marker(
                engine.current_position,
                popup="Xe (GPS hiện tại)",
                icon=folium.Icon(color="blue", icon="truck", prefix="fa"),
            ).add_to(gmap)
        st_folium(gmap, use_container_width=True, height=500, returned_objects=[], key="gps_map")
elif gps_enabled:
    st.info("Nhấn **'Khởi tạo / Reset theo dõi GPS cho xe đã chọn'** ở trên để bắt đầu.")

st.divider()

# ============================================================================
# FLEET REAL-TIME SIMULATION — nhiều xe cùng lúc, đầy tải/hỏng xe/chặn đường
# (Bổ sung theo mục 14–21, 34, 38 của đặc tả gốc — dùng module simulation.py)
# ============================================================================
st.subheader("🚦 Mô phỏng vận hành đội xe thời gian thực (Fleet Real-time Simulation)")
st.caption(
    "Mô phỏng TẤT CẢ xe trong tuyến Optimized ở trên di chuyển cùng lúc theo road geometry OSRM. "
    "Hệ thống tự động xử lý đầy tải, lượng rác thực tế khác dự báo, xe hỏng và đường bị chặn — khi "
    "trạng thái thay đổi, tự động tái tối ưu (OR-Tools + GLS) phần còn lại. Đồng hồ mô phỏng là ảo "
    "(tăng tốc để tiện demo), không phải thời gian thực tuyệt đối."
)

FLEET_STATE_ICON = {
    STATE_IDLE: "⚪", STATE_MOVING: "🚛", STATE_COLLECTING: "📥", STATE_FULL: "🟠",
    STATE_RETURNING_TO_DEPOT: "↩️", STATE_BROKEN: "🔴", STATE_BLOCKED: "⚠️", STATE_COMPLETED: "🎉",
}
FLEET_STATE_COLOR = {
    STATE_IDLE: "#999999", STATE_MOVING: "#1f77b4", STATE_COLLECTING: "#2ca02c", STATE_FULL: "#ff7f0e",
    STATE_RETURNING_TO_DEPOT: "#9467bd", STATE_BROKEN: "#d62728", STATE_BLOCKED: "#d62728",
    STATE_COMPLETED: "#2ca02c",
}


def _fleet_geometry_fetch_factory(base_url):
    def _fetch(a, b):
        ordered = (coords[a], coords[b])
        geom = get_osrm_route_geometry(ordered, base_url=base_url)
        return geom if geom else [coords[a], coords[b]]
    return _fetch


def _fleet_reoptimize_nodes(node_indices, base_url, capacity_kg):
    """Giải CVRP 1 xe cho danh sách node (dùng khi 1 xe cụ thể tái tối ưu
    phần pending còn lại của chính nó, hoặc để phân bổ pool node chung cho
    1 xe đang rảnh). Trả về (global_sequence | None, message)."""
    sub_indices = [depot_index] + list(node_indices)
    sub_dist = [[dist_m_full[a][b] for b in sub_indices] for a in sub_indices]
    sub_dur = [[dur_s_full[a][b] for b in sub_indices] for a in sub_indices]
    sub_demands = [demands_full[i] for i in sub_indices]
    sub_service = [service_times_s_full[i] for i in sub_indices]
    cfg = OptimizeConfig(
        num_vehicles=1, vehicle_capacity_kg=capacity_kg, depot_index=0,
        use_gls=use_gls, first_solution_strategy=first_solution_strategy,
        time_limit_sec=min(time_limit_sec, 10),
    )
    routes, solved, msg, _n = solve_cvrp_with_auto_scaling(
        sub_dist, sub_dur, sub_demands, sub_service, cfg, max_extra_vehicles=0,
    )
    if not solved or not routes:
        return None, msg
    global_seq = [sub_indices[i] for i in routes[0].node_sequence]
    return global_seq, "ok"


with st.expander("⚙️ Cấu hình mô phỏng đội xe", expanded=st.session_state.get("fleet_sim_enabled", False)):
    fleet_sim_enabled = st.checkbox("Bật mô phỏng đội xe thời gian thực", key="fleet_sim_enabled")
    col1, col2, col3 = st.columns(3)
    with col1:
        fleet_speed_kmh = st.slider("Tốc độ xe mô phỏng (km/h)", 10, 60, 25, key="fleet_speed_kmh")
    with col2:
        fleet_sim_accel = st.slider("Giây mô phỏng / lần refresh", 5, 120, 30, key="fleet_sim_accel")
    with col3:
        fleet_full_threshold = st.slider("Ngưỡng coi là đầy tải (%)", 80, 100, 95, key="fleet_full_threshold")

    st.markdown("**Chi phí vận hành (giả định, có thể chỉnh — mục 22 đặc tả)**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fleet_fuel_cost_per_km = st.number_input("Fuel cost/km (VNĐ)", value=4500, step=100)
    with c2:
        fleet_driver_cost_per_hour = st.number_input("Driver cost/giờ (VNĐ)", value=45000, step=1000)
    with c3:
        fleet_maint_cost_per_km = st.number_input("Maintenance/km (VNĐ)", value=800, step=50)
    with c4:
        fleet_other_cost = st.number_input("Chi phí khác (VNĐ)", value=0, step=1000)
    st.caption("Cost parameters are configurable assumptions — không phải chi phí thực tế của doanh nghiệp cụ thể nào.")

    if st.button("🚀 Khởi tạo / Reset mô phỏng đội xe (dùng tuyến Optimized ở trên)"):
        fleet_sim_obj = FleetSimulator(
            depot_index=depot_index,
            depot_coords=coords[depot_index],
            demands_predicted={i: demands_full[i] for i in range(len(demands_full))},
            service_times_s={i: service_times_s_full[i] for i in range(len(service_times_s_full))},
            vehicle_capacity_kg=vehicle_capacity_kg,
            full_load_threshold_pct=float(fleet_full_threshold),
        )
        geometry_fetch = _fleet_geometry_fetch_factory(res["osrm_base_url"])
        with st.spinner("Đang lấy road geometry từng chặng (OSRM) cho tất cả xe..."):
            for r in optimized_routes:
                fleet_sim_obj.add_truck(r.vehicle_id, capacity_kg=vehicle_capacity_kg, speed_kmh=fleet_speed_kmh)
                legs = build_route_legs(r.node_sequence, geometry_fetch)
                fleet_sim_obj.assign_route(r.vehicle_id, r.node_sequence, legs)
        st.session_state.fleet_sim = fleet_sim_obj
        st.session_state.fleet_sim_running = True
        st.rerun()

if st.session_state.get("fleet_sim_enabled") and "fleet_sim" in st.session_state:
    fsim: FleetSimulator = st.session_state.fleet_sim

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        running_now = st.session_state.get("fleet_sim_running", True)
        if st.button("⏸️ Pause" if running_now else "▶️ Resume"):
            st.session_state.fleet_sim_running = not running_now
    with ctrl2:
        broken_target = st.selectbox("Chọn xe", [t.vehicle_id for t in fsim.trucks.values()], key="fleet_broken_pick")
    with ctrl3:
        if st.button("🔴 Đánh dấu xe hỏng"):
            fsim.mark_broken(broken_target)
    with ctrl4:
        if st.button("🔧 Sửa xong, đưa xe về Depot"):
            fsim.repair_truck(broken_target)

    with st.expander("⚠️ Mô phỏng sự cố: chặn đường / rác thực tế vượt dự báo", expanded=False):
        bc1, bc2 = st.columns(2)
        with bc1:
            node_a = st.selectbox("Điểm A", node_ids_full, key="fleet_block_a")
        with bc2:
            node_b = st.selectbox(
                "Điểm B", node_ids_full, key="fleet_block_b", index=min(1, len(node_ids_full) - 1)
            )
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🚧 Chặn đoạn đường A↔B"):
                fsim.block_road(index_of_node[node_a], index_of_node[node_b])
        with bcol2:
            if st.button("✅ Gỡ chặn A↔B"):
                fsim.unblock_road(index_of_node[node_a], index_of_node[node_b])

        variance_node = st.selectbox("Điểm cần cập nhật rác thực tế", node_ids_full, key="fleet_variance_node")
        variance_factor = st.slider(
            "Rác thực tế = dự báo × hệ số", 0.5, 3.0, 1.5, step=0.1, key="fleet_variance_factor"
        )
        if st.button("📈 Cập nhật lượng rác thực tế tại điểm này"):
            vi = index_of_node[variance_node]
            predicted = fsim.demands_predicted.get(vi, 0.0)
            fsim.set_actual_waste(vi, predicted * variance_factor)

    # ---- Tick mô phỏng tự động + AUTO RE-OPTIMIZATION (mục 20) ----
    if st.session_state.get("fleet_sim_running", True):
        st_autorefresh(interval=1800, key="fleet_sim_autorefresh")
        fsim.tick(float(fleet_sim_accel))

        geometry_fetch = _fleet_geometry_fetch_factory(res["osrm_base_url"])
        for t in list(fsim.trucks.values()):
            if t.status == STATE_FULL and t.needs_reopt:
                # Xe vừa xác định sẽ vượt tải giữa tuyến -> cần leg OSRM thật
                # từ vị trí hiện tại về depot (prototype: dùng vị trí xe làm
                # điểm xuất phát cho OSRM Route Service).
                depot_coords_now = coords[depot_index]
                geom = get_osrm_route_geometry(
                    (t.position, depot_coords_now), base_url=res["osrm_base_url"]
                ) or [t.position, depot_coords_now]
                leg = RouteLeg(from_node=t.current_node or depot_index, to_node=depot_index, geometry=geom)
                fsim.redirect_to_depot(t.vehicle_id, leg)
                t.needs_reopt = False
            elif t.status == STATE_IDLE and t.needs_reopt and t.pending_nodes:
                seq, msg = _fleet_reoptimize_nodes(t.pending_nodes, res["osrm_base_url"], t.capacity_kg)
                if seq:
                    legs = build_route_legs(seq, geometry_fetch)
                    fsim.assign_route(t.vehicle_id, seq, legs)
                else:
                    fsim.log(f"⚠️ Truck {t.vehicle_id}: tái tối ưu thất bại ({msg}), chờ vòng sau.")
                t.needs_reopt = False

        # ---- Phân bổ pool node chưa gán (xe hỏng / phát sinh mới) cho xe rảnh ----
        pool = fsim.pending_pool()
        idle = fsim.idle_trucks()
        if pool and idle:
            seq, msg = _fleet_reoptimize_nodes(pool, res["osrm_base_url"], vehicle_capacity_kg)
            if seq:
                legs = build_route_legs(seq, geometry_fetch)
                fsim.assign_route(idle[0], seq, legs)
                fsim.clear_pending_pool()
            else:
                fsim.log(f"⚠️ Không thể phân bổ pool node chờ ({msg}).")

    # ---- DASHBOARD KPI (mục 21) ----
    cost_params = {
        "fuel_cost_per_km": fleet_fuel_cost_per_km, "driver_cost_per_hour": fleet_driver_cost_per_hour,
        "maintenance_cost_per_km": fleet_maint_cost_per_km, "other_cost": fleet_other_cost,
    }
    kpi = fsim.dashboard_kpis(cost_params)
    st.markdown(f"**⏱️ Đồng hồ mô phỏng: {kpi['clock']}** — Vehicle operating window: 06:00–09:00 & 16:00–20:00")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total predicted", f"{kpi['total_predicted_waste_kg']:,.0f} kg")
    k2.metric("Collected", f"{kpi['collected_waste_kg']:,.0f} kg")
    k3.metric("Remaining", f"{kpi['remaining_waste_kg']:,.0f} kg")
    k4.metric("Vehicles (active/broken)", f"{kpi['active_vehicles']}/{kpi['broken_vehicles']}")
    k5.metric("Total distance", f"{kpi['total_distance_km']:,.2f} km")
    k6.metric("Service level", f"{kpi['service_level_pct']:.1f}%")
    k7, k8, k9 = st.columns(3)
    k7.metric("Avg utilization", f"{kpi['avg_utilization_pct']:.1f}%")
    k8.metric("Total cost (ước tính)", f"{kpi['total_cost']:,.0f} VNĐ")
    k9.metric("Total time", f"{kpi['total_time_h']:.2f} giờ")

    # ---- MAP (mục 23) ----
    fleet_map = folium.Map(
        location=[coords[depot_index][0], coords[depot_index][1]], zoom_start=15, tiles="cartodbpositron"
    )
    folium.Marker(coords[depot_index], popup="DEPOT", icon=folium.Icon(color="green", icon="home")).add_to(fleet_map)
    for i, row in df_points.iterrows():
        if row["is_depot"]:
            continue
        folium.CircleMarker(
            [row["latitude"], row["longitude"]], radius=4, color="#888888", fill=True, fill_opacity=0.6,
            popup=f"{row['node_id']}",
        ).add_to(fleet_map)
    for a, b in fsim.blocked_edges:
        folium.PolyLine(
            [coords[a], coords[b]], color="red", weight=6, opacity=0.6, dash_array="4,8",
            tooltip="⚠️ Road blockage",
        ).add_to(fleet_map)
    for t in fsim.trucks.values():
        if t.position is None:
            continue
        icon_char = FLEET_STATE_ICON.get(t.status, "🚛")
        color = FLEET_STATE_COLOR.get(t.status, "#1f77b4")
        folium.Marker(
            t.position,
            popup=f"Truck {t.vehicle_id} — {t.status} — {t.current_load_kg:.0f}/{t.capacity_kg:.0f}kg",
            icon=folium.DivIcon(html=f'<div style="font-size:22px;">{icon_char}</div>'),
        ).add_to(fleet_map)
        if t.leg_idx < len(t.legs):
            leg = t.legs[t.leg_idx]
            if leg.geometry:
                folium.PolyLine(leg.geometry, color=color, weight=3, opacity=0.6).add_to(fleet_map)
    st_folium(fleet_map, use_container_width=True, height=550, returned_objects=[], key="fleet_sim_map")

    # ---- VEHICLE TABLE (mục 25) ----
    st.markdown("**🚛 Vehicle Status**")
    st.dataframe(pd.DataFrame(fsim.vehicle_table()), use_container_width=True, hide_index=True)

    # ---- ROUTE TABLE (mục 26) ----
    st.markdown("**🗺️ Route Status (còn lại)**")
    st.dataframe(
        pd.DataFrame(fsim.route_table(lambda idx: node_ids_full[idx] if idx is not None else "-")),
        use_container_width=True, hide_index=True,
    )

    # ---- EVENT LOG (mục 34) ----
    with st.expander("📋 Event Log", expanded=True):
        for e in fsim.recent_events(30):
            st.text(f"{format_hhmm(e.ts_min)}  {e.text}")
elif st.session_state.get("fleet_sim_enabled"):
    st.info("Nhấn **'Khởi tạo / Reset mô phỏng đội xe'** ở trên để bắt đầu.")
