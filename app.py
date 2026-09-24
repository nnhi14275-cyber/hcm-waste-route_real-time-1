"""
app.py — LoGVN
---------------
Streamlit UI - Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt
bằng Google OR-Tools + Guided Local Search (GLS), dùng OSRM/OSM làm mạng
lưới đường thực tế (đường đi THẬT theo road network, KHÔNG phải đường chim
bay — Haversine chỉ là fallback dự phòng, TẮT theo mặc định).

PHẠM VI NGHIÊN CỨU: Tuyến Lê Văn Việt và khu vực lân cận, TP. Thủ Đức, TP.HCM.

Layout 2 khu vực:
  I  (trái, có thể thu gọn)  — Dữ liệu học máy + Dự báo nhu cầu + nguồn điểm thu gom
  II (phải)                  — Routing (OSRM) + ràng buộc tải trọng + OR-Tools +
                                hệ số chi phí/phát thải + Dashboard so sánh +
                                bảng phân bổ xe + mô phỏng sự cố giao thông
"""

from __future__ import annotations

import io
import math
import time

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

from baseline import nearest_neighbor_baseline, clarke_wright_savings, BASELINE_METHODS
from data_generator import DemoConfig, generate_demo_data, load_points_from_dataframe, DEPOT_LOCATION
from clustering import run_clustering_cvrp, split_time_budget
from dynamic_routing import DynamicRoutingEngine, GPSTrackerConfig, interpolate_along_route
from optimizer import OptimizeConfig, solve_cvrp, solve_cvrp_with_auto_scaling
from routing import (
    DEFAULT_OSRM_BASE_URL, OSRMError, check_point_count_limit,
    get_osrm_matrices, get_osrm_route_geometry,
)
from waste_streams import WASTE_STREAMS, build_stream_problem, run_stream_optimization, aggregate_kpi
from forecasting import (
    WASTE_TYPES,
    prepare_history_from_dataframe,
    train_and_forecast_uploaded_history,
    estimate_vehicles_needed,
    build_forecast_excel,
    build_template_history_excel,
)

st.set_page_config(page_title="LoGVN – Tối ưu tuyến thu gom rác", page_icon="🌱", layout="wide")

# ============================================================================
# GREEN THEME (bổ sung cho .streamlit/config.toml — CSS này chỉnh thêm các
# chi tiết mà theme config không với tới, VD màu nền thẻ metric/expander)
# ============================================================================
st.markdown(
    """
    <style>
    :root { --logvn-green: #1E7A34; --logvn-green-light: #E8F5E9; }
    .stButton>button[kind="primary"] { background-color: var(--logvn-green); border-color: var(--logvn-green); }
    .stButton>button[kind="primary"]:hover { background-color: #16602A; border-color: #16602A; }
    div[data-testid="stMetric"] {
        background-color: var(--logvn-green-light);
        border: 1px solid #C8E6C9;
        border-radius: 8px;
        padding: 10px 14px;
    }
    div[data-testid="stExpander"] { border: 1px solid #C8E6C9 !important; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("## 🌱 LoGVN")
st.caption("Green Logistics Vietnam — Prototype tối ưu tuyến thu gom rác thải sinh hoạt")

st.markdown(
    """
| Hạng mục | Nội dung |
|---|---|
| **Khu vực thử nghiệm** | Tuyến Lê Văn Việt – TP. Thủ Đức, TP.HCM (khu vực Quận 9 cũ) |
| **Routing engine** | OpenStreetMap + OSRM (đường đi thật theo mạng lưới đường bộ) |
| **Optimization** | Google OR-Tools + Guided Local Search |
| **Real-time traffic** | Mô phỏng thủ công (mô-đun "Sự cố giao thông" bên dưới) — KHÔNG phải dữ liệu traffic thời gian thực từ nhà cung cấp bản đồ |
| **Baseline** | Nearest Neighbor / Greedy hoặc Clarke-Wright Savings |
"""
)
st.caption(
    "Đây là mô hình nghiên cứu/prototype phục vụ mục đích học thuật (Green Logistics), "
    "KHÔNG phải hệ thống điều hành xe thu gom rác thực tế."
)
st.divider()

# ============================================================================
# SESSION STATE MẶC ĐỊNH
# ============================================================================
_DEFAULTS = {
    "df_points": None, "results": None,
    "forecast_result": None, "forecast_history": None,
    "forecast_df": None, "forecast_selected_date": None,
    "stream_results": None, "stream_errors": None,
    "collapse_left": False,
    "incident_state": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================================
# HÀM TIỆN ÍCH DÙNG CHUNG
# ============================================================================
def auto_fleet_size(total_demand_kg: float, capacity_kg: float, overload_pct: float, buffer_vehicles: int = 1) -> int:
    """Tự động tính số xe cần thiết = ceil(tổng khối lượng / (capacity × overload%))
    + buffer_vehicles xe dự phòng (giúp OR-Tools/baseline có dư địa tìm lời giải
    khả thi ngay từ đầu, tránh phải tăng dần số xe qua auto-scaling nhiều vòng).
    KHÔNG giới hạn thủ công số xe — người dùng chỉ chỉnh overload_pct (mục 4)."""
    eff_capacity = capacity_kg * (overload_pct / 100.0)
    if eff_capacity <= 0:
        return max(1, buffer_vehicles)
    return max(1, math.ceil(total_demand_kg / eff_capacity)) + buffer_vehicles


def _aggregate_kpi_routes(routes, fuel_rate, fuel_price, emission_factor):
    total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0 if routes else 0.0
    travel_time_min = sum(r.travel_time_s for r in routes) / 60.0 if routes else 0.0
    service_time_min = sum(r.service_time_s for r in routes) / 60.0 if routes else 0.0
    total_time_min = sum(r.total_route_time_s for r in routes) / 60.0 if routes else 0.0
    num_vehicles_used = len(routes)
    total_waste_kg = sum(r.collected_waste_kg for r in routes) if routes else 0.0
    avg_util = (sum(r.capacity_utilization_pct for r in routes) / len(routes)) if routes else 0.0
    fuel_l = total_distance_km * fuel_rate
    co2_kg = fuel_l * emission_factor
    cost_vnd = fuel_l * fuel_price
    return {
        "Tổng quãng đường (km)": round(total_distance_km, 2),
        "Thời gian di chuyển (phút)": round(travel_time_min, 1),
        "Thời gian phục vụ (phút)": round(service_time_min, 1),
        "Tổng thời gian tuyến (phút)": round(total_time_min, 1),
        "Số xe sử dụng": num_vehicles_used,
        "Khối lượng thu gom (kg)": round(total_waste_kg, 1),
        "Tải trọng TB (%)": round(avg_util, 1),
        "Nhiên liệu tiêu thụ (lít)": round(fuel_l, 1),
        "Chi phí nhiên liệu (VNĐ)": round(cost_vnd, 0),
        "Phát thải CO2 (kg)": round(co2_kg, 1),
    }


def _vehicle_allocation_table(routes, df_points_local):
    rows = []
    for r in routes:
        stop_ids = [
            df_points_local.iloc[i]["node_id"]
            for i in r.node_sequence
            if not df_points_local.iloc[i]["is_depot"]
        ]
        rows.append({
            "Xe": f"Vehicle {r.vehicle_id}",
            "Số điểm": r.num_stops,
            "Danh sách điểm": ", ".join(stop_ids),
            "Khối lượng (kg)": round(r.collected_waste_kg, 1),
            "Tải trọng (%)": round(r.capacity_utilization_pct, 1),
            "Quãng đường (km)": round(r.total_distance_m / 1000.0, 2),
            "Thời gian tuyến (phút)": round(r.total_route_time_s / 60.0, 1),
        })
    return pd.DataFrame(rows)


def _build_routing_report_excel(results: dict) -> bytes:
    """Xuất báo cáo Excel: KPI baseline vs optimized, bảng phân bổ xe, chi phí/phát thải."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        kpi_df = pd.DataFrame([
            {"Kịch bản": "Baseline", **results["kpi_base"]},
            {"Kịch bản": "Optimized (OR-Tools + GLS)", **results["kpi_opt"]},
        ])
        kpi_df.to_excel(writer, sheet_name="KPI so sánh", index=False)
        results["alloc_df"].to_excel(writer, sheet_name="Phân bổ xe", index=False)
        savings_df = pd.DataFrame([results["savings"]])
        savings_df.to_excel(writer, sheet_name="Tiết kiệm", index=False)
    buf.seek(0)
    return buf.getvalue()

# ============================================================================
# TOGGLE THU GỌN KHU I + LAYOUT 2 KHU VỰC (I trái | II phải)
# ============================================================================
collapse_left = st.toggle("◀ Thu gọn khu I (Dữ liệu)", value=st.session_state.collapse_left)
st.session_state.collapse_left = collapse_left
zone_ratio = [0.07, 0.93] if collapse_left else [1, 2.1]
zone1, zone2 = st.columns(zone_ratio, gap="large")

# ----------------------------------------------------------------------------
# KHU I — DỮ LIỆU HỌC MÁY + DỰ BÁO + NGUỒN ĐIỂM THU GOM
# ----------------------------------------------------------------------------
with zone1:
    if collapse_left:
        st.markdown("**I**")
        st.caption("Bấm toggle phía trên để mở lại.")
    else:
        st.markdown("### I. Dữ liệu")

        st.markdown("**1. Dữ liệu học máy**")
        history_file = st.file_uploader(
            "📂 Đẩy dữ liệu lịch sử (Excel/CSV, khuyến nghị 180 ngày)",
            type=["xlsx", "xls", "csv"], key="history_upload",
            help=(
                "Cột tối thiểu: date, node_id, waste_kg. Có thể dùng thêm num_households. "
                "Nếu có 3 loại rác: waste_organic_kg, waste_recyclable_kg, waste_other_kg."
            ),
        )
        forecast_method = st.selectbox("Mô hình dự báo", ["XGBoost", "Prophet"], index=0, key="forecast_method")
        forecast_horizon = st.slider("Số ngày dự báo tối đa", 1, 14, 7, key="forecast_horizon")
        train_btn = st.button("🎯 Huấn luyện mô hình", use_container_width=True)

        tmpl_bytes = build_template_history_excel()
        st.download_button(
            "⬇️ Tải file mẫu lịch sử", data=tmpl_bytes,
            file_name="mau_lich_su_rac.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.markdown("**2. Dự báo**")
        if st.session_state.forecast_result is not None:
            _fdf = st.session_state.forecast_result["forecast_df"]
            _available_dates = sorted(pd.to_datetime(_fdf["date"]).dt.date.unique())
            forecast_date = st.selectbox(
                "📅 Chọn ngày dự báo (time window theo lịch thực tế)",
                _available_dates,
                format_func=lambda d: d.strftime("%d/%m/%Y (%A)"),
                key="forecast_date_pick",
            )
            st.session_state.forecast_selected_date = forecast_date
        else:
            st.caption("Huấn luyện mô hình ở mục 1 trước để có danh sách ngày dự báo.")
            forecast_date = None
        run_forecast_btn = st.button("▶️ Chạy dự báo", use_container_width=True)

        st.divider()
        st.markdown("**Điểm thu gom (toạ độ)**")
        data_mode = st.radio(
            "Nguồn dữ liệu điểm", ["Dữ liệu demo (Lê Văn Việt)", "Upload CSV/XLSX"], key="data_mode"
        )
        if data_mode == "Dữ liệu demo (Lê Văn Việt)":
            num_points = st.slider("Số điểm thu gom (demo)", 20, 30, 25, key="num_points")
            seed = st.number_input("Random seed", value=42, step=1, key="seed")
            use_tw_demo = st.checkbox("Sinh time window demo (VRPTW)", value=False, key="use_tw_demo")
            st.caption("Case study quy mô nhỏ: 20–30 điểm thu gom + 1 depot.")
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader("Upload file (CSV hoặc XLSX)", type=["csv", "xlsx"], key="points_upload")
            st.caption(
                "Cột bắt buộc: node_id, latitude, longitude, waste_kg. "
                "Tuỳ chọn: service_time, time_window_start, time_window_end, is_depot."
            )
            num_points, seed, use_tw_demo = 25, 42, False

# giá trị mặc định khi khu I đang thu gọn (không render lại control -> giữ nguyên giá trị đã chọn trước đó qua session_state)
if collapse_left:
    forecast_method = st.session_state.get("forecast_method", "XGBoost")
    forecast_horizon = st.session_state.get("forecast_horizon", 7)
    data_mode = st.session_state.get("data_mode", "Dữ liệu demo (Lê Văn Việt)")
    num_points = st.session_state.get("num_points", 25)
    seed = st.session_state.get("seed", 42)
    use_tw_demo = st.session_state.get("use_tw_demo", False)
    uploaded_file = None
    train_btn = False
    run_forecast_btn = False
    forecast_date = st.session_state.get("forecast_selected_date")
    history_file = None

# ----------------------------------------------------------------------------
# KHU II — ROUTING (OSRM) + RÀNG BUỘC TẢI TRỌNG + OR-TOOLS + CHI PHÍ/PHÁT THẢI
# ----------------------------------------------------------------------------
with zone2:
    st.markdown("### II. Routing & Tối ưu")

    with st.expander("3. Routing (OSRM)", expanded=True):
        osrm_base_url = st.text_input("OSRM base URL", value=DEFAULT_OSRM_BASE_URL, key="osrm_base_url_input")
        allow_fallback = st.checkbox(
            "Cho phép fallback Haversine nếu OSRM lỗi (KHÔNG khuyến nghị)",
            value=False, key="allow_fallback",
            help=(
                "TẮT theo mặc định: ứng dụng luôn tính khoảng cách/thời gian theo ĐƯỜNG ĐI THẬT "
                "trên mạng lưới đường bộ qua OSRM, không phải đường chim bay. Haversine (đường chim "
                "bay) chỉ được dùng khi bạn chủ động bật ô này VÀ OSRM không gọi được."
            ),
        )
        if allow_fallback:
            st.warning("Fallback Haversine đã BẬT — nếu OSRM lỗi, khoảng cách sẽ là đường chim bay, không còn là đường đi thật.")
            fallback_speed = st.slider("Tốc độ giả định cho fallback (km/h)", 10, 50, 25, key="fallback_speed")
        else:
            fallback_speed = 25
        vehicle_capacity_kg = st.number_input("Vehicle capacity định mức (kg)", value=1000, step=50, key="vehicle_capacity_kg")
        max_route_hours = st.slider("Max route duration (giờ)", 1.0, 8.0, 4.0, step=0.5, key="max_route_hours")

    with st.expander("4. Ràng buộc tải trọng xe", expanded=True):
        overload_pct = st.slider(
            "Tải trọng tối đa cho phép / xe (%)", 100, 130, 115, step=1, key="overload_pct",
            help="Xe được phép chở tối đa (%) × Vehicle capacity định mức ở mục 3, trước khi phải điều thêm xe.",
        )
        st.caption(
            "**Không giới hạn thủ công số xe.** Hệ thống tự tính số xe cần thiết dựa trên "
            f"tổng khối lượng rác và tải trọng tối đa {overload_pct}% × capacity định mức — "
            "OR-Tools sẽ tự thêm xe nếu vẫn chưa khả thi (xem `solve_cvrp_with_auto_scaling`)."
        )

    with st.expander("5. Thuật toán tối ưu (OR-Tools)", expanded=True):
        use_gls = st.checkbox("Bật Guided Local Search (GLS)", value=True, key="use_gls")
        first_solution_strategy = st.selectbox(
            "Chiến lược khởi tạo (initial solution)",
            ["PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC"],
            key="first_solution_strategy",
        )
        time_limit_sec = st.slider("Thời gian chạy tối ưu (giây)", 5, 120, 20, key="time_limit_sec")
        baseline_method_label = st.selectbox(
            "Baseline dùng để so sánh", list(BASELINE_METHODS.keys()), index=0, key="baseline_method_label",
        )

    with st.expander("6. Hệ số tiêu hao, chi phí & phát thải", expanded=True):
        fuel_rate_l_per_km = st.number_input("Fuel rate (lít/km)", value=0.35, step=0.01, format="%.2f", key="fuel_rate")
        fuel_price_vnd_per_l = st.number_input(
            "Đơn giá nhiên liệu (VNĐ/lít)", value=22000, step=500, key="fuel_price",
            help="Giả định để quy đổi ra chi phí (VNĐ) trong Dashboard so sánh — điều chỉnh theo giá dầu diesel thực tế tại thời điểm chạy.",
        )
        emission_factor_kg_per_l = st.number_input(
            "Emission factor (kg CO2 / lít nhiên liệu)", value=2.68, step=0.01, format="%.2f", key="emission_factor",
        )
        if st.session_state.results is not None:
            _export_bytes = _build_routing_report_excel(st.session_state.results)
            st.download_button(
                "⬇️ Xuất báo cáo tối ưu (Excel)", data=_export_bytes,
                file_name="bao_cao_toi_uu_LoGVN.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.caption("Chạy tối ưu xong sẽ có nút xuất báo cáo Excel ở đây.")

    run_btn = st.button("🚀 Chạy tối ưu", type="primary", use_container_width=True)

st.divider()

# ============================================================================
# LOAD DỮ LIỆU ĐIỂM THU GOM
# ============================================================================
def _load_data() -> pd.DataFrame | None:
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


_new_points = _load_data()
if _new_points is not None:
    st.session_state.df_points = _new_points

if st.session_state.df_points is None:
    st.info("Vui lòng chọn dữ liệu demo hoặc upload điểm thu gom ở Khu I, sau đó nhấn **Chạy tối ưu** ở Khu II.")
    st.stop()

df_points = st.session_state.df_points

with st.expander("Xem dữ liệu điểm thu gom", expanded=False):
    st.dataframe(df_points, use_container_width=True)

# ============================================================================
# XỬ LÝ DỰ BÁO (train_btn / run_forecast_btn)
# ============================================================================
if train_btn:
    if history_file is None:
        st.error("Hãy đẩy file dữ liệu lịch sử ở mục 1 (Khu I) trước khi huấn luyện.")
    else:
        try:
            if history_file.name.lower().endswith(".csv"):
                raw_hist = pd.read_csv(history_file)
            else:
                raw_hist = pd.read_excel(history_file)
            history = prepare_history_from_dataframe(raw_hist, df_points=df_points)
            with st.spinner(f"Đang huấn luyện mô hình {forecast_method}..."):
                result = train_and_forecast_uploaded_history(
                    history, method=forecast_method.lower(), forecast_days=forecast_horizon
                )
            st.session_state.forecast_history = history
            st.session_state.forecast_result = result
            st.success(f"Đã huấn luyện xong mô hình {forecast_method} và dự báo {forecast_horizon} ngày tới.")
        except Exception as exc:
            st.error(f"Huấn luyện/dự báo thất bại: {exc}")

if run_forecast_btn:
    if st.session_state.forecast_result is None:
        st.error("Chưa có mô hình đã huấn luyện — bấm 'Huấn luyện mô hình' ở mục 1 trước.")
    elif forecast_date is None:
        st.error("Hãy chọn ngày dự báo trước khi bấm 'Chạy dự báo'.")
    else:
        _fdf_all = st.session_state.forecast_result["forecast_df"].copy()
        _fdf_all["_d"] = pd.to_datetime(_fdf_all["date"]).dt.date
        _selected = _fdf_all[_fdf_all["_d"] == forecast_date].drop(columns=["_d"])
        st.session_state.forecast_df = _selected
        st.success(f"Đã lấy dự báo cho ngày {forecast_date.strftime('%d/%m/%Y')} — {len(_selected)} điểm.")

if st.session_state.forecast_result is not None:
    with st.expander("🔮 Kết quả dự báo nhu cầu", expanded=(st.session_state.forecast_df is not None)):
        _result = st.session_state.forecast_result
        _fdf = _result["forecast_df"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Số điểm dự báo", _fdf["node_id"].nunique())
        with c2:
            st.metric("Tổng khối lượng dự báo (kg, cả horizon)", f"{_fdf['total_kg'].sum():,.0f}")
        with c3:
            _, _n_veh_est = estimate_vehicles_needed(
                {r["node_id"]: {"total_kg": r["total_kg"]} for _, r in _fdf.iterrows()},
                vehicle_capacity_kg,
            )
            st.metric("Số xe ước tính (theo dự báo)", _n_veh_est)

        _fig = px.line(_fdf, x="date", y="total_kg", color="node_id", title="Dự báo khối lượng rác theo ngày/điểm")
        _fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(_fig, use_container_width=True)

        if _result.get("validation_df") is not None and len(_result["validation_df"]) > 0:
            st.caption("Kiểm định mô hình (validation) trên phần lịch sử giữ lại:")
            st.dataframe(_result["validation_df"], use_container_width=True, hide_index=True)

        if st.session_state.forecast_df is not None:
            st.markdown("**Dự báo được chọn để định tuyến:**")
            st.dataframe(st.session_state.forecast_df, use_container_width=True, hide_index=True)
            _export_fc = build_forecast_excel(
                st.session_state.forecast_history, _fdf, _result.get("validation_df")
            )
            st.download_button(
                "⬇️ Xuất Excel dự báo đầy đủ", data=_export_fc,
                file_name="du_bao_rac_LoGVN.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if st.checkbox("Dùng khối lượng dự báo này thay cho waste_kg hiện tại của điểm thu gom", value=False):
                _fc_map = dict(zip(st.session_state.forecast_df["node_id"], st.session_state.forecast_df["total_kg"]))
                df_points = df_points.copy()
                df_points["waste_kg"] = df_points["node_id"].map(_fc_map).fillna(df_points["waste_kg"])
                st.session_state.df_points = df_points
                st.caption("✅ Đã áp dụng khối lượng dự báo vào bảng điểm thu gom (dùng cho phần Routing bên dưới).")

st.divider()

# ============================================================================
# CHẠY TỐI ƯU (run_btn) — đội xe tự động theo overload_pct, KHÔNG giới hạn thủ công
# ============================================================================
if run_btn:
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    node_ids_full = df_points["node_id"].tolist()
    depot_index = int(df_points.index[df_points["is_depot"]][0])
    demands_full = df_points["waste_kg"].tolist()
    service_times_s_full = (df_points["service_time"] * 60).tolist()
    total_demand = sum(demands_full)

    _warn = check_point_count_limit(len(coords), osrm_base_url)
    if _warn:
        st.warning(_warn)

    try:
        with st.spinner("Đang lấy ma trận khoảng cách/thời gian THẬT từ OSRM (road network)..."):
            matrix_result = get_osrm_matrices(
                coords, base_url=osrm_base_url, allow_haversine_fallback=allow_fallback,
                fallback_avg_speed_kmh=fallback_speed,
            )
    except OSRMError as exc:
        st.error(str(exc))
        st.stop()

    if matrix_result.source == "HAVERSINE_FALLBACK":
        st.warning(matrix_result.warning or "Đang dùng Haversine fallback (đường chim bay), không phải đường đi thật.")
    else:
        st.success("✅ Đã lấy khoảng cách/thời gian THEO ĐƯỜNG ĐI THẬT từ OSRM (không phải chim bay).")

    dist_m_full = matrix_result.distance_matrix_m
    dur_s_full = matrix_result.duration_matrix_s

    n_fleet = auto_fleet_size(total_demand, vehicle_capacity_kg, overload_pct, buffer_vehicles=1)
    effective_capacity_kg = vehicle_capacity_kg * (overload_pct / 100.0)
    st.caption(
        f"🚚 Đội xe tự động: tổng khối lượng {total_demand:,.0f} kg ÷ (capacity {vehicle_capacity_kg:,.0f} kg × "
        f"{overload_pct}%) → **{n_fleet} xe** (đã cộng 1 xe dự phòng). OR-Tools sẽ tự thêm xe nếu vẫn chưa khả thi."
    )

    baseline_fn = BASELINE_METHODS[baseline_method_label]
    with st.spinner(f"Đang xây dựng baseline ({baseline_method_label})..."):
        try:
            baseline_routes = baseline_fn(
                dist_m_full, dur_s_full, demands_full, service_times_s_full,
                vehicle_capacity_kg=effective_capacity_kg, depot_index=depot_index,
                max_route_time_s=max_route_hours * 3600, max_vehicles=max(n_fleet, 20),
            )
        except RuntimeError as exc:
            st.error(f"Baseline thất bại: {exc}")
            st.stop()

    use_tw = bool(df_points["time_window_start"].sum() > 0) if "time_window_start" in df_points.columns else False
    time_windows_s = None
    if use_tw:
        time_windows_s = list(zip(df_points["time_window_start"] * 60, df_points["time_window_end"] * 60))

    cfg = OptimizeConfig(
        num_vehicles=n_fleet,
        vehicle_capacity_kg=effective_capacity_kg,
        depot_index=depot_index,
        use_gls=use_gls,
        first_solution_strategy=first_solution_strategy,
        time_limit_sec=time_limit_sec,
        max_route_time_s=max_route_hours * 3600,
        use_time_windows=use_tw,
        time_windows_s=time_windows_s,
    )
    with st.spinner("Đang tối ưu bằng OR-Tools + GLS..."):
        optimized_routes, solved, msg, n_used = solve_cvrp_with_auto_scaling(
            dist_m_full, dur_s_full, demands_full, service_times_s_full, cfg, max_extra_vehicles=4,
        )
    if not solved:
        st.error(msg)
        st.stop()
    if n_used > n_fleet:
        st.info(msg)

    kpi_base = _aggregate_kpi_routes(baseline_routes, fuel_rate_l_per_km, fuel_price_vnd_per_l, emission_factor_kg_per_l)
    kpi_opt = _aggregate_kpi_routes(optimized_routes, fuel_rate_l_per_km, fuel_price_vnd_per_l, emission_factor_kg_per_l)
    alloc_df = _vehicle_allocation_table(optimized_routes, df_points)

    def _pct_change(base, opt):
        if base == 0:
            return 0.0
        return 100.0 * (base - opt) / base

    savings = {
        "Giảm quãng đường (km)": round(kpi_base["Tổng quãng đường (km)"] - kpi_opt["Tổng quãng đường (km)"], 2),
        "Giảm quãng đường (%)": round(_pct_change(kpi_base["Tổng quãng đường (km)"], kpi_opt["Tổng quãng đường (km)"]), 1),
        "Giảm chi phí nhiên liệu (VNĐ)": round(kpi_base["Chi phí nhiên liệu (VNĐ)"] - kpi_opt["Chi phí nhiên liệu (VNĐ)"], 0),
        "Giảm chi phí nhiên liệu (%)": round(_pct_change(kpi_base["Chi phí nhiên liệu (VNĐ)"], kpi_opt["Chi phí nhiên liệu (VNĐ)"]), 1),
        "Giảm phát thải CO2 (kg)": round(kpi_base["Phát thải CO2 (kg)"] - kpi_opt["Phát thải CO2 (kg)"], 1),
        "Chênh lệch số xe": kpi_opt["Số xe sử dụng"] - kpi_base["Số xe sử dụng"],
    }

    st.session_state.results = {
        "df_points": df_points, "coords": coords, "node_ids_full": node_ids_full,
        "depot_index": depot_index, "demands_full": demands_full,
        "service_times_s_full": service_times_s_full, "dist_m_full": dist_m_full,
        "dur_s_full": dur_s_full, "baseline_routes": baseline_routes,
        "optimized_routes": optimized_routes, "kpi_base": kpi_base, "kpi_opt": kpi_opt,
        "alloc_df": alloc_df, "savings": savings, "osrm_base_url": osrm_base_url,
        "matrix_source": matrix_result.source, "n_fleet": n_fleet,
        "effective_capacity_kg": effective_capacity_kg,
        "vehicle_capacity_kg": vehicle_capacity_kg, "overload_pct": overload_pct,
        "use_gls": use_gls, "first_solution_strategy": first_solution_strategy,
        "time_limit_sec": time_limit_sec, "max_route_hours": max_route_hours,
        "baseline_method_label": baseline_method_label,
        "fuel_rate_l_per_km": fuel_rate_l_per_km, "fuel_price_vnd_per_l": fuel_price_vnd_per_l,
        "emission_factor_kg_per_l": emission_factor_kg_per_l,
    }
    st.rerun()

# ============================================================================
# HIỂN THỊ KẾT QUẢ: DASHBOARD SO SÁNH + PHÂN BỔ XE + BẢN ĐỒ + SỰ CỐ GIAO THÔNG
# ============================================================================
if st.session_state.results is None:
    st.info("Cấu hình xong ở Khu II rồi nhấn **🚀 Chạy tối ưu** để xem Dashboard so sánh.")
    st.stop()

res = st.session_state.results
df_points = res["df_points"]
coords = res["coords"]
node_ids_full = res["node_ids_full"]
depot_index = res["depot_index"]
demands_full = res["demands_full"]
service_times_s_full = res["service_times_s_full"]
dist_m_full = res["dist_m_full"]
dur_s_full = res["dur_s_full"]
baseline_routes = res["baseline_routes"]
optimized_routes = res["optimized_routes"]
kpi_base = res["kpi_base"]
kpi_opt = res["kpi_opt"]
alloc_df = res["alloc_df"]
savings = res["savings"]
vehicle_capacity_kg = res["vehicle_capacity_kg"]
overload_pct = res["overload_pct"]
use_gls = res["use_gls"]
first_solution_strategy = res["first_solution_strategy"]
time_limit_sec = res["time_limit_sec"]
max_route_hours = res["max_route_hours"]
index_of_node = {nid: i for i, nid in enumerate(node_ids_full)}
depot_node_id = df_points.loc[df_points["is_depot"], "node_id"].iloc[0]

if res["matrix_source"] == "HAVERSINE_FALLBACK":
    st.warning("⚠️ Kết quả đang hiển thị dựa trên Haversine fallback (đường chim bay), không phải đường đi thật.")
else:
    st.success("✅ Toàn bộ khoảng cách/thời gian bên dưới lấy từ OSRM — đường đi THẬT theo mạng lưới đường bộ.")

st.subheader("📊 Dashboard so sánh: Baseline vs Optimized")
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric(
        "Quãng đường (km)", f"{kpi_opt['Tổng quãng đường (km)']:,.1f}",
        delta=f"-{savings['Giảm quãng đường (%)']:.1f}%" if savings["Giảm quãng đường (%)"] > 0 else f"+{-savings['Giảm quãng đường (%)']:.1f}%",
        delta_color="normal" if savings["Giảm quãng đường (%)"] > 0 else "inverse",
    )
with d2:
    st.metric(
        "Chi phí nhiên liệu (VNĐ)", f"{kpi_opt['Chi phí nhiên liệu (VNĐ)']:,.0f}",
        delta=f"-{savings['Giảm chi phí nhiên liệu (%)']:.1f}%" if savings["Giảm chi phí nhiên liệu (%)"] > 0 else f"+{-savings['Giảm chi phí nhiên liệu (%)']:.1f}%",
        delta_color="normal" if savings["Giảm chi phí nhiên liệu (%)"] > 0 else "inverse",
    )
with d3:
    st.metric("Số xe sử dụng", kpi_opt["Số xe sử dụng"], delta=savings["Chênh lệch số xe"], delta_color="inverse")
with d4:
    st.metric("Phát thải CO2 (kg)", f"{kpi_opt['Phát thải CO2 (kg)']:,.1f}", delta=f"-{savings['Giảm phát thải CO2 (kg)']:,.1f} kg")

kpi_compare_df = pd.DataFrame([
    {"Kịch bản": f"Baseline ({res.get('baseline_method_label', '')})", **kpi_base},
    {"Kịch bản": "Optimized (OR-Tools + GLS)", **kpi_opt},
])
st.dataframe(kpi_compare_df, use_container_width=True, hide_index=True)
st.caption(
    f"💰 Tiết kiệm: {savings['Giảm quãng đường (km)']:,.2f} km ({savings['Giảm quãng đường (%)']:.1f}%), "
    f"{savings['Giảm chi phí nhiên liệu (VNĐ)']:,.0f} VNĐ nhiên liệu ({savings['Giảm chi phí nhiên liệu (%)']:.1f}%), "
    f"{savings['Giảm phát thải CO2 (kg)']:,.1f} kg CO2 so với baseline. "
    "Chi phí quy đổi từ đơn giá nhiên liệu ở mục 6 (Khu II) — là GIẢ ĐỊNH có thể chỉnh, không phải giá thị trường theo thời gian thực."
)

st.markdown("**🚛 Bảng phân bổ xe theo điểm (Optimized)**")
st.dataframe(alloc_df, use_container_width=True, hide_index=True)
st.caption(
    f"Cột **Tải trọng (%)** tính trên ngưỡng tối đa cho phép "
    f"({vehicle_capacity_kg:,.0f} kg × {overload_pct}% = {res['effective_capacity_kg']:,.0f} kg/xe), "
    "KHÔNG phải trên capacity định mức — 100% ở đây nghĩa là xe đã chở đúng mức trần cho phép ở mục 4 (Khu II)."
)

st.markdown("**🗺️ Bản đồ tuyến (đường đi thật qua OSRM)**")
fmap = folium.Map(location=[coords[depot_index][0], coords[depot_index][1]], zoom_start=14, tiles="cartodbpositron")
folium.Marker(coords[depot_index], popup="DEPOT", icon=folium.Icon(color="black", icon="home")).add_to(fmap)
_palette = ["#1E7A34", "#2E86AB", "#E67E22", "#8E44AD", "#C0392B", "#16A085", "#D35400", "#2C3E50"]
for _i, r in enumerate(optimized_routes):
    color = _palette[_i % len(_palette)]
    ordered_coords = tuple(coords[n] for n in r.node_sequence)
    geometry = get_osrm_route_geometry(ordered_coords, base_url=res["osrm_base_url"])
    if not geometry:
        geometry = list(ordered_coords)
    folium.PolyLine(geometry, color=color, weight=4, opacity=0.8, tooltip=f"Vehicle {r.vehicle_id}").add_to(fmap)
    for n in r.node_sequence:
        if df_points.iloc[n]["is_depot"]:
            continue
        folium.CircleMarker(
            coords[n], radius=5, color=color, fill=True, fill_opacity=0.9,
            popup=f"{df_points.iloc[n]['node_id']} — Vehicle {r.vehicle_id}",
        ).add_to(fmap)
st_folium(fmap, use_container_width=True, height=480, returned_objects=[], key="main_route_map")

st.divider()

# ============================================================================
# MÔ PHỎNG SỰ CỐ GIAO THÔNG (REAL-TIME TRAFFIC — MÔ PHỎNG THỦ CÔNG)
# ============================================================================
st.subheader("🚧 Mô phỏng sự cố giao thông (real-time traffic)")
st.caption(
    "OSRM demo server công khai KHÔNG có dữ liệu traffic thời gian thực miễn phí. Mô-đun này mô "
    "phỏng: chọn 1 điểm (VD gần đó có sự cố/kẹt xe), tăng thời gian di chuyển qua các cung đường "
    "quanh điểm đó lên theo hệ số, rồi tính lại tuyến NGAY LẬP TỨC — giống cách một hệ thống có "
    "traffic feed thật sẽ phản ứng, chỉ khác nguồn dữ liệu traffic là do người dùng nhập thủ công "
    "thay vì lấy tự động từ nhà cung cấp bản đồ."
)

incident_node_label = st.selectbox(
    "📍 Vị trí xảy ra sự cố",
    options=node_ids_full,
    format_func=lambda nid: f"{nid} (DEPOT)" if nid == depot_node_id else nid,
    key="incident_node_label",
)
incident_severity = st.slider(
    "Mức độ nghiêm trọng (hệ số nhân thời gian di chuyển qua điểm này)",
    1.2, 3.0, 1.8, step=0.1, key="incident_severity",
)
ic1, ic2 = st.columns(2)
with ic1:
    apply_incident_btn = st.button("🚧 Áp dụng sự cố & tính lại tuyến", use_container_width=True)
with ic2:
    clear_incident_btn = st.button("♻️ Xoá sự cố, khôi phục tuyến gốc", use_container_width=True)

if clear_incident_btn:
    st.session_state.incident_state = None
    st.rerun()

if apply_incident_btn:
    incident_idx = index_of_node[incident_node_label]
    n = len(dur_s_full)
    dur_incident = [row[:] for row in dur_s_full]
    for a in range(n):
        for b in range(n):
            if a == incident_idx or b == incident_idx:
                dur_incident[a][b] = dur_s_full[a][b] * incident_severity

    incident_cfg = OptimizeConfig(
        num_vehicles=res["n_fleet"],
        vehicle_capacity_kg=res["effective_capacity_kg"],
        depot_index=depot_index,
        use_gls=use_gls,
        first_solution_strategy=first_solution_strategy,
        time_limit_sec=time_limit_sec,
        max_route_time_s=max_route_hours * 3600,
    )
    with st.spinner(f"Đang tính lại tuyến do sự cố tại {incident_node_label} (×{incident_severity} thời gian di chuyển)..."):
        incident_routes, incident_solved, incident_msg, _n2 = solve_cvrp_with_auto_scaling(
            dist_m_full, dur_incident, demands_full, service_times_s_full, incident_cfg, max_extra_vehicles=4,
        )
    if not incident_solved:
        st.error(f"Không tính lại được tuyến sau sự cố: {incident_msg}")
    else:
        kpi_incident = _aggregate_kpi_routes(
            incident_routes, res["fuel_rate_l_per_km"], res["fuel_price_vnd_per_l"], res["emission_factor_kg_per_l"]
        )
        st.session_state.incident_state = {
            "node": incident_node_label, "severity": incident_severity,
            "routes": incident_routes, "kpi": kpi_incident,
        }
        st.rerun()

if st.session_state.incident_state is not None:
    ist = st.session_state.incident_state
    st.error(
        f"🚧 Đang có sự cố tại **{ist['node']}** (×{ist['severity']} thời gian di chuyển) — "
        "tuyến bên dưới đã được TÍNH LẠI để tránh/giảm thiểu ảnh hưởng."
    )
    ic_before, ic_after = st.columns(2)
    with ic_before:
        st.markdown("**Trước sự cố (tuyến gốc)**")
        st.metric("Thời gian tổng (phút)", f"{kpi_opt['Tổng thời gian tuyến (phút)']:,.1f}")
        st.metric("Quãng đường (km)", f"{kpi_opt['Tổng quãng đường (km)']:,.1f}")
    with ic_after:
        st.markdown("**Sau khi tính lại vì sự cố**")
        _dt = ist["kpi"]["Tổng thời gian tuyến (phút)"] - kpi_opt["Tổng thời gian tuyến (phút)"]
        _dd = ist["kpi"]["Tổng quãng đường (km)"] - kpi_opt["Tổng quãng đường (km)"]
        st.metric("Thời gian tổng (phút)", f"{ist['kpi']['Tổng thời gian tuyến (phút)']:,.1f}", delta=f"{_dt:+.1f} phút", delta_color="inverse")
        st.metric("Quãng đường (km)", f"{ist['kpi']['Tổng quãng đường (km)']:,.1f}", delta=f"{_dd:+.1f} km", delta_color="inverse")

    imap = folium.Map(location=[coords[depot_index][0], coords[depot_index][1]], zoom_start=14, tiles="cartodbpositron")
    folium.Marker(coords[depot_index], popup="DEPOT", icon=folium.Icon(color="black", icon="home")).add_to(imap)
    folium.CircleMarker(
        coords[index_of_node[ist["node"]]], radius=14, color="red", fill=True, fill_opacity=0.3,
        popup=f"🚧 Sự cố: {ist['node']}",
    ).add_to(imap)
    for _i, r in enumerate(ist["routes"]):
        color = _palette[_i % len(_palette)]
        ordered_coords = tuple(coords[n] for n in r.node_sequence)
        geometry = get_osrm_route_geometry(ordered_coords, base_url=res["osrm_base_url"])
        if not geometry:
            geometry = list(ordered_coords)
        folium.PolyLine(geometry, color=color, weight=4, opacity=0.8, tooltip=f"Vehicle {r.vehicle_id} (đã tính lại)").add_to(imap)
    st_folium(imap, use_container_width=True, height=440, returned_objects=[], key="incident_map")

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
    stream_errors = []
    for key in WASTE_STREAMS:
        problem = build_stream_problem(
            df_points, dist_m_full, dur_s_full, service_times_s_full, key, depot_idx_global
        )
        if problem is None:
            stream_results[key] = None
            continue
        with st.spinner(f"Đang tối ưu luồng '{WASTE_STREAMS[key]['label']}'..."):
            try:
                stream_results[key] = run_stream_optimization(
                    problem,
                    num_vehicles=stream_vehicle_cfg[key]["num_vehicles"],
                    vehicle_capacity_kg=stream_vehicle_cfg[key]["vehicle_capacity_kg"],
                    use_gls=use_gls,
                    first_solution_strategy=first_solution_strategy,
                    time_limit_sec=min(time_limit_sec, 20),
                    max_route_time_s=max_route_hours * 3600,
                )
            except RuntimeError as exc:
                # VD: baseline Nearest Neighbor không đủ xe/tải trọng để phục vụ
                # hết điểm của luồng này -> báo lỗi rõ ràng cho đúng luồng thay
                # vì làm crash toàn bộ phần "Tối ưu đa luồng".
                stream_results[key] = None
                stream_errors.append(f"Luồng '{WASTE_STREAMS[key]['label']}': {exc}")
    st.session_state.stream_results = stream_results
    st.session_state.stream_errors = stream_errors

if st.session_state.get("stream_errors"):
    for err in st.session_state.stream_errors:
        st.error(
            f"Không tối ưu được {err} Hãy tăng 'Số xe' hoặc 'Capacity xe (kg)' cho "
            "luồng này ở phần cấu hình đội xe phía trên rồi chạy lại."
        )

if st.session_state.get("stream_results"):
    stream_results = st.session_state.stream_results
    total_after_km = 0.0
    total_after_vehicles = 0
    stream_kpi_rows = []

    for key, meta in WASTE_STREAMS.items():
        result = stream_results.get(key)
        st.markdown(f"**Luồng: {meta['label']}**")
        if result is None:
            # dùng `depot_index` (đã có sẵn ở scope module, tính từ pipeline chính)
            # thay vì `depot_idx_global` vì biến đó chỉ tồn tại trong nhánh
            # `if run_streams_btn:` phía trên - ở lần rerun mà nút không được
            # bấm lại (chỉ hiển thị kết quả đã lưu trong session_state),
            # `depot_idx_global` sẽ chưa được gán và gây NameError.
            problem_exists = build_stream_problem(
                df_points, dist_m_full, dur_s_full, service_times_s_full, key, depot_index
            ) is not None
            if problem_exists:
                st.caption("Luồng này không tối ưu được ở lần chạy vừa rồi — xem thông báo lỗi ở trên.")
            else:
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

    if stream_kpi_rows:
        st.markdown("**Tổng hợp theo từng luồng (Optimized)**")
        st.dataframe(pd.DataFrame(stream_kpi_rows), use_container_width=True, hide_index=True)

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
# DYNAMIC ROUTING (GPS TỰ ĐỘNG) + XỬ LÝ VẬT CẢN/RÁC CỒNG KỀNH (CHẶNG PHỤ)
# ============================================================================
st.subheader("🛰️ Dynamic Routing – GPS tự động (Auto completion, Auto re-optimize & Chặng phụ)")
st.caption(
    "Xe được theo dõi qua GPS điện thoại. Khi xe vào bán kính điểm thu gom và đứng đủ lâu "
    "→ điểm tự động chuyển pending → completed. Khi xe vào bán kính DEPOT và đứng đủ lâu "
    "→ hệ thống tự xác nhận 'xe đã về DEPOT' và TỰ ĐỘNG tái tối ưu (OR-Tools + GLS) cho các "
    "điểm còn pending. **Mới:** nếu ngoài hiện trường có vật cản/rác cồng kềnh/sự cố khiến xe "
    "không thể tới một điểm (VD không quay đầu được), tài xế báo 'sự cố' cho điểm đó → hệ thống "
    "loại điểm khỏi tuyến hiện tại NGAY, tính lại phần còn lại; khi xe về DEPOT mà vẫn còn điểm bị "
    "hoãn và xe CHƯA đầy tải theo dự báo, hệ thống TỰ ĐỘNG tạo một **chặng phụ** (DEPOT → các điểm "
    "bị hoãn → DEPOT) để xử lý nốt."
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
        st.session_state.gps_vehicle_capacity = chosen_route.vehicle_capacity_kg
        st.rerun()

if gps_enabled and "gps_engine" in st.session_state:
    engine: DynamicRoutingEngine = st.session_state.gps_engine

    # ---- Báo sự cố / vật cản tại 1 điểm đang pending ----
    _pending_now = engine.pending_node_ids()
    if _pending_now:
        st.markdown("**🚧 Báo sự cố / vật cản tại điểm (không thể thu gom ngay)**")
        oc1, oc2, oc3 = st.columns([2, 2, 1])
        with oc1:
            obstacle_node = st.selectbox("Điểm gặp sự cố", _pending_now, key="obstacle_node_select")
        with oc2:
            obstacle_reason = st.text_input(
                "Lý do", value="Vật cản / rác cồng kềnh / không quay đầu được", key="obstacle_reason"
            )
        with oc3:
            st.write("")
            st.write("")
            report_obstacle_btn = st.button("🚧 Báo sự cố", use_container_width=True)
    else:
        report_obstacle_btn = False
        obstacle_node = None
        obstacle_reason = ""

    def _reoptimize_current_route(pending_ids: list[str], num_vehicles: int = 1):
        """Tính lại tuyến hiện tại (DEPOT -> các điểm pending_ids -> DEPOT) cho
        cùng 1 xe vật lý đang chạy. Trả về (new_route_global_nodes, new_geometry, total_km) hoặc None nếu thất bại."""
        sub_indices = [index_of_node[depot_node_id]] + [index_of_node[nid] for nid in pending_ids]
        sub_dist = [[dist_m_full[a][b] for b in sub_indices] for a in sub_indices]
        sub_dur = [[dur_s_full[a][b] for b in sub_indices] for a in sub_indices]
        sub_demands = [demands_full[i] for i in sub_indices]
        sub_service = [service_times_s_full[i] for i in sub_indices]
        reopt_cfg = OptimizeConfig(
            num_vehicles=num_vehicles,
            vehicle_capacity_kg=st.session_state.get("gps_vehicle_capacity", vehicle_capacity_kg),
            depot_index=0, use_gls=use_gls, first_solution_strategy=first_solution_strategy,
            time_limit_sec=min(time_limit_sec, 15), max_route_time_s=max_route_hours * 3600,
        )
        new_routes, solved, msg, _n = solve_cvrp_with_auto_scaling(
            sub_dist, sub_dur, sub_demands, sub_service, reopt_cfg, max_extra_vehicles=0
        )
        if not (solved and new_routes):
            return None, None, None, msg
        new_route_global_nodes = [sub_indices[i] for i in new_routes[0].node_sequence]
        new_active_coords = tuple(coords[i] for i in new_route_global_nodes)
        new_geom = get_osrm_route_geometry(new_active_coords, base_url=res["osrm_base_url"])
        return new_route_global_nodes, (new_geom or list(new_active_coords)), new_routes[0].total_distance_m / 1000.0, "success"

    if report_obstacle_btn and obstacle_node:
        if engine.mark_deferred(obstacle_node, obstacle_reason):
            st.session_state.gps_event_log.append(f"🚧 Sự cố tại {obstacle_node}: {obstacle_reason} — điểm bị hoãn, tính lại tuyến.")
            remaining_pending = engine.pending_node_ids()
            if remaining_pending:
                new_nodes, new_geom, new_km, _msg = _reoptimize_current_route(remaining_pending)
                if new_nodes:
                    st.session_state.gps_active_route_nodes = new_nodes
                    st.session_state.gps_active_geometry = new_geom
                    st.session_state.gps_sim_progress_m = 0.0
                    engine.sync_pending_after_reoptimize(remaining_pending)
                    st.session_state.gps_event_log.append(f"🔁 Đã tính lại tuyến, bỏ qua điểm hoãn ({new_km:.2f} km còn lại).")
                else:
                    st.session_state.gps_event_log.append(f"⚠️ Tính lại tuyến thất bại: {_msg}")
            else:
                st.session_state.gps_event_log.append("Không còn điểm pending nào khác trên tuyến chính — xe có thể về DEPOT.")
            st.rerun()

    # Tick tự động ~1.5s/lần để mô phỏng/đọc GPS liên tục (không cần bấm nút)
    st_autorefresh(interval=1500, key="gps_autorefresh_tick")

    def _handle_event(ev: dict):
        if ev["newly_completed"]:
            for nid in ev["newly_completed"]:
                st.session_state.gps_event_log.append(f"✅ Auto-completed: {nid}")
        if ev["depot_confirmed"]:
            st.session_state.gps_event_log.append("🏠 Auto depot detected: xe đã về DEPOT")
            pending_ids = engine.pending_node_ids()
            deferred_ids = engine.deferred_node_ids()

            if pending_ids:
                # ---- AUTO RE-OPTIMIZATION: chỉ các điểm pending, DEPOT là điểm xuất phát ----
                new_nodes, new_geom, new_km, msg = _reoptimize_current_route(pending_ids)
                if new_nodes:
                    st.session_state.gps_active_route_nodes = new_nodes
                    st.session_state.gps_active_geometry = new_geom
                    st.session_state.gps_sim_progress_m = 0.0
                    engine.sync_pending_after_reoptimize(pending_ids)
                    st.session_state.gps_event_log.append(
                        f"🔁 Auto re-optimized: DEPOT → {len(pending_ids)} điểm pending còn lại ({new_km:.2f} km)"
                    )
                else:
                    st.session_state.gps_event_log.append(f"⚠️ Re-optimize thất bại: {msg}")
                return

            if deferred_ids:
                # ---- THUẬT TOÁN DYNAMIC: tự động tính CHẶNG PHỤ cho các điểm bị hoãn ----
                # (vật cản/rác cồng kềnh) vì xe chưa đầy tải theo dự báo ban đầu.
                current_load = sum(demands_full[index_of_node[nid]] for nid in engine.completed_node_ids())
                capacity_ref = st.session_state.get("gps_vehicle_capacity", vehicle_capacity_kg)
                if current_load < capacity_ref:
                    for nid in deferred_ids:
                        engine.requeue_deferred(nid)
                    new_nodes, new_geom, new_km, msg = _reoptimize_current_route(deferred_ids)
                    if new_nodes:
                        st.session_state.gps_active_route_nodes = new_nodes
                        st.session_state.gps_active_geometry = new_geom
                        st.session_state.gps_sim_progress_m = 0.0
                        engine.sync_pending_after_reoptimize(deferred_ids)
                        st.session_state.gps_event_log.append(
                            f"🔧 Chặng phụ tự động: DEPOT → {len(deferred_ids)} điểm từng bị hoãn → DEPOT "
                            f"({new_km:.2f} km) — xe mới chở {current_load:.0f}/{capacity_ref:.0f} kg, còn dư tải."
                        )
                    else:
                        st.session_state.gps_event_log.append(f"⚠️ Không tính được chặng phụ: {msg}")
                else:
                    st.session_state.gps_event_log.append(
                        f"ℹ️ Xe đã đầy tải ({current_load:.0f}/{capacity_ref:.0f} kg) — "
                        f"{len(deferred_ids)} điểm bị hoãn để lại cho ca sau, KHÔNG tạo chặng phụ."
                    )
                    st.session_state.gps_tour_finished = True
                return

            st.session_state.gps_tour_finished = True
            st.session_state.gps_event_log.append("🎉 Đã hoàn thành toàn bộ tuyến (kể cả chặng phụ nếu có).")

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
            {"node_id": nid, "status": s.status, "lý do hoãn": s.deferred_reason}
            for nid, s in engine.point_status.items()
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True, height=250)
        m_pending, m_deferred = st.columns(2)
        m_pending.metric("Điểm còn pending", len(engine.pending_node_ids()))
        m_deferred.metric("Điểm bị hoãn", len(engine.deferred_node_ids()))
        if st.session_state.get("gps_tour_finished"):
            st.success("Xe đã hoàn thành ca (không còn điểm pending, và không còn chặng phụ khả thi).")
        with st.expander("Nhật ký sự kiện", expanded=True):
            for line in st.session_state.gps_event_log[-20:][::-1]:
                st.text(line)

    with col_map:
        gmap = folium.Map(location=[DEPOT_LOCATION[0], DEPOT_LOCATION[1]], zoom_start=15, tiles="cartodbpositron")
        folium.Marker(DEPOT_LOCATION, popup="DEPOT", icon=folium.Icon(color="black", icon="home")).add_to(gmap)
        _status_color = {"completed": "#2ca02c", "pending": "#ff7f0e", "deferred": "#d62728"}
        for nid, s in engine.point_status.items():
            plat, plon = engine.point_coords[nid]
            color = _status_color.get(s.status, "#7f7f7f")
            folium.CircleMarker(
                [plat, plon], radius=6, color=color, fill=True, fill_opacity=0.9,
                popup=f"{nid} - {s.status}" + (f" ({s.deferred_reason})" if s.status == "deferred" else ""),
            ).add_to(gmap)
        folium.PolyLine(
            st.session_state.gps_active_geometry, color="#9467bd", weight=4, opacity=0.7,
            tooltip="Tuyến đang chạy (auto re-optimize / chặng phụ khi có sự kiện)",
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
