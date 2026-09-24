"""
test_harness.py
----------------
Chạy lại (không qua UI Streamlit) đúng các bước mà app.py thực hiện, dùng
Haversine fallback thay OSRM (không cần internet), để phát hiện lỗi tích hợp
giữa các module (mismatched dict keys, thiếu hàm, sai signature, v.v.)
"""
import math
import time
import traceback

import numpy as np
import pandas as pd

FAILURES = []

def check(label, fn):
    print(f"\n=== {label} ===")
    try:
        result = fn()
        print(f"OK: {label}")
        return result
    except Exception as exc:
        print(f"FAIL: {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        FAILURES.append((label, exc))
        return None


# ---------------------------------------------------------------------------
# 1. IMPORTS - đúng như app.py
# ---------------------------------------------------------------------------
def step_imports():
    global nearest_neighbor_baseline, DemoConfig, generate_demo_data, load_points_from_dataframe, DEPOT_LOCATION
    global run_clustering_cvrp, split_time_budget
    global DynamicRoutingEngine, GPSTrackerConfig, interpolate_along_route
    global OptimizeConfig, solve_cvrp, solve_cvrp_with_auto_scaling
    global DEFAULT_OSRM_BASE_URL, OSRMError, check_point_count_limit, get_osrm_matrices, get_osrm_route_geometry
    global WASTE_STREAMS, build_stream_problem, run_stream_optimization, aggregate_kpi
    global WASTE_TYPES, prepare_history_from_dataframe, train_and_forecast_uploaded_history
    global estimate_vehicles_needed, build_forecast_excel, build_template_history_excel

    from baseline import nearest_neighbor_baseline
    from data_generator import DemoConfig, generate_demo_data, load_points_from_dataframe, DEPOT_LOCATION
    from clustering import run_clustering_cvrp, split_time_budget
    from dynamic_routing import DynamicRoutingEngine, GPSTrackerConfig, interpolate_along_route
    from optimizer import OptimizeConfig, solve_cvrp, solve_cvrp_with_auto_scaling
    from routing import DEFAULT_OSRM_BASE_URL, OSRMError, check_point_count_limit, get_osrm_matrices, get_osrm_route_geometry
    from waste_streams import WASTE_STREAMS, build_stream_problem, run_stream_optimization, aggregate_kpi
    from forecasting import (
        WASTE_TYPES,
        prepare_history_from_dataframe,
        train_and_forecast_uploaded_history,
        estimate_vehicles_needed,
        build_forecast_excel,
        build_template_history_excel,
    )

check("1. imports giống app.py", step_imports)


# ---------------------------------------------------------------------------
# 2. LOAD DỮ LIỆU DEMO (như app.py _load_data, nhánh demo)
# ---------------------------------------------------------------------------
def step_load_data():
    cfg = DemoConfig(num_points=25, seed=42, use_time_windows=False)
    df_points = generate_demo_data(cfg)
    assert df_points.shape[0] == 26, f"expect 25 points + depot = 26 rows, got {df_points.shape[0]}"
    assert df_points.iloc[0]["is_depot"] == True
    for col in ["waste_recyclable_kg", "waste_food_kg", "waste_other_kg", "is_major_food_generator", "requires_small_vehicle"]:
        assert col in df_points.columns, f"missing col {col}"
    return df_points

df_points = check("2. generate_demo_data(25 điểm)", step_load_data)


# ---------------------------------------------------------------------------
# 2b. LOAD DỮ LIỆU UPLOAD (load_points_from_dataframe) - kiểm tra nhánh upload
# ---------------------------------------------------------------------------
def step_load_upload():
    raw = pd.DataFrame({
        "node_id": ["A", "B", "C"],
        "latitude": [10.84, 10.845, 10.83],
        "longitude": [106.78, 106.783, 106.79],
        "waste_kg": [100.0, 200.0, 150.0],
    })
    out = load_points_from_dataframe(raw)
    assert out.iloc[0]["is_depot"] == True
    assert "waste_recyclable_kg" in out.columns
    return out

check("2b. load_points_from_dataframe (upload tối thiểu)", step_load_upload)


# ---------------------------------------------------------------------------
# 3. OSRM MATRIX (Haversine fallback, không cần internet)
# ---------------------------------------------------------------------------
def step_matrix():
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    matrix_result = get_osrm_matrices(
        coords, base_url="http://invalid.invalid",
        allow_haversine_fallback=True, fallback_avg_speed_kmh=25,
    )
    assert matrix_result.source == "HAVERSINE_FALLBACK"
    return matrix_result

matrix_result = check("3. get_osrm_matrices (haversine fallback)", step_matrix)


# ---------------------------------------------------------------------------
# 4. BASELINE (Nearest Neighbor)
# ---------------------------------------------------------------------------
def step_baseline():
    dist_m = matrix_result.distance_matrix_m
    dur_s = matrix_result.duration_matrix_s
    service_times_s = (df_points["service_time"] * 60).tolist()
    demands = df_points["waste_kg"].tolist()
    depot_index = int(df_points.index[df_points["is_depot"]][0])
    routes = nearest_neighbor_baseline(
        dist_m, dur_s, demands, service_times_s,
        vehicle_capacity_kg=1000, depot_index=depot_index,
        max_route_time_s=4 * 3600, max_vehicles=20,
    )
    assert len(routes) > 0
    return routes, dist_m, dur_s, service_times_s, demands, depot_index

baseline_out = check("4. nearest_neighbor_baseline", step_baseline)
if baseline_out:
    baseline_routes, dist_m, dur_s, service_times_s, demands, depot_index = baseline_out


# ---------------------------------------------------------------------------
# 5. OR-TOOLS CVRP (optimized)
# ---------------------------------------------------------------------------
def step_cvrp():
    cfg = OptimizeConfig(
        num_vehicles=max(3, len(baseline_routes)),
        vehicle_capacity_kg=1000, depot_index=depot_index,
        use_gls=True, first_solution_strategy="PATH_CHEAPEST_ARC",
        time_limit_sec=5, max_route_time_s=4 * 3600,
    )
    routes, solved, msg = solve_cvrp(dist_m, dur_s, demands, service_times_s, cfg)
    assert solved, msg
    return routes

optimized_routes = check("5. solve_cvrp (OR-Tools + GLS)", step_cvrp)


# ---------------------------------------------------------------------------
# 6. TABLE 4 / KPI AGGREGATION (app.py _aggregate, _daily_operation_row)
# ---------------------------------------------------------------------------
def step_kpi():
    def _aggregate(routes, fuel_rate=0.35, emission_factor=2.68):
        total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0
        travel_time_min = sum(r.travel_time_s for r in routes) / 60.0
        service_time_min = sum(r.service_time_s for r in routes) / 60.0
        total_time_min = sum(r.total_route_time_s for r in routes) / 60.0
        num_vehicles_used = len(routes)
        total_waste_kg = sum(r.collected_waste_kg for r in routes)
        avg_util = sum(r.capacity_utilization_pct for r in routes) / len(routes) if routes else 0.0
        fuel_l = total_distance_km * fuel_rate
        co2_kg = fuel_l * emission_factor
        return {
            "Total distance (km)": round(total_distance_km, 2),
            "Number of vehicles": num_vehicles_used,
            "Total waste collected (kg)": round(total_waste_kg, 1),
        }
    b = _aggregate(baseline_routes)
    o = _aggregate(optimized_routes)

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
        return {"Tổng điểm thu gom": total_points, "extra_weight": extra_weight, "carrying_avg": carrying_avg}
    base_ops = _daily_operation_row(baseline_routes, 1000)
    prop_ops = _daily_operation_row(optimized_routes, 1000)
    return b, o, base_ops, prop_ops

check("6. KPI aggregation + Table4 (_aggregate, _daily_operation_row)", step_kpi)


# ---------------------------------------------------------------------------
# 7. OSRM ROUTE GEOMETRY (dùng cho vẽ bản đồ) - fallback network sẽ lỗi;
#    kiểm tra hàm trả về [] gracefully khi mạng không có (không raise).
# ---------------------------------------------------------------------------
def step_route_geometry():
    coords_pair = ((10.84, 106.78), (10.845, 106.783))
    geom = get_osrm_route_geometry(coords_pair, base_url="http://invalid.invalid")
    assert geom == [], f"expect [] khi OSRM không khả dụng, got {geom}"
    return geom

check("7. get_osrm_route_geometry (mạng không khả dụng -> trả về [])", step_route_geometry)


# ---------------------------------------------------------------------------
# 8. CLUSTERING COMPARE (K-means++ + CVRP theo cụm) - bộ 50 điểm cho nhanh
# ---------------------------------------------------------------------------
def step_clustering():
    cfg = DemoConfig(num_points=50, seed=42, use_time_windows=False)
    cmp_df = generate_demo_data(cfg)
    cmp_coords = tuple(zip(cmp_df["latitude"], cmp_df["longitude"]))
    cmp_demands = cmp_df["waste_kg"].tolist()
    cmp_service_s = (cmp_df["service_time"] * 60).tolist()
    cmp_node_ids = cmp_df["node_id"].tolist()
    cmp_depot_idx = int(cmp_df.index[cmp_df["is_depot"]][0])

    cmp_matrix = get_osrm_matrices(
        cmp_coords, base_url="http://invalid.invalid",
        allow_haversine_fallback=True, fallback_avg_speed_kmh=25,
    )
    dist_cmp = cmp_matrix.distance_matrix_m
    dur_cmp = cmp_matrix.duration_matrix_s
    cmp_capacity = 1500
    max_route_s = 8 * 3600

    nn_routes = nearest_neighbor_baseline(
        dist_cmp, dur_cmp, cmp_demands, cmp_service_s, cmp_capacity, cmp_depot_idx,
        max_route_s, max_vehicles=60,
    )

    direct_cfg = OptimizeConfig(
        num_vehicles=max(3, math.ceil(sum(cmp_demands) / cmp_capacity) + 2),
        vehicle_capacity_kg=cmp_capacity, depot_index=0, use_gls=True,
        first_solution_strategy="PATH_CHEAPEST_ARC", time_limit_sec=5,
        max_route_time_s=max_route_s,
    )
    direct_routes, direct_solved, direct_msg, _n = solve_cvrp_with_auto_scaling(
        dist_cmp, dur_cmp, cmp_demands, cmp_service_s, direct_cfg, max_extra_vehicles=5,
    )
    assert direct_solved, direct_msg

    cluster_result = run_clustering_cvrp(
        list(cmp_coords), cmp_depot_idx, cmp_demands, cmp_service_s, dist_cmp, dur_cmp, cmp_node_ids,
        vehicle_capacity_kg=cmp_capacity, use_gls=True,
        first_solution_strategy="PATH_CHEAPEST_ARC", time_limit_sec=5,
        max_route_time_s=max_route_s,
    )
    assert cluster_result.solved, cluster_result.messages
    assert len(cluster_result.routes) > 0
    return nn_routes, direct_routes, cluster_result

clustering_out = check("8. Clustering compare (NN vs CVRP trực tiếp vs K-means++ CVRP)", step_clustering)


# ---------------------------------------------------------------------------
# 9. WASTE STREAMS (multi-stream optimization) — ĐÂY LÀ NƠI PHÁT HIỆN BUG
# ---------------------------------------------------------------------------
def step_waste_streams_broken_config():
    """Tái hiện đúng config mặc định (2 xe, 800kg) mà UI cho phép người dùng
    bấm nút ngay mà không chỉnh gì - PHẢI báo lỗi rõ ràng (RuntimeError được
    bắt lại), KHÔNG được crash cả app. Mô phỏng đúng try/except mới trong app.py."""
    depot_idx_global = int(df_points.index[df_points["is_depot"]][0])
    stream_results = {}
    stream_errors = []
    stream_vehicle_cfg = {k: {"num_vehicles": 2, "vehicle_capacity_kg": 800} for k in WASTE_STREAMS}
    for key in WASTE_STREAMS:
        problem = build_stream_problem(df_points, dist_m, dur_s, service_times_s, key, depot_idx_global)
        if problem is None:
            stream_results[key] = None
            continue
        try:
            stream_results[key] = run_stream_optimization(
                problem,
                num_vehicles=stream_vehicle_cfg[key]["num_vehicles"],
                vehicle_capacity_kg=stream_vehicle_cfg[key]["vehicle_capacity_kg"],
                use_gls=True, first_solution_strategy="PATH_CHEAPEST_ARC",
                time_limit_sec=5, max_route_time_s=4 * 3600,
            )
        except RuntimeError as exc:
            stream_results[key] = None
            stream_errors.append(f"Luồng '{WASTE_STREAMS[key]['label']}': {exc}")
    # Với config mặc định (800kg x 2 xe) trên 25 điểm demo, ít nhất 1 luồng
    # (thường là 'other') sẽ KHÔNG đủ tải -> phải thấy lỗi được bắt lại ở đây,
    # không phải một exception văng thẳng lên ngoài hàm này.
    assert stream_errors, "Kỳ vọng có ít nhất 1 lỗi capacity được bắt lại (không phải crash)"
    return stream_results, stream_errors

check("9a. Waste streams - config mặc định thiếu tải (PHẢI báo lỗi, KHÔNG crash)", step_waste_streams_broken_config)


def step_waste_streams_ok_config():
    """Config đủ tải (10 xe, 5000kg) - toàn bộ luồng phải solve được và
    stream_kpi_rows phải build được KHÔNG lỗi KeyError (đây là bug đã fix)."""
    depot_idx_global = int(df_points.index[df_points["is_depot"]][0])
    stream_results = {}
    for key in WASTE_STREAMS:
        problem = build_stream_problem(df_points, dist_m, dur_s, service_times_s, key, depot_idx_global)
        if problem is None:
            stream_results[key] = None
            continue
        stream_results[key] = run_stream_optimization(
            problem, num_vehicles=10, vehicle_capacity_kg=5000,
            use_gls=True, first_solution_strategy="PATH_CHEAPEST_ARC",
            time_limit_sec=5, max_route_time_s=4 * 3600,
        )

    total_after_km = 0.0
    total_after_vehicles = 0
    stream_kpi_rows = []
    for key, meta in WASTE_STREAMS.items():
        result = stream_results.get(key)
        if result is None:
            continue
        assert result.solved, f"luồng {key} không solve được: {result.message}"
        kpi_base = aggregate_kpi(result.baseline_routes)
        kpi_opt = aggregate_kpi(result.optimized_routes)
        total_after_km += kpi_opt["Tổng quãng đường (km)"]
        total_after_vehicles += kpi_opt["Số xe"]
        # ĐÚNG NHƯ app.py (đã fix: waste_streams.py giờ trả về đúng key này)
        stream_kpi_rows.append({
            "Luồng": meta["label"],
            "Số xe (optimized)": kpi_opt["Số xe"],
            "Quãng đường (km)": kpi_opt["Tổng quãng đường (km)"],
            "Khối lượng (kg)": kpi_opt["Khối lượng thu gom (kg)"],
        })
    assert len(stream_kpi_rows) == 3, f"expect 3 luồng có dữ liệu, got {len(stream_kpi_rows)}"
    return stream_results, stream_kpi_rows

check("9b. Waste streams - config đủ tải (PHẢI solve OK, KHÔNG KeyError)", step_waste_streams_ok_config)


# ---------------------------------------------------------------------------
# 10. DYNAMIC ROUTING (GPS simulation) - app.py GPS engine init + tick loop
# ---------------------------------------------------------------------------
def step_dynamic_routing():
    node_ids_full = df_points["node_id"].tolist()
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    chosen_route = optimized_routes[0]
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
        points_for_engine, depot_latlon=DEPOT_LOCATION,
        config=GPSTrackerConfig(completion_radius_m=40, depot_radius_m=40, dwell_seconds_required=10),
    )
    active_coords = tuple(coords[i] for i in chosen_route.node_sequence)
    active_geometry = get_osrm_route_geometry(active_coords, base_url="http://invalid.invalid")
    if not active_geometry:
        active_geometry = list(active_coords)

    # mô phỏng xe chạy dọc route, tick từng giây, dùng interpolate_along_route
    sim_progress_m = 0.0
    speed_mps = 20 * 1000 / 3600
    events = []
    for t in range(1, 2000):
        sim_progress_m += speed_mps * 1.0
        lat, lon, finished = interpolate_along_route(active_geometry, sim_progress_m)
        if lat is None:
            break
        ev = engine.update_position(lat, lon, ts=float(t))
        if ev["newly_completed"] or ev["depot_confirmed"]:
            events.append(ev)
        if finished and t > 20:
            break
    assert len(points_for_engine) > 0
    return engine, events

check("10. DynamicRoutingEngine GPS simulation (auto completion loop)", step_dynamic_routing)


# ---------------------------------------------------------------------------
# 11. FORECASTING (XGBoost path via prepare_history_from_dataframe + train_and_forecast_uploaded_history)
# ---------------------------------------------------------------------------
def step_forecasting():
    from forecasting import generate_waste_history
    raw_history = generate_waste_history(df_points, days=180, seed=42)
    # simulate what an uploaded file with waste_organic/recyclable/other looks like
    raw_history = raw_history.rename(columns={})
    history = prepare_history_from_dataframe(raw_history, df_points=df_points)
    assert history["date"].nunique() >= 30
    result = train_and_forecast_uploaded_history(history, method="xgboost", forecast_days=7)
    assert "forecast_df" in result and "validation_df" in result
    forecast_df = result["forecast_df"]
    assert "total_kg" in forecast_df.columns
    total_kg, n_vehicles = estimate_vehicles_needed(
        {r["node_id"]: {"total_kg": r["total_kg"]} for _, r in forecast_df.iterrows()}, 1000
    )
    assert n_vehicles >= 1

    export_bytes = build_forecast_excel(history, forecast_df, result.get("validation_df"))
    assert len(export_bytes) > 0
    template_bytes = build_template_history_excel()
    assert len(template_bytes) > 0
    return result

check("11. Forecasting pipeline (XGBoost, đúng luồng app.py)", step_forecasting)


# ---------------------------------------------------------------------------
# TỔNG KẾT
# ---------------------------------------------------------------------------
print("\n\n================ TỔNG KẾT ================")
if FAILURES:
    print(f"{len(FAILURES)} bước THẤT BẠI:")
    for label, exc in FAILURES:
        print(f"  - {label}: {type(exc).__name__}: {exc}")
else:
    print("Tất cả các bước đều PASS.")
