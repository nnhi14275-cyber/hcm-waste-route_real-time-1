"""
test_harness2.py
-----------------
Kiểm tra các luồng MỚI trong app.py (LoGVN): auto fleet sizing 115%, dashboard
chi phí, bảng phân bổ xe, mô phỏng sự cố giao thông, và thuật toán Dynamic
(báo vật cản -> chặng phụ) trong dynamic_routing.py. Dùng Haversine fallback
(không cần internet).
"""
import math
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


from baseline import nearest_neighbor_baseline, clarke_wright_savings, BASELINE_METHODS
from data_generator import DemoConfig, generate_demo_data, DEPOT_LOCATION
from dynamic_routing import DynamicRoutingEngine, GPSTrackerConfig, interpolate_along_route
from optimizer import OptimizeConfig, solve_cvrp_with_auto_scaling
from routing import get_osrm_matrices


def auto_fleet_size(total_demand_kg, capacity_kg, overload_pct, buffer_vehicles=1):
    eff_capacity = capacity_kg * (overload_pct / 100.0)
    if eff_capacity <= 0:
        return max(1, buffer_vehicles)
    return max(1, math.ceil(total_demand_kg / eff_capacity)) + buffer_vehicles


# ---------------------------------------------------------------------------
# 1. Auto fleet sizing (115% overload rule)
# ---------------------------------------------------------------------------
def step_fleet_size():
    cfg = DemoConfig(num_points=25, seed=42)
    df_points = generate_demo_data(cfg)
    total_demand = sum(df_points["waste_kg"])
    n = auto_fleet_size(total_demand, 1000, 115)
    print(f"total_demand={total_demand:.1f}kg, capacity=1000kg, overload=115% -> n_fleet={n}")
    assert n >= 1
    return df_points, total_demand, n

out = check("1. auto_fleet_size", step_fleet_size)
df_points, total_demand, n_fleet = out


# ---------------------------------------------------------------------------
# 2. Full pipeline with effective_capacity = capacity * overload%, BASELINE_METHODS both
# ---------------------------------------------------------------------------
def step_pipeline():
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    depot_index = int(df_points.index[df_points["is_depot"]][0])
    demands_full = df_points["waste_kg"].tolist()
    service_times_s_full = (df_points["service_time"] * 60).tolist()
    node_ids_full = df_points["node_id"].tolist()

    mr = get_osrm_matrices(coords, base_url="http://invalid.invalid", allow_haversine_fallback=True, fallback_avg_speed_kmh=25)
    dist_m_full, dur_s_full = mr.distance_matrix_m, mr.duration_matrix_s

    effective_capacity_kg = 1000 * 1.15

    for label, fn in BASELINE_METHODS.items():
        routes = fn(dist_m_full, dur_s_full, demands_full, service_times_s_full,
                    vehicle_capacity_kg=effective_capacity_kg, depot_index=depot_index,
                    max_route_time_s=4 * 3600, max_vehicles=max(n_fleet, 20))
        assert len(routes) > 0, f"{label} returned no routes"
        print(f"  baseline[{label}] -> {len(routes)} routes")

    cfg = OptimizeConfig(
        num_vehicles=n_fleet, vehicle_capacity_kg=effective_capacity_kg, depot_index=depot_index,
        use_gls=True, first_solution_strategy="PATH_CHEAPEST_ARC", time_limit_sec=8, max_route_time_s=4 * 3600,
    )
    optimized_routes, solved, msg, n_used = solve_cvrp_with_auto_scaling(
        dist_m_full, dur_s_full, demands_full, service_times_s_full, cfg, max_extra_vehicles=4
    )
    assert solved, msg
    print(f"  optimized -> {len(optimized_routes)} routes, n_used={n_used}")
    return coords, depot_index, demands_full, service_times_s_full, node_ids_full, dist_m_full, dur_s_full, optimized_routes

pipe_out = check("2. pipeline (baseline both methods + OR-Tools, effective_capacity 115%)", step_pipeline)
coords, depot_index, demands_full, service_times_s_full, node_ids_full, dist_m_full, dur_s_full, optimized_routes = pipe_out


# ---------------------------------------------------------------------------
# 3. Cost dashboard math (no NaN/negative sanity)
# ---------------------------------------------------------------------------
def step_cost_dashboard():
    def _aggregate_kpi_routes(routes, fuel_rate, fuel_price, emission_factor):
        total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0 if routes else 0.0
        fuel_l = total_distance_km * fuel_rate
        return {
            "Tổng quãng đường (km)": round(total_distance_km, 2),
            "Nhiên liệu tiêu thụ (lít)": round(fuel_l, 1),
            "Chi phí nhiên liệu (VNĐ)": round(fuel_l * fuel_price, 0),
            "Phát thải CO2 (kg)": round(fuel_l * emission_factor, 1),
            "Số xe sử dụng": len(routes),
        }
    kpi = _aggregate_kpi_routes(optimized_routes, 0.35, 22000, 2.68)
    assert kpi["Chi phí nhiên liệu (VNĐ)"] >= 0
    assert kpi["Tổng quãng đường (km)"] > 0
    print(" ", kpi)
    return kpi

check("3. cost dashboard math", step_cost_dashboard)


# ---------------------------------------------------------------------------
# 4. Real-time traffic incident simulation (localized multiplier + reroute)
# ---------------------------------------------------------------------------
def step_incident():
    index_of_node = {nid: i for i, nid in enumerate(node_ids_full)}
    incident_node = node_ids_full[5] if node_ids_full[5] != node_ids_full[depot_index] else node_ids_full[6]
    incident_idx = index_of_node[incident_node]
    severity = 1.8
    n = len(dur_s_full)
    dur_incident = [row[:] for row in dur_s_full]
    for a in range(n):
        for b in range(n):
            if a == incident_idx or b == incident_idx:
                dur_incident[a][b] = dur_s_full[a][b] * severity

    # sanity: incident matrix should differ from original exactly at rows/cols touching incident_idx
    diffs = sum(1 for a in range(n) for b in range(n) if dur_incident[a][b] != dur_s_full[a][b])
    assert diffs > 0

    cfg = OptimizeConfig(
        num_vehicles=n_fleet, vehicle_capacity_kg=1000 * 1.15, depot_index=depot_index,
        use_gls=True, first_solution_strategy="PATH_CHEAPEST_ARC", time_limit_sec=8, max_route_time_s=4 * 3600,
    )
    incident_routes, solved, msg, _n = solve_cvrp_with_auto_scaling(
        dist_m_full, dur_incident, demands_full, service_times_s_full, cfg, max_extra_vehicles=4
    )
    assert solved, msg
    print(f"  incident reroute solved with {len(incident_routes)} routes")
    return incident_routes

check("4. traffic incident simulation (localized multiplier + reroute)", step_incident)


# ---------------------------------------------------------------------------
# 5. Dynamic obstacle handling: mark_deferred -> reoptimize -> depot -> supplementary trip
# ---------------------------------------------------------------------------
def step_dynamic_obstacle():
    index_of_node = {nid: i for i, nid in enumerate(node_ids_full)}
    depot_node_id = df_points.loc[df_points["is_depot"], "node_id"].iloc[0]
    chosen_route = optimized_routes[0]
    points_for_engine = [
        {"node_id": node_ids_full[i], "latitude": df_points.iloc[i]["latitude"], "longitude": df_points.iloc[i]["longitude"]}
        for i in chosen_route.node_sequence if not df_points.iloc[i]["is_depot"]
    ]
    assert len(points_for_engine) >= 2, "cần route có >=2 điểm để test obstacle có ý nghĩa"

    engine = DynamicRoutingEngine(points_for_engine, depot_latlon=DEPOT_LOCATION,
                                   config=GPSTrackerConfig(completion_radius_m=40, depot_radius_m=40, dwell_seconds_required=10))

    obstacle_node = points_for_engine[0]["node_id"]
    assert engine.mark_deferred(obstacle_node, "Vật cản test") is True
    assert obstacle_node not in engine.pending_node_ids()
    assert obstacle_node in engine.deferred_node_ids()
    # mark_deferred trên điểm KHÔNG pending (VD điểm đã completed hoặc không tồn tại) phải trả False, không raise
    assert engine.mark_deferred("khong_ton_tai", "x") is False

    # reoptimize phần còn lại (giống app.py _reoptimize_current_route)
    remaining_pending = engine.pending_node_ids()

    def reopt(pending_ids, num_vehicles=1):
        sub_indices = [index_of_node[depot_node_id]] + [index_of_node[nid] for nid in pending_ids]
        sub_dist = [[dist_m_full[a][b] for b in sub_indices] for a in sub_indices]
        sub_dur = [[dur_s_full[a][b] for b in sub_indices] for a in sub_indices]
        sub_demands = [demands_full[i] for i in sub_indices]
        sub_service = [service_times_s_full[i] for i in sub_indices]
        cfg = OptimizeConfig(num_vehicles=num_vehicles, vehicle_capacity_kg=1000 * 1.15, depot_index=0,
                              use_gls=True, first_solution_strategy="PATH_CHEAPEST_ARC", time_limit_sec=8, max_route_time_s=4 * 3600)
        routes, solved, msg, _n = solve_cvrp_with_auto_scaling(sub_dist, sub_dur, sub_demands, sub_service, cfg, max_extra_vehicles=0)
        return routes, solved, msg, sub_indices

    if remaining_pending:
        routes, solved, msg, sub_indices = reopt(remaining_pending)
        assert solved, msg
        new_nodes = [sub_indices[i] for i in routes[0].node_sequence]
        engine.sync_pending_after_reoptimize(remaining_pending)
        # deferred điểm KHÔNG được bị đẩy lại pending bởi sync
        assert obstacle_node in engine.deferred_node_ids(), "sync_pending_after_reoptimize không được revert deferred -> pending"

    # simulate xe hoàn thành hết remaining_pending (auto-complete thủ công cho test, không chạy GPS thật)
    for nid in list(engine.pending_node_ids()):
        st_ = engine.point_status[nid]
        st_.status = "completed"

    assert engine.pending_node_ids() == []
    assert engine.deferred_node_ids() == [obstacle_node]

    # ---- Chặng phụ: requeue deferred rồi reopt ----
    deferred_ids = engine.deferred_node_ids()
    for nid in deferred_ids:
        assert engine.requeue_deferred(nid) is True
    assert engine.requeue_deferred("khong_ton_tai") is False  # phải trả False, không raise

    routes2, solved2, msg2, sub_indices2 = reopt(deferred_ids)
    assert solved2, msg2
    engine.sync_pending_after_reoptimize(deferred_ids)
    assert deferred_ids[0] in engine.pending_node_ids(), "requeue_deferred + sync phải đưa điểm về pending cho chặng phụ"
    print(f"  Chặng phụ tính được route qua {len(deferred_ids)} điểm bị hoãn trước đó, OK.")
    return engine

check("5. Dynamic obstacle handling (mark_deferred -> reopt -> requeue -> chặng phụ)", step_dynamic_obstacle)


# ---------------------------------------------------------------------------
# 6. Vehicle allocation table build (no crash, correct columns)
# ---------------------------------------------------------------------------
def step_alloc_table():
    def _vehicle_allocation_table(routes, df_points_local):
        rows = []
        for r in routes:
            stop_ids = [df_points_local.iloc[i]["node_id"] for i in r.node_sequence if not df_points_local.iloc[i]["is_depot"]]
            rows.append({
                "Xe": f"Vehicle {r.vehicle_id}", "Số điểm": r.num_stops,
                "Danh sách điểm": ", ".join(stop_ids),
                "Khối lượng (kg)": round(r.collected_waste_kg, 1),
                "Tải trọng (%)": round(r.capacity_utilization_pct, 1),
                "Quãng đường (km)": round(r.total_distance_m / 1000.0, 2),
                "Thời gian tuyến (phút)": round(r.total_route_time_s / 60.0, 1),
            })
        return pd.DataFrame(rows)
    df = _vehicle_allocation_table(optimized_routes, df_points)
    assert len(df) == len(optimized_routes)
    assert set(["Xe", "Số điểm", "Danh sách điểm", "Khối lượng (kg)", "Tải trọng (%)", "Quãng đường (km)", "Thời gian tuyến (phút)"]) <= set(df.columns)
    print(df.to_string(index=False))
    return df

check("6. vehicle allocation table", step_alloc_table)


# ---------------------------------------------------------------------------
# 7. Excel report export (routing report)
# ---------------------------------------------------------------------------
def step_excel_export():
    import io
    def _aggregate_kpi_routes(routes, fuel_rate, fuel_price, emission_factor):
        total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0 if routes else 0.0
        fuel_l = total_distance_km * fuel_rate
        return {"Tổng quãng đường (km)": round(total_distance_km, 2), "Chi phí nhiên liệu (VNĐ)": round(fuel_l * fuel_price, 0)}

    kpi_opt = _aggregate_kpi_routes(optimized_routes, 0.35, 22000, 2.68)
    alloc_df = pd.DataFrame([{"Xe": f"Vehicle {r.vehicle_id}"} for r in optimized_routes])
    results = {
        "kpi_base": kpi_opt, "kpi_opt": kpi_opt, "alloc_df": alloc_df,
        "savings": {"Giảm quãng đường (km)": 1.0},
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        kpi_df = pd.DataFrame([{"Kịch bản": "Baseline", **results["kpi_base"]}, {"Kịch bản": "Optimized", **results["kpi_opt"]}])
        kpi_df.to_excel(writer, sheet_name="KPI so sánh", index=False)
        results["alloc_df"].to_excel(writer, sheet_name="Phân bổ xe", index=False)
        pd.DataFrame([results["savings"]]).to_excel(writer, sheet_name="Tiết kiệm", index=False)
    buf.seek(0)
    data = buf.getvalue()
    assert len(data) > 0
    print(f"  excel bytes: {len(data)}")

check("7. Excel export (routing report)", step_excel_export)


print("\n\n================ TỔNG KẾT ================")
if FAILURES:
    print(f"{len(FAILURES)} bước THẤT BẠI:")
    for label, exc in FAILURES:
        print(f"  - {label}: {type(exc).__name__}: {exc}")
else:
    print("Tất cả các bước đều PASS.")
