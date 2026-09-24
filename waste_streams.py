"""
waste_streams.py
----------------
Xử lý bài toán tối ưu tuyến theo từng luồng rác (waste stream).

API được thiết kế khớp trực tiếp với app.py:
    WASTE_STREAMS
    build_stream_problem(...)
    run_stream_optimization(...)
    aggregate_kpi(...)

Mỗi luồng rác được xem như một bài toán CVRP độc lập nhưng tái sử dụng
ma trận OSRM đã được tính ở pipeline chính.
"""
from __future__ import annotations

from dataclasses import dataclass

from baseline import RouteResult, compute_route_metrics, nearest_neighbor_baseline
from optimizer import OptimizeConfig, solve_cvrp_with_auto_scaling


WASTE_STREAMS = {
    "recyclable": {
        "label": "Tái chế",
        "column": "waste_recyclable_kg",
    },
    "food": {
        "label": "Thực phẩm (nhóm phát sinh nhiều)",
        "column": "waste_food_kg",
    },
    "other": {
        "label": "Còn lại",
        "column": "waste_other_kg",
    },
}


@dataclass
class StreamProblem:
    """Bài toán con của một waste stream, dùng index cục bộ."""

    stream_key: str
    node_ids: list[str]
    global_indices: list[int]
    distance_matrix_m: list
    duration_matrix_s: list
    demands: list[float]
    service_times_s: list[float]
    depot_index: int = 0


@dataclass
class StreamOptimizationResult:
    problem: StreamProblem
    baseline_routes: list[RouteResult]
    optimized_routes: list[RouteResult]
    solved: bool
    message: str


def _submatrix(matrix, indices):
    return [[matrix[i][j] for j in indices] for i in indices]


def build_stream_problem(
    df_points,
    dist_m_full,
    dur_s_full,
    service_times_s_full,
    stream_key: str,
    depot_idx_global: int,
):
    """Tạo bài toán con cho một luồng rác.

    Depot luôn được đưa lên index 0. Các điểm có demand của luồng <= 0
    được loại khỏi bài toán con. Nếu không còn điểm khách hàng nào thì
    trả về None để app bỏ qua luồng đó.
    """
    if stream_key not in WASTE_STREAMS:
        raise ValueError(f"Luồng rác không hợp lệ: {stream_key}")

    demand_col = WASTE_STREAMS[stream_key]["column"]
    if demand_col not in df_points.columns:
        raise ValueError(
            f"Thiếu cột '{demand_col}'. Hãy dùng dữ liệu có đủ 3 cột waste_recyclable_kg, "
            "waste_food_kg và waste_other_kg hoặc để data_generator tự bổ sung."
        )

    customer_indices = [
        int(i)
        for i, row in df_points.iterrows()
        if int(i) != depot_idx_global and float(row[demand_col]) > 1e-9
    ]
    if not customer_indices:
        return None

    global_indices = [depot_idx_global] + customer_indices
    node_ids = [str(df_points.iloc[i]["node_id"]) for i in global_indices]
    demands = [0.0] + [float(df_points.iloc[i][demand_col]) for i in customer_indices]
    service = [float(service_times_s_full[i]) for i in global_indices]

    return StreamProblem(
        stream_key=stream_key,
        node_ids=node_ids,
        global_indices=global_indices,
        distance_matrix_m=_submatrix(dist_m_full, global_indices),
        duration_matrix_s=_submatrix(dur_s_full, global_indices),
        demands=demands,
        service_times_s=service,
        depot_index=0,
    )


def run_stream_optimization(
    problem: StreamProblem,
    num_vehicles: int,
    vehicle_capacity_kg: float,
    use_gls: bool = True,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    time_limit_sec: int = 15,
    max_route_time_s: float | None = None,
) -> StreamOptimizationResult:
    """Chạy baseline Nearest Neighbor và OR-Tools + GLS cho một luồng."""
    baseline_routes = nearest_neighbor_baseline(
        problem.distance_matrix_m,
        problem.duration_matrix_s,
        problem.demands,
        problem.service_times_s,
        vehicle_capacity_kg=vehicle_capacity_kg,
        depot_index=problem.depot_index,
        max_route_time_s=max_route_time_s,
        max_vehicles=max(1, int(num_vehicles)),
    )

    cfg = OptimizeConfig(
        num_vehicles=max(1, int(num_vehicles)),
        vehicle_capacity_kg=float(vehicle_capacity_kg),
        depot_index=problem.depot_index,
        use_gls=use_gls,
        first_solution_strategy=first_solution_strategy,
        time_limit_sec=max(1, int(time_limit_sec)),
        max_route_time_s=max_route_time_s,
    )

    optimized_routes, solved, message, _ = solve_cvrp_with_auto_scaling(
        problem.distance_matrix_m,
        problem.duration_matrix_s,
        problem.demands,
        problem.service_times_s,
        cfg,
        max_extra_vehicles=4,
    )

    return StreamOptimizationResult(
        problem=problem,
        baseline_routes=baseline_routes,
        optimized_routes=optimized_routes,
        solved=solved,
        message=message,
    )


def aggregate_kpi(routes: list[RouteResult]) -> dict:
    """Tổng hợp KPI cho một waste stream, cùng kiểu hiển thị mà app.py cần."""
    if not routes:
        return {
            "Số xe": 0,
            "Tổng quãng đường (km)": 0.0,
            "Tổng thời gian tuyến (phút)": 0.0,
            "Khối lượng thu gom (kg)": 0.0,
            "Tải trọng TB (%)": 0.0,
        }

    total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0
    total_time_min = sum(r.total_route_time_s for r in routes) / 60.0
    total_waste = sum(r.collected_waste_kg for r in routes)
    avg_util = sum(r.capacity_utilization_pct for r in routes) / len(routes)

    return {
        "Số xe": len(routes),
        "Tổng quãng đường (km)": round(total_distance_km, 2),
        "Tổng thời gian tuyến (phút)": round(total_time_min, 1),
        "Khối lượng thu gom (kg)": round(total_waste, 1),
        "Tải trọng TB (%)": round(avg_util, 1),
    }
