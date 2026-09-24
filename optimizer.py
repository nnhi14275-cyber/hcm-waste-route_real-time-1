"""
optimizer.py
------------
Tối ưu hoá tuyến thu gom bằng Google OR-Tools (CVRP, mở rộng VRPTW khi có
time window), dùng Guided Local Search (GLS) làm metaheuristic chính.

Khoảng cách/thời gian sử dụng trong mô hình PHẢI lấy từ ma trận OSRM (được
truyền vào từ routing.py), KHÔNG tính lại bằng distance / average_speed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from baseline import RouteResult, compute_route_metrics

FIRST_SOLUTION_STRATEGIES = {
    "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    "GLOBAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
}


@dataclass
class OptimizeConfig:
    num_vehicles: int
    vehicle_capacity_kg: float
    depot_index: int = 0
    use_gls: bool = True
    first_solution_strategy: str = "PATH_CHEAPEST_ARC"
    time_limit_sec: int = 20
    max_route_time_s: float | None = None
    use_time_windows: bool = False
    time_windows_s: list | None = None  # list[(start_s, end_s)] theo node index

    # ---- Đội xe không đồng nhất (heterogeneous fleet) ----
    # Nếu vehicle_capacities_kg được cung cấp (độ dài = num_vehicles), giá trị
    # này sẽ ĐƯỢC ƯU TIÊN thay cho vehicle_capacity_kg đồng nhất phía trên.
    vehicle_capacities_kg: list | None = None
    # Đánh dấu xe nào là "xe nhỏ" (đủ nhỏ để vào hẻm sâu). Độ dài = num_vehicles.
    small_vehicle_flags: list | None = None
    # Các node (hẻm sâu/đường nhỏ) CHỈ được phục vụ bởi xe nhỏ.
    small_vehicle_only_nodes: list | None = None


def solve_cvrp(
    distance_matrix,
    duration_matrix,
    demands: list,
    service_times_s: list,
    config: OptimizeConfig,
):
    """Giải CVRP / VRPTW bằng OR-Tools + Guided Local Search.

    Trả về (routes: list[RouteResult], solved: bool, info_message: str)
    """
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, config.num_vehicles, config.depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # ---- Chi phí quãng đường (khoảng cách OSRM, mét) là mục tiêu chính ----
    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(distance_matrix[i][j]))

    distance_cb_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_cb_index)

    # ---- Ràng buộc tải trọng xe (vehicle capacity, hỗ trợ đội xe không đồng nhất) ----
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(round(demands[node]))

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)
    if config.vehicle_capacities_kg:
        capacities = [int(round(c)) for c in config.vehicle_capacities_kg]
    else:
        capacities = [int(round(config.vehicle_capacity_kg))] * config.num_vehicles
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index,
        0,  # không cho slack tải trọng
        capacities,
        True,  # start cumul to zero
        "Capacity",
    )

    # ---- Thời gian: travel time (OSRM) + service time ----
    def time_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(duration_matrix[i][j] + service_times_s[i]))

    time_cb_index = routing.RegisterTransitCallback(time_callback)

    horizon = config.max_route_time_s
    if horizon is None:
        horizon = int(sum(max(row) for row in duration_matrix)) + int(sum(service_times_s)) + 3600
    horizon = int(round(horizon))

    routing.AddDimension(
        time_cb_index,
        int(horizon),   # slack tối đa (cho phép chờ time window)
        int(horizon),   # thời gian tuyến tối đa mỗi xe
        False,          # không ép cumul start = 0 (cần cho time window)
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # ---- VRPTW: time window theo từng điểm (nếu bật) ----
    if config.use_time_windows and config.time_windows_s:
        for node in range(n):
            index = manager.NodeToIndex(node)
            start_s, end_s = config.time_windows_s[node]
            time_dimension.CumulVar(index).SetRange(int(start_s), int(end_s))
        for v in range(config.num_vehicles):
            start_index = routing.Start(v)
            end_index = routing.End(v)
            depot_start, depot_end = config.time_windows_s[config.depot_index]
            time_dimension.CumulVar(start_index).SetRange(int(depot_start), int(depot_end))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))

    # ---- Ràng buộc "xe nhỏ cho hẻm sâu" ----
    # Một số điểm thu gom nằm trong hẻm nhỏ mà xe lớn không thể vào được;
    # các điểm này chỉ được phép gán cho các xe được đánh dấu là "xe nhỏ".
    # (Dùng solver.MemberCt trên VehicleVar thay vì SetAllowedVehiclesForIndex
    # vì binding SWIG của một số bản OR-Tools không nhận đúng kiểu Span<int>.)
    if config.small_vehicle_only_nodes and config.small_vehicle_flags:
        small_vehicle_indices = [
            v for v, is_small in enumerate(config.small_vehicle_flags) if is_small
        ]
        if small_vehicle_indices:
            solver = routing.solver()
            for node in config.small_vehicle_only_nodes:
                node_index = manager.NodeToIndex(node)
                vehicle_var = routing.VehicleVar(node_index)
                solver.Add(solver.MemberCt(vehicle_var, small_vehicle_indices))

    # ---- Tham số tìm kiếm ----
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    strategy = FIRST_SOLUTION_STRATEGIES.get(
        config.first_solution_strategy,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    )
    search_parameters.first_solution_strategy = strategy

    if config.use_gls:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
    search_parameters.time_limit.FromSeconds(int(config.time_limit_sec))

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return [], False, (
            "OR-Tools không tìm được lời giải khả thi với các ràng buộc hiện tại "
            "(capacity/time window/max route time). Hãy nới lỏng ràng buộc hoặc "
            "tăng số xe."
        )

    routes: list[RouteResult] = []
    for vehicle_id in range(config.num_vehicles):
        index = routing.Start(vehicle_id)
        sequence = []
        while not routing.IsEnd(index):
            sequence.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        sequence.append(manager.IndexToNode(index))  # depot cuối

        if len(sequence) <= 2:
            continue  # xe không được sử dụng

        vehicle_capacity = capacities[vehicle_id]
        route = compute_route_metrics(
            sequence, distance_matrix, duration_matrix,
            demands, service_times_s, vehicle_id + 1, vehicle_capacity,
        )
        routes.append(route)

    if not routes:
        return [], False, "OR-Tools trả về lời giải rỗng (không xe nào được sử dụng)."

    return routes, True, "success"


def solve_cvrp_with_auto_scaling(
    distance_matrix,
    duration_matrix,
    demands: list,
    service_times_s: list,
    config: OptimizeConfig,
    max_extra_vehicles: int = 4,
):
    """Gọi solve_cvrp(); nếu infeasible do thiếu xe, tự động thử tăng dần số xe
    (giữ nguyên capacity/loại xe hiện có, nhân bản thêm xe cùng loại) trước khi
    báo lỗi hẳn cho người dùng. Giúp tránh tình trạng báo "infeasible" chung
    chung trong khi chỉ cần thêm 1-2 xe là giải được.

    Trả về (routes, solved, message, num_vehicles_used_final).
    """
    base_num_vehicles = config.num_vehicles
    base_capacities = config.vehicle_capacities_kg
    base_small_flags = config.small_vehicle_flags

    for extra in range(0, max_extra_vehicles + 1):
        trial_num_vehicles = base_num_vehicles + extra
        trial_config = OptimizeConfig(
            num_vehicles=trial_num_vehicles,
            vehicle_capacity_kg=config.vehicle_capacity_kg,
            depot_index=config.depot_index,
            use_gls=config.use_gls,
            first_solution_strategy=config.first_solution_strategy,
            time_limit_sec=config.time_limit_sec,
            max_route_time_s=config.max_route_time_s,
            use_time_windows=config.use_time_windows,
            time_windows_s=config.time_windows_s,
            small_vehicle_only_nodes=config.small_vehicle_only_nodes,
        )
        if extra == 0:
            trial_config.vehicle_capacities_kg = base_capacities
            trial_config.small_vehicle_flags = base_small_flags
        elif base_capacities:
            # Nhân bản thêm xe cùng loại với xe cuối cùng trong danh sách hiện có
            trial_config.vehicle_capacities_kg = base_capacities + [base_capacities[-1]] * extra
            if base_small_flags:
                trial_config.small_vehicle_flags = base_small_flags + [base_small_flags[-1]] * extra

        routes, solved, msg = solve_cvrp(distance_matrix, duration_matrix, demands, service_times_s, trial_config)
        if solved:
            if extra > 0:
                msg = (
                    f"Không tìm được lời giải với {base_num_vehicles} xe ban đầu. "
                    f"Đã tự động tăng lên {trial_num_vehicles} xe để tìm được lời giải khả thi."
                )
            return routes, True, msg, trial_num_vehicles

    return [], False, (
        f"OR-Tools không tìm được lời giải khả thi ngay cả khi tăng lên "
        f"{base_num_vehicles + max_extra_vehicles} xe. Hãy nới lỏng ràng buộc "
        f"(capacity, max_route_time, time window) hoặc kiểm tra lại dữ liệu."
    ), base_num_vehicles


def run_multiple_for_stability(
    distance_matrix,
    duration_matrix,
    demands: list,
    service_times_s: list,
    config: OptimizeConfig,
    n_runs: int = 3,
):
    """Chạy OR-Tools+GLS nhiều lần với cùng cấu hình để đánh giá ĐỘ ỔN ĐỊNH
    của lời giải (do GLS dừng theo time_limit nên kết quả có thể dao động
    giữa các lần chạy). Trả về:

        best_routes, stats

    stats = {
        "distances_km": [...],   # tổng quãng đường mỗi lần chạy
        "best_km": ..., "mean_km": ..., "std_km": ..., "n_runs": n_runs,
    }
    """
    import statistics

    results = []
    for _ in range(max(1, n_runs)):
        routes, solved, _msg, _n = solve_cvrp_with_auto_scaling(
            distance_matrix, duration_matrix, demands, service_times_s, config
        )
        if solved:
            total_km = sum(r.total_distance_m for r in routes) / 1000.0
            results.append((total_km, routes))

    if not results:
        return [], {"distances_km": [], "best_km": None, "mean_km": None, "std_km": None, "n_runs": n_runs}

    results.sort(key=lambda x: x[0])
    best_km, best_routes = results[0]
    distances = [r[0] for r in results]
    stats = {
        "distances_km": [round(d, 2) for d in distances],
        "best_km": round(best_km, 2),
        "mean_km": round(statistics.mean(distances), 2),
        "std_km": round(statistics.stdev(distances), 2) if len(distances) > 1 else 0.0,
        "n_runs": len(distances),
    }
    return best_routes, stats
