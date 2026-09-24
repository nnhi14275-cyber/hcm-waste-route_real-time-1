"""
clustering.py
--------------
Giai đoạn PHÂN CỤM (Clustering) trong mô hình đề xuất "Cluster-first,
Route-second":

1. Số lượng cụm k được xác định TỰ ĐỘNG theo tổng khối lượng rác và sức chứa
   xe:  k = ceil( sum(T_i) / Q ).
2. Dùng K-means với khởi tạo k-means++ để nhóm các điểm thu gom thành k cụm
   đồng nhất về không gian.
3. Cân bằng tải trọng (capacity balancing): K-means gốc không xét ràng buộc
   tải trọng, nên sau khi phân cụm không gian, các điểm ở "biên" cụm quá tải
   sẽ được chuyển sang cụm liền kề còn dư sức chứa, lặp lại đến khi mọi cụm
   đều thoả capacity (hoặc không thể cân bằng thêm).
4. Mỗi cụm sau đó được giải như MỘT bài toán CVRP độc lập bằng OR-Tools +
   Guided Local Search (tái sử dụng optimizer.py), rồi gộp kết quả lại thành
   lời giải tổng thể.

Đây là mô hình đề xuất (K-means clustering + CVRP theo cụm) để so sánh với:
- Thuật toán đường đi ngắn nhất (Nearest Neighbor), KHÔNG phân cụm.
- CVRP trực tiếp bằng OR-Tools+GLS trên toàn bộ điểm, KHÔNG phân cụm.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from baseline import RouteResult, compute_route_metrics
from optimizer import OptimizeConfig, solve_cvrp_with_auto_scaling


def determine_num_clusters(
    demands: list, vehicle_capacity_kg: float, safety_margin: float = 0.90
) -> int:
    """k = ceil( tổng khối lượng rác / (sức chứa xe × safety_margin) ).

    safety_margin < 1 chừa dư địa để bước cân bằng tải trọng (capacity
    balancing) có thể dịch chuyển điểm giữa các cụm mà không vượt sức chứa —
    nếu k tính đúng khít theo tổng/capacity (safety_margin=1), trung bình tải
    mỗi cụm đã sát ngưỡng capacity, hầu như không còn dư địa để cân bằng các
    cụm bị lệch tải do phân cụm theo không gian (K-means không biết capacity).
    """
    total_demand = sum(demands)
    if vehicle_capacity_kg <= 0:
        return 1
    effective_capacity = max(1.0, vehicle_capacity_kg * safety_margin)
    return max(1, math.ceil(total_demand / effective_capacity))


@dataclass
class ClusteringResult:
    k: int
    labels: dict            # {node_index (khách hàng, không gồm depot): cluster_id}
    cluster_loads: dict     # {cluster_id: tổng khối lượng (kg)}
    balanced: bool          # True nếu cân bằng capacity thành công cho mọi cụm
    iterations: int


def capacitated_kmeans(
    coords: list,
    demands: list,
    customer_indices: list,
    vehicle_capacity_kg: float,
    k: int | None = None,
    random_state: int = 42,
    max_balance_iterations: int = 500,
) -> ClusteringResult:
    """Phân cụm K-means++ trên toạ độ các điểm khách hàng (không gồm depot),
    sau đó cân bằng tải trọng để mỗi cụm không vượt sức chứa xe.

    coords: list toạ độ (lat, lon) đầy đủ theo index gốc (bao gồm cả depot).
    customer_indices: các index (trong coords/demands) LÀ khách hàng, không
    gồm depot.
    """
    if k is None:
        k = determine_num_clusters([demands[i] for i in customer_indices], vehicle_capacity_kg)
    k = max(1, min(k, len(customer_indices)))

    X = np.array([coords[i] for i in customer_indices])

    if k == 1 or len(customer_indices) <= 1:
        labels_arr = np.zeros(len(customer_indices), dtype=int)
    else:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels_arr = km.fit_predict(X)

    labels = {customer_indices[i]: int(labels_arr[i]) for i in range(len(customer_indices))}

    def _cluster_loads(lbl: dict) -> dict:
        loads: dict[int, float] = {c: 0.0 for c in range(k)}
        for idx, c in lbl.items():
            loads[c] += demands[idx]
        return loads

    def _centroid(cluster_id: int, lbl: dict):
        members_c = [j for j, cc in lbl.items() if cc == cluster_id]
        if not members_c:
            return None
        clat = sum(coords[j][0] for j in members_c) / len(members_c)
        clon = sum(coords[j][1] for j in members_c) / len(members_c)
        return clat, clon

    # ---- Cân bằng tải trọng: chuyển điểm từ cụm quá tải sang cụm còn dư ----
    # Ưu tiên di chuyển vào cụm còn ĐỦ dư sức chứa (feasible) và gần nhất;
    # nếu không cụm nào đủ dư, chuyển "best-effort" vào cụm còn dư NHIỀU NHẤT
    # để giảm dần mức vượt tải, tránh bế tắc giữa chừng. `balanced` luôn được
    # tính lại từ tải trọng thực tế sau vòng lặp, không suy luận qua cờ nội bộ.
    iterations = 0
    for _ in range(max_balance_iterations):
        loads = _cluster_loads(labels)
        overloaded = [c for c, load in loads.items() if load > vehicle_capacity_kg + 1e-6]
        if not overloaded:
            break
        iterations += 1

        c_over = max(overloaded, key=lambda c: loads[c])
        members = sorted(
            [idx for idx, c in labels.items() if c == c_over],
            key=lambda idx: demands[idx],
        )

        moved = False
        for idx in members:
            plat, plon = coords[idx]
            feasible_candidates = []
            all_candidates = []
            for c in range(k):
                if c == c_over:
                    continue
                residual = vehicle_capacity_kg - loads[c]
                centroid = _centroid(c, labels) or (plat, plon)
                d = math.hypot(plat - centroid[0], plon - centroid[1])
                all_candidates.append((residual, d, c))
                if residual >= demands[idx] - 1e-6:
                    feasible_candidates.append((d, c))
            if feasible_candidates:
                feasible_candidates.sort(key=lambda x: x[0])
                labels[idx] = feasible_candidates[0][1]
                moved = True
                break
            elif all_candidates:
                # best-effort: chuyển điểm NHỎ NHẤT của cụm quá tải sang cụm
                # còn dư NHIỀU NHẤT (giảm dần mức vượt tải dù chưa hết hẳn)
                all_candidates.sort(key=lambda x: -x[0])
                if all_candidates[0][0] > 0:
                    labels[idx] = all_candidates[0][2]
                    moved = True
                    break

        if not moved:
            break  # không còn nước đi nào cải thiện được nữa

    final_loads = _cluster_loads(labels)
    balanced = all(load <= vehicle_capacity_kg + 1e-6 for load in final_loads.values())
    return ClusteringResult(k=k, labels=labels, cluster_loads=final_loads, balanced=balanced, iterations=iterations)


@dataclass
class ClusterSubproblem:
    cluster_id: int
    sub_indices: list       # index gốc, sub_indices[0] luôn = depot
    node_ids: list
    sub_dist_m: list
    sub_dur_s: list
    demands: list
    service_times_s: list


def build_cluster_subproblems(
    clustering: ClusteringResult,
    depot_index: int,
    dist_m_full,
    dur_s_full,
    demands_full: list,
    service_times_s_full: list,
    node_ids_full: list,
) -> list:
    problems = []
    for c in range(clustering.k):
        members = [idx for idx, cc in clustering.labels.items() if cc == c]
        if not members:
            continue
        sub_indices = [depot_index] + members
        sub_dist = [[dist_m_full[a][b] for b in sub_indices] for a in sub_indices]
        sub_dur = [[dur_s_full[a][b] for b in sub_indices] for a in sub_indices]
        demands = [demands_full[i] for i in sub_indices]
        service = [service_times_s_full[i] for i in sub_indices]
        node_ids = [node_ids_full[i] for i in sub_indices]
        problems.append(ClusterSubproblem(
            cluster_id=c, sub_indices=sub_indices, node_ids=node_ids,
            sub_dist_m=sub_dist, sub_dur_s=sub_dur, demands=demands, service_times_s=service,
        ))
    return problems


@dataclass
class ClusterRoutingResult:
    clustering: ClusteringResult
    cluster_problems: list
    routes: list             # list[RouteResult], đã gộp tất cả cụm, vehicle_id đánh lại liên tục
    solved: bool
    messages: list
    runtime_sec: float


def split_time_budget(total_budget_sec: float, k: int, min_per_cluster_sec: float = 2.0) -> float:
    """Chia công bằng ngân sách thời gian tính toán cho từng cụm, để so sánh
    runtime với phương pháp KHÔNG phân cụm (vốn dùng trọn `total_budget_sec`
    cho toàn bộ bài toán) là công bằng — mỗi cụm không được cấp lại "full
    budget" một cách vô lý, mà chia theo số cụm k."""
    if k <= 0:
        return total_budget_sec
    return max(min_per_cluster_sec, total_budget_sec / k)


def run_clustering_cvrp(
    coords: list,
    depot_index: int,
    demands_full: list,
    service_times_s_full: list,
    dist_m_full,
    dur_s_full,
    node_ids_full: list,
    vehicle_capacity_kg: float,
    use_gls: bool = True,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    time_limit_sec: int = 15,
    max_route_time_s: float | None = None,
    k: int | None = None,
    random_state: int = 42,
) -> ClusterRoutingResult:
    """Mô hình đề xuất: Giai đoạn 1 - phân cụm K-means++ (tự động k theo tải
    trọng); Giai đoạn 2 - giải CVRP cho TỪNG cụm bằng OR-Tools + GLS
    (PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH), rồi gộp kết quả.

    `time_limit_sec` là TỔNG ngân sách thời gian tính toán (giống ý nghĩa của
    tham số cùng tên ở phương pháp KHÔNG phân cụm), được CHIA ĐỀU cho từng
    cụm (qua split_time_budget) để so sánh runtime giữa 2 phương pháp là công
    bằng — không cấp "full budget" cho từng cụm một cách vô lý."""
    t0 = time.time()
    customer_indices = [i for i in range(len(coords)) if i != depot_index]

    clustering = capacitated_kmeans(
        coords, demands_full, customer_indices, vehicle_capacity_kg, k=k, random_state=random_state,
    )
    cluster_problems = build_cluster_subproblems(
        clustering, depot_index, dist_m_full, dur_s_full, demands_full, service_times_s_full, node_ids_full,
    )
    per_cluster_time_limit = split_time_budget(time_limit_sec, clustering.k)

    all_routes: list[RouteResult] = []
    messages = []
    solved_all = True
    next_vehicle_id = 1

    for problem in cluster_problems:
        cfg = OptimizeConfig(
            num_vehicles=2,  # thường 1 xe/cụm là đủ (đã cân bằng capacity); dự phòng 2
            vehicle_capacity_kg=vehicle_capacity_kg,
            depot_index=0,
            use_gls=use_gls,
            first_solution_strategy=first_solution_strategy,
            time_limit_sec=max(1, int(round(per_cluster_time_limit))),
            max_route_time_s=max_route_time_s,
        )
        routes, solved, msg, _n = solve_cvrp_with_auto_scaling(
            problem.sub_dist_m, problem.sub_dur_s, problem.demands, problem.service_times_s,
            cfg, max_extra_vehicles=2,
        )
        if not solved:
            solved_all = False
            messages.append(f"Cụm {problem.cluster_id}: {msg}")
            continue

        for r in routes:
            global_sequence = [problem.sub_indices[i] for i in r.node_sequence]
            new_route = compute_route_metrics(
                global_sequence, dist_m_full, dur_s_full, demands_full, service_times_s_full,
                next_vehicle_id, vehicle_capacity_kg,
            )
            all_routes.append(new_route)
            next_vehicle_id += 1

    runtime = time.time() - t0
    return ClusterRoutingResult(
        clustering=clustering, cluster_problems=cluster_problems, routes=all_routes,
        solved=solved_all, messages=messages, runtime_sec=runtime,
    )
