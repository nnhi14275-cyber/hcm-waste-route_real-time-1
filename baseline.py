"""
baseline.py
-----------
Tuyến cơ sở (baseline) = mô phỏng bằng heuristic đơn giản, KHÔNG dùng
OR-Tools/GLS, để so sánh công bằng với tuyến tối ưu.

Cung cấp 2 heuristic:
- Nearest Neighbor / Greedy (đơn giản, thường cho kết quả kém nhất -> dùng
  làm "baseline yếu" mặc định).
- Clarke-Wright Savings (heuristic kinh điển cho VRP, thường cho kết quả tốt
  hơn Nearest Neighbor đáng kể) -> dùng làm "baseline mạnh" để so sánh công
  bằng hơn với OR-Tools+GLS, tránh việc % cải thiện bị thổi phồng do baseline
  quá yếu.

Cả hai đều có xét:
- Khả năng chứa của xe (capacity_kg)
- Thời gian phục vụ (service_time)
- Thời gian di chuyển lấy từ OSRM (duration_matrix)
- Depot (xe xuất phát từ depot và quay về depot)

Baseline và tuyến tối ưu (optimizer.py) PHẢI dùng chung ma trận OSRM để đảm
bảo so sánh công bằng.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteResult:
    vehicle_id: int
    node_sequence: list          # index các node, bắt đầu và kết thúc = depot_index
    total_distance_m: float = 0.0
    travel_time_s: float = 0.0
    service_time_s: float = 0.0
    total_route_time_s: float = 0.0
    collected_waste_kg: float = 0.0
    vehicle_capacity_kg: float = 0.0

    @property
    def capacity_utilization_pct(self) -> float:
        if self.vehicle_capacity_kg <= 0:
            return 0.0
        return 100.0 * self.collected_waste_kg / self.vehicle_capacity_kg

    @property
    def num_stops(self) -> int:
        """Số điểm dừng thu gom trên tuyến (không tính depot)."""
        return max(0, len(self.node_sequence) - 2)


def compute_route_metrics(
    node_sequence: list,
    distance_matrix,
    duration_matrix,
    demands: list,
    service_times_s: list,
    vehicle_id: int,
    vehicle_capacity_kg: float,
) -> RouteResult:
    total_distance = 0.0
    travel_time = 0.0
    service_time = 0.0
    waste = 0.0
    for idx in range(len(node_sequence) - 1):
        a, b = node_sequence[idx], node_sequence[idx + 1]
        total_distance += distance_matrix[a][b]
        travel_time += duration_matrix[a][b]
    for node in node_sequence[1:-1]:
        service_time += service_times_s[node]
        waste += demands[node]

    return RouteResult(
        vehicle_id=vehicle_id,
        node_sequence=list(node_sequence),
        total_distance_m=total_distance,
        travel_time_s=travel_time,
        service_time_s=service_time,
        total_route_time_s=travel_time + service_time,
        collected_waste_kg=waste,
        vehicle_capacity_kg=vehicle_capacity_kg,
    )


def nearest_neighbor_baseline(
    distance_matrix,
    duration_matrix,
    demands: list,
    service_times_s: list,
    vehicle_capacity_kg: float,
    depot_index: int = 0,
    max_route_time_s: float | None = None,
    max_vehicles: int = 20,
) -> list:
    """Xây dựng baseline bằng Nearest Neighbor / Greedy có xét tải trọng, thời
    gian phục vụ và thời gian di chuyển (OSRM). Xe xuất phát và quay về depot.

    Trả về danh sách RouteResult (mỗi phần tử = 1 xe / 1 tuyến).
    """
    n = len(distance_matrix)
    unvisited = set(range(n)) - {depot_index}
    routes: list[RouteResult] = []
    vehicle_id = 0

    while unvisited and vehicle_id < max_vehicles:
        vehicle_id += 1
        current = depot_index
        load = 0.0
        elapsed_time = 0.0
        sequence = [depot_index]

        while True:
            # Tìm điểm chưa thăm gần nhất (theo thời gian di chuyển OSRM) mà
            # vẫn khả thi về tải trọng và thời gian tuyến tối đa.
            best_node = None
            best_time = None
            for node in unvisited:
                new_load = load + demands[node]
                if new_load > vehicle_capacity_kg:
                    continue
                travel = duration_matrix[current][node]
                arrival_service_time = elapsed_time + travel + service_times_s[node]
                return_time = duration_matrix[node][depot_index]
                projected_total = arrival_service_time + return_time
                if max_route_time_s is not None and projected_total > max_route_time_s:
                    continue
                if best_time is None or travel < best_time:
                    best_time = travel
                    best_node = node

            if best_node is None:
                break

            travel = duration_matrix[current][best_node]
            elapsed_time += travel + service_times_s[best_node]
            load += demands[best_node]
            sequence.append(best_node)
            unvisited.discard(best_node)
            current = best_node

        sequence.append(depot_index)

        if len(sequence) > 2:  # có ít nhất 1 điểm thu gom
            routes.append(
                compute_route_metrics(
                    sequence, distance_matrix, duration_matrix,
                    demands, service_times_s, vehicle_id, vehicle_capacity_kg,
                )
            )
        else:
            # Không thêm được điểm nào (VD do capacity quá nhỏ) -> tránh vòng lặp vô hạn
            break

    if unvisited:
        raise RuntimeError(
            f"Baseline (Nearest Neighbor) không thể phục vụ hết {len(unvisited)} điểm "
            f"trong giới hạn {max_vehicles} xe. Hãy tăng số xe, tải trọng xe, hoặc "
            f"max_route_time."
        )

    return routes


def clarke_wright_savings(
    distance_matrix,
    duration_matrix,
    demands: list,
    service_times_s: list,
    vehicle_capacity_kg: float,
    depot_index: int = 0,
    max_route_time_s: float | None = None,
    max_vehicles: int = 20,
) -> list:
    """Baseline "mạnh" bằng thuật toán Clarke-Wright Savings.

    Ý tưởng: bắt đầu với mỗi điểm là 1 tuyến riêng (Depot-i-Depot), sau đó lần
    lượt gộp 2 tuyến lại nếu "savings" = d(depot,i)+d(depot,j)-d(i,j) lớn, và
    vẫn thoả ràng buộc capacity + max_route_time. Kết quả thường tốt hơn đáng
    kể so với Nearest Neighbor, giúp so sánh với OR-Tools+GLS công bằng hơn
    (tránh % cải thiện bị thổi phồng do baseline quá yếu).
    """
    n = len(distance_matrix)
    customers = [i for i in range(n) if i != depot_index]

    # Mỗi khách hàng ban đầu là 1 tuyến riêng: [depot, i, depot]
    route_of = {c: [depot_index, c, depot_index] for c in customers}
    load_of = {c: demands[c] for c in customers}
    time_of = {
        c: duration_matrix[depot_index][c] + service_times_s[c] + duration_matrix[c][depot_index]
        for c in customers
    }

    # Tính savings cho mọi cặp (i, j)
    savings = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = (
                distance_matrix[depot_index][i]
                + distance_matrix[depot_index][j]
                - distance_matrix[i][j]
            )
            savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    # route_id để tránh gộp một tuyến vào chính nó qua các bí danh khác nhau
    route_id_of = {c: c for c in customers}
    routes_by_id = {c: route_of[c] for c in customers}

    def _route_ends(route):
        # điểm đầu/cuối (không tính depot) của 1 tuyến [depot, ..., depot]
        return route[1], route[-2]

    for s, i, j in savings:
        ri = route_id_of[i]
        rj = route_id_of[j]
        if ri == rj:
            continue
        route_i = routes_by_id[ri]
        route_j = routes_by_id[rj]

        start_i, end_i = _route_ends(route_i)
        start_j, end_j = _route_ends(route_j)

        # Chỉ gộp được khi i và j đang là đầu/cuối tuyến của chúng (đặc trưng
        # của Clarke-Wright: nối đuôi tuyến này với đầu tuyến kia)
        merged = None
        if end_i == i and start_j == j:
            merged = route_i[:-1] + route_j[1:]
        elif end_j == j and start_i == i:
            merged = route_j[:-1] + route_i[1:]
        else:
            continue

        new_load = load_of[ri] + load_of[rj]
        if new_load > vehicle_capacity_kg:
            continue

        new_time = 0.0
        for k in range(len(merged) - 1):
            a, b = merged[k], merged[k + 1]
            new_time += duration_matrix[a][b]
        new_time += sum(service_times_s[node] for node in merged[1:-1])
        if max_route_time_s is not None and new_time > max_route_time_s:
            continue

        new_id = min(ri, rj)
        routes_by_id[new_id] = merged
        load_of[new_id] = new_load
        time_of[new_id] = new_time
        other_id = rj if new_id == ri else ri
        del routes_by_id[other_id]
        del load_of[other_id]
        del time_of[other_id]
        for c in customers:
            if route_id_of[c] == ri or route_id_of[c] == rj:
                route_id_of[c] = new_id

    final_sequences = list(routes_by_id.values())
    if len(final_sequences) > max_vehicles:
        raise RuntimeError(
            f"Baseline (Clarke-Wright Savings) cần {len(final_sequences)} xe, vượt quá "
            f"giới hạn {max_vehicles} xe. Hãy tăng số xe hoặc tải trọng xe."
        )

    routes = []
    for vid, seq in enumerate(final_sequences, start=1):
        routes.append(
            compute_route_metrics(
                seq, distance_matrix, duration_matrix,
                demands, service_times_s, vid, vehicle_capacity_kg,
            )
        )
    return routes


BASELINE_METHODS = {
    "Nearest Neighbor / Greedy (baseline yếu)": nearest_neighbor_baseline,
    "Clarke-Wright Savings (baseline mạnh hơn)": clarke_wright_savings,
}
