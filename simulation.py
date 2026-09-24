"""
simulation.py
--------------
Mô phỏng vận hành ĐỘI XE thời gian thực (Fleet Real-time Simulation), dùng
bởi app.py ở mục "Mô phỏng vận hành đội xe thời gian thực".

Module này KHÔNG phụ thuộc Streamlit (chỉ Python thuần + dynamic_routing.py
để tái sử dụng hàm nội suy vị trí dọc road geometry), để có thể unit-test
độc lập giống dynamic_routing.py.

API bắt buộc (được app.py import trực tiếp):
    FleetSimulator, RouteLeg, build_route_legs, format_hhmm,
    STATE_IDLE, STATE_MOVING, STATE_COLLECTING, STATE_FULL,
    STATE_RETURNING_TO_DEPOT, STATE_BROKEN, STATE_BLOCKED, STATE_COMPLETED

Nguyên tắc:
- Mỗi xe (Truck) di chuyển dọc theo `legs` (danh sách RouteLeg, mỗi leg có
  road geometry lấy từ OSRM) bằng cách nội suy theo quãng đường đã đi
  (progress_m), giống hệt cơ chế mô phỏng GPS ở dynamic_routing.py.
- Khi xe đến 1 điểm thu gom: cộng dồn tải trọng (waste thực tế nếu có, nếu
  không dùng waste dự báo), đánh dấu điểm đã thu gom. Nếu tải trọng vượt
  ngưỡng full_load_threshold_pct -> đánh dấu FULL + needs_reopt=True để
  app.py điều hướng xe quay về DEPOT.
- Khi xe đến DEPOT: đổ tải (reset load = 0). Nếu còn điểm pending (VD do
  vừa đầy tải giữa tuyến) -> IDLE + needs_reopt=True để app.py tái tối ưu
  phần còn lại. Nếu hết điểm -> COMPLETED.
- Xe hỏng (mark_broken): các điểm CHƯA thu gom của xe đó được đẩy vào
  "pending pool" dùng chung, để app.py phân bổ lại cho xe khác đang rảnh.
- Đường bị chặn (block_road): xe đang ở leg đi qua cạnh bị chặn sẽ dừng lại
  (BLOCKED) cho tới khi được gỡ chặn hoặc được tái tối ưu sang tuyến khác.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynamic_routing import interpolate_along_route

# ---------------------------------------------------------------------------
# Trạng thái xe (string constants - app.py dùng để tra icon/màu và so sánh)
# ---------------------------------------------------------------------------
STATE_IDLE = "idle"
STATE_MOVING = "moving"
STATE_COLLECTING = "collecting"
STATE_FULL = "full"
STATE_RETURNING_TO_DEPOT = "returning_to_depot"
STATE_BROKEN = "broken"
STATE_BLOCKED = "blocked"
STATE_COMPLETED = "completed"


def format_hhmm(minutes: float) -> str:
    """Định dạng số phút (tính từ 00:00) thành chuỗi HH:MM, cuộn vòng 24h."""
    total_min = int(round(minutes)) % (24 * 60)
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"


@dataclass
class RouteLeg:
    """Một chặng đường (node -> node kế tiếp) kèm road geometry OSRM thật."""
    from_node: int
    to_node: int
    geometry: list  # list[(lat, lon)]


def build_route_legs(node_sequence: list, geometry_fetch) -> list:
    """Dựng danh sách RouteLeg cho 1 tuyến, dùng `geometry_fetch(a, b)` (hàm
    do app.py cung cấp, gọi OSRM Route Service) để lấy road geometry thật
    cho từng cặp node liên tiếp trong node_sequence."""
    legs = []
    for i in range(len(node_sequence) - 1):
        a, b = node_sequence[i], node_sequence[i + 1]
        geometry = geometry_fetch(a, b)
        legs.append(RouteLeg(from_node=a, to_node=b, geometry=geometry or []))
    return legs


@dataclass
class Event:
    ts_min: float
    text: str


@dataclass
class Truck:
    vehicle_id: int
    capacity_kg: float
    speed_kmh: float = 25.0
    status: str = STATE_IDLE
    position: tuple | None = None
    current_node: int | None = None
    node_sequence: list = field(default_factory=list)
    legs: list = field(default_factory=list)
    leg_idx: int = 0
    leg_progress_m: float = 0.0
    pending_nodes: list = field(default_factory=list)  # điểm CHƯA thu gom trên tuyến hiện tại
    current_load_kg: float = 0.0
    needs_reopt: bool = False
    total_distance_km: float = 0.0


class FleetSimulator:
    """Mô phỏng nhiều xe cùng lúc, dùng bởi app.py (mục Fleet Real-time
    Simulation). Gọi `tick(accel_seconds)` mỗi lần app.py refresh để tiến
    mô phỏng thêm `accel_seconds` giây (ảo, có thể tăng tốc để demo)."""

    def __init__(
        self,
        depot_index: int,
        depot_coords: tuple,
        demands_predicted: dict,
        service_times_s: dict,
        vehicle_capacity_kg: float,
        full_load_threshold_pct: float = 95.0,
        start_clock_min: float = 6 * 60.0,  # mặc định bắt đầu ca sáng 06:00
    ):
        self.depot_index = depot_index
        self.depot_coords = depot_coords
        self.demands_predicted = dict(demands_predicted)
        self.demands_actual: dict[int, float] = {}
        self.service_times_s = dict(service_times_s)
        self.vehicle_capacity_kg = vehicle_capacity_kg
        self.full_load_threshold_pct = full_load_threshold_pct

        self.trucks: dict[int, Truck] = {}
        self.blocked_edges: set[frozenset] = set()
        self._pending_pool: list[int] = []

        self.clock_min = start_clock_min
        self.elapsed_min = 0.0

        self.collected_nodes: set[int] = set()
        self.collected_kg_total = 0.0
        self.events: list[Event] = []
        self._collecting_wait_s: dict[int, float] = {}

    # ------------------------------------------------------------ tiện ích
    def log(self, text: str) -> None:
        self.events.append(Event(ts_min=self.clock_min, text=text))

    def _actual_demand(self, node: int) -> float:
        if node in self.demands_actual:
            return self.demands_actual[node]
        return float(self.demands_predicted.get(node, 0.0))

    def _edge_blocked(self, a: int, b: int) -> bool:
        return frozenset((a, b)) in self.blocked_edges

    # -------------------------------------------------------- quản lý xe
    def add_truck(self, vehicle_id: int, capacity_kg: float | None = None, speed_kmh: float = 25.0) -> Truck:
        truck = Truck(
            vehicle_id=vehicle_id,
            capacity_kg=float(capacity_kg) if capacity_kg else float(self.vehicle_capacity_kg),
            speed_kmh=float(speed_kmh),
            status=STATE_IDLE,
            position=self.depot_coords,
            current_node=self.depot_index,
        )
        self.trucks[vehicle_id] = truck
        return truck

    def assign_route(self, vehicle_id: int, node_sequence: list, legs: list) -> None:
        """Gán (hoặc thay) tuyến cho 1 xe. node_sequence gồm cả depot đầu/cuối."""
        truck = self.trucks.get(vehicle_id)
        if truck is None:
            truck = self.add_truck(vehicle_id)

        truck.node_sequence = list(node_sequence)
        truck.legs = list(legs)
        truck.leg_idx = 0
        truck.leg_progress_m = 0.0
        truck.pending_nodes = [n for n in node_sequence if n != self.depot_index]
        truck.current_load_kg = 0.0
        truck.needs_reopt = False
        if truck.position is None:
            truck.position = self.depot_coords
        truck.current_node = node_sequence[0] if node_sequence else self.depot_index
        truck.status = STATE_MOVING if truck.legs else STATE_COMPLETED
        self._collecting_wait_s.pop(vehicle_id, None)
        self.log(f"🚚 Vehicle {vehicle_id}: được gán tuyến mới ({len(truck.pending_nodes)} điểm).")

    def mark_broken(self, vehicle_id: int) -> None:
        truck = self.trucks.get(vehicle_id)
        if truck is None:
            return
        if truck.pending_nodes:
            self._pending_pool.extend(truck.pending_nodes)
            truck.pending_nodes = []
        truck.status = STATE_BROKEN
        truck.needs_reopt = False
        self.log(f"🔴 Vehicle {vehicle_id}: báo hỏng. Các điểm còn lại đã đưa vào pool chờ phân bổ cho xe khác.")

    def repair_truck(self, vehicle_id: int) -> None:
        truck = self.trucks.get(vehicle_id)
        if truck is None:
            return
        truck.status = STATE_IDLE
        truck.position = self.depot_coords
        truck.current_node = self.depot_index
        truck.current_load_kg = 0.0
        truck.legs = []
        truck.leg_idx = 0
        truck.leg_progress_m = 0.0
        truck.needs_reopt = False
        self._collecting_wait_s.pop(vehicle_id, None)
        self.log(f"🔧 Vehicle {vehicle_id}: sửa xong, đã đưa về DEPOT, sẵn sàng nhận tuyến mới.")

    def redirect_to_depot(self, vehicle_id: int, leg: RouteLeg) -> None:
        """Chuyển hướng xe (đang đầy tải giữa tuyến) trực tiếp về DEPOT bằng
        1 leg OSRM thật tính từ vị trí hiện tại của xe."""
        truck = self.trucks.get(vehicle_id)
        if truck is None:
            return
        truck.legs = [leg]
        truck.leg_idx = 0
        truck.leg_progress_m = 0.0
        truck.status = STATE_RETURNING_TO_DEPOT
        self._collecting_wait_s.pop(vehicle_id, None)
        self.log(f"↩️ Vehicle {vehicle_id}: đầy tải, đang quay về DEPOT để đổ tải.")

    # --------------------------------------------------------- đường/pool
    def block_road(self, a: int, b: int) -> None:
        self.blocked_edges.add(frozenset((a, b)))
        self.log(f"🚧 Đã chặn đoạn đường {a} ↔ {b}.")

    def unblock_road(self, a: int, b: int) -> None:
        self.blocked_edges.discard(frozenset((a, b)))
        self.log(f"✅ Đã gỡ chặn đoạn đường {a} ↔ {b}.")

    def set_actual_waste(self, node_index: int, actual_kg: float) -> None:
        self.demands_actual[node_index] = max(0.0, float(actual_kg))
        self.log(f"📈 Cập nhật rác thực tế tại node {node_index}: {actual_kg:.1f} kg.")

    def pending_pool(self) -> list:
        return list(self._pending_pool)

    def clear_pending_pool(self) -> None:
        self._pending_pool = []

    def idle_trucks(self) -> list:
        return [vid for vid, t in self.trucks.items() if t.status == STATE_IDLE and not t.pending_nodes]

    # --------------------------------------------------------------- tick
    def tick(self, accel_seconds: float) -> None:
        """Tiến mô phỏng thêm `accel_seconds` giây (ảo) cho TẤT CẢ xe."""
        accel_seconds = max(0.0, float(accel_seconds))
        self.clock_min += accel_seconds / 60.0
        self.elapsed_min += accel_seconds / 60.0
        for truck in self.trucks.values():
            self._tick_truck(truck, accel_seconds)

    def _tick_truck(self, truck: Truck, accel_seconds: float) -> None:
        if truck.status in (STATE_BROKEN, STATE_COMPLETED, STATE_IDLE):
            return

        if truck.leg_idx >= len(truck.legs):
            # Không còn leg nào để đi (VD vừa gán tuyến rỗng) -> để yên.
            if truck.current_node == self.depot_index and not truck.pending_nodes:
                truck.status = STATE_COMPLETED
            return

        leg = truck.legs[truck.leg_idx]

        # ---- Đường bị chặn: dừng xe lại, không tiêu hao thời gian di chuyển ----
        if self._edge_blocked(leg.from_node, leg.to_node):
            if truck.status != STATE_BLOCKED:
                truck.status = STATE_BLOCKED
                self.log(f"⚠️ Vehicle {truck.vehicle_id}: đường bị chặn ({leg.from_node}↔{leg.to_node}), tạm dừng.")
            return
        if truck.status == STATE_BLOCKED:
            truck.status = STATE_MOVING  # đường vừa được gỡ chặn -> tiếp tục

        # ---- Đang dừng phục vụ tại điểm (service time) ----
        if truck.status == STATE_COLLECTING:
            remaining = self._collecting_wait_s.get(truck.vehicle_id, 0.0) - accel_seconds
            if remaining > 0:
                self._collecting_wait_s[truck.vehicle_id] = remaining
                return
            self._collecting_wait_s.pop(truck.vehicle_id, None)
            truck.status = STATE_MOVING

        if not leg.geometry:
            # Không có road geometry (OSRM lỗi cho đoạn này) -> coi như đến
            # thẳng điểm cuối leg ngay lập tức để mô phỏng không bị kẹt.
            finished, lat, lon = True, None, None
        else:
            speed_ms = max(0.1, truck.speed_kmh) * 1000.0 / 3600.0
            move_m = speed_ms * accel_seconds
            truck.leg_progress_m += move_m
            truck.total_distance_km += move_m / 1000.0
            lat, lon, finished = interpolate_along_route(leg.geometry, truck.leg_progress_m)
            if lat is not None:
                truck.position = (lat, lon)

        if not finished:
            return

        # ---- Xe vừa đến điểm cuối của leg hiện tại ----
        truck.current_node = leg.to_node
        truck.leg_idx += 1
        truck.leg_progress_m = 0.0

        if leg.to_node == self.depot_index:
            truck.current_load_kg = 0.0
            if truck.pending_nodes:
                truck.status = STATE_IDLE
                truck.needs_reopt = True
                self.log(f"🏠 Vehicle {truck.vehicle_id}: đã về DEPOT, còn {len(truck.pending_nodes)} điểm pending -> chờ tái tối ưu.")
            elif truck.leg_idx < len(truck.legs):
                truck.status = STATE_MOVING
            else:
                truck.status = STATE_COMPLETED
                self.log(f"🎉 Vehicle {truck.vehicle_id}: hoàn thành toàn bộ tuyến, đã về DEPOT.")
        else:
            if leg.to_node in truck.pending_nodes:
                truck.pending_nodes.remove(leg.to_node)
            collected = self._actual_demand(leg.to_node)
            truck.current_load_kg += collected
            if leg.to_node not in self.collected_nodes:
                self.collected_nodes.add(leg.to_node)
                self.collected_kg_total += collected

            svc_s = float(self.service_times_s.get(leg.to_node, 0.0))
            if svc_s > 0:
                self._collecting_wait_s[truck.vehicle_id] = svc_s
                truck.status = STATE_COLLECTING
            else:
                truck.status = STATE_MOVING

            threshold_kg = truck.capacity_kg * (self.full_load_threshold_pct / 100.0)
            if truck.current_load_kg >= threshold_kg:
                truck.status = STATE_FULL
                truck.needs_reopt = True
                self._collecting_wait_s.pop(truck.vehicle_id, None)
                self.log(
                    f"🟠 Vehicle {truck.vehicle_id}: đầy tải tại node {leg.to_node} "
                    f"({truck.current_load_kg:.0f}/{truck.capacity_kg:.0f} kg) -> cần quay về DEPOT."
                )

    # ------------------------------------------------------------ dashboard
    def dashboard_kpis(self, cost_params: dict) -> dict:
        total_predicted = sum(
            v for k, v in self.demands_predicted.items() if k != self.depot_index
        )
        collected = self.collected_kg_total
        remaining = max(0.0, total_predicted - collected)

        active = sum(1 for t in self.trucks.values() if t.status != STATE_BROKEN)
        broken = sum(1 for t in self.trucks.values() if t.status == STATE_BROKEN)
        total_distance_km = sum(t.total_distance_km for t in self.trucks.values())

        total_customers = sum(1 for k in self.demands_predicted if k != self.depot_index)
        service_level_pct = (
            len(self.collected_nodes) / total_customers * 100.0 if total_customers else 0.0
        )

        active_trucks = [t for t in self.trucks.values() if t.status != STATE_BROKEN]
        avg_util = (
            sum((t.current_load_kg / t.capacity_kg * 100.0) if t.capacity_kg else 0.0 for t in active_trucks)
            / len(active_trucks)
        ) if active_trucks else 0.0

        total_time_h = self.elapsed_min / 60.0
        total_cost = (
            total_distance_km * float(cost_params.get("fuel_cost_per_km", 0))
            + total_time_h * float(cost_params.get("driver_cost_per_hour", 0))
            + total_distance_km * float(cost_params.get("maintenance_cost_per_km", 0))
            + float(cost_params.get("other_cost", 0))
        )

        return {
            "clock": format_hhmm(self.clock_min),
            "total_predicted_waste_kg": round(total_predicted, 1),
            "collected_waste_kg": round(collected, 1),
            "remaining_waste_kg": round(remaining, 1),
            "active_vehicles": active,
            "broken_vehicles": broken,
            "total_distance_km": round(total_distance_km, 2),
            "service_level_pct": round(service_level_pct, 1),
            "avg_utilization_pct": round(avg_util, 1),
            "total_cost": round(total_cost, 0),
            "total_time_h": round(total_time_h, 2),
        }

    def vehicle_table(self) -> list:
        rows = []
        for t in self.trucks.values():
            rows.append({
                "Xe": f"Vehicle {t.vehicle_id}",
                "Trạng thái": t.status,
                "Tải hiện tại (kg)": round(t.current_load_kg, 1),
                "Sức chứa (kg)": round(t.capacity_kg, 1),
                "Tải trọng (%)": round((t.current_load_kg / t.capacity_kg * 100.0) if t.capacity_kg else 0.0, 1),
                "Điểm còn lại": len(t.pending_nodes),
                "Quãng đường đã chạy (km)": round(t.total_distance_km, 2),
            })
        return rows

    def route_table(self, node_label_fn) -> list:
        rows = []
        for t in self.trucks.values():
            rows.append({
                "Xe": f"Vehicle {t.vehicle_id}",
                "Trạng thái": t.status,
                "Node hiện tại": node_label_fn(t.current_node),
                "Điểm còn lại": ", ".join(node_label_fn(n) for n in t.pending_nodes) if t.pending_nodes else "-",
            })
        return rows

    def recent_events(self, n: int = 30) -> list:
        """Trả về tối đa `n` sự kiện GẦN NHẤT, mới nhất trước (app.py không
        tự reverse khi hiển thị)."""
        return list(reversed(self.events[-n:]))
