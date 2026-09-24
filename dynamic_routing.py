"""
dynamic_routing.py
-------------------
Dynamic routing dựa trên GPS xe (điện thoại đặt trên xe), KHÔNG yêu cầu tài
xế bấm nút xác nhận.

Nguyên tắc bắt buộc:
1. AUTO COMPLETION: xe vào bán kính 30-50m quanh 1 điểm thu gom và duy trì
   >= dwell_seconds_required (mặc định 10s) -> điểm đó pending -> completed.
2. AUTO DEPOT DETECTION: xe vào bán kính 30-50m quanh DEPOT và duy trì đủ
   thời gian -> xác nhận "xe đã về DEPOT", KHÔNG cần bấm nút.
3. AUTO RE-OPTIMIZATION: ngay khi xác nhận xe về DEPOT, lấy toàn bộ điểm còn
   pending, đặt DEPOT làm điểm xuất phát, chạy lại OR-Tools + GLS cho các
   điểm này (việc gọi optimizer thực hiện ở app.py, module này chỉ phát tín
   hiệu "depot_confirmed").
4. KHÔNG dùng GPS để suy luận "xe đầy" - dwell tại DEPOT là tín hiệu DUY
   NHẤT để tái tối ưu. Không có cảm biến tải trọng trong phạm vi prototype.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class GPSTrackerConfig:
    completion_radius_m: float = 40.0  # trong khoảng 30-50m theo yêu cầu
    depot_radius_m: float = 40.0       # trong khoảng 30-50m theo yêu cầu
    dwell_seconds_required: float = 10.0


@dataclass
class PointStatus:
    node_id: str
    status: str = "pending"  # "pending" | "completed" | "deferred"
    dwell_start_ts: float | None = None
    completed_ts: float | None = None
    last_distance_m: float | None = None
    deferred_reason: str = ""


@dataclass
class DepotDwellState:
    dwell_start_ts: float | None = None
    inside_now: bool = False


class DynamicRoutingEngine:
    """Theo dõi GPS xe theo thời gian (mỗi lần đọc GPS mới gọi update_position
    một lần) và tự động cập nhật trạng thái điểm thu gom + phát hiện việc xe
    quay về DEPOT, KHÔNG cần thao tác thủ công của tài xế."""

    def __init__(
        self,
        points: list[dict],
        depot_latlon: tuple[float, float],
        config: GPSTrackerConfig | None = None,
    ):
        """points: list các dict {"node_id", "latitude", "longitude"} (không
        gồm depot)."""
        self.config = config or GPSTrackerConfig()
        self.depot_latlon = depot_latlon
        self.point_status: dict[str, PointStatus] = {
            p["node_id"]: PointStatus(node_id=p["node_id"]) for p in points
        }
        self.point_coords: dict[str, tuple[float, float]] = {
            p["node_id"]: (p["latitude"], p["longitude"]) for p in points
        }
        self.depot_dwell = DepotDwellState()
        self.position_history: list[tuple[float, float, float]] = []
        self.current_position: tuple[float, float] | None = None
        self.last_update_ts: float | None = None

    # ---------------------------------------------------------------- utils
    def pending_node_ids(self) -> list[str]:
        return [nid for nid, s in self.point_status.items() if s.status == "pending"]

    def deferred_node_ids(self) -> list[str]:
        """Các điểm bị hoãn do vật cản/rác cồng kềnh/sự cố tại chỗ (KHÔNG
        completed, KHÔNG còn pending trên tuyến hiện tại) - cần một chuyến
        phụ riêng để xử lý sau."""
        return [nid for nid, s in self.point_status.items() if s.status == "deferred"]

    def mark_deferred(self, node_id: str, reason: str = "") -> bool:
        """Đánh dấu 1 điểm đang pending là 'deferred' (vật cản, rác cồng kềnh,
        tai nạn không quay đầu được, ...). Điểm bị loại khỏi pending_node_ids()
        ngay lập tức để tuyến hiện tại được tính lại KHÔNG còn đi qua điểm này.
        Trả về True nếu đánh dấu thành công (điểm tồn tại và đang pending)."""
        status = self.point_status.get(node_id)
        if status is None or status.status != "pending":
            return False
        status.status = "deferred"
        status.dwell_start_ts = None
        status.deferred_reason = reason
        return True

    def requeue_deferred(self, node_id: str) -> bool:
        """Đưa 1 điểm đang 'deferred' trở lại 'pending' - dùng khi bắt đầu một
        chặng phụ (supplementary trip) quay lại xử lý các điểm đã bị hoãn.
        Trả về True nếu điểm tồn tại và đang ở trạng thái deferred."""
        status = self.point_status.get(node_id)
        if status is None or status.status != "deferred":
            return False
        status.status = "pending"
        status.dwell_start_ts = None
        return True

    def completed_node_ids(self) -> list[str]:
        return [nid for nid, s in self.point_status.items() if s.status == "completed"]

    def all_completed(self) -> bool:
        return len(self.pending_node_ids()) == 0

    def sync_pending_after_reoptimize(self, node_ids_in_new_route: list[str]) -> None:
        """Sau khi tái tối ưu, đảm bảo các điểm trong tuyến mới ở trạng thái
        'pending' (không đổi các điểm đã completed HOẶC đã bị deferred - một
        điểm deferred chỉ được đưa lại vào pending qua chuyến phụ riêng, không
        tự động lẫn vào tuyến chính)."""
        for node_id in node_ids_in_new_route:
            status = self.point_status.get(node_id)
            if status and status.status not in ("completed", "deferred"):
                status.status = "pending"
                status.dwell_start_ts = None

    # ------------------------------------------------------------- core loop
    def update_position(self, lat: float, lon: float, ts: float | None = None) -> dict:
        """Nạp 1 điểm GPS mới. Trả về sự kiện xảy ra tại lần cập nhật này:

            {
              "newly_completed": [node_id, ...],
              "depot_confirmed": bool,   # True CHỈ đúng 1 lần tại thời điểm
                                          # xác nhận xe về depot (không lặp
                                          # lại khi xe vẫn đứng yên ở depot)
            }
        """
        ts = ts if ts is not None else time.time()
        self.current_position = (lat, lon)
        self.last_update_ts = ts
        self.position_history.append((lat, lon, ts))

        newly_completed: list[str] = []

        # ---- 1. AUTO COMPLETION cho từng điểm còn pending ----
        for node_id, status in self.point_status.items():
            if status.status == "completed":
                continue
            plat, plon = self.point_coords[node_id]
            d = haversine_m(lat, lon, plat, plon)
            status.last_distance_m = d
            if d <= self.config.completion_radius_m:
                if status.dwell_start_ts is None:
                    status.dwell_start_ts = ts
                elif ts - status.dwell_start_ts >= self.config.dwell_seconds_required:
                    status.status = "completed"
                    status.completed_ts = ts
                    newly_completed.append(node_id)
            else:
                status.dwell_start_ts = None  # ra khỏi bán kính -> reset dwell timer

        # ---- 2. AUTO DEPOT DETECTION ----
        depot_confirmed = False
        dlat, dlon = self.depot_latlon
        d_depot = haversine_m(lat, lon, dlat, dlon)
        if d_depot <= self.config.depot_radius_m:
            if not self.depot_dwell.inside_now:
                # vừa mới vào vùng depot -> bắt đầu đếm dwell
                self.depot_dwell.dwell_start_ts = ts
                self.depot_dwell.inside_now = True
            elif (
                self.depot_dwell.dwell_start_ts is not None
                and ts - self.depot_dwell.dwell_start_ts >= self.config.dwell_seconds_required
            ):
                # Chỉ báo depot_confirmed đúng 1 lần cho mỗi lượt "ra rồi vào lại"
                depot_confirmed = True
                self.depot_dwell.dwell_start_ts = None  # tránh báo lặp lại liên tục
        else:
            self.depot_dwell.inside_now = False
            self.depot_dwell.dwell_start_ts = None

        return {"newly_completed": newly_completed, "depot_confirmed": depot_confirmed}


def interpolate_along_route(route_latlon_points: list[tuple[float, float]], progress_m: float):
    """Nội suy vị trí (lat, lon) dọc theo 1 chuỗi điểm road-geometry theo
    khoảng cách progress_m (mét) đã đi được từ điểm đầu tiên. Dùng để MÔ
    PHỎNG GPS xe di chuyển dọc tuyến khi chưa có thiết bị GPS thật (demo).

    Trả về (lat, lon, finished: bool) — finished=True khi đã đi hết tuyến.
    """
    if not route_latlon_points:
        return None, None, True
    if len(route_latlon_points) == 1:
        return route_latlon_points[0][0], route_latlon_points[0][1], True

    remaining = progress_m
    for i in range(len(route_latlon_points) - 1):
        lat1, lon1 = route_latlon_points[i]
        lat2, lon2 = route_latlon_points[i + 1]
        seg_len = haversine_m(lat1, lon1, lat2, lon2)
        if seg_len <= 1e-6:
            continue
        if remaining <= seg_len:
            frac = remaining / seg_len
            lat = lat1 + (lat2 - lat1) * frac
            lon = lon1 + (lon2 - lon1) * frac
            return lat, lon, False
        remaining -= seg_len

    last = route_latlon_points[-1]
    return last[0], last[1], True
