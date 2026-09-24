"""
data_generator.py
------------------
Sinh dữ liệu demo CHỈ trong phạm vi tuyến đường Lê Văn Việt và khu vực lân cận,
TP. Thủ Đức, TP.HCM (khu vực Quận 9 cũ).

KHÔNG sinh dữ liệu rải toàn TP.HCM hoặc toàn TP. Thủ Đức.

Cấu trúc dữ liệu mỗi điểm:
    node_id, latitude, longitude, waste_kg, service_time,
    time_window_start, time_window_end, is_depot
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Toạ độ tham chiếu dọc tuyến Lê Văn Việt (xấp xỉ, chỉ dùng để mô phỏng khu vực
# nghiên cứu - KHÔNG phải toạ độ khảo sát thực địa chính xác từng mét).
# Thứ tự đi từ phía Ngã tư Thủ Đức về phía Nguyễn Duy Trinh.
# ---------------------------------------------------------------------------
LE_VAN_VIET_WAYPOINTS = [
    (10.8483, 106.7828),
    (10.8460, 106.7845),
    (10.8430, 106.7860),
    (10.8400, 106.7870),
    (10.8370, 106.7885),
    (10.8340, 106.7900),
    (10.8310, 106.7912),
    (10.8285, 106.7920),
]

# Depot / điểm tập kết rác giả định, đặt gần khu vực giữa tuyến Lê Văn Việt.
DEPOT_LOCATION = (10.8400, 106.7870)

STUDY_AREA_LABEL = "Khu vực thử nghiệm: Tuyến Lê Văn Việt – TP. Thủ Đức, TP.HCM"

# Bán kính lệch tối đa (độ) để mô phỏng các điểm nằm trên hẻm/đường nhánh
# kết nối TRỰC TIẾP với trục Lê Văn Việt. ~0.002 độ ~ 200m, đủ gần để mọi
# điểm demo đều bám sát trục nghiên cứu chính, không lan ra khu vực khác.
_JITTER_LAT = 0.0020
_JITTER_LON = 0.0020


@dataclass
class DemoConfig:
    num_points: int = 25       # mặc định trong khoảng 20-30 điểm theo phạm vi case study
    num_vehicles: int = 3      # mặc định 2-3 xe thu gom cho case study quy mô nhỏ
    seed: int = 42
    min_waste_kg: float = 80.0
    max_waste_kg: float = 450.0
    min_service_time_min: float = 3.0
    max_service_time_min: float = 12.0
    use_time_windows: bool = False
    tw_start_min: int = 0          # phút tính từ giờ xuất phát
    tw_end_min: int = 240
    tw_window_len_min: int = 90
    # Tỉ lệ điểm nằm trong hẻm sâu (lệch xa khỏi trục Lê Văn Việt) chỉ được
    # phục vụ bởi xe nhỏ - mô phỏng ràng buộc "xe lớn không vào được hẻm nhỏ".
    small_alley_jitter_ratio: float = 0.6  # điểm có |jitter| > tỉ lệ này x max jitter -> cần xe nhỏ

    # ---- Phân loại rác tại nguồn (theo Luật BVMT 2020 / lộ trình TP.HCM) ----
    # Đa số điểm (hộ gia đình) chỉ phân 2 nhóm: tái chế + còn lại (rác thực
    # phẩm lẫn trong "còn lại"). Riêng nhóm "chủ nguồn thải phát sinh nhiều
    # rác thực phẩm" (chợ, nhà hàng, khách sạn, TTTM có dịch vụ ăn uống) theo
    # diện thí điểm phân 3 nhóm: tái chế + thực phẩm riêng + còn lại.
    major_food_generator_ratio: float = 0.16  # % điểm là chợ/nhà hàng/khách sạn
    recyclable_ratio_range: tuple = (0.12, 0.28)   # % khối lượng là rác tái chế
    food_ratio_range: tuple = (0.35, 0.55)         # % phần còn lại (sau tái chế) là rác thực phẩm, CHỈ áp dụng cho nhóm phát sinh nhiều thực phẩm


def generate_demo_data(config: DemoConfig | None = None) -> pd.DataFrame:
    """Sinh DataFrame gồm 1 depot + N điểm thu gom quanh Lê Văn Việt.

    Toạ độ các điểm thu gom được lấy quanh các waypoint dọc Lê Văn Việt (mô
    phỏng các hẻm / đường nhánh kết nối vào trục chính), KHÔNG bắt buộc nằm
    chính xác trên tim đường.
    """
    if config is None:
        config = DemoConfig()

    rng = random.Random(config.seed)
    # Case study chính: 20-30 điểm. Cho phép tối đa 150 điểm để phục vụ các
    # phân tích mở rộng (VD so sánh clustering) cần bộ dữ liệu lớn hơn.
    n = max(20, min(config.num_points, 150))

    rows = []
    # Depot luôn là node_id = "DEPOT"
    rows.append(
        {
            "node_id": "DEPOT",
            "latitude": DEPOT_LOCATION[0],
            "longitude": DEPOT_LOCATION[1],
            "waste_kg": 0.0,
            "service_time": 0.0,
            "time_window_start": 0,
            "time_window_end": 24 * 60,
            "is_depot": True,
            "requires_small_vehicle": False,
            "is_major_food_generator": False,
            "waste_recyclable_kg": 0.0,
            "waste_food_kg": 0.0,
            "waste_other_kg": 0.0,
        }
    )

    for i in range(1, n + 1):
        base_lat, base_lon = rng.choice(LE_VAN_VIET_WAYPOINTS)
        jitter_lat = rng.uniform(-_JITTER_LAT, _JITTER_LAT)
        jitter_lon = rng.uniform(-_JITTER_LON, _JITTER_LON)
        lat = base_lat + jitter_lat
        lon = base_lon + jitter_lon
        waste = round(rng.uniform(config.min_waste_kg, config.max_waste_kg), 1)
        service_time = round(
            rng.uniform(config.min_service_time_min, config.max_service_time_min), 1
        )

        # Điểm lệch xa khỏi trục chính (hẻm sâu) -> giả định xe lớn khó vào,
        # chỉ xe nhỏ mới phục vụ được (mô phỏng ràng buộc vận hành thực tế).
        jitter_ratio = math.hypot(jitter_lat / _JITTER_LAT, jitter_lon / _JITTER_LON) / math.sqrt(2)
        requires_small_vehicle = jitter_ratio > config.small_alley_jitter_ratio

        # ---- Phân luồng rác theo lộ trình thực tế ----
        is_major_food_generator = rng.random() < config.major_food_generator_ratio
        recyclable_ratio = rng.uniform(*config.recyclable_ratio_range)
        waste_recyclable_kg = round(waste * recyclable_ratio, 1)
        remaining = waste - waste_recyclable_kg
        if is_major_food_generator:
            food_ratio = rng.uniform(*config.food_ratio_range)
            waste_food_kg = round(remaining * food_ratio, 1)
        else:
            waste_food_kg = 0.0  # nhóm hộ gia đình: chưa tách riêng rác thực phẩm (2 nhóm)
        waste_other_kg = round(waste - waste_recyclable_kg - waste_food_kg, 1)

        if config.use_time_windows:
            latest_start = max(config.tw_start_min, config.tw_end_min - config.tw_window_len_min)
            tw_start = rng.randint(config.tw_start_min, latest_start)
            tw_end = min(config.tw_end_min, tw_start + config.tw_window_len_min)
        else:
            tw_start, tw_end = 0, 24 * 60

        rows.append(
            {
                "node_id": f"P{i:02d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "waste_kg": waste,
                "service_time": service_time,
                "time_window_start": tw_start,
                "time_window_end": tw_end,
                "is_depot": False,
                "requires_small_vehicle": requires_small_vehicle,
                "is_major_food_generator": is_major_food_generator,
                "waste_recyclable_kg": waste_recyclable_kg,
                "waste_food_kg": waste_food_kg,
                "waste_other_kg": waste_other_kg,
            }
        )

    return pd.DataFrame(rows)


def load_points_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá dữ liệu người dùng upload (CSV/XLSX) về đúng schema.

    Yêu cầu tối thiểu các cột: node_id, latitude, longitude, waste_kg.
    Các cột service_time, time_window_start/end, is_depot là tuỳ chọn.
    Nếu không có node nào is_depot=True, node đầu tiên sẽ được coi là depot.
    """
    required = {"node_id", "latitude", "longitude", "waste_kg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong dữ liệu upload: {sorted(missing)}")

    out = df.copy()
    if "service_time" not in out.columns:
        out["service_time"] = 5.0
    if "time_window_start" not in out.columns:
        out["time_window_start"] = 0
    if "time_window_end" not in out.columns:
        out["time_window_end"] = 24 * 60
    if "is_depot" not in out.columns:
        out["is_depot"] = False
        out.loc[out.index[0], "is_depot"] = True

    if not out["is_depot"].any():
        out.loc[out.index[0], "is_depot"] = True

    # ---- Phân luồng rác (tái chế / thực phẩm / còn lại) ----
    # Nếu người dùng không cung cấp sẵn, suy ra mặc định hợp lý: 20% tái chế,
    # phần còn lại gộp vào "còn lại" (coi như chưa tách riêng rác thực phẩm -
    # đúng với đa số điểm theo lộ trình 2 nhóm hiện hành của TP.HCM).
    if "is_major_food_generator" not in out.columns:
        out["is_major_food_generator"] = False
    if "waste_recyclable_kg" not in out.columns:
        out["waste_recyclable_kg"] = (out["waste_kg"] * 0.2).round(1)
    if "waste_food_kg" not in out.columns:
        out["waste_food_kg"] = 0.0
        out.loc[out["is_major_food_generator"], "waste_food_kg"] = (
            (out.loc[out["is_major_food_generator"], "waste_kg"]
             - out.loc[out["is_major_food_generator"], "waste_recyclable_kg"]) * 0.45
        ).round(1)
    if "waste_other_kg" not in out.columns:
        out["waste_other_kg"] = (
            out["waste_kg"] - out["waste_recyclable_kg"] - out["waste_food_kg"]
        ).round(1)

    # Đưa depot lên hàng đầu để các module khác luôn giả định index 0 = depot
    depot_rows = out[out["is_depot"]]
    other_rows = out[~out["is_depot"]]
    out = pd.concat([depot_rows, other_rows], ignore_index=True)
    return out
