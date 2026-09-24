"""
routing.py
----------
Toàn bộ chức năng liên quan tới OpenStreetMap (OSM) + OSRM.

Nguyên tắc bắt buộc:
- OSRM là nguồn khoảng cách/thời gian di chuyển CHÍNH.
- KHÔNG âm thầm chuyển sang Haversine nếu OSRM lỗi.
- Haversine chỉ được dùng khi người dùng CHỦ ĐỘNG bật allow_haversine_fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import requests
import streamlit as st

DEFAULT_OSRM_BASE_URL = "https://router.project-osrm.org"
DEFAULT_PROFILE = "driving"
OSRM_TIMEOUT_SEC = 15
OSRM_MAX_RETRIES = 3
OSRM_RETRY_BACKOFF_SEC = 1.5

# OSRM demo server công khai giới hạn số điểm trong 1 lần gọi Table Service
# (mặc định máy chủ demo ~100). Cảnh báo sớm cho người dùng thay vì để lỗi
# HTTP khó hiểu khi vượt ngưỡng.
OSRM_PUBLIC_SERVER_SOFT_LIMIT = 100


class OSRMError(Exception):
    """Lỗi khi gọi OSRM - không được âm thầm nuốt lỗi này."""


@dataclass
class MatrixResult:
    distance_matrix_m: list  # mét
    duration_matrix_s: list  # giây
    source: str  # "OSRM" hoặc "HAVERSINE_FALLBACK"
    warning: str | None = None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _haversine_fallback_matrices(coordinates: Sequence[tuple], avg_speed_kmh: float):
    n = len(coordinates)
    dist = [[0.0] * n for _ in range(n)]
    dur = [[0.0] * n for _ in range(n)]
    speed_ms = max(1.0, avg_speed_kmh) * 1000 / 3600
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            d = _haversine_m(lat1, lon1, lat2, lon2)
            dist[i][j] = d
            dur[i][j] = d / speed_ms
    return dist, dur


def _request_with_retry(url: str, params: dict) -> requests.Response:
    """Gọi OSRM có retry với exponential backoff để giảm rủi ro OSRM demo
    server công khai bị timeout/rate-limit tạm thời (điểm yếu đã ghi nhận:
    phụ thuộc vào server công khai không có SLA)."""
    last_exc = None
    for attempt in range(OSRM_MAX_RETRIES):
        try:
            return requests.get(url, params=params, timeout=OSRM_TIMEOUT_SEC)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < OSRM_MAX_RETRIES - 1:
                time.sleep(OSRM_RETRY_BACKOFF_SEC * (attempt + 1))
    raise last_exc


def check_point_count_limit(num_points: int, base_url: str) -> str | None:
    """Trả về cảnh báo (str) nếu số điểm có nguy cơ vượt giới hạn của OSRM
    demo server công khai, hoặc None nếu ổn / đang dùng server riêng."""
    if base_url.rstrip("/") == DEFAULT_OSRM_BASE_URL and num_points > OSRM_PUBLIC_SERVER_SOFT_LIMIT:
        return (
            f"Số điểm ({num_points}) gần/vượt giới hạn khuyến nghị "
            f"({OSRM_PUBLIC_SERVER_SOFT_LIMIT}) của OSRM demo server công khai. "
            "Có thể gặp lỗi hoặc bị từ chối request. Khuyến nghị tự dựng OSRM "
            "server riêng (xem README) nếu cần chạy với số điểm lớn hơn."
        )
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def get_osrm_matrices(
    coordinates: tuple,
    base_url: str = DEFAULT_OSRM_BASE_URL,
    profile: str = DEFAULT_PROFILE,
    allow_haversine_fallback: bool = False,
    fallback_avg_speed_kmh: float = 25.0,
    traffic_sensitivity_multiplier: float = 1.0,
) -> MatrixResult:
    """Lấy ma trận khoảng cách (m) và thời gian (s) đường bộ giữa mọi cặp điểm
    bằng OSRM Table Service.

    coordinates: tuple các (lat, lon) - dùng tuple để có thể cache_data hash được.

    traffic_sensitivity_multiplier: hệ số nhân THỦ CÔNG lên ma trận thời gian
    (KHÔNG áp lên khoảng cách) để làm SENSITIVITY ANALYSIS mô phỏng giờ cao
    điểm/thấp điểm (VD 1.3 = thời gian di chuyển tăng 30%). Đây KHÔNG phải dữ
    liệu traffic thời gian thực - mặc định = 1.0 (không áp dụng).
    """
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    url = f"{base_url}/table/v1/{profile}/{coord_str}"
    params = {"annotations": "distance,duration"}

    try:
        resp = _request_with_retry(url, params)
    except requests.exceptions.RequestException as exc:
        if allow_haversine_fallback:
            dist, dur = _haversine_fallback_matrices(coordinates, fallback_avg_speed_kmh)
            return MatrixResult(
                dist, dur, source="HAVERSINE_FALLBACK",
                warning="Fallback mode – không sử dụng mạng lưới đường thực tế "
                        f"(lý do: không kết nối được OSRM: {exc}).",
            )
        raise OSRMError(
            "Không thể lấy dữ liệu mạng lưới đường từ OSRM. Vui lòng thử lại."
        ) from exc

    if resp.status_code != 200:
        if allow_haversine_fallback:
            dist, dur = _haversine_fallback_matrices(coordinates, fallback_avg_speed_kmh)
            return MatrixResult(
                dist, dur, source="HAVERSINE_FALLBACK",
                warning=f"Fallback mode – không sử dụng mạng lưới đường thực tế "
                        f"(OSRM trả về HTTP {resp.status_code}).",
            )
        raise OSRMError(
            f"Không thể lấy dữ liệu mạng lưới đường từ OSRM (HTTP {resp.status_code}). "
            "Vui lòng thử lại."
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        if allow_haversine_fallback:
            dist, dur = _haversine_fallback_matrices(coordinates, fallback_avg_speed_kmh)
            return MatrixResult(
                dist, dur, source="HAVERSINE_FALLBACK",
                warning="Fallback mode – không sử dụng mạng lưới đường thực tế "
                        "(phản hồi OSRM không phải JSON hợp lệ).",
            )
        raise OSRMError(
            "Không thể lấy dữ liệu mạng lưới đường từ OSRM. Vui lòng thử lại."
        ) from exc

    if payload.get("code") != "Ok":
        msg = payload.get("message", "unknown error")
        if allow_haversine_fallback:
            dist, dur = _haversine_fallback_matrices(coordinates, fallback_avg_speed_kmh)
            return MatrixResult(
                dist, dur, source="HAVERSINE_FALLBACK",
                warning=f"Fallback mode – không sử dụng mạng lưới đường thực tế "
                        f"(OSRM báo lỗi: {msg}).",
            )
        raise OSRMError(
            f"Không thể lấy dữ liệu mạng lưới đường từ OSRM ({msg}). Vui lòng thử lại."
        )

    dist = payload.get("distances")
    dur = payload.get("durations")
    n = len(coordinates)

    def _has_none(m):
        return m is None or any(m[i][j] is None for i in range(n) for j in range(n))

    if _has_none(dist) or _has_none(dur):
        if allow_haversine_fallback:
            fdist, fdur = _haversine_fallback_matrices(coordinates, fallback_avg_speed_kmh)
            return MatrixResult(
                fdist, fdur, source="HAVERSINE_FALLBACK",
                warning="Fallback mode – không sử dụng mạng lưới đường thực tế "
                        "(ma trận OSRM chứa giá trị None, có thể do điểm nằm ngoài "
                        "mạng lưới đường)."
            )
        raise OSRMError(
            "Ma trận khoảng cách/thời gian từ OSRM chứa giá trị thiếu (None). "
            "Một số điểm có thể nằm ngoài mạng lưới đường hoặc không thể tới được. "
            "Vui lòng kiểm tra lại toạ độ hoặc thử lại."
        )

    warning = None
    if traffic_sensitivity_multiplier != 1.0:
        dur = [[v * traffic_sensitivity_multiplier for v in row] for row in dur]
        warning = (
            f"Đã áp hệ số sensitivity ×{traffic_sensitivity_multiplier:g} lên thời gian di "
            "chuyển (KHÔNG phải traffic thời gian thực) để phân tích độ nhạy."
        )

    return MatrixResult(dist, dur, source="OSRM", warning=warning)


@st.cache_data(show_spinner=False, ttl=3600)
def get_osrm_route_geometry(
    ordered_coords: tuple,
    base_url: str = DEFAULT_OSRM_BASE_URL,
    profile: str = DEFAULT_PROFILE,
) -> list:
    """Lấy geometry đường đi thực tế (road geometry) cho một chuỗi điểm theo
    thứ tự tuyến (Depot -> P.. -> P.. -> Depot) bằng OSRM Route Service.

    Trả về danh sách (lat, lon) để vẽ trên Folium. Nếu OSRM lỗi, trả về danh
    sách rỗng (map sẽ tự fallback vẽ đường thẳng nối các điểm - đã xử lý ở app.py)
    và không được coi là dữ liệu mạng lưới đường thực tế.
    """
    if len(ordered_coords) < 2:
        return []

    coord_str = ";".join(f"{lon},{lat}" for lat, lon in ordered_coords)
    url = f"{base_url}/route/v1/{profile}/{coord_str}"
    params = {"overview": "full", "geometries": "geojson"}

    try:
        resp = _request_with_retry(url, params)
        if resp.status_code != 200:
            return []
        payload = resp.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            return []
        coords = payload["routes"][0]["geometry"]["coordinates"]  # [lon, lat]
        return [(lat, lon) for lon, lat in coords]
    except requests.exceptions.RequestException:
        return []
    except (ValueError, KeyError, IndexError):
        return []
