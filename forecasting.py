"""
forecasting.py
--------------
Dự báo khối lượng rác thải sinh hoạt phát sinh theo từng điểm thu gom
(node_id), phục vụ bài toán tối ưu định tuyến VRP (OR-Tools + GLS).

Đầu vào: lịch sử khối lượng rác 3 loại (hữu cơ / tái chế / còn lại) theo
ngày, theo từng điểm thu gom.

Đầu ra chính (dùng trực tiếp cho optimizer.py):
    get_demand_forecast(df_points, ...) -> dict
        {
          "P01": {"organic": 172.0, "recyclable": 40.0, "other": 55.0,
                   "total_kg": 267.0},
          "P02": {...},
          ...
        }
    estimate_vehicles_needed(demand_dict, vehicle_capacity_kg) -> (tổng_kg, so_xe)

Nguyên tắc:
- Xử lý dự báo như bài toán time series theo TỪNG node_id.
- Chia train/test THEO THỜI GIAN (không random split).
- Cung cấp song song 2 phương án: Facebook Prophet và XGBoost, đánh giá
  bằng MAE trên tập test để so sánh (giống tinh thần đối chứng baseline
  của phần VRP: không chỉ chạy 1 model rồi kết luận).
- KHÔNG dùng biến nhiệt độ / thời tiết theo yêu cầu đề tài.
"""

from __future__ import annotations

import math
import random
from io import BytesIO
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WASTE_TYPES = ["organic", "recyclable", "other"]

# Ngày Tết mẫu (có thể chỉnh theo năm thực tế) - dùng cho cả sinh dữ liệu demo
# lẫn khai báo "holiday" cho Prophet.
DEFAULT_TET_DATES = pd.date_range("2025-01-27", "2025-02-02")


# =============================================================================
# BƯỚC 1A: CHUẨN HOÁ FILE LỊCH SỬ THỰC TẾ / UPLOAD
# =============================================================================
def prepare_history_from_dataframe(
    raw_df: pd.DataFrame,
    df_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Chuẩn hoá dữ liệu lịch sử upload cho pipeline dự báo.

    Hỗ trợ 2 dạng:
    1) date, node_id, waste_kg (+ num_households tuỳ chọn)
    2) date, node_id, waste_organic_kg, waste_recyclable_kg, waste_other_kg

    Nếu file chỉ có waste_kg, 3 luồng rác được suy ra theo tỷ trọng mặc định
    để vẫn có thể chạy mô hình đa luồng. Nếu file đã có 3 cột thành phần,
    hệ thống dùng trực tiếp các giá trị đó.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("File lịch sử đang rỗng.")

    df = raw_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "ngày": "date", "day": "date", "datetime": "date",
        "điểm": "node_id", "node": "node_id", "id": "node_id",
        "khối lượng": "waste_kg", "waste": "waste_kg", "total_kg": "waste_kg",
        "số hộ": "num_households", "households": "num_households",
        "waste_food_kg": "waste_organic_kg",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})

    required = {"date", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(sorted(missing))}. Cần tối thiểu date và node_id.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["node_id"] = df["node_id"].astype(str).str.strip()
    df = df.dropna(subset=["date"])
    df = df[df["node_id"].ne("")].copy()

    component_cols = ["waste_organic_kg", "waste_recyclable_kg", "waste_other_kg"]
    if not set(component_cols).issubset(df.columns):
        if "waste_kg" not in df.columns:
            raise ValueError(
                "File cần có waste_kg hoặc đủ 3 cột waste_organic_kg, "
                "waste_recyclable_kg, waste_other_kg."
            )
        total = pd.to_numeric(df["waste_kg"], errors="coerce").fillna(0).clip(lower=0)
        df["waste_organic_kg"] = total * 0.55
        df["waste_recyclable_kg"] = total * 0.20
        df["waste_other_kg"] = total * 0.25

    for col in component_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    if "num_households" not in df.columns:
        if df_points is not None and "num_households" in df_points.columns:
            hh = df_points[["node_id", "num_households"]].copy()
            hh["node_id"] = hh["node_id"].astype(str)
            df = df.merge(hh.drop_duplicates("node_id"), on="node_id", how="left")
        else:
            df["num_households"] = 1
    df["num_households"] = pd.to_numeric(df["num_households"], errors="coerce").fillna(1).clip(lower=1)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["date"].isin(DEFAULT_TET_DATES).astype(int)
    df = df.sort_values(["node_id", "date"]).drop_duplicates(["node_id", "date"], keep="last")
    df = df[["date", "node_id", "num_households", "day_of_week", "is_weekend", "is_holiday"] + component_cols]
    if df["date"].nunique() < 14:
        raise ValueError("Cần ít nhất 14 ngày lịch sử để tạo lag_7 và rolling mean_7.")
    return df.reset_index(drop=True)


def train_and_forecast_uploaded_history(
    history_df: pd.DataFrame,
    method: str = "xgboost",
    forecast_days: int = 7,
) -> dict:
    """Huấn luyện trên chính history_df đã upload và trả về forecast dạng long."""
    history = history_df.copy().sort_values(["node_id", "date"])
    cutoff = (history["date"].max() - pd.Timedelta(days=min(14, max(7, history["date"].nunique() // 5)))).strftime("%Y-%m-%d")
    method = method.lower()
    validation_rows = []
    forecast_parts = []

    for waste_type in WASTE_TYPES:
        target_col = f"waste_{waste_type}_kg"
        if method == "xgboost":
            result = train_xgboost_forecast(history, target_col, cutoff)
            future = predict_next_days_xgboost(history, target_col, result, num_days=forecast_days)
            validation_rows.append({
                "Loại rác": waste_type,
                "MAE (kg)": round(result.mae, 2),
                "Feature quan trọng nhất": result.feature_importance.index[0],
                "Mức độ ảnh hưởng (%)": round(result.feature_importance.iloc[0] * 100, 1),
            })
        elif method == "prophet":
            # Prophet: đánh giá theo hold-out cuối kỳ rồi fit toàn bộ lịch sử cho dự báo.
            rows = []
            for node_id in history["node_id"].unique():
                node_hist = history[history["node_id"] == node_id].copy()
                cutoff_node = node_hist["date"].max() - pd.Timedelta(days=min(14, max(7, len(node_hist)//5)))
                train_node = node_hist[node_hist["date"] < cutoff_node]
                test_node = node_hist[node_hist["date"] >= cutoff_node]
                if len(train_node) < 14:
                    continue
                fc_test = forecast_prophet(train_node, node_id, target_col, periods=len(test_node))
                rows.extend((test_node[target_col].to_numpy(), fc_test["yhat"].to_numpy()))
            if rows:
                actual = np.concatenate([x[0] for x in rows])
                pred = np.concatenate([x[1] for x in rows])
                mae = float(np.mean(np.abs(actual - pred)))
            else:
                mae = float("nan")
            validation_rows.append({"Loại rác": waste_type, "MAE (kg)": round(mae, 2) if np.isfinite(mae) else "-", "Feature quan trọng nhất": "Seasonality/Trend", "Mức độ ảnh hưởng (%)": "-"})
            for node_id in history["node_id"].unique():
                fc = forecast_prophet(history, node_id, target_col, periods=forecast_days)
                forecast_parts.append(fc.assign(node_id=node_id, waste_type=waste_type)[["ds", "node_id", "yhat", "waste_type"]].rename(columns={"ds":"date", "yhat":"predicted_kg"}))
            continue
        else:
            raise ValueError("method phải là xgboost hoặc prophet")

        future["waste_type"] = waste_type
        forecast_parts.append(future.rename(columns={"predicted_kg": "predicted_kg"})[["date", "node_id", "predicted_kg", "waste_type"]])

    long_fc = pd.concat(forecast_parts, ignore_index=True)
    pivot = long_fc.pivot_table(index=["date", "node_id"], columns="waste_type", values="predicted_kg", aggfunc="sum").reset_index()
    for col in WASTE_TYPES:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["total_kg"] = pivot[WASTE_TYPES].sum(axis=1).round(1)
    pivot.columns.name = None
    validation_df = pd.DataFrame(validation_rows)
    return {"forecast_df": pivot.sort_values(["date", "node_id"]).reset_index(drop=True), "validation_df": validation_df, "cutoff_date": cutoff, "method": method}


def build_forecast_excel(history_df: pd.DataFrame, forecast_df: pd.DataFrame, validation_df: pd.DataFrame | None = None) -> bytes:
    """Xuất workbook để người dùng lưu/đưa vào báo cáo."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        history_df.to_excel(writer, index=False, sheet_name="Lich_su_180_ngay")
        forecast_df.to_excel(writer, index=False, sheet_name="Du_bao_theo_diem")
        daily = forecast_df.groupby("date", as_index=False).agg(
            Tong_demand_kg=("total_kg", "sum"),
            So_diem=("node_id", "nunique"),
        )
        daily.to_excel(writer, index=False, sheet_name="Du_bao_theo_ngay")
        if validation_df is not None:
            validation_df.to_excel(writer, index=False, sheet_name="Kiem_dinh_mo_hinh")
    return output.getvalue()


def build_template_history_excel() -> bytes:
    """Template Excel tối thiểu cho dữ liệu lịch sử."""
    sample = pd.DataFrame(columns=[
        "date", "node_id", "num_households", "waste_kg",
        "waste_organic_kg", "waste_recyclable_kg", "waste_other_kg"
    ])
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample.to_excel(writer, index=False, sheet_name="Lich_su_180_ngay")
    return output.getvalue()


# =============================================================================
# BƯỚC 1: SINH DỮ LIỆU LỊCH SỬ MÔ PHỎNG (dùng khi chưa có dữ liệu cân rác thật)
# =============================================================================
def generate_waste_history(
    df_points: pd.DataFrame,
    days: int = 180,
    end_date: str | None = None,
    seed: int = 42,
    tet_dates: pd.DatetimeIndex = DEFAULT_TET_DATES,
) -> pd.DataFrame:
    """Sinh dữ liệu lịch sử khối lượng rác theo ngày cho từng điểm thu gom
    (không tính depot), dựa trên `waste_kg` hiện có trong df_points làm mức
    nền (baseline trung bình mỗi điểm), rồi thêm biến động theo:
    - thứ trong tuần / cuối tuần
    - ngày Tết (rác hữu cơ tăng mạnh, rác tái chế giảm nhẹ)
    - nhiễu ngẫu nhiên

    Trả về DataFrame dạng long: date, node_id, num_households, day_of_week,
    is_weekend, is_holiday, waste_organic_kg, waste_recyclable_kg,
    waste_other_kg.
    """
    rng = random.Random(seed)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")

    points = df_points[~df_points["is_depot"]].reset_index(drop=True)
    rows = []
    for _, p in points.iterrows():
        node_id = p["node_id"]
        # waste_kg hiện tại của điểm được coi là mức phát sinh "1 lần thu gom"
        # trung bình -> quy đổi thành base theo ngày (giả định 1 hộ ~ 1.2kg/ngày
        # rác hữu cơ, phần còn lại chia theo tỉ lệ tham khảo 55%/20%/25%).
        base_total = max(30.0, float(p["waste_kg"]) / 3.0)  # ước lượng phát sinh/ngày
        base_organic = base_total * 0.55
        base_recyclable = base_total * 0.20
        base_other = base_total * 0.25
        num_households = max(20, int(base_total / 1.2))

        for date in dates:
            dow = date.dayofweek
            is_weekend = 1 if dow >= 5 else 0
            is_holiday = 1 if date in tet_dates else 0

            organic = (
                base_organic
                + is_holiday * base_organic * 1.6
                + is_weekend * base_organic * 0.25
                + rng.gauss(0, base_organic * 0.06)
            )
            recyclable = (
                base_recyclable
                - is_holiday * base_recyclable * 0.15
                + rng.gauss(0, base_recyclable * 0.08)
            )
            other = base_other + rng.gauss(0, base_other * 0.08)

            rows.append(
                {
                    "date": date,
                    "node_id": node_id,
                    "num_households": num_households,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "waste_organic_kg": max(0.0, round(organic, 1)),
                    "waste_recyclable_kg": max(0.0, round(recyclable, 1)),
                    "waste_other_kg": max(0.0, round(other, 1)),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# BƯỚC 2: FEATURE ENGINEERING (lag, rolling mean) - dùng cho XGBoost
# =============================================================================
def build_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Thêm các đặc trưng chuỗi thời gian cho MỖI node_id riêng biệt:
    - lag_1: giá trị 1 ngày trước
    - lag_7: giá trị 7 ngày trước (bắt chu kỳ theo tuần)
    - rolling_mean_7: trung bình 7 ngày gần nhất (bắt xu hướng ngắn hạn)
    """
    df = df.sort_values(["node_id", "date"]).copy()
    df["lag_1"] = df.groupby("node_id")[target_col].shift(1)
    df["lag_7"] = df.groupby("node_id")[target_col].shift(7)
    df["rolling_mean_7"] = df.groupby("node_id")[target_col].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"])


def split_train_test_by_time(df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia train/test THEO THỜI GIAN (không random split) - bắt buộc cho bài
    toán time series để tránh rò rỉ thông tin tương lai vào tập train."""
    cutoff = pd.Timestamp(cutoff_date)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    return train, test


# =============================================================================
# BƯỚC 3A: MÔ HÌNH PROPHET (bắt seasonality tuần/năm + ngày Tết)
# =============================================================================
def forecast_prophet(
    history_df: pd.DataFrame,
    node_id: str,
    target_col: str,
    periods: int = 14,
    tet_dates: pd.DatetimeIndex = DEFAULT_TET_DATES,
) -> pd.DataFrame:
    """Dự báo cho 1 điểm (node_id) bằng Facebook Prophet.
    Trả về DataFrame: ds, yhat, yhat_lower, yhat_upper cho `periods` ngày tới.
    """
    from prophet import Prophet  # import trong hàm để tránh lỗi nếu chưa cài đặt

    node_df = history_df[history_df["node_id"] == node_id][["date", target_col]].rename(
        columns={"date": "ds", target_col: "y"}
    )
    holidays = pd.DataFrame({
        "holiday": "tet",
        "ds": tet_dates,
        "lower_window": 0,
        "upper_window": 0,
    })
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays=holidays)
    model.fit(node_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)


# =============================================================================
# BƯỚC 3B: MÔ HÌNH XGBOOST (đa điểm, dùng feature engineering đầy đủ)
# =============================================================================
@dataclass
class XGBForecastResult:
    model: object
    mae: float
    feature_importance: pd.Series
    test_predictions: pd.DataFrame  # date, node_id, actual, predicted
    feature_columns: list = field(default_factory=list)


def train_xgboost_forecast(
    history_df: pd.DataFrame,
    target_col: str,
    cutoff_date: str,
) -> XGBForecastResult:
    """Huấn luyện 1 mô hình XGBoost DÙNG CHUNG cho mọi node_id (phân biệt
    bằng one-hot encoding node_id) - giúp mô hình học được cả đặc trưng
    riêng của từng điểm lẫn quy luật chung (thứ trong tuần, ngày lễ...).
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error

    feat_df = build_features(history_df, target_col)
    feat_df = pd.get_dummies(feat_df, columns=["node_id"], prefix="node")
    node_cols = [c for c in feat_df.columns if c.startswith("node_")]

    feature_cols = [
        "num_households", "day_of_week", "is_weekend", "is_holiday",
        "month", "day", "lag_1", "lag_7", "rolling_mean_7",
    ] + node_cols

    train, test = split_train_test_by_time(feat_df, cutoff_date)
    if len(test) == 0:
        raise ValueError(
            f"Không có dữ liệu test sau cutoff_date={cutoff_date}. "
            "Hãy chọn cutoff sớm hơn hoặc sinh thêm dữ liệu lịch sử."
        )

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    test_pred_df = test[["date"]].copy()
    # khôi phục lại node_id gốc từ one-hot để hiển thị bảng kết quả
    test_pred_df["node_id"] = test[node_cols].idxmax(axis=1).str.replace("node_", "", regex=False)
    test_pred_df["actual"] = y_test.values
    test_pred_df["predicted"] = np.round(pred, 1)

    return XGBForecastResult(
        model=model, mae=mae, feature_importance=importance,
        test_predictions=test_pred_df, feature_columns=feature_cols,
    )


def predict_next_days_xgboost(
    history_df: pd.DataFrame,
    target_col: str,
    result: XGBForecastResult,
    num_days: int = 7,
) -> pd.DataFrame:
    """Dự báo lặp (recursive forecasting) cho `num_days` ngày tiếp theo, cho
    TẤT CẢ node_id, dùng model XGBoost đã huấn luyện ở train_xgboost_forecast.
    Mỗi bước dự báo xong sẽ được nối vào lịch sử để tính lag/rolling cho bước
    kế tiếp (vì ngày mai chưa có dữ liệu thật để tính lag_1)."""
    df = history_df.copy()
    node_cols_template = [c for c in result.feature_columns if c.startswith("node_")]
    all_nodes = [c.replace("node_", "") for c in node_cols_template]

    last_date = df["date"].max()
    predictions = []

    for step in range(1, num_days + 1):
        target_date = last_date + pd.Timedelta(days=step)
        dow = target_date.dayofweek
        is_weekend = 1 if dow >= 5 else 0
        is_holiday = 1 if target_date in DEFAULT_TET_DATES else 0

        feat_rows = []
        for node_id in all_nodes:
            node_hist = df[df["node_id"] == node_id].sort_values("date")
            last_row = node_hist.iloc[-1]
            lag_1 = node_hist[target_col].iloc[-1]
            lag_7 = node_hist[target_col].iloc[-7] if len(node_hist) >= 7 else lag_1
            rolling_mean_7 = node_hist[target_col].tail(7).mean()

            row = {
                "date": target_date,
                "node_id": node_id,
                "num_households": last_row["num_households"],
                "day_of_week": dow,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "month": target_date.month,
                "day": target_date.day,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_mean_7": rolling_mean_7,
            }
            feat_rows.append(row)

        step_df = pd.DataFrame(feat_rows)
        step_encoded = pd.get_dummies(step_df, columns=["node_id"], prefix="node")
        for col in node_cols_template:
            if col not in step_encoded.columns:
                step_encoded[col] = False
        X_pred = step_encoded[result.feature_columns]

        preds = result.model.predict(X_pred)
        step_df["predicted_kg"] = np.round(np.maximum(0.0, preds), 1)
        predictions.append(step_df[["date", "node_id", "predicted_kg"]])

        # Nối kết quả dự báo vào lịch sử để bước tiếp theo tính lag đúng
        append_df = step_df[["date", "node_id", "num_households", "day_of_week",
                              "is_weekend", "is_holiday"]].copy()
        append_df[target_col] = step_df["predicted_kg"]
        df = pd.concat([df, append_df], ignore_index=True)

    return pd.concat(predictions, ignore_index=True)


# =============================================================================
# BƯỚC 4: API TỔNG HỢP - dùng trực tiếp cho VRP (optimizer.py)
# =============================================================================
def get_demand_forecast(
    df_points: pd.DataFrame,
    history_days: int = 180,
    forecast_day_offset: int = 1,
    method: str = "xgboost",
    seed: int = 42,
) -> dict:
    """Hàm API chính: trả về demand DỰ BÁO cho `forecast_day_offset` ngày tới
    (mặc định = ngày mai), dưới dạng dict để truyền trực tiếp vào
    demand_callback của OR-Tools (thay cho df_points["waste_kg"] cố định).

    Trả về:
        {
          "P01": {"organic": 172.0, "recyclable": 40.0, "other": 55.0, "total_kg": 267.0},
          ...
        }
    """
    history = generate_waste_history(df_points, days=history_days, seed=seed)
    cutoff_date = (history["date"].max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    result_by_type: dict[str, dict] = {}
    for waste_type in WASTE_TYPES:
        target_col = f"waste_{waste_type}_kg"
        if method == "xgboost":
            xgb_result = train_xgboost_forecast(history, target_col, cutoff_date)
            future_pred = predict_next_days_xgboost(
                history, target_col, xgb_result, num_days=forecast_day_offset
            )
            last_day_pred = future_pred[future_pred["date"] == future_pred["date"].max()]
            for _, row in last_day_pred.iterrows():
                result_by_type.setdefault(row["node_id"], {})[waste_type] = float(row["predicted_kg"])
        else:  # prophet - chạy riêng cho từng node
            for node_id in history["node_id"].unique():
                fc = forecast_prophet(history, node_id, target_col, periods=forecast_day_offset)
                val = max(0.0, float(fc.iloc[-1]["yhat"]))
                result_by_type.setdefault(node_id, {})[waste_type] = round(val, 1)

    demand_dict = {}
    for node_id, types in result_by_type.items():
        total = sum(types.values())
        demand_dict[node_id] = {**types, "total_kg": round(total, 1)}

    return demand_dict


def estimate_vehicles_needed(demand_dict: dict, vehicle_capacity_kg: float) -> tuple[float, int]:
    """Tính tổng khối lượng rác dự báo và số xe TỐI THIỂU cần huy động, dựa
    trên tổng demand dự báo và tải trọng mỗi xe (làm tròn LÊN - ceiling)."""
    total_kg = sum(v["total_kg"] for v in demand_dict.values())
    num_vehicles = max(1, math.ceil(total_kg / vehicle_capacity_kg))
    return round(total_kg, 1), num_vehicles


def evaluate_all_types(history_df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Chạy XGBoost cho cả 3 loại rác, trả về bảng tổng hợp MAE + feature
    quan trọng nhất của mỗi loại - dùng để đưa vào báo cáo/slide."""
    rows = []
    for waste_type in WASTE_TYPES:
        target_col = f"waste_{waste_type}_kg"
        result = train_xgboost_forecast(history_df, target_col, cutoff_date)
        top_feature = result.feature_importance.index[0]
        rows.append({
            "Loại rác": waste_type,
            "MAE (kg)": round(result.mae, 2),
            "Feature quan trọng nhất": top_feature,
            "Mức độ ảnh hưởng (%)": round(result.feature_importance.iloc[0] * 100, 1),
        })
    return pd.DataFrame(rows)
