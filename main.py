import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 흥행 예측기")
st.write(
    "박스오피스 데이터와 영화 정보를 이용하여 "
    "영화의 총 관객 수를 다중 회귀로 예측합니다."
)


# =========================================================
# 데이터 주소
# =========================================================

DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "kobis_daily.csv"
)

MOVIES_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "kobis_movies.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")

    return daily, movies


try:
    daily, movies = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 데이터 전처리
# =========================================================

# 날짜
daily["날짜"] = pd.to_datetime(
    daily["날짜"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

movies["openDt"] = pd.to_datetime(
    movies["openDt"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

movies["first_date"] = pd.to_datetime(
    movies["first_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# 영화 코드 문자열 통일
daily["영화코드"] = daily["영화코드"].astype(str).str.strip()
movies["movieCd"] = movies["movieCd"].astype(str).str.strip()


# =========================================================
# daily 데이터에서 영화별 첫 관측일 정보 만들기
# =========================================================

daily_sorted = daily.sort_values(
    ["영화코드", "날짜", "순위"]
).copy()

# 영화별 첫 관측 기록
first_daily = (
    daily_sorted
    .groupby("영화코드", as_index=False)
    .first()
)

first_daily = first_daily[
    [
        "영화코드",
        "일관객",
        "스크린수",
        "상영횟수",
        "날짜"
    ]
].rename(
    columns={
        "일관객": "첫관측_일관객",
        "스크린수": "첫관측_스크린수",
        "상영횟수": "첫관측_상영횟수",
        "날짜": "첫관측일"
    }
)


# =========================================================
# 두 데이터 합치기
# =========================================================

df = movies.merge(
    first_daily,
    left_on="movieCd",
    right_on="영화코드",
    how="left"
)

# 영화 정보표의 모든 영화를 사용하기 위해
# movies를 기준으로 left join을 사용함.


# =========================================================
# 장르 전처리
# 여러 장르가 | 로 연결되어 있으면 첫 번째 장르만 사용
# =========================================================

df["genre"] = (
    df["genre"]
    .fillna("미상")
    .astype(str)
    .str.split("|")
    .str[0]
    .str.strip()
)

df["nation"] = (
    df["nation"]
    .fillna("미상")
    .astype(str)
    .str.strip()
)


# =========================================================
# 개봉월 / 개봉요일 등 파생변수
# =========================================================

df["개봉월"] = df["openDt"].dt.month

df["개봉요일"] = df["openDt"].dt.dayofweek

weekday_map = {
    0: "월요일",
    1: "화요일",
    2: "수요일",
    3: "목요일",
    4: "금요일",
    5: "토요일",
    6: "일요일"
}

df["개봉요일"] = df["개봉요일"].map(weekday_map)


# =========================================================
# 사용할 수치형 변수
# =========================================================

numeric_features = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 등장일수",
    "peak": "성수기 개봉 여부",
    "첫관측_일관객": "박스오피스 첫 관측일 관객",
    "첫관측_스크린수": "박스오피스 첫 관측일 스크린수",
    "첫관측_상영횟수": "박스오피스 첫 관측일 상영횟수"
}

categorical_features = {
    "genre": "장르",
    "nation": "국가",
    "개봉요일": "개봉요일"
}

date_features = {
    "개봉월": "개봉월"
}


# =========================================================
# 화면 - 데이터 기간
# =========================================================

all_dates = daily["날짜"].dropna()

if len(all_dates) > 0:
    min_date = all_dates.min()
    max_date = all_dates.max()

    st.info(
        f"📅 기준 기간: "
        f"{min_date.strftime('%Y-%m-%d')} ~ "
        f"{max_date.strftime('%Y-%m-%d')}"
    )


# =========================================================
# 변수 선택
# =========================================================

st.subheader("1️⃣ 예측에 사용할 변수 선택")

st.write(
    "체크한 변수만 회귀 모델의 입력값으로 사용됩니다. "
    "목표값은 영화의 `total_audi(총 관객)`입니다."
)

selected_numeric = []
selected_categorical = []
selected_date = []


col1, col2, col3 = st.columns(3)


with col1:
    st.markdown("### 🔢 수치형 변수")

    for column, label in numeric_features.items():

        default = column in [
            "first_scrn",
            "first_show",
            "first_week_audi",
            "days_in_top10",
            "peak"
        ]

        if st.checkbox(
            label,
            value=default,
            key=f"numeric_{column}"
        ):
            selected_numeric.append(column)


with col2:
    st.markdown("### 🏷️ 범주형 변수")

    for column, label in categorical_features.items():

        default = column in [
            "genre",
            "nation"
        ]

        if st.checkbox(
            label,
            value=default,
            key=f"categorical_{column}"
        ):
            selected_categorical.append(column)


with col3:
    st.markdown("### 📅 날짜 관련 변수")

    for column, label in date_features.items():

        default = True

        if st.checkbox(
            label,
            value=default,
            key=f"date_{column}"
        ):
            selected_date.append(column)


selected_features = (
    selected_numeric
    + selected_categorical
    + selected_date
)


if len(selected_features) == 0:
    st.warning("⚠️ 최소 1개 이상의 변수를 선택해 주세요.")
    st.stop()


st.success(
    "선택된 변수: "
    + ", ".join(
        [numeric_features.get(x, categorical_features.get(x, date_features.get(x, x)))
         for x in selected_features]
    )
)


# =========================================================
# 목표값 확인
# =========================================================

df["total_audi"] = pd.to_numeric(
    df["total_audi"],
    errors="coerce"
)

# 목표값이 없는 영화는 모델을 만들 수 없으므로 제외
model_df = df.dropna(subset=["total_audi"]).copy()


# =========================================================
# 영화코드 순으로 정렬
# =========================================================

model_df["movieCd_sort"] = model_df["movieCd"].astype(str)

model_df = model_df.sort_values(
    "movieCd_sort"
).reset_index(drop=True)


# =========================================================
# 10편마다 앞의 3편을 시험용으로 지정
# 나머지 7편은 학습용
#
# 0,1,2 -> 시험
# 3,4,5,6,7,8,9 -> 학습
# 반복
# =========================================================

model_df["순번"] = np.arange(len(model_df))

model_df["데이터구분"] = np.where(
    model_df["순번"] % 10 < 3,
    "시험용",
    "학습용"
)


train_df = model_df[
    model_df["데이터구분"] == "학습용"
].copy()

test_df = model_df[
    model_df["데이터구분"] == "시험용"
].copy()


# =========================================================
# X / y
# =========================================================

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["total_audi"].copy()
y_test = test_df["total_audi"].copy()


# =========================================================
# 전처리 파이프라인
# =========================================================

numeric_selected = [
    x for x in selected_features
    if x in numeric_features or x in date_features
]

categorical_selected = [
    x for x in selected_features
    if x in categorical_features
]


transformers = []


if len(numeric_selected) > 0:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_selected
        )
    )


if len(categorical_selected) > 0:

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_selected
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# =========================================================
# 다중 회귀 모델
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regression",
            LinearRegression()
        )
    ]
)


# =========================================================
# 학습
# =========================================================

model.fit(
    X_train,
    y_train
)


# =========================================================
# 시험용 데이터 예측
# =========================================================

pred = model.predict(X_test)

pred = np.asarray(pred)

test_df["예측_총관객"] = pred

test_df["예측_총관객"] = (
    test_df["예측_총관객"]
    .replace([np.inf, -np.inf], np.nan)
)


# =========================================================
# 점수 계산
# =========================================================

valid_mask = (
    test_df["예측_총관객"].notna()
    & test_df["total_audi"].notna()
)

actual = test_df.loc[
    valid_mask,
    "total_audi"
].astype(float)

predicted = test_df.loc[
    valid_mask,
    "예측_총관객"
].astype(float)


if len(actual) > 0:

    r2 = r2_score(
        actual,
        predicted
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # 실제 관객이 0인 경우 MAPE 계산에서 제외
    mape_mask = actual != 0

    if mape_mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[mape_mask] - predicted[mape_mask])
                / actual[mape_mask]
            )
        ) * 100

    else:
        mape = np.nan

else:

    r2 = np.nan
    mae = np.nan
    rmse = np.nan
    mape = np.nan


# =========================================================
# 결과 요약
# =========================================================

st.subheader("2️⃣ 모델 평가 결과")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "학습에 사용한 영화",
        f"{len(train_df)}편"
    )

with c2:
    st.metric(
        "점수를 잰 영화",
        f"{len(test_df)}편"
    )

with c3:
    st.metric(
        "R² 예측 점수",
        f"{r2:.3f}"
        if not np.isnan(r2)
        else "-"
    )

with c4:
    st.metric(
        "평균 절대 오차",
        f"{mae:,.0f}명"
        if not np.isnan(mae)
        else "-"
    )


st.write(
    f"📌 기준 기간: "
    f"{min_date.strftime('%Y-%m-%d')} ~ "
    f"{max_date.strftime('%Y-%m-%d')}"
)


st.markdown("### 📊 추가 평가 지표")

m1, m2 = st.columns(2)

with m1:
    st.write(
        f"**RMSE:** "
        f"{rmse:,.0f}명"
        if not np.isnan(rmse)
        else "**RMSE:** -"
    )

with m2:
    st.write(
        f"**MAPE:** "
        f"{mape:.2f}%"
        if not np.isnan(mape)
        else "**MAPE:** -"
    )


# =========================================================
# 시험용 영화 중 1,000명 미만 예측
# =========================================================

low_prediction_mask = (
    test_df["예측_총관객"] < 1000
)

low_count = low_prediction_mask.sum()


st.info(
    f"📍 시험용 영화 중 예측 관객 수가 "
    f"1,000명보다 작은 영화: **{low_count}편**"
)


# =========================================================
# 실제 vs 예측 산점도
# =========================================================

st.subheader(
    "3️⃣ 실제 총 관객 수와 예측 총 관객 수"
)

st.write(
    "가로축은 시험용 영화의 실제 총 관객 수, "
    "세로축은 모델이 예측한 총 관객 수입니다. "
    "점선은 예측이 실제와 정확히 같은 경우입니다."
)


plot_df = test_df.copy()

# 로그축을 사용하려면 양수여야 하므로
# 실제값이 0 이하인 경우 표시에서 제외
plot_df = plot_df[
    (plot_df["total_audi"] > 0)
    & plot_df["예측_총관객"].notna()
].copy()


# 예측값이 1,000 미만인 영화는 그래프 바닥에 붙임
plot_df["그래프_예측값"] = plot_df["예측_총관객"].clip(
    lower=1
)

low_df = plot_df[
    plot_df["예측_총관객"] < 1000
].copy()


fig = go.Figure()


# 일반 시험용 영화
normal_df = plot_df[
    plot_df["예측_총관객"] >= 1000
].copy()


fig.add_trace(
    go.Scatter(
        x=normal_df["total_audi"],
        y=normal_df["그래프_예측값"],
        mode="markers",
        name="시험용 영화",
        text=normal_df["movieNm"],
        customdata=np.column_stack(
            [
                normal_df["movieCd"],
                normal_df["예측_총관객"],
                normal_df["total_audi"]
            ]
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "영화코드: %{customdata[0]}<br>"
            "실제 총 관객: %{customdata[2]:,.0f}명<br>"
            "예측 총 관객: %{customdata[1]:,.0f}명"
            "<extra></extra>"
        ),
        marker=dict(
            size=9
        )
    )
)


# 1,000명 미만 예측 영화
if len(low_df) > 0:

    fig.add_trace(
        go.Scatter(
            x=low_df["total_audi"],
            y=np.full(
                len(low_df),
                1
            ),
            mode="markers",
            name="예측 1,000명 미만",
            text=low_df["movieNm"],
            customdata=np.column_stack(
                [
                    low_df["movieCd"],
                    low_df["예측_총관객"],
                    low_df["total_audi"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 총 관객: %{customdata[2]:,.0f}명<br>"
                "예측 총 관객: %{customdata[1]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                size=11,
                symbol="triangle-down"
            )
        )
    )


# =========================================================
# 실제 = 예측 기준선
# =========================================================

if len(plot_df) > 0:

    min_value = max(
        1,
        plot_df["total_audi"].min()
    )

    max_value = max(
        plot_df["total_audi"].max(),
        plot_df["그래프_예측값"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="실제 = 예측",
            line=dict(
                dash="dash"
            ),
            hoverinfo="skip"
        )
    )


fig.update_xaxes(
    type="log",
    title="실제 총 관객 수",
    showgrid=True
)

fig.update_yaxes(
    type="log",
    title="예측한 총 관객 수",
    showgrid=True
)

fig.update_layout(
    height=700,
    hovermode="closest",
    legend_title="구분"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 예측 영화 설명
# =========================================================

if low_count > 0:

    st.caption(
        "▼ 삼각형으로 표시된 영화는 예측값이 1,000명 미만이라 "
        "로그 그래프의 바닥에 붙여 표시했습니다."
    )


# =========================================================
# 영화별 실제 / 예측 결과
# =========================================================

st.subheader("4️⃣ 시험용 영화별 예측 결과")

result_table = test_df[
    [
        "movieCd",
        "movieNm",
        "total_audi",
        "예측_총관객",
        "데이터구분"
    ]
].copy()

result_table = result_table.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객",
        "예측_총관객": "예측 총 관객",
        "데이터구분": "구분"
    }
)

result_table["예측 오차"] = (
    result_table["예측 총 관객"]
    - result_table["실제 총 관객"]
)

result_table["절대 오차"] = (
    result_table["예측 오차"]
    .abs()
)

result_table = result_table.sort_values(
    "영화코드"
).reset_index(drop=True)


st.dataframe(
    result_table.style.format(
        {
            "실제 총 관객": "{:,.0f}",
            "예측 총 관객": "{:,.0f}",
            "예측 오차": "{:,.0f}",
            "절대 오차": "{:,.0f}"
        }
    ),
    use_container_width=True,
    height=500
)


# =========================================================
# 전체 데이터 사용 현황
# =========================================================

st.subheader("5️⃣ 데이터 사용 현황")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        "영화 정보표 전체 영화",
        f"{len(movies)}편"
    )

with info_col2:
    st.metric(
        "모델에 사용된 영화",
        f"{len(model_df)}편"
    )

with info_col3:
    st.metric(
        "박스오피스 데이터 행",
        f"{len(daily):,}개"
    )


st.write(
    "영화 정보표를 기준으로 데이터를 결합했기 때문에 "
    "영화 정보표에 존재하는 영화들을 모두 대상으로 사용했습니다."
)

st.write(
    "영화코드 순으로 정렬한 뒤 10편씩 묶어 각 묶음의 "
    "앞 3편을 시험용, 뒤 7편을 학습용으로 지정했습니다."
)


# =========================================================
# 사용한 변수
# =========================================================

with st.expander("🔎 현재 모델에 사용된 변수 보기"):

    st.write(
        "선택한 입력 변수:"
    )

    for feature in selected_features:

        label = (
            numeric_features.get(feature)
            or categorical_features.get(feature)
            or date_features.get(feature)
            or feature
        )

        st.write(
            f"- **{label}** (`{feature}`)"
        )

    st.write(
        "**예측 대상:** `total_audi` (총 관객)"
    )


# =========================================================
# 주의사항
# =========================================================

with st.expander("ℹ️ 예측 방법 설명"):

    st.write(
        """
        이 앱은 다중 선형회귀(Linear Regression)를 사용합니다.

        영화코드 순으로 영화들을 정렬한 뒤 10편 단위로 나누고,
        각 10편 중 앞의 3편은 모델 학습에서 제외하여 시험용으로 사용합니다.

        따라서 시험용 영화의 실제 총 관객 수는 모델 학습에 사용되지 않습니다.

        선택한 장르·국가·개봉요일 같은 범주형 변수는
        원-핫 인코딩을 통해 숫자로 변환합니다.

        또한 입력 변수의 크기 차이를 줄이기 위해 수치형 변수는
        표준화한 뒤 회귀 모델에 넣습니다.

        산점도의 점선은 '실제 관객 수 = 예측 관객 수'를 의미합니다.
        점선에 가까울수록 예측이 실제에 가깝습니다.
        """
    )
