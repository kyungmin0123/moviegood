import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")

    return daily, movies


# =========================================================
# 데이터 전처리
# =========================================================

@st.cache_data
def prepare_data(daily, movies):

    daily = daily.copy()
    movies = movies.copy()

    # 날짜 변환
    daily["날짜"] = pd.to_datetime(
        daily["날짜"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    # 영화 코드 문자열 통일
    daily["영화코드"] = daily["영화코드"].astype(str).str.strip()
    movies["movieCd"] = movies["movieCd"].astype(str).str.strip()

    # 숫자형 변환
    daily["일관객"] = pd.to_numeric(daily["일관객"], errors="coerce")
    daily["상영횟수"] = pd.to_numeric(daily["상영횟수"], errors="coerce")

    numeric_columns = [
        "first_scrn",
        "first_show",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
        "peak"
    ]

    for col in numeric_columns:
        if col in movies.columns:
            movies[col] = pd.to_numeric(
                movies[col],
                errors="coerce"
            )

    # -----------------------------------------------------
    # 각 영화의 첫 관측일 찾기
    # -----------------------------------------------------
    # 날짜가 가장 빠른 행을 영화별로 하나씩 선택
    daily_sorted = daily.sort_values(
        ["영화코드", "날짜", "순위"]
    ).copy()

    first_daily = daily_sorted.drop_duplicates(
        subset="영화코드",
        keep="first"
    ).copy()

    # -----------------------------------------------------
    # 첫 관측일의 상영당 관객 수 계산
    # -----------------------------------------------------
    first_daily["상영당_관객수"] = np.where(
        first_daily["상영횟수"] > 0,
        first_daily["일관객"] / first_daily["상영횟수"],
        np.nan
    )

    # 필요한 열만 사용
    first_daily = first_daily[
        [
            "영화코드",
            "날짜",
            "일관객",
            "상영횟수",
            "상영당_관객수"
        ]
    ]

    # -----------------------------------------------------
    # 영화 정보와 결합
    # -----------------------------------------------------

    movies = movies.merge(
        first_daily,
        left_on="movieCd",
        right_on="영화코드",
        how="left"
    )

    # 영화코드 기준 정렬
    movies = movies.sort_values(
        "movieCd"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # 10개 단위로 3개 테스트 / 7개 학습
    # -----------------------------------------------------

    movies["순번"] = np.arange(len(movies))

    movies["데이터구분"] = np.where(
        movies["순번"] % 10 < 3,
        "테스트",
        "학습"
    )

    return movies


# =========================================================
# 모델 만들기
# =========================================================

def make_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "regression",
                LinearRegression()
            )
        ]
    )


# =========================================================
# 예측 실행
# =========================================================

def train_and_predict(data, features):

    model_data = data.dropna(
        subset=["total_audi"]
    ).copy()

    train_data = model_data[
        model_data["데이터구분"] == "학습"
    ].copy()

    test_data = model_data[
        model_data["데이터구분"] == "테스트"
    ].copy()

    X_train = train_data[features]
    y_train = train_data["total_audi"]

    X_test = test_data[features]
    y_test = test_data["total_audi"]

    model = make_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    result = test_data[
        [
            "movieCd",
            "movieNm",
            "total_audi"
        ]
    ].copy()

    result["예측관객"] = predictions
    result["오차"] = (
        result["예측관객"] -
        result["total_audi"]
    )
    result["절대오차"] = result["오차"].abs()

    return model, result, r2, mae


# =========================================================
# 데이터 로딩
# =========================================================

try:
    daily, movies = load_data()
    data = prepare_data(
        daily,
        movies
    )

except Exception as e:

    st.error(
        "데이터를 불러오는 중 오류가 발생했습니다."
    )

    st.exception(e)

    st.stop()


# =========================================================
# 제목
# =========================================================

st.title("🎬 영화 흥행 예측기")

st.markdown(
    """
    영화가 **처음 TOP 10에 관측된 시점**까지의 정보를 이용하여
    최종 누적 관객 수를 예측해 봅니다.
    """
)

st.divider()


# =========================================================
# 데이터 정보
# =========================================================

st.subheader("📊 데이터 정보")

col1, col2, col3, col4 = st.columns(4)

total_movies = len(data)

train_count = len(
    data[data["데이터구분"] == "학습"]
)

test_count = len(
    data[data["데이터구분"] == "테스트"]
)

first_date = data["날짜"].min()
last_date = data["날짜"].max()

with col1:
    st.metric(
        "전체 영화 수",
        f"{total_movies:,}편"
    )

with col2:
    st.metric(
        "학습 영화",
        f"{train_count:,}편"
    )

with col3:
    st.metric(
        "테스트 영화",
        f"{test_count:,}편"
    )

with col4:
    st.metric(
        "데이터 기간",
        f"{first_date:%Y-%m-%d} ~ {last_date:%Y-%m-%d}"
    )


st.divider()


# =========================================================
# 상영당 관객 수 확인
# =========================================================

st.subheader("🎯 새로 만든 변수")

st.markdown(
    """
    **상영당 관객 수**는 영화가 처음 TOP 10에 관측된 날의

    `일관객 ÷ 상영횟수`

    로 계산했습니다.

    즉, **처음 관측된 날 한 번의 상영에서 평균적으로 몇 명의 관객이
    관람했는지**를 나타냅니다.
    """
)

valid_per_show = data["상영당_관객수"].dropna()

if len(valid_per_show) > 0:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "계산 가능한 영화",
            f"{len(valid_per_show):,}편"
        )

    with c2:
        st.metric(
            "평균 상영당 관객",
            f"{valid_per_show.mean():,.1f}명"
        )

    with c3:
        st.metric(
            "최대 상영당 관객",
            f"{valid_per_show.max():,.1f}명"
        )


st.divider()


# =========================================================
# 사용할 변수
# =========================================================

basic_features = [
    "first_scrn",
    "first_show",
    "peak"
]

extended_features = [
    "first_scrn",
    "first_show",
    "peak",
    "상영당_관객수"
]


# =========================================================
# 모델 계산
# =========================================================

basic_model, basic_result, basic_r2, basic_mae = train_and_predict(
    data,
    basic_features
)

extended_model, extended_result, extended_r2, extended_mae = train_and_predict(
    data,
    extended_features
)


# =========================================================
# 모델 비교
# =========================================================

st.subheader("📈 기본 변수 3개 vs 상영당 관객 수 추가")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 기본 변수 3개")

    st.caption(
        "첫 관측 스크린 수 + 첫 관측 상영횟수 + peak"
    )

    st.metric(
        "R²",
        f"{basic_r2:.4f}"
    )

    st.metric(
        "MAE",
        f"{basic_mae:,.0f}명"
    )


with col2:

    st.markdown("### 상영당 관객 수 추가")

    st.caption(
        "기본 변수 3개 + 첫 관측일 상영당 관객 수"
    )

    st.metric(
        "R²",
        f"{extended_r2:.4f}"
    )

    st.metric(
        "MAE",
        f"{extended_mae:,.0f}명"
    )


r2_difference = extended_r2 - basic_r2

if r2_difference > 0:

    st.success(
        f"상영당 관객 수를 추가했을 때 R²가 "
        f"{r2_difference:+.4f} 증가했습니다."
    )

elif r2_difference < 0:

    st.warning(
        f"상영당 관객 수를 추가했을 때 R²가 "
        f"{r2_difference:+.4f} 감소했습니다."
    )

else:

    st.info(
        "두 모델의 R²가 동일합니다."
    )


st.divider()


# =========================================================
# 예측값 비교 그래프
# =========================================================

st.subheader("🔍 실제 관객 수와 예측 관객 수 비교")

st.caption(
    "가로축 = 실제 최종 관객 수 / 세로축 = 예측 최종 관객 수"
)

fig = go.Figure()


# ---------------------------------------------------------
# 1,000명 이상 기본 모델
# ---------------------------------------------------------

basic_normal = basic_result[
    basic_result["예측관객"] >= 1000
].copy()

if len(basic_normal) > 0:

    fig.add_trace(
        go.Scatter(
            x=basic_normal["total_audi"],
            y=np.maximum(
                basic_normal["예측관객"],
                1
            ),
            mode="markers",
            name="기본 모델",
            marker=dict(
                size=9,
                symbol="circle"
            ),
            customdata=basic_normal[
                ["movieNm", "예측관객"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "실제 관객: %{x:,.0f}명<br>"
                "예측 관객: %{customdata[1]:,.0f}명"
                "<extra></extra>"
            )
        )
    )


# ---------------------------------------------------------
# 1,000명 미만 기본 모델
# ---------------------------------------------------------

basic_low = basic_result[
    basic_result["예측관객"] < 1000
].copy()

if len(basic_low) > 0:

    fig.add_trace(
        go.Scatter(
            x=basic_low["total_audi"],
            y=[1] * len(basic_low),
            mode="markers",
            name="기본 모델 (<1,000명)",
            marker=dict(
                size=11,
                symbol="circle-open"
            ),
            customdata=basic_low[
                ["movieNm", "예측관객"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "실제 관객: %{x:,.0f}명<br>"
                "예측 관객: %{customdata[1]:,.0f}명"
                "<extra></extra>"
            )
        )
    )


# ---------------------------------------------------------
# 1,000명 이상 확장 모델
# ---------------------------------------------------------

extended_normal = extended_result[
    extended_result["예측관객"] >= 1000
].copy()

if len(extended_normal) > 0:

    fig.add_trace(
        go.Scatter(
            x=extended_normal["total_audi"],
            y=np.maximum(
                extended_normal["예측관객"],
                1
            ),
            mode="markers",
            name="상영당 관객 수 추가 모델",
            marker=dict(
                size=10,
                symbol="diamond"
            ),
            customdata=extended_normal[
                ["movieNm", "예측관객"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "실제 관객: %{x:,.0f}명<br>"
                "예측 관객: %{customdata[1]:,.0f}명"
                "<extra></extra>"
            )
        )
    )


# ---------------------------------------------------------
# 1,000명 미만 확장 모델
# ---------------------------------------------------------

extended_low = extended_result[
    extended_result["예측관객"] < 1000
].copy()

if len(extended_low) > 0:

    fig.add_trace(
        go.Scatter(
            x=extended_low["total_audi"],
            y=[1] * len(extended_low),
            mode="markers",
            name="상영당 관객 수 추가 모델 (<1,000명)",
            marker=dict(
                size=12,
                symbol="diamond-open"
            ),
            customdata=extended_low[
                ["movieNm", "예측관객"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "실제 관객: %{x:,.0f}명<br>"
                "예측 관객: %{customdata[1]:,.0f}명"
                "<extra></extra>"
            )
        )
    )


# ---------------------------------------------------------
# 실제 = 예측 기준선
# ---------------------------------------------------------

max_actual = max(
    basic_result["total_audi"].max(),
    extended_result["total_audi"].max()
)

max_pred = max(
    basic_result["예측관객"].max(),
    extended_result["예측관객"].max(),
    1000
)

line_max = max(
    max_actual,
    max_pred
)

fig.add_trace(
    go.Scatter(
        x=[1, line_max],
        y=[1, line_max],
        mode="lines",
        name="실제 = 예측",
        line=dict(
            dash="dash",
            width=2
        ),
        hoverinfo="skip"
    )
)


fig.update_layout(
    height=700,
    xaxis=dict(
        title="실제 최종 관객 수",
        type="log",
        range=[
            0,
            np.log10(max_actual * 1.2)
        ]
    ),
    yaxis=dict(
        title="예측 최종 관객 수",
        type="log",
        range=[
            0,
            np.log10(max_pred * 1.2)
        ]
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(
        l=60,
        r=40,
        t=80,
        b=60
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 예측 개수
# =========================================================

basic_low_count = (
    basic_result["예측관객"] < 1000
).sum()

extended_low_count = (
    extended_result["예측관객"] < 1000
).sum()

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "기본 모델 1,000명 미만 예측",
        f"{basic_low_count}편"
    )

with c2:

    st.metric(
        "상영당 관객 수 추가 모델 1,000명 미만 예측",
        f"{extended_low_count}편"
    )


st.divider()


# =========================================================
# 테스트 영화 예측 결과
# =========================================================

st.subheader("🎞️ 테스트 영화 예측 결과")

st.caption(
    "영화 코드 순으로 정렬했을 때 10개마다 앞의 3개 영화를 테스트 데이터로 사용했습니다."
)

display_result = extended_result.copy()

display_result["실제 관객"] = (
    display_result["total_audi"]
)

display_result["상영당 관객 수"] = (
    data.set_index("movieCd")
    .reindex(display_result["movieCd"])["상영당_관객수"]
    .values
)

display_result = display_result[
    [
        "movieCd",
        "movieNm",
        "실제 관객",
        "예측관객",
        "오차",
        "절대오차"
    ]
].copy()

display_result.columns = [
    "영화코드",
    "영화명",
    "실제 최종 관객",
    "예측 최종 관객",
    "예측 오차",
    "절대 오차"
]

display_result = display_result.sort_values(
    "영화코드"
)

st.dataframe(
    display_result.style.format(
        {
            "실제 최종 관객": "{:,.0f}",
            "예측 최종 관객": "{:,.0f}",
            "예측 오차": "{:+,.0f}",
            "절대 오차": "{:,.0f}"
        }
    ),
    use_container_width=True,
    height=500
)


st.divider()


# =========================================================
# 데이터 확인
# =========================================================

with st.expander("📋 전체 영화 데이터 보기"):

    view_columns = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_show",
        "peak",
        "상영당_관객수",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
        "데이터구분"
    ]

    available_columns = [
        col
        for col in view_columns
        if col in data.columns
    ]

    st.dataframe(
        data[available_columns],
        use_container_width=True,
        height=500
    )


# =========================================================
# 주의사항
# =========================================================

st.warning(
    """
    ⚠️ **해석할 때 주의하세요**

    이 분석의 R²와 MAE는 실제 개봉 전에 아무 정보도 없는 상태에서
    흥행을 예측한 성능이 아닙니다.

    이번 분석은 **영화가 처음 TOP 10에 관측된 날의 집계가 끝난 시점**을
    기준으로 삼는다고 가정했습니다.

    또한 `total_audi`는 해당 영화의 전체 기간이 끝난 뒤 집계된
    최종 누적 관객 수이므로, 이 데이터로 계산한 R²를
    실제 서비스에서의 사전 흥행 예측 성능으로 해석하면 안 됩니다.
    """
)


st.caption(
    "🎬 영화 흥행 데이터 분석 · Linear Regression"
)
