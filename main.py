```python
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
    "박스오피스 데이터와 영화 정보를 결합하여 "
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

    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8"
    )

    movies = pd.read_csv(
        MOVIES_URL,
        encoding="utf-8"
    )

    return daily, movies


try:
    daily, movies = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 날짜 전처리
# =========================================================

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


# =========================================================
# 영화 코드 통일
# =========================================================

daily["영화코드"] = (
    daily["영화코드"]
    .astype(str)
    .str.strip()
)

movies["movieCd"] = (
    movies["movieCd"]
    .astype(str)
    .str.strip()
)


# =========================================================
# 박스오피스 데이터에서 영화별 첫 관측값 추출
# =========================================================

daily_sorted = daily.sort_values(
    ["영화코드", "날짜", "순위"]
).copy()

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
# 두 데이터 결합
# =========================================================

df = movies.merge(
    first_daily,
    left_on="movieCd",
    right_on="영화코드",
    how="left"
)


# =========================================================
# 장르
# =========================================================

df["genre"] = (
    df["genre"]
    .fillna("미상")
    .astype(str)
    .str.split("|")
    .str[0]
    .str.strip()
)


# =========================================================
# 국가
# =========================================================

df["nation"] = (
    df["nation"]
    .fillna("미상")
    .astype(str)
    .str.strip()
)


# =========================================================
# 파생 변수
# =========================================================

df["개봉월"] = df["openDt"].dt.month

df["개봉요일번호"] = df["openDt"].dt.dayofweek

weekday_map = {
    0: "월요일",
    1: "화요일",
    2: "수요일",
    3: "목요일",
    4: "금요일",
    5: "토요일",
    6: "일요일"
}

df["개봉요일"] = (
    df["개봉요일번호"]
    .map(weekday_map)
)


# =========================================================
# 숫자형 변환
# =========================================================

numeric_columns = [
    "first_scrn",
    "first_show",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
    "peak",
    "첫관측_일관객",
    "첫관측_스크린수",
    "첫관측_상영횟수"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# =========================================================
# 기준 기간
# =========================================================

valid_dates = daily["날짜"].dropna()

if len(valid_dates) > 0:

    min_date = valid_dates.min()
    max_date = valid_dates.max()

else:

    min_date = None
    max_date = None


st.info(
    f"📅 기준 기간: "
    f"{min_date.strftime('%Y-%m-%d')} ~ "
    f"{max_date.strftime('%Y-%m-%d')}"
)


# =========================================================
# 사용할 영화 전체
# =========================================================

model_df = df.dropna(
    subset=["total_audi"]
).copy()


# =========================================================
# 영화코드 순 정렬
# =========================================================

model_df["movieCd_sort"] = (
    model_df["movieCd"]
    .astype(str)
)

model_df = (
    model_df
    .sort_values("movieCd_sort")
    .reset_index(drop=True)
)


# =========================================================
# 시험용 / 학습용 구분
#
# 10편마다 앞 3편 = 시험용
# 나머지 7편 = 학습용
# =========================================================

model_df["순번"] = np.arange(
    len(model_df)
)

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
# 모델별 변수
# =========================================================

# 기본 3가지 변수
basic_features = [
    "first_scrn",
    "first_show",
    "peak"
]

# 기본 3가지 + 첫 주 관객
extended_features = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi"
]


feature_labels = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객"
}


# =========================================================
# 모델 만드는 함수
# =========================================================

def make_model(features):

    numeric_features = features

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        ]
    )

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

    return model


# =========================================================
# 모델 학습 및 평가 함수
# =========================================================

def train_and_evaluate(features):

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    y_train = train_df["total_audi"].copy()
    y_test = test_df["total_audi"].copy()

    model = make_model(features)

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    prediction = np.asarray(
        prediction
    )

    result = test_df[
        [
            "movieCd",
            "movieNm",
            "total_audi"
        ]
    ].copy()

    result["예측_총관객"] = prediction

    valid = (
        result["total_audi"].notna()
        & result["예측_총관객"].notna()
    )

    actual = result.loc[
        valid,
        "total_audi"
    ].astype(float)

    predicted = result.loc[
        valid,
        "예측_총관객"
    ].astype(float)

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

    return model, result, r2, mae, rmse


# =========================================================
# 두 모델 실행
# =========================================================

basic_model, basic_result, basic_r2, basic_mae, basic_rmse = (
    train_and_evaluate(
        basic_features
    )
)

extended_model, extended_result, extended_r2, extended_mae, extended_rmse = (
    train_and_evaluate(
        extended_features
    )
)


# =========================================================
# 모델 비교
# =========================================================

st.subheader(
    "1️⃣ 두 모델의 예측 성능 비교"
)

st.write(
    "두 모델 모두 같은 영화들을 학습용과 시험용으로 나누어 "
    "동일한 조건에서 평가했습니다."
)


col1, col2 = st.columns(2)


with col1:

    st.markdown("### 🔵 기본 모델")

    st.write(
        "**사용 변수**"
    )

    st.write(
        "• 첫 관측일 스크린수  \n"
        "• 첫 관측일 상영횟수  \n"
        "• 성수기 개봉 여부"
    )

    st.metric(
        "R² 예측 점수",
        f"{basic_r2:.3f}"
    )

    st.metric(
        "평균 절대 오차",
        f"{basic_mae:,.0f}명"
    )


with col2:

    st.markdown("### 🟢 첫 주 관객 추가 모델")

    st.write(
        "**사용 변수**"
    )

    st.write(
        "• 첫 관측일 스크린수  \n"
        "• 첫 관측일 상영횟수  \n"
        "• 성수기 개봉 여부  \n"
        "• **첫 주 관객**"
    )

    st.metric(
        "R² 예측 점수",
        f"{extended_r2:.3f}"
    )

    st.metric(
        "평균 절대 오차",
        f"{extended_mae:,.0f}명"
    )


# =========================================================
# 성능 변화
# =========================================================

r2_change = extended_r2 - basic_r2
mae_change = extended_mae - basic_mae


st.markdown("### 📈 첫 주 관객을 추가했을 때")

c1, c2 = st.columns(2)


with c1:

    st.metric(
        "R² 변화",
        f"{r2_change:+.3f}"
    )


with c2:

    st.metric(
        "MAE 변화",
        f"{mae_change:+,.0f}명"
    )


if r2_change > 0:

    st.success(
        "첫 주 관객 수를 추가했을 때 R²가 높아져 "
        "시험용 데이터에서의 예측 설명력이 좋아졌습니다."
    )

elif r2_change < 0:

    st.warning(
        "첫 주 관객 수를 추가했을 때 R²가 낮아졌습니다."
    )

else:

    st.info(
        "두 모델의 R²가 동일합니다."
    )


# =========================================================
# 데이터 사용 현황
# =========================================================

st.subheader(
    "2️⃣ 학습 및 평가 데이터"
)

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "전체 영화",
        f"{len(model_df)}편"
    )


with c2:

    st.metric(
        "학습에 사용한 영화",
        f"{len(train_df)}편"
    )


with c3:

    st.metric(
        "점수를 잰 영화",
        f"{len(test_df)}편"
    )


st.write(
    f"📅 **기준 기간:** "
    f"{min_date.strftime('%Y-%m-%d')} ~ "
    f"{max_date.strftime('%Y-%m-%d')}"
)


# =========================================================
# 중요 안내
# =========================================================

st.warning(
    """
    ⚠️ **중요: 이 결과는 실제 개봉 전 흥행 예측 성능이 아닙니다.**

    이 데이터의 `first_scrn`, `first_show`, `first_week_audi` 등의 값은
    영화가 이미 개봉하고 박스오피스에 등장한 뒤 집계된 사후 데이터입니다.

    특히 `첫 주 관객`은 영화의 첫 주 흥행 결과가 이미 발생한 뒤 알 수 있는 값입니다.

    따라서 이 모델의 점수는
    **'개봉 전에 흥행을 얼마나 정확하게 예측할 수 있는가'**를 의미하지 않습니다.

    여기서는 주어진 사후 집계 데이터를 이용하여
    **선택한 변수들이 총 관객 수와 얼마나 관련되어 있는지**를 비교하는
    분석용 모델로 해석하는 것이 적절합니다.
    """
)


# =========================================================
# 산점도
# =========================================================

st.subheader(
    "3️⃣ 실제 총 관객 수 vs 예측 총 관객 수"
)

st.write(
    "가로축은 시험용 영화의 실제 총 관객 수, "
    "세로축은 예측한 총 관객 수입니다."
)

plot_df = basic_result.copy()

plot_df["기본_예측"] = basic_result[
    "예측_총관객"
]

plot_df["첫주추가_예측"] = extended_result[
    "예측_총관객"
]

plot_df = plot_df[
    (plot_df["total_audi"] > 0)
    & plot_df["기본_예측"].notna()
    & plot_df["첫주추가_예측"].notna()
].copy()


# =========================================================
# 그래프 범위
# =========================================================

all_values = pd.concat(
    [
        plot_df["total_audi"],
        plot_df["기본_예측"].clip(lower=1),
        plot_df["첫주추가_예측"].clip(lower=1)
    ]
)

min_value = max(
    1,
    all_values.min()
)

max_value = max(
    all_values.max(),
    1
)


# =========================================================
# Plotly 산점도
# =========================================================

fig = go.Figure()


# 기본 모델
fig.add_trace(
    go.Scatter(
        x=plot_df["total_audi"],
        y=plot_df["기본_예측"].clip(lower=1),
        mode="markers",
        name="기본 모델",
        text=plot_df["movieNm"],
        customdata=np.column_stack(
            [
                plot_df["movieCd"],
                plot_df["total_audi"],
                plot_df["기본_예측"]
            ]
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "영화코드: %{customdata[0]}<br>"
            "실제 총 관객: %{customdata[1]:,.0f}명<br>"
            "기본 모델 예측: %{customdata[2]:,.0f}명"
            "<extra></extra>"
        ),
        marker=dict(
            size=9
        )
    )
)


# 첫 주 관객 추가 모델
fig.add_trace(
    go.Scatter(
        x=plot_df["total_audi"],
        y=plot_df["첫주추가_예측"].clip(lower=1),
        mode="markers",
        name="첫 주 관객 추가 모델",
        text=plot_df["movieNm"],
        customdata=np.column_stack(
            [
                plot_df["movieCd"],
                plot_df["total_audi"],
                plot_df["첫주추가_예측"]
            ]
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "영화코드: %{customdata[0]}<br>"
            "실제 총 관객: %{customdata[1]:,.0f}명<br>"
            "첫 주 관객 추가 모델 예측: %{customdata[2]:,.0f}명"
            "<extra></extra>"
        ),
        marker=dict(
            size=9,
            symbol="diamond"
        )
    )
)


# 실제 = 예측 기준선
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
    height=750,
    hovermode="closest",
    legend_title="모델"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 예측
# =========================================================

basic_low = (
    plot_df["기본_예측"] < 1000
).sum()

extended_low = (
    plot_df["첫주추가_예측"] < 1000
).sum()


c1, c2 = st.columns(2)


with c1:

    st.info(
        f"🔵 기본 모델: 예측 1,000명 미만 **{basic_low}편**"
    )


with c2:

    st.info(
        f"🟢 첫 주 관객 추가 모델: "
        f"예측 1,000명 미만 **{extended_low}편**"
    )


# =========================================================
# 시험용 영화 결과표
# =========================================================

st.subheader(
    "4️⃣ 시험용 영화별 예측 결과"
)

comparison = basic_result[
    [
        "movieCd",
        "movieNm",
        "total_audi"
    ]
].copy()

comparison["기본 모델 예측"] = (
    basic_result["예측_총관객"]
)

comparison["첫 주 관객 추가 모델 예측"] = (
    extended_result["예측_총관객"]
)

comparison["기본 모델 오차"] = (
    comparison["기본 모델 예측"]
    - comparison["total_audi"]
)

comparison["첫 주 관객 추가 모델 오차"] = (
    comparison["첫 주 관객 추가 모델 예측"]
    - comparison["total_audi"]
)

comparison = comparison.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객"
    }
)

comparison = comparison.sort_values(
    "영화코드"
).reset_index(drop=True)


st.dataframe(
    comparison.style.format(
        {
            "실제 총 관객": "{:,.0f}",
            "기본 모델 예측": "{:,.0f}",
            "첫 주 관객 추가 모델 예측": "{:,.0f}",
            "기본 모델 오차": "{:,.0f}",
            "첫 주 관객 추가 모델 오차": "{:,.0f}"
        }
    ),
    use_container_width=True,
    height=550
)


# =========================================================
# 사용 변수
# =========================================================

st.subheader(
    "5️⃣ 모델에 사용한 변수"
)

v1, v2 = st.columns(2)


with v1:

    st.markdown("### 🔵 기본 모델")

    for feature in basic_features:

        st.write(
            f"- {feature_labels[feature]}"
        )


with v2:

    st.markdown("### 🟢 첫 주 관객 추가 모델")

    for feature in extended_features:

        st.write(
            f"- {feature_labels[feature]}"
        )


# =========================================================
# 모델 설명
# =========================================================

with st.expander("🔎 모델 평가 방법 자세히 보기"):

    st.write(
        """
        영화 정보표의 모든 영화를 영화코드 순으로 정렬합니다.

        정렬된 영화를 10편씩 묶은 뒤 각 묶음의 앞 3편을
        시험용 데이터로 떼어 놓고, 나머지 7편을 학습용 데이터로 사용합니다.

        이 과정을 반복하여 전체 영화 중 약 30%를 시험용으로 사용합니다.

        두 모델은 정확히 같은 시험용 영화를 사용하기 때문에
        R²와 MAE를 비교하여 '첫 주 관객' 변수를 추가했을 때
        성능이 어떻게 달라지는지 확인할 수 있습니다.

        R²는 1에 가까울수록 실제 총 관객 수의 변동을
        모델이 잘 설명한다는 의미입니다.

        MAE는 예측값과 실제값의 평균적인 차이입니다.
        따라서 MAE는 작을수록 좋습니다.
        """
    )
```
