import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================

# 페이지 설정

# =========================================================

st.set_page_config(
page_title="영화 흥행 예측기",
page_icon="🎬",
layout="wide"
)

st.title("🎬 영화 흥행 예측기")

st.write(
"영화가 처음 10위권에 진입한 날의 정보를 이용하여 "
"최종 총 관객 수를 예측하는 회귀 모델을 비교합니다."
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

```
daily = pd.read_csv(
    DAILY_URL,
    encoding="utf-8"
)

movies = pd.read_csv(
    MOVIES_URL,
    encoding="utf-8"
)

return daily, movies
```

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

# 영화코드 통일

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

# 숫자형 변환

# =========================================================

for col in [
"일관객",
"상영횟수",
"스크린수",
"순위",
"first_scrn",
"first_show",
"first_week_audi",
"total_audi",
"days_in_top10",
"peak"
]:

```
if col in daily.columns:
    daily[col] = pd.to_numeric(
        daily[col],
        errors="coerce"
    )

if col in movies.columns:
    movies[col] = pd.to_numeric(
        movies[col],
        errors="coerce"
    )
```

# =========================================================

# 영화마다 '10위권에 처음 등장한 날' 찾기

# =========================================================

daily_sorted = daily.sort_values(
[
"영화코드",
"날짜",
"순위"
]
).copy()

first_daily = (
daily_sorted
.groupby(
"영화코드",
as_index=False
)
.first()
)

# =========================================================

# 첫 등장일 정보만 추출

# =========================================================

first_daily = first_daily[
[
"영화코드",
"날짜",
"일관객",
"스크린수",
"상영횟수"
]
].rename(
columns={
"날짜": "첫관측일",
"일관객": "첫관측_일관객",
"스크린수": "첫관측_스크린수",
"상영횟수": "첫관측_상영횟수"
}
)

# =========================================================

# ★ 새 변수: 상영당 관객 수

#

# 첫 등장일의 일관객 / 첫 등장일의 상영횟수

# =========================================================

first_daily["상영당_관객수"] = np.where(
first_daily["첫관측_상영횟수"] > 0,
first_daily["첫관측_일관객"]
/ first_daily["첫관측_상영횟수"],
np.nan
)

# =========================================================

# 영화 정보와 결합

# =========================================================

df = movies.merge(
first_daily,
left_on="movieCd",
right_on="영화코드",
how="left"
)

# =========================================================

# 영화코드 순 정렬

# =========================================================

df["movieCd_sort"] = (
df["movieCd"]
.astype(str)
.str.strip()
)

df = (
df
.sort_values("movieCd_sort")
.reset_index(drop=True)
)

# =========================================================

# 목표값이 없는 영화 제거

# =========================================================

model_df = df.dropna(
subset=["total_audi"]
).copy()

# =========================================================

# 학습용 / 시험용 분리

#

# 영화코드 순으로 정렬한 상태에서

#

# 10편마다

# 앞 3편 → 시험용

# 뒤 7편 → 학습용

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

# 기준 기간

# =========================================================

valid_dates = daily["날짜"].dropna()

if len(valid_dates) > 0:

```
min_date = valid_dates.min()
max_date = valid_dates.max()

st.info(
    f"📅 기준 기간: "
    f"{min_date.strftime('%Y-%m-%d')} ~ "
    f"{max_date.strftime('%Y-%m-%d')}"
)
```

# =========================================================

# 데이터 사용 현황

# =========================================================

st.subheader("📊 데이터 사용 현황")

c1, c2, c3 = st.columns(3)

with c1:
st.metric(
"전체 영화",
f"{len(model_df)}편"
)

with c2:
st.metric(
"학습용 영화",
f"{len(train_df)}편"
)

with c3:
st.metric(
"시험용 영화",
f"{len(test_df)}편"
)

st.write(
"영화코드 순으로 정렬한 뒤 10편마다 앞의 3편을 시험용으로 "
"떼어 놓고, 나머지 7편을 학습용으로 사용했습니다."
)

# =========================================================

# 새 변수 설명

# =========================================================

st.subheader("🆕 새로 만든 변수")

st.info(
"상영당 관객 수 = "
"**10위권에 처음 등장한 날의 일관객 ÷ 그날의 상영횟수**"
)

st.write(
"즉, 영화가 처음 10위권에 들어왔을 때 "
"한 번 상영할 때 평균적으로 몇 명의 관객이 들어왔는지를 나타냅니다."
)

# =========================================================

# 변수 목록

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

feature_names = {
"first_scrn": "첫 관측일 스크린수",
"first_show": "첫 관측일 상영횟수",
"peak": "성수기 개봉 여부",
"상영당_관객수": "상영당 관객 수"
}

# =========================================================

# 회귀 모델 함수

# =========================================================

def run_regression(features):

```
X_train = train_df[features].copy()
X_test = test_df[features].copy()

y_train = train_df["total_audi"].copy()
y_test = test_df["total_audi"].copy()

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
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

pipeline.fit(
    X_train,
    y_train
)

prediction = pipeline.predict(
    X_test
)

r2 = r2_score(
    y_test,
    prediction
)

mae = mean_absolute_error(
    y_test,
    prediction
)

return pipeline, prediction, r2, mae
```

# =========================================================

# 두 모델 실행

# =========================================================

basic_model, basic_prediction, basic_r2, basic_mae = (
run_regression(
basic_features
)
)

extended_model, extended_prediction, extended_r2, extended_mae = (
run_regression(
extended_features
)
)

# =========================================================

# R² 비교

# =========================================================

st.subheader(
"1️⃣ 기본 모델과 새 변수 추가 모델 비교"
)

col1, col2 = st.columns(2)

with col1:

```
st.markdown(
    "### 🔵 기본 변수 3개"
)

st.write(
    "• 첫 관측일 스크린수"
)

st.write(
    "• 첫 관측일 상영횟수"
)

st.write(
    "• 성수기 개봉 여부"
)

st.metric(
    "R²",
    f"{basic_r2:.3f}"
)
```

with col2:

```
st.markdown(
    "### 🟢 상영당 관객 수 추가"
)

st.write(
    "• 첫 관측일 스크린수"
)

st.write(
    "• 첫 관측일 상영횟수"
)

st.write(
    "• 성수기 개봉 여부"
)

st.write(
    "• **상영당 관객 수**"
)

st.metric(
    "R²",
    f"{extended_r2:.3f}"
)
```

# =========================================================

# R² 변화

# =========================================================

r2_difference = (
extended_r2
- basic_r2
)

st.markdown(
"### 📈 상영당 관객 수를 추가했을 때"
)

if r2_difference > 0:

```
st.success(
    f"R²가 **{r2_difference:+.3f}** 변화했습니다. "
    "상영당 관객 수를 추가한 모델의 R²가 더 높습니다."
)
```

elif r2_difference < 0:

```
st.warning(
    f"R²가 **{r2_difference:+.3f}** 변화했습니다. "
    "상영당 관객 수를 추가한 모델의 R²가 더 낮습니다."
)
```

else:

```
st.info(
    "두 모델의 R²가 동일합니다."
)
```

# =========================================================

# MAE 비교

# =========================================================

st.subheader(
"2️⃣ 평균 절대 오차 비교"
)

mae_col1, mae_col2 = st.columns(2)

with mae_col1:

```
st.metric(
    "기본 모델 MAE",
    f"{basic_mae:,.0f}명"
)
```

with mae_col2:

```
st.metric(
    "상영당 관객 수 추가 모델 MAE",
    f"{extended_mae:,.0f}명"
)
```

# =========================================================

# 산점도

# =========================================================

st.subheader(
"3️⃣ 실제 총 관객 수와 예측 총 관객 수"
)

st.write(
"시험용 영화의 실제 총 관객 수를 가로축, "
"예측 총 관객 수를 세로축으로 표시했습니다."
)

plot_df = test_df[
[
"movieCd",
"movieNm",
"total_audi"
]
].copy()

plot_df["기본_예측"] = basic_prediction
plot_df["확장_예측"] = extended_prediction

plot_df = plot_df[
(plot_df["total_audi"] > 0)
].copy()

# =========================================================

# 로그축에서 사용할 수 있도록 예측값 처리

# =========================================================

plot_df["기본_그래프값"] = (
plot_df["기본_예측"]
.clip(lower=1)
)

plot_df["확장_그래프값"] = (
plot_df["확장_예측"]
.clip(lower=1)
)

all_values = pd.concat(
[
plot_df["total_audi"],
plot_df["기본_그래프값"],
plot_df["확장_그래프값"]
]
)

min_value = max(
1,
all_values.min()
)

max_value = max(
1,
all_values.max()
)

fig = go.Figure()

# =========================================================

# 기본 모델

# =========================================================

fig.add_trace(
go.Scatter(
x=plot_df["total_audi"],
y=plot_df["기본_그래프값"],
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

# =========================================================

# 상영당 관객 수 추가 모델

# =========================================================

fig.add_trace(
go.Scatter(
x=plot_df["total_audi"],
y=plot_df["확장_그래프값"],
mode="markers",
name="상영당 관객 수 추가",
text=plot_df["movieNm"],
customdata=np.column_stack(
[
plot_df["movieCd"],
plot_df["total_audi"],
plot_df["확장_예측"]
]
),
hovertemplate=(
"<b>%{text}</b><br>"
"영화코드: %{customdata[0]}<br>"
"실제 총 관객: %{customdata[1]:,.0f}명<br>"
"확장 모델 예측: %{customdata[2]:,.0f}명"
"<extra></extra>"
),
marker=dict(
size=9,
symbol="diamond"
)
)
)

# =========================================================

# 실제 = 예측 기준선

# =========================================================

fig.add_trace(
go.Scatter(
x=[
min_value,
max_value
],
y=[
min_value,
max_value
],
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
plot_df["확장_예측"] < 1000
).sum()

low1, low2 = st.columns(2)

with low1:

```
st.info(
    f"🔵 기본 모델: "
    f"예측 1,000명 미만 **{basic_low}편**"
)
```

with low2:

```
st.info(
    f"🟢 상영당 관객 수 추가 모델: "
    f"예측 1,000명 미만 **{extended_low}편**"
)
```

# =========================================================

# 시험용 영화 결과표

# =========================================================

st.subheader(
"4️⃣ 시험용 영화별 결과"
)

result = test_df[
[
"movieCd",
"movieNm",
"total_audi",
"상영당_관객수"
]
].copy()

result["기본 모델 예측"] = basic_prediction

result["상영당 관객 수 추가 모델 예측"] = (
extended_prediction
)

result["기본 모델 오차"] = (
result["기본 모델 예측"]
- result["total_audi"]
)

result["추가 모델 오차"] = (
result["상영당 관객 수 추가 모델 예측"]
- result["total_audi"]
)

result = result.rename(
columns={
"movieCd": "영화코드",
"movieNm": "영화명",
"total_audi": "실제 총 관객"
}
)

result = result.sort_values(
"영화코드"
).reset_index(drop=True)

st.dataframe(
result.style.format(
{
"상영당_관객수": "{:,.2f}",
"실제 총 관객": "{:,.0f}",
"기본 모델 예측": "{:,.0f}",
"상영당 관객 수 추가 모델 예측": "{:,.0f}",
"기본 모델 오차": "{:,.0f}",
"추가 모델 오차": "{:,.0f}"
}
),
use_container_width=True,
height=550
)

# =========================================================

# 모델 해석 안내

# =========================================================

st.subheader(
"💡 이번 분석에서 확인하는 것"
)

st.write(
"""
이번 분석에서는 영화가 처음 10위권에 진입하여 해당 날짜의
집계가 끝난 시점까지 알 수 있는 정보를 사용합니다.

```
기본 변수 3개에 **상영당 관객 수**를 추가했을 때 R²가 얼마나
달라지는지를 비교함으로써, 처음 등장한 날의 상영 효율을
나타내는 변수가 최종 총 관객 수를 설명하는 데 얼마나 도움이
되는지 살펴볼 수 있습니다.
"""
```

)

# =========================================================

# 주의사항

# =========================================================

st.warning(
"""
⚠️ **주의**

```
이 데이터는 실제 영화가 개봉된 뒤 수집된 박스오피스
사후 집계 데이터입니다.

따라서 이 결과를 실제 개봉 전에 영화를 대상으로 한
'흥행 예측 성능'으로 해석해서는 안 됩니다.

이번 분석은 영화가 처음 10위권에 등장한 시점까지
관측된 정보를 이용했을 때, 최종 총 관객 수와 어떤 관계가
있는지를 비교하는 분석입니다.
"""
```

)

# =========================================================

# 변수 설명

# =========================================================

with st.expander("🔎 사용한 변수 자세히 보기"):

```
st.write(
    "### 기본 모델"
)

for feature in basic_features:

    st.write(
        f"- **{feature_names[feature]}**"
    )

st.write(
    "### 확장 모델"
)

for feature in extended_features:

    st.write(
        f"- **{feature_names[feature]}**"
    )

st.write(
    "### 상영당 관객 수 계산"
)

st.write(
    "**첫 등장일 일관객 ÷ 첫 등장일 상영횟수**"
)

st.write(
    "예를 들어 처음 10위권에 등장한 날 10,000명이 관람했고 "
    "500회 상영했다면 상영당 관객 수는 20명입니다."
)
```

````

그리고 `requirements.txt`는 그대로 아래 5개면 돼.

```text
streamlit
pandas
numpy
scikit-learn
plotly
````

**이번 버전에서 중요한 변경점**은 `first_week_audi`를 쓰지 않고, **첫 10위권 등장일의 `일관객 ÷ 상영횟수`로 만든 `상영당_관객수`**를 추가했다는 거야.
따라서 이번에는 "첫 주가 끝난 뒤"가 아니라 **첫 관측일의 집계가 끝난 시점**이라는 가정에 맞춰 비교할 수 있어.
