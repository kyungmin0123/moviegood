import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
page_title="영화 흥행 예측기",
page_icon="🎬",
layout="wide"
)

st.title("🎬 영화 흥행 예측기")
st.write(
"영화가 처음 10위권에 진입한 날의 정보를 이용하여 "
"최종 총 관객 수를 예측합니다."
)

DAILY_URL = (
"https://raw.githubusercontent.com/greatsong/modudata/main/data/"
"kobis_daily.csv"
)

MOVIES_URL = (
"https://raw.githubusercontent.com/greatsong/modudata/main/data/"
"kobis_movies.csv"
)

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

for col in [
"일관객",
"상영횟수",
"스크린수",
"순위"
]:
daily[col] = pd.to_numeric(
daily[col],
errors="coerce"
)

for col in [
"first_scrn",
"first_show",
"first_week_audi",
"total_audi",
"days_in_top10",
"peak"
]:
movies[col] = pd.to_numeric(
movies[col],
errors="coerce"
)

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

first_daily["상영당_관객수"] = np.where(
first_daily["첫관측_상영횟수"] > 0,
first_daily["첫관측_일관객"]
/ first_daily["첫관측_상영횟수"],
np.nan
)

df = movies.merge(
first_daily,
left_on="movieCd",
right_on="영화코드",
how="left"
)

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

model_df = df.dropna(
subset=["total_audi"]
).copy()

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

valid_dates = daily["날짜"].dropna()

if len(valid_dates) > 0:
min_date = valid_dates.min()
max_date = valid_dates.max()

```
st.info(
    f"📅 기준 기간: "
    f"{min_date.strftime('%Y-%m-%d')} ~ "
    f"{max_date.strftime('%Y-%m-%d')}"
)
```

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
"영화코드 순으로 정렬한 뒤 10편마다 앞 3편을 시험용으로 "
"떼어 놓고, 나머지 7편을 학습용으로 사용했습니다."
)

st.subheader("🆕 새 변수: 상영당 관객 수")

st.info(
"상영당 관객 수 = "
"첫 10위권 등장일의 일관객 ÷ "
"첫 10위권 등장일의 상영횟수"
)

st.write(
"영화가 처음 10위권에 들어온 날 한 번 상영할 때 "
"평균적으로 몇 명의 관객이 관람했는지를 나타냅니다."
)

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

def run_regression(features):

```
X_train = train_df[features].copy()
X_test = test_df[features].copy()

y_train = train_df["total_audi"].copy()
y_test = test_df["total_audi"].copy()

model = Pipeline(
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

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
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

return model, prediction, r2, mae
```

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

st.subheader("1️⃣ 두 모델의 R² 비교")

col1, col2 = st.columns(2)

with col1:

```
st.markdown("### 🔵 기본 변수 3개")

st.write(
    "첫 관측일 스크린수"
)

st.write(
    "첫 관측일 상영횟수"
)

st.write(
    "성수기 개봉 여부"
)

st.metric(
    "R²",
    f"{basic_r2:.3f}"
)
```

with col2:

```
st.markdown("### 🟢 상영당 관객 수 추가")

st.write(
    "첫 관측일 스크린수"
)

st.write(
    "첫 관측일 상영횟수"
)

st.write(
    "성수기 개봉 여부"
)

st.write(
    "상영당 관객 수"
)

st.metric(
    "R²",
    f"{extended_r2:.3f}"
)
```

r2_difference = (
extended_r2
- basic_r2
)

st.markdown("### 📈 R² 변화")

if r2_difference > 0:

```
st.success(
    f"상영당 관객 수를 추가하면 "
    f"R²가 {r2_difference:+.3f} 변화합니다."
)
```

elif r2_difference < 0:

```
st.warning(
    f"상영당 관객 수를 추가하면 "
    f"R²가 {r2_difference:+.3f} 변화합니다."
)
```

else:

```
st.info(
    "두 모델의 R²가 같습니다."
)
```

st.subheader("2️⃣ 평균 절대 오차")

mae1, mae2 = st.columns(2)

with mae1:

```
st.metric(
    "기본 모델 MAE",
    f"{basic_mae:,.0f}명"
)
```

with mae2:

```
st.metric(
    "상영당 관객 수 추가 모델 MAE",
    f"{extended_mae:,.0f}명"
)
```

st.warning(
"⚠️ 이 데이터는 영화가 개봉된 뒤 수집된 "
"사후 집계 데이터입니다. "
"따라서 이 결과는 실제 개봉 전에 흥행을 예측했을 때의 "
"성능을 의미하지 않습니다. "
"이번 분석은 영화가 처음 10위권에 등장한 시점까지 "
"알 수 있는 정보를 이용했을 때 최종 총 관객 수와의 "
"관계를 비교하는 분석입니다."
)

st.subheader(
"3️⃣ 실제 총 관객 수와 예측 총 관객 수"
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
plot_df["total_audi"] > 0
].copy()

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
hovermode="closest"
)

st.plotly_chart(
fig,
use_container_width=True
)

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
    f"🔵 기본 모델: 예측 1,000명 미만 "
    f"{basic_low}편"
)
```

with low2:

```
st.info(
    f"🟢 확장 모델: 예측 1,000명 미만 "
    f"{extended_low}편"
)
```

st.subheader("4️⃣ 시험용 영화별 결과")

result = test_df[
[
"movieCd",
"movieNm",
"total_audi",
"상영당_관객수"
]
].copy()

result["기본 모델 예측"] = (
basic_prediction
)

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

st.subheader("5️⃣ 변수 설명")

st.write(
"기본 모델: "
"첫 관측일 스크린수, 첫 관측일 상영횟수, 성수기 개봉 여부"
)

st.write(
"확장 모델: "
"기본 모델의 세 변수 + 상영당 관객 수"
)

st.write(
"상영당 관객 수: "
"첫 10위권 등장일의 일관객을 그날의 상영횟수로 나눈 값"
)
