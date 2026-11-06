from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="타이타닉 생존 대시보드", layout="wide")

app_folder = Path(__file__).resolve().parent
model_path = app_folder / "titanic_model.pkl"
csv_path = app_folder / "titanic.csv"

st.title("타이타닉 생존 대시보드")

titanic = pd.read_csv(csv_path)
sex_labels = {"female": "여성", "male": "남성"}


def summarize_survival(records, group_column):
    summary = records.groupby(group_column)["survived"].agg(
        total="count",
        survivors="sum",
    )
    summary = summary.loc[summary["total"] > 0].copy()
    summary["rate"] = summary["survivors"] / summary["total"] * 100
    return summary


def show_survival_chart(summary):
    st.bar_chart(summary["rate"].rename("생존율 (%)"))

    for label, row in summary.iterrows():
        st.write(
            f"**{label}** · 전체 {row['total']:,.0f}명 중 "
            f"{row['survivors']:,.0f}명 생존 ({row['rate']:.1f}%)"
        )


# 첫째 줄: 성별과 탑승항
sex_column, port_column = st.columns(2)

with sex_column:
    st.header("성별 생존율")

    selected_class = st.selectbox(
        "객실 등급",
        ["전체", 1, 2, 3],
        format_func=lambda value: "전체" if value == "전체" else f"{value}등급",
        key="survival_class_filter",
    )

    filtered = titanic
    if selected_class != "전체":
        filtered = titanic.loc[titanic["pclass"] == selected_class]

    sex_survival = summarize_survival(filtered, "sex").rename(index=sex_labels)

    st.caption("선택한 등급에서 성별과 생존 여부가 기록된 승객 기준")
    show_survival_chart(sex_survival)

with port_column:
    st.header("탑승항별 생존율")

    known_ports = titanic.dropna(subset=["embark_town", "survived"])
    port_survival = summarize_survival(known_ports, "embark_town")

    st.caption(f"탑승항과 생존 여부가 모두 기록된 {len(known_ports):,}명 기준")
    show_survival_chart(port_survival.sort_values("rate", ascending=False))

st.divider()

# 둘째 줄: 가족 동반 여부와 연령
family_column, age_column = st.columns(2)

with family_column:
    st.header("가족 동반 여부별 생존율")

    family_records = titanic.dropna(subset=["alone", "survived"])
    family_survival = summarize_survival(family_records, "alone").rename(
        index={
            True: "가족 동반 기록 없음",
            False: "가족과 함께 탑승",
        }
    )

    st.caption(f"가족 동반 여부와 생존 여부가 기록된 {len(family_records):,}명 기준")
    show_survival_chart(family_survival)

with age_column:
    st.header("어린이와 성인의 생존율")

    age_records = titanic.dropna(subset=["age", "survived"]).copy()
    age_records["age_group"] = "성인"
    age_records.loc[age_records["age"] < 18, "age_group"] = "어린이"

    age_survival = summarize_survival(age_records, "age_group")
    excluded_count = len(titanic) - len(age_records)

    st.caption(
        f"어린이 만 18세 미만 · 성인 만 18세 이상 · 집계 {len(age_records):,}명 · "
        f"나이 또는 생존 여부 미기록 {excluded_count:,}명 제외"
    )
    show_survival_chart(age_survival)

    older_passengers = age_records.loc[age_records["age"] >= 60]
    older_total = len(older_passengers)
    older_survivors = int(older_passengers["survived"].eq(1).sum())

    if older_total > 0:
        older_rate = older_survivors / older_total * 100
        st.metric("60세 이상 생존율", f"{older_rate:.1f}%")
        st.write(f"전체 **{older_total}명** 중 **{older_survivors}명** 생존")
    else:
        st.info("집계할 수 있는 60세 이상 승객 기록이 없어.")

st.divider()

# 전체 너비: 순위 비교
st.header("생존율 1위와 생존자 수 1위는 같을까?")
st.caption("출처 titanic.csv · 전체 승객 기록을 성별과 객실 등급으로 비교")

groups = (
    titanic.groupby(["pclass", "sex"])["survived"]
    .agg(total="count", survivors="sum")
    .reset_index()
)
groups = groups.loc[groups["total"] > 0].copy()
groups["rate"] = groups["survivors"] / groups["total"] * 100

groups["집단"] = (
    groups["pclass"].astype(str)
    + "등급 "
    + groups["sex"].replace(sex_labels)
)
groups["생존자 / 전체"] = (
    groups["survivors"].astype(int).astype(str)
    + "명 / "
    + groups["total"].astype(str)
    + "명"
)
groups["생존율"] = groups["rate"].map("{:.1f}%".format)

display_columns = ["집단", "생존자 / 전체", "생존율"]
rate_column, count_column = st.columns(2)

with rate_column:
    st.subheader("생존율 순위")
    st.dataframe(
        groups.sort_values("rate", ascending=False)[display_columns],
        hide_index=True,
    )

with count_column:
    st.subheader("생존자 수 순위")
    st.dataframe(
        groups.sort_values("survivors", ascending=False)[display_columns],
        hide_index=True,
    )

with st.expander("답 확인하기"):
    rate_leaders = groups.loc[groups["rate"] == groups["rate"].max(), "집단"]
    count_leaders = groups.loc[
        groups["survivors"] == groups["survivors"].max(), "집단"
    ]

    st.write(f"**생존율 1위** · {', '.join(rate_leaders)}")
    st.write(f"**생존자 수 1위** · {', '.join(count_leaders)}")

    if set(rate_leaders) == set(count_leaders):
        st.success("두 기준에서 1위인 집단이 같아.")
    else:
        st.info("두 기준에서 1위인 집단이 달라.")

st.divider()

# 전체 너비: 모델 예측
st.header("모델로 생존 여부 예측하기")
st.caption("입력한 승객 정보에 대한 학습 모델의 예측이야.")

with st.form("passenger_form"):
    age = st.number_input("나이", min_value=0.0, value=22.0)
    sex = st.selectbox("성별", ["male", "female"], format_func=sex_labels.get)
    pclass = st.selectbox("객실 등급", [1, 2, 3], index=2)
    fare = st.number_input("요금", min_value=0.0, value=7.25)

    submitted = st.form_submit_button("예측하기")

if submitted:
    model = joblib.load(model_path)

    passenger = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "pclass": [pclass],
            "fare": [fare],
        }
    )

    prediction = model.predict(passenger)[0]
    result = "생존" if prediction == 1 else "사망"

    st.info(f"모델 예측: {result}")