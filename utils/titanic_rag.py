from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

MODEL_GUIDE = {
    "id": "M01",
    "title": "생존 예측 모델 안내",
    "text": (
        "이 앱의 예측 모델은 승객의 나이, 성별, 객실 등급, 요금을 입력받아 "
        "생존 또는 사망을 분류한다. 예측은 학습 데이터에서 찾은 패턴을 적용한 "
        "결과이며 실제 개인의 운명을 확정하거나 생존 원인을 설명하지 않는다. "
        "모델 예측과 데이터에서 집계한 실제 생존율은 서로 다른 결과다."
    ),
    "kind": "model",
}

INTERPRETATION_GUIDE = {
    "id": "R01",
    "title": "분석 결과 해석 기준",
    "text": (
        "이 앱은 제공된 타이타닉 데이터의 관찰 결과를 설명한다. 집단 간 생존율 "
        "차이만으로 원인과 결과를 단정하지 않는다. 두 생존율의 차이는 퍼센트(%)가 "
        "아니라 퍼센트포인트(%p)로 표시한다. 나이 정보가 없는 승객은 나이별 "
        "집계에서 제외하며, 제외 인원과 분모를 함께 밝힌다. 자료에 근거가 없는 "
        "질문에는 확인할 수 없다고 답한다."
    ),
    "kind": "guide",
}


def _rate_text(survivors: int, total: int) -> str:
    if total == 0:
        return "계산할 수 없음(해당 기록 없음)"
    return f"{survivors / total * 100:.2f}%"


def _group_fact(
    data: pd.DataFrame,
    column: str,
    labels: dict[Any, str],
    fact_id: str,
    title: str,
) -> dict[str, str]:
    lines = []
    for value, label in labels.items():
        group = data.loc[data[column] == value, "survived"].dropna()
        total = len(group)
        survivors = int(group.sum()) if total else 0
        lines.append(
            f"{label}: 생존 {survivors}명 / 전체 {total}명, "
            f"생존율 {_rate_text(survivors, total)}."
        )
    return {
        "id": fact_id,
        "title": title,
        "text": " ".join(lines),
        "kind": "fact",
    }


def build_fact_documents(data: pd.DataFrame) -> list[dict[str, str]]:
    """Create cited evidence snippets directly from the app's Titanic CSV."""
    records = data.copy()
    records["survived"] = pd.to_numeric(records["survived"], errors="coerce")

    facts = [
        _group_fact(
            records,
            "sex",
            {"female": "여성 승객", "male": "남성 승객"},
            "F01",
            "성별 생존율",
        ),
        _group_fact(
            records,
            "pclass",
            {1: "1등급 승객", 2: "2등급 승객", 3: "3등급 승객"},
            "F02",
            "객실 등급별 생존율",
        ),
    ]

    if "embark_town" in records.columns:
        known_ports = records.dropna(subset=["embark_town", "survived"])
        port_lines = []
        for port, group in known_ports.groupby("embark_town", sort=False):
            total = len(group)
            survivors = int(group["survived"].sum())
            port_lines.append(
                f"{port}: 생존 {survivors}명 / 전체 {total}명, "
                f"생존율 {_rate_text(survivors, total)}."
            )
        facts.append({
            "id": "F03",
            "title": "탑승항별 생존율",
            "text": " ".join(port_lines)
            + f" 탑승항 정보가 확인되는 전체 인원은 {len(known_ports)}명이다.",
            "kind": "fact",
        })

    if "alone" in records.columns:
        family_data = records.dropna(subset=["alone", "survived"])
        alone_lines = []
        for value, label in [(False, "가족과 함께 탑승"), (True, "혼자 탑승")]:
            group = family_data.loc[family_data["alone"] == value, "survived"]
            total = len(group)
            survivors = int(group.sum()) if total else 0
            alone_lines.append(
                f"{label}: 생존 {survivors}명 / 전체 {total}명, "
                f"생존율 {_rate_text(survivors, total)}."
            )
        facts.append({
            "id": "F04",
            "title": "가족 동반 여부별 생존율",
            "text": " ".join(alone_lines),
            "kind": "fact",
        })

    if "age" in records.columns:
        age_data = records.dropna(subset=["age", "survived"]).copy()
        age_data["연령대"] = pd.cut(
            age_data["age"],
            bins=[0, 18, 30, 45, 60, math.inf],
            labels=["18세 미만", "18~29세", "30~44세", "45~59세", "60세 이상"],
            right=False,
        )
        age_lines = []
        for label, group in age_data.groupby("연령대", observed=False):
            total = len(group)
            survivors = int(group["survived"].sum()) if total else 0
            age_lines.append(
                f"{label}: 생존 {survivors}명 / 전체 {total}명, "
                f"생존율 {_rate_text(survivors, total)}."
            )
        missing_age = int(records["age"].isna().sum())
        facts.append({
            "id": "F05",
            "title": "연령대별 생존율과 결측",
            "text": " ".join(age_lines)
            + f" 나이 정보가 없어 연령별 집계에서 제외된 인원은 {missing_age}명이다.",
            "kind": "fact",
        })

        deaths = records.loc[records["survived"] == 0].dropna(subset=["age"]).copy()
        if not deaths.empty:
            deaths["연령대"] = pd.cut(
                deaths["age"],
                bins=[0, 18, 30, 45, 60, math.inf],
                labels=["18세 미만", "18~29세", "30~44세", "45~59세", "60세 이상"],
                right=False,
            )
            death_counts = deaths["연령대"].value_counts(sort=False)
            death_lines = [
                f"{age_group}: 사망 {int(count)}명, "
                f"나이가 확인되는 사망자 {len(deaths)}명 중 {count / len(deaths) * 100:.2f}%."
                for age_group, count in death_counts.items()
            ]
            facts.append({
                "id": "F06",
                "title": "나이가 확인되는 사망자의 연령대 구성",
                "text": " ".join(death_lines)
                + " 이 비율은 연령대 안에서의 사망률이 아니라, 나이가 확인되는 사망자 중 각 연령대가 차지하는 비율이다.",
                "kind": "fact",
            })

    total = int(records["survived"].notna().sum())
    survivors = int(records["survived"].sum())
    facts.append({
        "id": "F07",
        "title": "전체 생존 인원",
        "text": (
            f"생존 여부가 기록된 전체 승객은 {total}명이며, "
            f"생존자는 {survivors}명, 생존율은 {_rate_text(survivors, total)}이다."
        ),
        "kind": "fact",
    })
    return facts


def documents_for_channel(
    facts: list[dict[str, str]], channel: str
) -> list[dict[str, str]]:
    documents = list(facts)
    if channel in {"B", "C"}:
        documents.append(MODEL_GUIDE)
    if channel == "C":
        documents.append(INTERPRETATION_GUIDE)
    return documents


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve_documents(
    client: OpenAI,
    question: str,
    documents: list[dict[str, str]],
    limit: int = 4,
) -> list[dict[str, Any]]:
    document_texts = [f"{doc['title']}\n{doc['text']}" for doc in documents]
    vectors = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=document_texts,
    ).data
    query_vector = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    ).data[0].embedding

    ranked = []
    for document, vector in zip(documents, vectors):
        ranked.append({
            **document,
            "score": cosine_similarity(query_vector, vector.embedding),
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def answer_question(
    client: OpenAI,
    question: str,
    retrieved: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    evidence = "\n\n".join(
        f"[{item['id']}] {item['title']}\n{item['text']}"
        for item in retrieved
    )
    system_prompt = (
        "너는 타이타닉 데이터 분석 Q&A 도우미야. 답은 아래 검색된 근거 안에서만 해. "
        "근거에 필요한 정보가 없으면 '제공된 자료로는 확인할 수 없습니다.'라고 말해. "
        "수치 질문에는 분자와 분모 또는 계산 방법을 밝혀. 집단 간 생존율 차이는 "
        "퍼센트포인트(%p)로 표현해. 실제 관찰값과 모델 예측을 혼동하지 말고, "
        "근거 ID를 문장 끝에 [F01]처럼 붙여. 검색 근거에 없는 ID를 만들지 마. "
        "인과관계는 단정하지 말고 한국어로 간결하게 답해.\n\n"
        f"검색된 근거:\n{evidence}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=messages,
    )
    return response.choices[0].message.content or "답변을 만들지 못했어."


def cache_key_for_documents(documents: list[dict[str, str]]) -> str:
    return json.dumps(documents, ensure_ascii=False, sort_keys=True)
