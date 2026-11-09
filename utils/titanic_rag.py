from __future__ import annotations

import json
import math
from pathlib import Path
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


def load_fact_documents(path: str | Path) -> list[dict[str, str]]:
    """Read the assignment-style Titanic fact table for RAG."""
    table = pd.read_csv(path, keep_default_na=False)
    required_columns = {
        "사실ID",
        "내용",
        "숫자",
        "기준(대상·기간·단위)",
    }
    missing = required_columns.difference(table.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"사실표에 필요한 열이 없어: {missing_names}")

    documents = []
    for row in table.to_dict(orient="records"):
        fact_id = str(row["사실ID"]).strip()
        content = str(row["내용"]).strip()
        numbers = str(row["숫자"]).strip()
        basis = str(row["기준(대상·기간·단위)"]).strip()
        documents.append({
            "id": fact_id,
            "title": content,
            "text": f"내용: {content} 숫자: {numbers} 기준: {basis}",
            "kind": "fact",
            "category": fact_id,
        })
    return documents


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
