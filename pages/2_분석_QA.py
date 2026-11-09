from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from openai import OpenAI

from utils.titanic_rag import (
    EMBEDDING_MODEL,
    answer_question,
    build_fact_documents,
    cache_key_for_documents,
    cosine_similarity,
    documents_for_channel,
)


st.set_page_config(page_title="타이타닉 분석 Q&A", page_icon="🚢", layout="wide")
st.title("타이타닉 분석 Q&A")
st.caption("질문과 가까운 사실을 임베딩 검색으로 찾고, 그 근거 안에서 답해.")

channel_labels = {
    "A": "A. 분석 사실 확인 (사실표)",
    "B": "B. 분석 + 모델 안내 (사실표 + 모델 설명)",
    "C": "C. 분석 + 해석 기준 (사실표 + 모델 설명 + 해석 기준)",
}
selected_channel = st.radio(
    "답변에 사용할 자료",
    options=list(channel_labels),
    format_func=channel_labels.get,
    horizontal=True,
)

app_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(app_folder, "titanic.csv")
titanic = pd.read_csv(csv_path)
facts = build_fact_documents(titanic)
documents = documents_for_channel(facts, selected_channel)

try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
except (FileNotFoundError, AttributeError):
    api_key = ""
api_key = api_key or os.getenv("OPENAI_API_KEY", "")

if not api_key:
    st.warning("RAG 검색과 답변 생성에 OpenAI API 키가 필요해.")
    with st.expander("API 키 설정 방법"):
        st.markdown(
            "로컬 실행은 프로젝트의 .streamlit/secrets.toml에 "
            "OPENAI_API_KEY = \"키값\" 을 추가해.\n\n"
            "Streamlit Community Cloud에서는 앱의 Settings → Secrets에 "
            "OPENAI_API_KEY = \"키값\" 을 등록해. 키는 코드나 GitHub에 올리지 마."
        )
    st.stop()

client = OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False, ttl=3600)
def embed_documents(
    key: str,
    serialized_documents: str,
) -> tuple[list[dict[str, str]], list[list[float]]]:
    docs = json.loads(serialized_documents)
    embedding_client = OpenAI(api_key=key)
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[f"{doc['title']}\n{doc['text']}" for doc in docs],
    )
    vectors = [item.embedding for item in response.data]
    return docs, vectors


serialized_documents = cache_key_for_documents(documents)

try:
    with st.spinner("자료를 불러오고 의미 검색용 임베딩을 준비하고 있어..."):
        indexed_documents, document_vectors = embed_documents(
            api_key,
            serialized_documents,
        )
except Exception as error:
    st.error(f"자료 임베딩을 준비하지 못했어. API 키와 연결 상태를 확인해 줘. ({error})")
    st.stop()

if (
    st.session_state.get("conversation_channel") != selected_channel
    or "messages" not in st.session_state
):
    st.session_state["conversation_channel"] = selected_channel
    st.session_state["messages"] = []

with st.expander("예시 질문"):
    st.markdown(
        "- 여성과 남성 승객의 생존율은 얼마나 차이가 나?\n"
        "- 3등급 승객의 생존율은 다른 등급과 비교해 어때?\n"
        "- 사망자 중 가장 높은 비율을 차지하는 연령대는?"
    )

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("찾아온 근거 보기"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['id']}] {source['title']}**  \n"
                        f"{source['text']}"
                    )

question = st.chat_input("타이타닉 데이터에 관해 질문해")
if question:
    st.session_state["messages"].append({
        "role": "user",
        "content": question,
    })
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("관련 근거를 찾고 답을 정리하고 있어..."):
            query_vector = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[question],
            ).data[0].embedding

            ranked_documents = []
            for document, vector in zip(indexed_documents, document_vectors):
                ranked_documents.append({
                    **document,
                    "score": cosine_similarity(query_vector, vector),
                })
            ranked_documents.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            retrieved = ranked_documents[:4]

            history = [
                {"role": item["role"], "content": item["content"]}
                for item in st.session_state["messages"][:-1]
            ]
            answer = answer_question(
                client,
                question,
                retrieved,
                history=history,
            )

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        })
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("찾아온 근거 보기"):
                for source in retrieved:
                    st.markdown(
                        f"**[{source['id']}] {source['title']}**  \n"
                        f"{source['text']}"
                    )
    except Exception as error:
        st.error(
            "검색이나 답변 생성 중 문제가 생겼어. API 사용량과 키 설정을 확인해 줘. "
            f"({error})"
        )
