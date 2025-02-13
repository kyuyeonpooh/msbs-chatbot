import textwrap

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Chat System Related ==================================================================

MODEL = "gpt-4o"
TEMPERATURE = 0.1

EMBEDDING_MODEL = "text-embedding-3-large"
FAISS_STORE_DIR = "faiss"
FAISS_STORE_INDEXES = [
    "demo_rag_txt",
    "demo_rag_ocr",
]
RETRIEVE_TOP_K = 8

EXAMPLE_QUESTION_FILE = "dataset/example_questions.txt"
PROMPT_TEMPLATE = """
    You are an assistant for answering questions based on equipment manuals.
    Use the following retrieved context to answer the question.
    If the answer is unclear, try to deduce it from the provided contexts, or state that you don't know.
    Avoid using the context if it appears irrelevant or deteriorated.
    Provide your answer in Korean and format it as Markdown if applicable.
    Please do not include the markdown table as code format in your answer.
    If possible, please mention the file name and its page number in code text.
    If you are unsure about the context, you can ask for the file name or machine name to user.

    ## Context 1:
    {context_ocr}

    ## Context_2:
    {context_txt}

    ## Question:
    {question}

    ## Answer:
"""  # noqa: E501
PROMPT_TEMPLATE = textwrap.dedent(PROMPT_TEMPLATE).strip()

# UI Related ===========================================================================

LOGO_URL_LARGE = "image/logo_large.png"
LOGO_URL_SMALL = "image/logo_small.png"

AVATAR = {
    "user": "🧑🏻‍🔧",
    "ai": "🤖",
}

# ======================================================================================


def main():
    # Set page config and logo
    st.set_page_config(
        page_title="MITSUBISHI Chatbot",
        page_icon=LOGO_URL_SMALL,
    )
    st.logo(
        image=LOGO_URL_LARGE,
        link="https://kr.mitsubishielectric.com/fa/ko/index.do",
        size="large",
    )

    # Left align the text inside the button
    st.markdown(
        """
            <style>
                div.stButton > button {
                    text-align: left;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

    # Create columns for title logo and title text
    title_logo, title_txt = st.columns([0.1, 0.9], vertical_alignment="bottom")
    with title_logo:
        st.image(LOGO_URL_SMALL, use_container_width=True)
    with title_txt:
        st.header("MITSUBISHI :red[ELECTRIC] :gray[Chatbot]", divider="rainbow")

    # Set sidebar
    with st.sidebar:
        st.markdown("### :bulb: **질문 작성 팁!**")

        st.write(
            "대상 기기나 문서명을 질문에 포함해주시면, "
            "더 정확하고 디테일한 답변을 받으실 수 있어요."
        )

        st.markdown("### :memo: **이런 질문은 어떠신가요?**")

        button_clicked = []  # Whether buttons were clicked or not before rerun
        with open(EXAMPLE_QUESTION_FILE, encoding="utf-8") as f:
            questions = f.readlines()
            for question in questions:
                button_clicked.append(
                    st.button(
                        question,
                        key=question,
                        use_container_width=True,
                    )
                )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "ai",
                "content": "안녕하세요. 무엇을 도와드릴까요?",
            }
        ]

    # Initialize RAG chain
    if "chain" not in st.session_state:
        st.session_state.chain = load_chain()

    # Display past chat messages from the history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATAR[message["role"]]):
            st.markdown(message["content"])

    # Accept user input if available
    if user_input := st.chat_input("궁금하신 내용을 여기에 적어주세요."):
        print("Got user input:", user_input)
        run_chain(user_input)
    elif any(button_clicked):
        question_index = button_clicked.index(True)
        question = questions[question_index]
        print(f"Index of button clicked: {question_index}")
        print(f"Selected question: {question}")
        run_chain(question)


def load_chain() -> RunnableSequence:
    faiss_txt = FAISS.load_local(
        folder_path=FAISS_STORE_DIR,
        embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        index_name=FAISS_STORE_INDEXES[0],
        allow_dangerous_deserialization=True,
    )
    faiss_ocr = FAISS.load_local(
        folder_path=FAISS_STORE_DIR,
        embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        index_name=FAISS_STORE_INDEXES[1],
        allow_dangerous_deserialization=True,
    )

    retriever_txt = faiss_txt.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
    retriever_ocr = faiss_ocr.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})

    llm = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {
            "context_ocr": retriever_ocr,
            "context_txt": retriever_txt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def run_chain(question: str) -> str:
    # Display user message in chat message container
    with st.chat_message("user", avatar=AVATAR["user"]):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )
    # Visualize AI response

    with (
        st.chat_message("ai", avatar=AVATAR["ai"]),
        st.container(),  # This `st.container()` removes double ghosting effect of AI response
        st.spinner("답변을 생성하고 있어요..."),
    ):
        response_gen = st.session_state.chain.stream(question)
        full_response = st.write_stream(response_gen)
        st.session_state.messages.append(
            {
                "role": "ai",
                "content": full_response,
            }
        )


if __name__ == "__main__":
    print("Main function called.")
    main()
