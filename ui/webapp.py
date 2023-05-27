import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st

from utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, get_mevzuat_url

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "Eve izinsiz girmenin cezası nedir?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "lol")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "3"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "3"))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", str(Path(__file__).parent / "eval_labels_example.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def main():

    st.set_page_config(page_title="DeepLex Demo")

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    
    st.markdown("""<h1 style='font-size: 5rem; font-weight: 600; letter-spacing: -.3rem; cursor: pointer; color: #2596f4;'>DeepLex</h1>
                    <h2 style='font-size: 3rem; font-weight: 400; letter-spacing: -.01rem; cursor: pointer; color: #2596f4;'>Akıllı Mevzuat Arama Motoru</h2>""", unsafe_allow_html=True)


    # Load csv into pandas dataframe
    try:
        df = pd.read_csv(EVAL_LABELS, sep=";")
    except Exception:
        st.error(
            f"The eval file was not found. Please check the demo's [README](https://github.com/deepset-ai/haystack/tree/main/ui/README.md) for more information."
        )
        sys.exit(
            f"The eval file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/main/ui/README.md) for more information."
        )

    # Search bar
    question = st.text_input(
        value=st.session_state.question,
        max_chars=100,
        on_change=reset_results,
        label="question",
        label_visibility="hidden",
    )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Ara")

    # Get next random question from the CSV
    if col2.button("Rastgele Soru"):
        reset_results()
        new_row = df.sample(1)
        while (
            new_row["Question Text"].values[0] == st.session_state.question
        ):  # Avoid picking the same question twice (the change is not visible on the UI)
            new_row = df.sample(1)
        st.session_state.question = new_row["Question Text"].values[0]
        st.session_state.answer = new_row["Answer"].values[0]
        st.session_state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        if hasattr(st, "scriptrunner"):
            raise st.scriptrunner.script_runner.RerunException(
                st.scriptrunner.script_requests.RerunData(widget_states=None)
            )
        raise st.runtime.scriptrunner.script_runner.RerunException(
            st.runtime.scriptrunner.script_requests.RerunData(widget_states=None)
        )
    st.session_state.random_question_requested = False

    run_query = (
        run_pressed or question != st.session_state.question
    ) and not st.session_state.random_question_requested

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Uygulama başlatılıyor..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Bağlantı hatası.")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question

        with st.spinner(
            "🧠 &nbsp;&nbsp; Binlerce doküman üzerinde yapay zeka arama yapıyor... \n "
           
        ):
            try:
                st.session_state.results = query(question)
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if st.session_state.results:

        st.write("## Sonuçlar:")

        for count, result in enumerate(st.session_state.results):
            if result["content"]:
                content = result["content"]
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.markdown(content, unsafe_allow_html=True)
                source = ""
                url, title = result['meta']['url'], result['meta']['mevAdi']
                if url and title:
                    source = f"[{title}]({get_mevzuat_url(url, content)})"
                else:
                    source = f"{result['source']}"
                st.markdown(f"**Kaynak:** {source}")
                st.markdown(f"Skor: {'{0:.3g}'.format(result['score'])}")
            else:
                st.info(
                    "🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
                st.write("**Relevance:** ", result["score"])
       
            st.write("___")


main()
