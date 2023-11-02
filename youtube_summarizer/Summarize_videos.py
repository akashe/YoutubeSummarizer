import os
import streamlit as st
import validators
import redirect as rd
import asyncio
import openai

from process_videos import process_videos


def ui_spacer(n=2, line=False, next_n=0):
    for _ in range(n):
        st.write('')
    if line:
        st.tabs([' '])
    for _ in range(next_n):
        st.write('')


with st.sidebar:
    ui_spacer(25)
    st.markdown(f"""
    ## YouTube Insight
    """)
    st.write("Made by [Akash Kumar](https://www.linkedin.com/in/akashkumar2/).", unsafe_allow_html=True)
    # ui_spacer(1)
    st.markdown('Source code can be found [here](https://github.com/akashe/YoutubeSummarizer/tree/dev).')

st.header("YouTube Insight: Summarizing Videos For You")
ui_spacer(1)

st.markdown(
    "Welcome to YouTube Insight! Extract key information from any YouTube video swiftly and efficiently. "
    "Simply paste the URL, plug in your search terms, and get either a general or "
    "specific summary. Handle multiple videos simultaneously and save time!"
)
ui_spacer(1)

with st.expander("Settings"):
    model_name = st.selectbox(
        'Which LLM you prefer to use?',
        ('GPT-3.5-turbo-16k: Cost effective', 'GPT-4: Precise but costly'))

    model_name = model_name.split(":")[0].lower()

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    openai.api_key = openai_api_key

with st.form("Analysis"):
    try:
        video_links = st.text_input("Enter Comma-Separated YouTube video urls",
                                    placeholder="https://www.youtube.com/watch?v=dBH3nMNKtaQ, "
                                                "https://youtu.be/pGsbEd6w7PI?si=QyK7UMybPx1_Was2")

        video_links = video_links.strip().split(",")
        video_links = [video.strip() for video in video_links if video != ""]
        video_links = [video.split("&")[0] for video in video_links]

        print(video_links)

        url_check = [validators.url(video) for video in video_links]
        video_check = ["watch" in video or "youtu.be" in video for video in video_links]

        if not all(url_check) or not all(video_check):
            raise Exception

    except Exception as e:
        st.error('Please enter valid youtube video urls separated by comma')

    search_terms = st.text_input("Enter Topic(s) For Custom Summary (leave blank for general summary)",
                                 placeholder="nutrition, OpenAI, Israel",
                                 help="Input topics, separated by commas, this will gather all related mentions from the "
                                      "videos for a focused summary.\n Try using GPT-4 for more than 1 topic.")

    return_sources = st.toggle("Return sources",
                               help="Get source urls in the combined summary")

    submitted = st.form_submit_button("Submit")

    to_out = st.empty()

    if not openai_api_key:
        st.info("Please add your OpenAI key in the Settings to continue.")
    elif submitted:

        if not search_terms == "":
            search_terms = search_terms.split(",")
            search_terms = [term.strip() for term in search_terms]
        else:
            search_terms = None

        with rd.stdout(to=to_out, format="markdown"):

            if len(video_links) == 0:
                print("Generating summaries using default options")
                video_links = ["https://www.youtube.com/watch?v=dBH3nMNKtaQ",
                               "https://youtu.be/pGsbEd6w7PI?si=QyK7UMybPx1_Was2"]

            _ = asyncio.run(
                process_videos(youtube_video_links=video_links,
                               search_terms=search_terms,
                               get_source=return_sources,
                               model_name=model_name)
            )
