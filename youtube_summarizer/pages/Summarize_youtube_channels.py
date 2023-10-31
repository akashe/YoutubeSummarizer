import os
import streamlit as st
from process_channels import process_channels
import validators
import redirect as rd
import asyncio
import openai


def ui_spacer(n=2, line=False, next_n=0):
    for _ in range(n):
        st.write('')
    if line:
        st.tabs([' '])
    for _ in range(next_n):
        st.write('')


with st.sidebar:
    ui_spacer(27)
    st.markdown(f"""
    ## YouTube Insight
    """)
    st.write("Made by [Akash Kumar](https://www.linkedin.com/in/akashkumar2/).", unsafe_allow_html=True)
    #ui_spacer(1)
    st.markdown('Source code can be found [here](https://github.com/akashe/YoutubeSummarizer/tree/dev).')

st.header("YouTube Insight: Summarizing Media For You")

st.markdown(
    "\nWelcome to YouTube Insight! Extract key information from any YouTube channel swiftly and efficiently. "
    "Simply paste the channel URL, specify timeframe, plug in your search terms, and get either a general or "
    "specific summary. Handle multiple channels simultaneously and save time with YouTube Insight!\n"
)

with st.expander("Settings"):
    model_name = st.selectbox(
        'Which LLM you prefer to use?',
        ('GPT-3.5-turbo-16k: Cost effective', 'GPT-4: Precise but costly'))

    model_name = model_name.split(":")[0].lower()

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    openai.api_key = openai_api_key

with st.form("YoutubeSummary"):
    try:
        youtube_channels = st.text_input("Enter list of youtube channels, separated by comma",
                                         placeholder="https://www.youtube.com/c/lexfridman, https://www.youtube.com/@hubermanlab")

        youtube_channels = youtube_channels.strip().split(",")
        youtube_channels = [channel.strip() for channel in youtube_channels if channel != ""]

        url_check = [validators.url(channel) for channel in youtube_channels]
        youtube_urls = ["youtube" in url for url in youtube_channels]

        if not all(url_check) or not all(youtube_urls):
            raise Exception
    except Exception as e:
        st.error('Please enter valid urls separated by comma')

    last_n_weeks = st.selectbox("Gather videos from the channels for last how many weeks?",
                                (1, 2, 3))
    try:
        search_terms = st.text_input("Enter topics you want summary for, separated by comma",
                                     placeholder="nutrition, OpenAI, Israel",
                                     help="Try using GPT-4 for more than 1 search term")

        if len(search_terms) == 0:
            raise Exception
    except Exception as e:
        st.error("If you don't enter search term, a general summary will be returned for all videos.")

    return_sources = st.toggle("Return sources")

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

            to_out.empty()

            if len(youtube_channels) == 0:
                print("Generating summaries using default options")
                youtube_channels = ["https://www.youtube.com/c/lexfridman",
                                    "https://www.youtube.com/@hubermanlab"]
                last_n_weeks = 3

            _ = asyncio.run(
                process_channels(youtube_channels,
                                 last_n_weeks,
                                 search_terms,
                                 return_sources,
                                 model_name)
            )
