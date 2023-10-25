import pdb
import os
import openai
import streamlit as st
from process_channels import process_channels
import validators
import contextlib
from functools import wraps
from io import StringIO
import redirect as rd

#TODO: give app name which displays in the tab

with st.sidebar:
    model_name = st.selectbox(
        'Which LLM you prefer to use?',
        ('GPT-3.5-turbo-16k: Cost effective', 'GPT-4: Precise but costly'))

    model_name = model_name.split(":")[0].lower()

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    os.environ["OPENAI_API_KEY"] = openai_api_key

st.title("InfoScribe: Your Personal Video News Reporter")

st.markdown(
    "Welcome to InfoScribe, your go-to web app for staying informed and up-to-date with the latest videos from your favorite YouTube channels. "
    "In today's fast-paced digital world, it can be challenging to keep track of the wealth of information available online. That's where InfoScribe comes to your rescue!")
st.markdown(
    "Are you tired of spending hours sifting through YouTube videos trying to find the information that matters most to you? Do you want a personalized news reporter"
    " that highlights the crucial details from the videos you care about? Look no further. InfoScribe is here to simplify your information consumption process and provide you with a tailored news experience like never before.")

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
                                     placeholder="nutrition, OpenAI, Israel")

        if len(search_terms) == 0:
            raise Exception
    except Exception as e:
        st.error("If you don't enter search term, a general summary will be returned for all videos.")

    return_sources = st.toggle("Return sources")

    submitted = st.form_submit_button("Submit")

    to_out = st.empty()

    if not openai_api_key:
        st.info("Please add your OpenAI key in the sidebar to continue.")
    elif submitted:

        if not search_terms == "":
            search_terms = search_terms.split(",")
            search_terms = [term.strip() for term in search_terms]
        else:
            search_terms = None

        with rd.stdout(to=to_out, format="markdown"):
            _ = process_channels(youtube_channels,
                                 last_n_weeks,
                                 search_terms,
                                 return_sources,
                                 model_name)
