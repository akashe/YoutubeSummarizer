import os
import streamlit as st
import validators
import redirect as rd
import asyncio

from process_videos import process_videos

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
    "Welcome to InfoScribe, your go-to web app for staying informed and up-to-date with the latest videos from your favorite YouTube videos. "
    "In today's fast-paced digital world, it can be challenging to keep track of the wealth of information available online. That's where InfoScribe comes to your rescue!")
st.markdown(
    "Are you tired of spending hours sifting through YouTube videos trying to find the information that matters most to you? Do you want a personalized news reporter"
    " that highlights the crucial details from the videos you care about? Look no further. InfoScribe is here to simplify your information consumption process and provide you with a tailored news experience like never before.")

with st.form("Process videos"):
    try:
        video_links = st.text_input("Enter list of youtube videos, separated by comma",
                       placeholder="https://www.youtube.com/watch?v=q8CHXefn7B4, https://www.youtube.com/watch?v=MVYrJJNdrEg")

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
            _ = process_videos(video_links,
                               search_terms,
                               return_sources,
                               model_name)
