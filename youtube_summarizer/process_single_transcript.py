import pdb
import tiktoken
import logging
import random
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound

from get_chain import get_model_max_len

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

from utils import chars_processed_dict_for_failed_cases_with_no_processing


def process_single_transcript(video_url: str,
                              model_name: str = "gpt-4o-mini"):

    try:
        if "m.youtube" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1].split("?")[0]
        else:
            video_id = video_url.split("?v=")[1].split('&')[0]
    except Exception as e:
        msg = "Enter valid urls"
        print(msg)
        logger.error(msg)
        return chars_processed_dict_for_failed_cases_with_no_processing, msg

    json_transcript = "Sorry! English transcripts unavailable for the video"
    try:
        username = st.secrets["proxy_username"]
        password = st.secrets["proxy_password"]

        ports = ["10001", "10002", "10003", "10004", "10005", "10006", "10007", "10008", "10009", "10010"]
        port = random.choice(ports)

        proxy = f"http://{username}:{password}@gate.smartproxy.com:{port}"
        logger.info(f'proxy: {proxy}')

        proxies = {
            'http': proxy,
            'https': proxy
        }

        json_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB'], proxies=proxies)
    except (TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound) as e:
        logger.info(f'Subtitle error {e}')
        logger.info(f'English Subtitles unavailable for the video')
        print("\n")
        print(f'English transcripts unavailable for the video')
    finally:
        logger.info(f'Using subtitles {len(json_transcript)}')

        input_chars_processed = 0
        if isinstance(json_transcript, list):
            text = [d['text'] for d in json_transcript]
            json_transcript = " ".join(text)

            # checking to see if the length is too long
            enc = tiktoken.encoding_for_model(model_name)
            model_max_token_len = get_model_max_len(model_name)

            # removing additional tokens to take care of past chat history
            model_max_token_len = model_max_token_len - 15000

            tokens = enc.encode(json_transcript)

            if len(tokens) > model_max_token_len:
                logger.info(f'The video is too long. Processing a smaller part of it.')
                print(f'The video is too long. Processing a smaller part of it.')

                json_transcript = enc.decode(tokens[:model_max_token_len])

            input_chars_processed += len(json_transcript)

        total_char_len_processed = {
            "input_chars": input_chars_processed,
            "output_chars": 0
        }

        return total_char_len_processed, json_transcript
