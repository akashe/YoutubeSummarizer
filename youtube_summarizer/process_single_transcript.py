import pdb
import tiktoken
import logging

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound

from get_chain import get_model_max_len

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_single_transcript(video_url: str,
                              model_name: str = "gpt-4-1106-preview"):

    try:
        if "m.youtube" in video_url:
            video_id = link.split("&v=")[-1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1].split("?")[0]
        else:
            video_id = video_url.split("?v=")[1]
    except Exception as e:
        logger.info("Please Enter valid urls")
        print("Please Enter valid urls")
        return "-1"

    json_transcript = "Sorry! English transcripts unavailable for the video"
    try:
        json_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB'])
    except (TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound):
        logger.info(f'English Subtitles unavailable for the video')
        print("\n")
        print(f'English transcripts unavailable for the video')
    finally:

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

        return json_transcript
