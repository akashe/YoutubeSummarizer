from datetime import datetime, timedelta
import json
from typing import List, Tuple
import openai

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_adjusted_iso_date_time(summary_of_n_weeks: int) -> str:
    current_date = datetime.now()
    adjusted_date = current_date - timedelta(weeks=summary_of_n_weeks)

    return adjusted_date.strftime('%Y-%m-%dT%H:%M:%SZ')


def get_transcript_from_xml(transcript_in_xml: dict) -> str:
    text_transcript = "\n".join([i["text"] for i in transcript_in_xml])

    return text_transcript


def check_supported_models(model_name: str) -> bool:
    with open("model_config.json","r") as f:
        supported_models = json.load(f)

    return model_name.lower() in supported_models


def get_transcripts(video_ids: List[str], video_titles: List[str]) -> List[List[dict]]:

    transcripts = []
    for video_id, video_title in zip(video_ids, video_titles):
        try:
            json_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB'])
            transcripts.append(json_transcript)
        except (TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound):
            logger.info(f'Subtitles unavailable for the video "{video_title}"')
            print("\n")
            print(f'English transcripts unavailable for the video "{video_title}"')

    return transcripts


def http_connection_decorator(func):
    async def inner(*args, **kwargs):
        try:
            from aiohttp import ClientSession
            openai.aiosession.set(ClientSession())

            return await func(*args, **kwargs)

        except Exception as e:
            print("Something bad happened with the request. Please retry :)")
        finally:
            await openai.aiosession.get().close()

    return inner
