import asyncio

import argparse
from typing import List

from youtube.get_information import YoutubeConnect
import openai

from utils import check_supported_models, http_connection_decorator, get_transcripts
from get_chain import aget_summary_of_each_video, get_documents, aget_summary_with_keywords

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

from openai_prompts import get_per_document_with_keyword_prompt_template,\
    get_combine_document_prompt_template,\
    get_per_document_prompt_template, \
    get_combine_document_with_source_prompt_template


@http_connection_decorator
async def process_a_playlist(
    youtube_playlist_name: str = "data",
    last_n_videos: int = 10,
    search_terms: List[str] = ["AGI", "history", "spirituality", "human pyschology", "new developments in science"],
    get_source: bool = False,
    model_name: str = "gpt-4"
) -> str:

    assert youtube_playlist_name.lower() != "watch later", "Watch later not accesible via YoutubeData API"

    youtube_connect = YoutubeConnect()

    playlist_id, total_videos = youtube_connect.get_playlist_id(youtube_playlist_name)

    total_video_to_process = (lambda x,y: x if y>=x else y)(total_videos,last_n_videos)

    logger.info(f"Found playlist id = {playlist_id} for the playlist name: {youtube_playlist_name}")
    logger.info(f"Playlist has {total_videos} videos")
    logger.info(f"Processing last {total_video_to_process} videos from the playlist")

    video_ids = youtube_connect.get_last_n_videos_from_playlist(playlist_id, total_video_to_process)

    video_titles = []
    for video_id in video_ids:
        video_titles.append(youtube_connect.get_video_title(video_id))

    transcripts = get_transcripts(video_ids, video_titles)

    documents = get_documents(video_ids, video_titles, transcripts, model_name)

    result = ""
    try:
        if search_terms:
            per_document_template = get_per_document_with_keyword_prompt_template(model_name)
            combine_document_template = get_combine_document_with_source_prompt_template(model_name) if get_source \
                else get_combine_document_prompt_template(model_name)

            result = await aget_summary_with_keywords(documents,
                                                      search_terms,
                                                      per_document_template,
                                                      combine_document_template,
                                                      model_name,
                                                      len(video_ids))
        else:
            per_document_template = get_per_document_prompt_template(model_name)
            result = await aget_summary_of_each_video(documents, per_document_template, model_name)
    except Exception as e:
        print("Something bad happened with the request. Please retry :)")
        return "-1"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('youtube_playlist_name', metavar='y', type=str,
                        help='name of your personal playlist')
    parser.add_argument('-n', '--last_n_videos', type=int, default=1,
                        help='Number of last n videos to process in the playlist')
    parser.add_argument('-s', '--search_terms', type=str, nargs='*',
                        help="design the summary around your topics of interest. If not given,"
                             "a general summary will be created.")
    parser.add_argument('--return_sources', action='store_true', default=False,
                        help="To return sources of information in the final summary.")
    parser.add_argument('--model_name', default='gpt-4',
                        help="model to use for generating summaries.")

    args = parser.parse_args()

    assert check_supported_models(args.model_name), "Model not available in config"

    asyncio.run(
        process_a_playlist(args.youtube_playlist_name, args.last_n_videos, args.search_terms, args.return_sources, args.model_name)
    )
