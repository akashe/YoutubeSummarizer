import pdb

import asyncio
import argparse
from typing import List

from youtube.get_information import YoutubeConnect
import openai

from utils import check_supported_models, get_transcripts, chars_processed_dict_for_failed_cases_with_some_processing, \
    chars_processed_dict_for_failed_cases_with_no_processing

from get_chain import get_summary_of_each_video, \
    aget_summary_of_each_video, \
    get_documents, \
    get_summary_with_keywords, \
    aget_summary_with_keywords

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

from openai_prompts import get_per_document_with_keyword_prompt_template, \
    get_combine_document_prompt_template, \
    get_per_document_prompt_template, \
    get_combine_document_with_source_prompt_template


# ["https://www.youtube.com/@BeerBiceps", "https://www.youtube.com/@hubermanlab","https://www.youtube.com/@MachineLearningStreetTalk"]
# ["AGI", "history", "spirituality", "human pyschology", "new developments in science"]
async def process_videos(
        youtube_video_links: List[str] = ["https://www.youtube.com/watch?v=MVYrJJNdrEg",
                                          "https://www.youtube.com/watch?v=e8qJsk1j2zE",
                                          "https://youtu.be/m8LnEp-4f2Y?si=ZgwnRQGp_DeztHdC",
                                          "https://m.youtube.com/watch?si=ZgwnRQGp_DeztHdC&v=m8LnEp-4f2Y&feature=youtu.be"],
        search_terms: List[str] = None,
        get_source: bool = False,
        model_name: str = "gpt-4o-mini"
) -> (dict, str):
    youtube_connect = YoutubeConnect()

    # TO do check each link for correctness
    # "https://m.youtube.com/watch?si=ZgwnRQGp_DeztHdC&v=m8LnEp-4f2Y&feature=youtu.be"
    # "https://youtu.be/m8LnEp-4f2Y?si=ZgwnRQGp_DeztHdC"
    try:
        video_ids = []
        for link in youtube_video_links:
            if "m.youtube" in link:
                video_id = link.split("v=")[-1].split("&")[0]
            elif "youtu.be" in link:
                video_id = link.split("/")[-1].split("?")[0]
            else:
                video_id = link.split("?v=")[1].split('&')[0]
            video_ids.append(video_id)
    except Exception as e:
        msg = "Enter valid urls"
        print(msg)
        logger.error(msg)
        return chars_processed_dict_for_failed_cases_with_no_processing, msg

    logger.info(f"Analyzing {len(video_ids)} videos")
    print("\n")
    if len(video_ids) > 1:
        print(f"Analyzing {len(video_ids)} videos")
    else:
        print(f"Analyzing {len(video_ids)} video")

    print("\n")
    video_titles = []
    for id_, video_id in enumerate(video_ids):
        video_title = youtube_connect.get_video_title(video_id)
        video_titles.append(video_title)
        print(f"{id_ + 1}. [{video_title}](https://www.youtube.com/watch?v={video_id})")

    transcripts = get_transcripts(video_ids, video_titles)
    if len(transcripts) == 0:
        return chars_processed_dict_for_failed_cases_with_no_processing, "Transcripts not available"

    documents = get_documents(video_ids, video_titles, transcripts, model_name)

    result = ""
    try:
        if search_terms:
            per_document_template = get_per_document_with_keyword_prompt_template(model_name)
            combine_document_template = get_combine_document_with_source_prompt_template(model_name) if get_source \
                else get_combine_document_prompt_template(model_name)

            chars_processed, result = await aget_summary_with_keywords(documents, search_terms,
                                                                       per_document_template,
                                                                       combine_document_template,
                                                                       model_name,
                                                                       len(video_ids))

        else:
            per_document_template = get_per_document_prompt_template(model_name)
            chars_processed, result = await aget_summary_of_each_video(documents, per_document_template, model_name)
    except Exception as e:
        logger.error(e)
        msg = "Something bad happened with the request. Please retry :)"
        print(msg)
        return chars_processed_dict_for_failed_cases_with_some_processing, msg

    return chars_processed, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--youtube_video_links', nargs='+', metavar='y',
                        help='a list youtube channel links that you want to process')
    parser.add_argument('-s', '--search_terms', type=str, nargs='*',
                        help="design the summary around your topics of interest. If not given,"
                             "a general summary will be created.")
    parser.add_argument('--return_sources', action='store_true', default=False,
                        help="To return sources of information in the final summary.")
    parser.add_argument('--model_name', default='gpt-3.5-turbo-16k',
                        help="model to use for generating summaries.")

    args = parser.parse_args()
    print(args)

    assert check_supported_models(args.model_name), "Model not available in config"

    asyncio.run(
        process_videos(args.youtube_video_links, args.search_terms, args.return_sources, args.model_name)
    )
