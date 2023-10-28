import pytest
import os

import sys
sys.path.append(os.path.join(os.getcwd(), "youtube_summarizer"))

pytest_plugins = ('pytest_asyncio',)

from youtube_summarizer.process_videos import process_videos

youtube_video_links = ["https://www.youtube.com/watch?v=MVYrJJNdrEg", "https://www.youtube.com/watch?v=e8qJsk1j2zE",
                       "https://youtu.be/m8LnEp-4f2Y?si=ZgwnRQGp_DeztHdC", "https://m.youtube.com/watch?si=ZgwnRQGp_DeztHdC&v=m8LnEp-4f2Y&feature=youtu.be"]
search_terms = None
get_source = False
model_name = "gpt-3.5-turbo-16k"


@pytest.mark.asyncio
async def test_process_videos_without_search_terms():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_videos(youtube_video_links, search_terms, get_source, model_name)

    assert result != ""
    assert len(result) > 0


@pytest.mark.asyncio
async def test_process_videos_with_search_terms():
    assert "OPENAI_API_KEY" in os.environ

    terms = ["India"]

    result = await process_videos(youtube_video_links, terms, get_source, model_name)

    assert result != ""
    assert len(result) > 0
    assert "India" in result


@pytest.mark.asyncio
async def test_process_videos_with_faulty_urls():
    assert "OPENAI_API_KEY" in os.environ

    url = ["www.akash.kumar"]

    result = await process_videos(url, search_terms, get_source, model_name)

    assert result == "-1"


@pytest.mark.asyncio
async def test_process_videos_with_sources():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_videos(youtube_video_links, search_terms, True, model_name)

    assert result != "-1"
    assert len(result) > 0


@pytest.mark.asyncio
async def test_process_videos_with_sources_and_search_terms():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_videos(youtube_video_links, ["India"], True, model_name)

    assert result != "-1"
    assert len(result) > 0
    assert "India" in result
    assert "youtube" in result


@pytest.mark.asyncio
async def test_process_videos_with_very_long_videos_with_combined_summary():
    assert "OPENAI_API_KEY" in os.environ

    result = await process_videos(["https://youtu.be/fUEjCXpOjPY?si=525ZRG13I4BO6Jdi",
                                   "https://youtu.be/eTBAxD6lt2g?si=1tNKf1zHSFFh7LfP",
                                   "https://youtu.be/34wA_bdG6QQ?si=pshVoVOvMb6hIaTE,"
                                   "https://youtu.be/co_MeKSnyAo?si=XDC0VCgM54xm-0Gx"],
                                  ["relationships", "Palestine"],
                                  True,
                                  model_name)

    assert result is not None
    assert len(result) > 10
    assert "Palestine" in result
    assert "youtube" in result

