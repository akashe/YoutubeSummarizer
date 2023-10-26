import pdb
from typing import Tuple, List


def get_channel_id_from_list_username_response(response: dict) -> str:
    assert response["pageInfo"]["totalResults"] == 1
    channel_id = response["items"][0]["id"]

    return channel_id


def get_latest_video_ids_from_list_channel_activity_response(response: dict):
    if "items" in response and len(response["items"]) > 0:
        channel_ids = []
        for item in response["items"]:
            if "upload" in item["contentDetails"]:
                channel_ids.append(item["contentDetails"]["upload"]["videoId"])

        return channel_ids
    else:
        return None


def get_playlist_id_from_list_playlists(response: dict, custom_playlist_name: str) -> Tuple[str, int]:
    playlist_items = response["items"]
    assert len(playlist_items) >= 1, "You have no playlists in your account"

    id, total_videos = None, 0
    for item in playlist_items:
        if item["snippet"]["title"].lower() == custom_playlist_name.lower():
            id = item["id"]
            total_videos = item["contentDetails"]["itemCount"]

    assert id != None, "Playlist ID not found."
    assert total_videos > 0, "You have 0 videos in the playlist"

    return id, total_videos


def get_video_ids_from_playlist(response: dict, n: int) -> List[str]:
    video_details = response["items"]

    assert len(video_details) > 0, "No videos in the playlist"

    video_ids = []
    for item in video_details:
        video_id = item["contentDetails"]["videoId"]
        video_ids.append(video_id)

    # Taking last n videos
    video_ids = video_ids[-n:]

    return video_ids


def get_video_title_from_list_video_id(response: dict) -> str:
    video_details = response["items"]

    assert len(video_details) == 1, "Multiple information for video ids"

    title = video_details[0]["snippet"]["title"]

    return title
