# -*- coding: utf-8 -*-
import json
# Sample Python code for youtube.captions.download
# NOTE: This sample code downloads a file and can't be executed via this
#       interface. To data this sample, you must run it locally using your
#       own API credentials.

# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os
import re
import streamlit as st

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError

from typing import List, Tuple
import requests

from .parsers import get_channel_id_from_list_username_response, \
    get_latest_video_ids_from_list_channel_activity_response, \
    get_playlist_id_from_list_playlists, get_video_ids_from_playlist, \
    get_video_title_from_list_video_id


class YoutubeConnect:
    """
    Class to connect and get information to YouTube Data API.
    The class won't work without a credentials.json that contains the:
        1. client_id and secret of your Google Workspace project
        2. project_id: Name of the project in your workspace

    Refer to README file for steps to get credentials.json
    """

    scopes: List[str] = ["https://www.googleapis.com/auth/youtube.readonly"]
    api_service_name: str = "youtube"
    api_version: str = "v3"
    client_secrets_file: str = "credentials.json"

    def __init__(self):
        """
        Authenticate credentials and create a client for YouTube Data API.
        """

        streamlit_secret_file_present = os.path.exists(os.path.join(".streamlit/", "secrets.toml")) or os.path.exists(
            os.path.join("/.streamlit/", "secrets.toml"))

        if not os.path.exists(self.client_secrets_file) and \
                streamlit_secret_file_present and \
                "client_id" in st.secrets:
            data = {
                "web": {
                    "client_id": st.secrets["client_id"],
                    "project_id": st.secrets["project_id"],
                    "auth_uri": st.secrets["auth_uri"],
                    "token_uri": st.secrets["token_uri"],
                    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
                    "client_secret": st.secrets["client_secret"],
                    "token": st.secrets["redirect_uris"],
                }
            }

            with open(self.client_secrets_file, "w") as f:
                json.dump(data, f)
        else:
            assert os.path.exists(
                self.client_secrets_file), "Download credentials.json from your Google Workspace Project"

        creds = None
        if streamlit_secret_file_present and "token" in st.secrets:
            mapping = {
                "token": st.secrets["token"],
                "refresh_token": st.secrets["refresh_token"],
                "token_uri": st.secrets["token_uri"],
                "client_id": st.secrets["client_id"],
                "client_secret": st.secrets["client_secret"],
                "scopes": st.secrets["scopes"],
                "expiry": st.secrets["expiry"],
            }
            creds = Credentials.from_authorized_user_info(mapping, self.scopes)
        elif os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets_file, self.scopes)
                creds = flow.run_local_server(port=51369)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        self.youtube = googleapiclient.discovery.build(
            self.api_service_name, self.api_version, credentials=creds)

    def get_channel_id_from_username(self, username: str) -> str:
        # TODO: better error handling for requests, it should not just be exit(-1)
        try:
            request = self.youtube.channels().list(
                part="id",
                forUsername=username
            )
            response = request.execute()
        except Exception as e:
            print(f'An error occurred: {e}')
            exit(-1)
        # parsing the request
        channel_id = get_channel_id_from_list_username_response(response)
        return channel_id

    @staticmethod
    def get_channel_id_from_channel_link(link: str) -> str:

        response = requests.get(link)
        possible_channel_ids = re.findall(r'\?channel_id=(.*?)"', str(response.content))

        assert len(set(possible_channel_ids)) == 1, "Having possibilities of multiple channel"

        channel_id = possible_channel_ids[0]

        return channel_id

    def get_latest_videos(
            self,
            channel_id: str,
            publish_date_after: str,
            max_results: int = 50) -> List[str]:

        try:
            request = self.youtube.activities().list(
                part="contentDetails",
                channelId=channel_id,
                maxResults=max_results,
                publishedAfter=publish_date_after
            )
            response = request.execute()
        except Exception as e:
            print(f'An error occurred: {e}')
            exit(-1)

        new_videos = get_latest_video_ids_from_list_channel_activity_response(response)

        return new_videos

    def get_playlist_id(self, custom_playlist_name) -> Tuple[str, int]:
        try:
            request = self.youtube.playlists().list(
                part="snippet,contentDetails",
                maxResults=50,
                mine=True
            )
            response = request.execute()
        except Exception as e:
            print(f'An error occurred: {e}')
            exit(-1)

        playlist_id, total_videos = get_playlist_id_from_list_playlists(response, custom_playlist_name)

        return playlist_id, total_videos

    def get_video_title(self, video_id: str) -> str:
        try:
            request = self.youtube.videos().list(
                part="snippet",
                id=video_id
            )
            response = request.execute()
        except Exception as e:
            print(f'An error occurred: {e}')
            exit(-1)

        video_title = get_video_title_from_list_video_id(response)

        return video_title

    def get_last_n_videos_from_playlist(self, playlist_id, n) -> List[str]:

        assert n <= 50, "Cant get more than 50 videos"

        try:
            request = self.youtube.playlistItems().list(
                part="snippet, contentDetails",
                maxResults=50,
                playlistId=playlist_id
            )
            response = request.execute()
        except Exception as e:
            print(f'An error occurred: {e}')
            exit(-1)

        video_ids = get_video_ids_from_playlist(response, n)

        return video_ids
