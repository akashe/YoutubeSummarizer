## YouTube Buddy: Streamline Your YouTube Experience

[Click here to try it out!](https://youtubebuddy.streamlit.app/)

Get information from youtube videos the way you want it.

1. **Video Summary**
2. **Keyword Specific Summary**
3. **Channel Summaries**
4. **Ask question about video via chat**
5. **Create clips**


### Community tokens:
To allow people try out youtubebuddy we allocate open access worth 2.5$ compute costs daily. Once the community tokens are exhausted, users can use their own personal OpenAI access keys.
If you like the app, consider [buying us a coffee](https://buymeacoffee.com/akashe). This would help us cover compute costs.


### Steps to Run the App Locally

Running the app locally involves the following steps:

1. [Set up a Google cloud project in Google workspaces.](https://developers.google.com/workspace/guides/get-started)
2. [Turn on Youtube Data API v3 for your Google project.](https://www.youtube.com/watch?v=fN8WwVQTWYk)
3. [Create OAuth configuration for your project.](https://developers.google.com/workspace/guides/configure-oauth-consent)
4. [Generate access credentials for your Google cloud project.](https://developers.google.com/workspace/guides/create-credentials)
5. [Download 'credentials.json' from your project and keep it locally.](https://techiejackieblogs.com/how-to-create-google-mail-api-credentials-json/)
6. Execute 'streamlit run youtube_summarizer/chatbot.py'.
7. On initial execution, authenticate the project via a Gmail account.
8. This will locally save a 'token.json' file, containing the access token for future access without having to authenticate again.
9. Open the streamlit app in UI and start interacting via chat.


#### Entry points for code
We mostly maintain the codebase for the streamlit entrpoints which allow interacting with YoutubeBuddy via chat. We use Assistants API from OpenAI for the chatbot and use the 'gpt-4o-mini' model for all processing.
- **chatbot.py**: Streamlit UI to get information from youtube using a chatbot.

[Retired entrypoints]
- **process_videos**: Use this entry point to process a single or multiple videos, employing or ignoring keywords.
- **process_channels**: Use this to process single or multiple channels with or without search terms.
- **process_a_playlist**: This entry point processes all the videos in a selected playlist.
- **process_clips**: This entry point creates clips for one or multiple videos.
