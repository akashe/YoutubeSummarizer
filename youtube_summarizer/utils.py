from datetime import datetime, timedelta
import json


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


def get_model_max_len(model_name:str) -> int:
    with open("model_config.json", "r") as f:
        config = json.load(f)

    model_max_len = config[model_name]["max_allowed_token_len"]
    tokens_for_prompt = config["tokens_for_prompt"]

    return model_max_len - tokens_for_prompt

