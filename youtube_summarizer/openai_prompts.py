from langchain.prompts import PromptTemplate

per_document_prompt = {
    "gpt-4o-mini": {
        "system": "Take a deep breadth. You are a news reporter whose job is to create summaries of videos."
                  "Given a transcript of a video, your job is to report everything that was mentioned.",
        "user": "Transcript: {context}"
    },
    "gpt-3.5-turbo-16k": {
        "system": "Summarize the following transcript of a youtube video. Highlight different topics that"
                  "were discussed in the video. ",
        "user": "Transcript: {context}"
    }
}


def get_per_document_prompt_template(model_name: str) -> dict:
    system_user_prompts = per_document_prompt[model_name]
    system_user_prompts["summary_keywords"] = False

    return system_user_prompts


per_document_with_keyword_prompt = {
    "gpt-4o-mini": {
        "system": "Take a deep breadth. You are a news reporter whose job is to cover the following topics: {summary_keywords}. "
                  "Given a transcript of a video, your job is to report everything that was mentioned about the topics. "
                  "If the topics are not covered in the transcript, clearly mention that the topics are not covered. ",
        "user": "Transcript: {context}"
    },
    "gpt-3.5-turbo-16k": {
        "system": "You are given a transcript of a youtube video. Summarize the video"
                  "if the topics: {summary_keywords} are discussed in it."
                  "If the topics are not covered in the transcript, output "
                  "that the topics are not covered.",
        "user": "Transcript: {context}"
    }
}


def get_per_document_with_keyword_prompt_template(model_name: str) -> dict:
    system_user_prompts = per_document_with_keyword_prompt[model_name]
    system_user_prompts["summary_keywords"] = True

    return system_user_prompts


combine_document_with_keyword_prompt = {
    "gpt-4o-mini": {
        "system": "Take a deep breadth. You are a news reporter. You are given summarized reports from different reporters."
                  "Each report is created as a discussion around these topics: {summary_keywords}."
                  "Every report does not contain information of all the topics. Some report cover a particular topic or multiple topics."
                  "Some report may contain no information about the topics also."
                  "Your job is to combine the information present in the smaller reports and create a big report.",
        "user": "Reports: {context} "
    },
    "gpt-3.5-turbo-16k": {
        "system": "You are given summarized reports from different youtube videos."
                  "Your job is to output information about the topics:{summary_keywords} present in these reports.",
        "user": "Reports: {context}"
    }
}


def get_combine_document_prompt_template(model_name: str) -> dict:
    system_user_prompts = combine_document_with_keyword_prompt[model_name]
    system_user_prompts["summary_keywords"] = True

    return system_user_prompts


combine_document_with_source_prompt = {
    "gpt-4o-mini": {
        "system": "Take a deep breadth. You are a news reporter. You are given summarized reports from different reporters."
                  "Each report is created as a discussion around these topics: {summary_keywords}."
                  "Each report contains the id of the source video of the report"
                  "Every report does not contain information of all the topics. Some report cover a particular topic or multiple topics."
                  "Some report may contain no information about the topics also."
                  "Your job is to combine the information present in the smaller reports and create a big report."
                  "Each information in the report should correctly attribute the source video.",
        "user": "Reports: {context} "
    },
    "gpt-3.5-turbo-16k": {
        "system": "You are given summarized reports from different youtube videos."
                  "Your job is to output information about the topics:{summary_keywords} present in these reports."
                  "Highlight the source video for each piece of information in the output",
        "user": "Reports: {context}"
    }
}


def get_combine_document_with_source_prompt_template(model_name: str) -> PromptTemplate:
    system_user_prompts = combine_document_with_source_prompt[model_name]
    system_user_prompts["summary_keywords"] = True

    return system_user_prompts
