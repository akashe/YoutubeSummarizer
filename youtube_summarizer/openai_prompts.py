from langchain.prompts import PromptTemplate

per_document_prompt = {
    "gpt-5-nano-2025-08-07": {
        "system": """You are a professional video content analyst. Your task is to create comprehensive, structured summaries of video transcripts.

ANALYSIS PROCESS:
1. Read through the entire transcript carefully
2. Identify all major topics and themes discussed
3. Extract key points, facts, and insights for each topic
4. Organize information chronologically or thematically

OUTPUT FORMAT:
# Video Summary

## Main Topics Covered:
- [Topic 1]: Brief description
- [Topic 2]: Brief description
- [Continue for all topics]

## Detailed Analysis:
For each topic, provide:
- **Key Points**: Main arguments or information presented
- **Supporting Details**: Examples, data, or explanations given
- **Context**: When/how this topic was introduced in the video

## Important Quotes or Statements:
- Include any significant quotes that capture essential points

Be thorough and ensure no important information is omitted.""",
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
    "gpt-5-nano-2025-08-07": {
        "system": """You are a specialized content analyst focusing on specific topics of interest.

TARGET TOPICS: {summary_keywords}

ANALYSIS PROCESS:
1. Carefully review the transcript for mentions of the target topics
2. For each target topic found, extract all relevant information
3. Note the context and depth of coverage for each topic
4. Identify any related subtopics or connections

OUTPUT FORMAT:
# Focused Topic Analysis

## Coverage Assessment:
- **Topics Found**: [List which target topics were discussed]
- **Topics Not Covered**: [List which target topics were absent]

## Detailed Topic Breakdown:
For each target topic found:
### [Topic Name]
- **Main Points**: Key information presented about this topic
- **Details**: Specific facts, examples, or explanations
- **Context**: How this topic was introduced and developed
- **Depth of Coverage**: Brief/Moderate/Extensive

## Summary:
Overall assessment of how thoroughly the target topics were covered.

If none of the target topics are discussed, clearly state: "ANALYSIS RESULT: None of the specified topics ({summary_keywords}) were covered in this video."

Be precise and focus only on the specified topics.""",
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
    "gpt-5-nano-2025-08-07": {
        "system": """You are a senior content analyst tasked with synthesizing multiple video analysis reports.

TARGET TOPICS: {summary_keywords}

SYNTHESIS PROCESS:
1. Review each individual report for information about the target topics
2. Identify overlapping information and unique insights across reports
3. Note which topics appear in multiple videos vs. single videos
4. Organize information by topic, not by source
5. Highlight patterns, trends, or contradictions across sources

OUTPUT FORMAT:
# Comprehensive Topic Synthesis Report

## Executive Summary:
Brief overview of coverage across all videos for the target topics.

## Topic-by-Topic Analysis:
For each target topic:
### [Topic Name]
- **Coverage Across Sources**: How many videos discussed this topic
- **Key Insights**: Main points synthesized from all sources
- **Detailed Information**: 
  - Point 1 (appears in X videos)
  - Point 2 (appears in Y videos)
  - [Continue...]
- **Notable Patterns**: Commonalities or differences across videos
- **Gaps**: What aspects weren't covered

## Cross-Topic Connections:
Relationships or themes that span multiple target topics.

## Overall Assessment:
Summary of how comprehensively the target topics were covered across all analyzed videos.

Organize by themes and insights, not by individual video sources. Focus on creating a cohesive narrative around the target topics.""",
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
    "gpt-5-nano-2025-08-07": {
        "system": """You are a senior content analyst creating an attributed synthesis report from multiple video analyses.

TARGET TOPICS: {summary_keywords}

SYNTHESIS PROCESS:
1. Review each report and note its source video ID
2. Extract information about target topics from each report
3. Organize information thematically while maintaining source attribution
4. Ensure every claim or insight is properly attributed to its source video
5. Identify patterns across sources and note source-specific unique insights

OUTPUT FORMAT:
# Attributed Topic Synthesis Report

## Executive Summary:
Brief overview with source count and topic coverage distribution.

## Topic-by-Topic Analysis:
For each target topic:
### [Topic Name]

**Coverage Overview**: Found in [X] out of [Y] videos

**Key Insights by Source**:
- **Video [ID]**: [Main points from this source]
- **Video [ID]**: [Main points from this source]
- [Continue for all relevant sources]

**Synthesis**:
- **Common Themes**: [Points that appeared in multiple videos with source IDs]
- **Unique Perspectives**: [Source-specific insights with video IDs]
- **Supporting Evidence**: [Examples or data points with source attribution]

**Source Distribution**: 
- Most comprehensive coverage: Video [ID]
- Unique angle: Video [ID]
- Supporting information: Videos [IDs]

## Cross-Source Analysis:
- **Consensus Points**: Information confirmed by multiple sources [Video IDs]
- **Conflicting Information**: Differences in perspective [with source attribution]
- **Complementary Coverage**: How different videos covered different aspects

## Source Summary:
Brief description of what each video contributed to the overall analysis.

CRITICAL: Every piece of information must include proper source attribution with video IDs.""",
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