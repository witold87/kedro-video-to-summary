"""
This is a boilerplate pipeline 'vid_to_summary'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_large_audio_transcription, transform,create_summary_with_mosaic,  create_summary_with_bart,  create_summary, get_credentials, send_summary_mail, fetch_video_data, send_mail_jinja, create_summary_redpajama


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fetch_video_data,
            inputs=None,
            outputs='preprocessed_audio',
            name='video_data_node'),
        node(
            func=get_large_audio_transcription,
            inputs='preprocessed_audio',
            outputs='transcribed_audio',
            name='audio_transcription_node'
        ),
        # node(
        #   func=clean_recordings,
        #   inputs=,
        #   outputs=,
        #   name='clean_recordings_node'
        # ),
        node(
            func=transform,
            inputs='transcribed_audio',
            outputs='selected_chunks',
            name='selected_chunks_node'
        ),
        node(
            func=get_credentials,
            inputs=None,
            outputs='openai_credentials',
            name='openai_credentials_node'
        ),
        node(
            func=create_summary,
            inputs=['selected_chunks', 'openai_credentials'],
            outputs='openai_answers',
            name='create_summary_node'
        ),
        node(
            func=create_summary_redpajama,
            inputs='selected_chunks',
            outputs='redpajama_answers',
            name='create_summary_redpajama_node'
        ),
        node(
            func=create_summary_with_bart,
            inputs='selected_chunks',
            outputs='bart_answers',
            name='create_summary_bart_node'
        ),
        # node(
        #     func=create_summary_with_mosaic,
        #     inputs='selected_chunks',
        #     outputs='mosaic_answers',
        #     name='create_summary_mosaic_node'
        # ),
        # ),
        # node(
        #     func=compare_answers,
        #     inputs=['openai_answers', 'redpajama_answers'],
        #     outputs=None,
        #     name='compare_outputs_node'
        # )
        # ,
        node(
            func=send_summary_mail,
            inputs='openai_answers',
            outputs=None,
            name='send_summary_node'
        )
        # node(
        #     func=send_mail_jinja,
        #     inputs='openai_answers',
        #     outputs=None,
        #     name='jinja_mail_node'
        # )
    ])
