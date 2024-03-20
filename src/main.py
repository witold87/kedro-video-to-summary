from speech_to_text import get_large_audio_transcription
from text_analysis import transform
from text_to_gpt import question
from video_to_audio import convert

if __name__ == '__main__':
    # video_filename = '../shape-shifting-dinos.mp4'
    # audio_file_name = f'shape-shifting-dinos'
    # #
    # # audio_file = convert(from_video_file=video_filename, to_audio_file=audio_file_name)
    #
    # audio_filepath = f'../{audio_file_name}'
    #
    # text = get_large_audio_transcription(audio_filepath)
    #
    with open('../shape-shifting-dinos.txt', 'r') as file:
         text = file.read()

    text = transform(text, max_tokens=2000)
    openai_resp = question(text)
    print(openai_resp)

