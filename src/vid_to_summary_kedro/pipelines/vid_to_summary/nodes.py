import os
import openai
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import yaml
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tqdm import tqdm
from kedro.io import PartitionedDataSet


def fetch_video_data():
    data_set = PartitionedDataSet(
        path="data/01_raw/",
        dataset="vid_to_summary_kedro.extras.datasets.video_dataset.VideoDataSet",
        filename_suffix='.mp4'
    )
    loaded = data_set.load()
    tracks = []
    for partition_id, partition_load_func in loaded.items():
        partition_data = partition_load_func()
        single_track = {'id': partition_id, 'track': partition_data}
        tracks.append(single_track)

    return tracks


def get_large_audio_transcription(data: list):
    r = sr.Recognizer()
    for item in data:
        id = item.get('id')
        track = item.get('track')
        filepath = 'data/02_intermediate/'
        file = track.export(filepath + id, format='wav')
        sound = AudioSegment.from_wav(file)

        chunks = split_on_silence(sound,
                                  min_silence_len=700,
                                  silence_thresh=sound.dBFS - 14,
                                  keep_silence=500,
                                  )
        folder_name = f'tmp'

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        whole_text = ""

        for i, audio_chunk in tqdm(enumerate(chunks, start=1)):

            chunk_filename = os.path.join(folder_name, f'chunk{i}_{id}.wav')
            audio_chunk.export(chunk_filename, format="wav")
            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)
                print(type(audio_listened))
                try:
                    text = r.recognize_google(audio_listened)
                    print(text)
                except sr.UnknownValueError as e:
                    print(e)
                else:
                    text = f"{text.capitalize()}. "
                    whole_text += text
                    item['transcript'] = whole_text
        yield {id: whole_text}


def _compose_summary_for_mail(summaries: dict):
    n_texts = len(summaries)
    combined_summary = ''
    for id, load_function in summaries.items():
        summary = load_function()
        combined_summary += summary + '<br><br><br>'
    return n_texts, combined_summary


def send_summary_mail(summaries: dict):
    n_recordings, complete_summary = _compose_summary_for_mail(summaries=summaries)
    me = "Witold Pawlak <zvit3k@gmail.com>"
    you = "Witold Pawlak<zvit3k@gmail.com>"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Link"
    msg['From'] = me
    msg['To'] = you

    html = f"""\
    <html>
      <head></head>
      <body>
        <p>Hi!<br>
           How are you?<br><br>
           Here is the summary of {n_recordings} recordings that you have missed:
           <h3>{complete_summary}</h3>
        </p>
        <table id="tbl_id" style="text-align:center" align="center" valign:"top">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Score</th>
                    <th>Points</th>
                    <th>Total</th>
                </tr>
            </thead>
        </table>
      </body>
    </html>
    """

    part2 = MIMEText(html, 'html')

    msg.attach(part2)

    with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as server:
        server.login("<>", "<>")
        server.sendmail(me, you, msg.as_string())


def render_template(template, **kwargs):
    import sys
    ''' renders a Jinja template into HTML '''
    # check if template exists
    if not os.path.exists(template):
        print(os.getcwd())
        print('No template file present: %s' % template)
        sys.exit()

    import jinja2
    template_dir = os.path.join(os.path.dirname(__file__))
    print(template_dir)
    templateLoader = jinja2.FileSystemLoader(searchpath=template_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    templ = templateEnv.get_template(template)
    return templ.render(**kwargs)

def send_mail_jinja(summaries: dict):
    _, summary = _compose_summary_for_mail(summaries)
    me = "Witold Pawlak <zvit3k@gmail.com>"
    you = "Witold Pawlak<zvit3k@gmail.com>"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Link"
    msg['From'] = me
    msg['To'] = you

    item1 = 'kryptonite'
    item2 = 'green clothing'

    # generate HTML from template
    html = render_template('template.html', **locals())

    part2 = MIMEText(html, 'html')

    msg.attach(part2)

    with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as server:
        server.login("<>", "<>")
        server.sendmail(me, you, msg.as_string())


def _text_to_array(text):
    words = text.split(' ')
    print(len(words))
    return words


def _get_first_chunk(text, max_tokens=300):
    chunked = text[:max_tokens]
    return ' '.join([chunk for chunk in chunked])


def transform(transcribed_texts: dict):
    for filename, partition_load_func in transcribed_texts.items():
        text = partition_load_func()
        text_as_array = _text_to_array(text)
        chunk = _get_first_chunk(text_as_array, 800)
        yield {filename: chunk}


def get_credentials():
    with open("conf/local/credentials.yml") as cred:
        cred_dict = yaml.safe_load(cred).get('openai')
    return cred_dict


def create_summary(prompts, credentials):
    for id, load_function in prompts.items():
        prompt = load_function()
        openai.api_key = credentials.get('api_key')
        q = f'Could you please summarize below text for me? Here is the text: {prompt}'
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": q},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        resp = response["choices"][0]["message"]["content"]
        yield {id: resp}


def create_summary_redpajama(prompts: dict):
    all_text = []
    for id, load_function in prompts.items():
        prompt = load_function()
        all_text.append(prompt)

    selected_prompt = all_text[0]

    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # init
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1",
                                                 torch_dtype=torch.bfloat16)
    # infer
    prompt = f"<human>: Could you please summarize below text for me? Here is the text: {selected_prompt}\n<bot>:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=256, do_sample=True, temperature=0.2, top_p=0.2, top_k=50, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    print(output_str)
    yield {'redpajama_output': output_str}


def create_summary_with_bart(prompts):
    from transformers import BartForConditionalGeneration, BartTokenizer

    all_text = []
    for id, load_function in prompts.items():
        prompt = load_function()
        all_text.append(prompt)

    selected_prompt = all_text[0]

    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-xsum-6-6")
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-xsum-6-6")

    input_ids = tokenizer.encode(selected_prompt, return_tensors="pt")
    generated_sequence = model.generate(num_beams=10, min_length=100, max_length=120, top_k=50, input_ids=input_ids)

    output_text = tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)
    print(output_text)
    yield {'bart_output': output_text}


def create_summary_with_mosaic(prompts: dict):
    all_text = []
    for id, load_function in prompts.items():
        prompt = load_function()
        all_text.append(prompt)

    selected_prompt = all_text[0]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b',
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    ppt = f'Could you please summarize below text for me? Here is the text: {selected_prompt}'

    input_ids = tokenizer.encode(ppt, return_tensors="pt")
    generated_sequence = model.generate(num_beams=10, min_length=100, max_length=120, top_k=50, input_ids=input_ids)
    output_text = tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)
    print(output_text)
    yield {'mosaic_output': output_text}