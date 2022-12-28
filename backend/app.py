import os
import tempfile
import flask
from flask import request
from flask_cors import CORS
from pyChatGPT import ChatGPT
import whisper

app = flask.Flask(__name__)
CORS(app)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':
        language = request.form['language']
        model = request.form['model_size']

        # there are no english models for large
        if model != 'large' and language == 'english':
            model = model + '.en'
        audio_model = whisper.load_model(model)

        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, 'temp.wav')

        wav_file = request.files['audio_data']
        wav_file.save(save_path)

        if language == 'english':
            result = audio_model.transcribe(save_path, language='english')
        else:
            result = audio_model.transcribe(save_path)

        return result['text']
    else:
        return "This endpoint only processes POST wav blob"

@app.route('/chat/test', methods=['GET'])
def chatgpt(): 
    session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..4p35M5JuePiCoiF7.SAV4FVsXoREh2Zj-teJpOBo9DrSks8GveNKcwXuigeFggJeIg1J9yRe3X_P6EtPicfEbbV5yfa-GkB-NDBbteW5lp45p1PZ9Uf_mmNJIrAocISswBwTd4J_ageGEb8sxxq79P5aJRi5xJsasXClu86_TauU9v1hhrnnPefBTfWS_OIu1mfbDGpWaqFDyObPbfd8jNJWixrD2vIbTMENRsVVMB1omhwxX1pMPCzlSLHByx757xegFGhLet-aDWVrUP9dnEuCZqs_mpsCrphFaWMc_6OghB2KWo1Ks22SSxoCShV7umNfOgtMQgxPIv-YInxb771mSOfo1toOhkwVBSBzR6UW3J-uTU6CGYwO2sAD4-_cEI8ha5rDkYB4m9xDheY5pQ3bnX1KdGvlxf3m_uA-0qN8Ydvvqo1X_jaYQctmepyRmPij0PIZmJ2i5zp5MP1Ej3SEGQI-0Brc_G8eiHn59j8NafCCtsk34GL5qdMA_3HSzW5HpETfz8gHswoH4qX2E9FepIxeGk-d-KovAf9lWHa6vCjDX7GA4_ko-6CwX1bDX1MQFbFR8yehqHeGA_66EO8kzJyPgQgjm5qEtEbJHsQkWgtxHhPjcmOD1M3Yl_KSFictEa6nBLFrKfq_zKMreu80ybGnw7Nz9fYyXri6NXfqVXyNvWOTpSucqbbx0XE5KGZ3P2ZbJgHQT4DnZfSOUHx9k0eOi0XoAJdzT9n60K0GgkoXG7dWLSfp6n63e4ROfu7g3Zo4jXyb9v3TbfRLG37fL9CSbpiiF_-kfrxrpWPLTjWku-9nYLsRw875V5aExSfVIJveBrfMmWe4ay3_9o65upBx-hFoDgNiT5gL55LCH34G3s-jDnVw8htrl0339pLkARTS4CF9Z7lJAb8UBZq3mITIy0Rxbcvmp91vgnkN34r0afZ_kqkiQDM_MFnX0ejCluAC5RVqaCowAFs0i3o087OZ7EjLQzco0Q6UlzH4X_Hbeb0jfIz6jAIM6QvFvyndxcMLkK6lHCureDI0IkAOe2s_hsCXkLIANWnxaZQK7k6in978R1ipBP6BpR6y3UUZIavLDJFYrMN16vSZCqYL7RW_qNWMSLmCuJaeCyKbn7q6ujgYUb1RcSPcVThVprNf4lGsDuslRmyV_ADB__bCgFe712zEgvJ-8f-DOq5h_5rsCJDnET15jGDDqa7Nv1d5pgsGGlJ1tSa8wtZVAKdQXj-IE8AcUGfYUQDTtJ1K-cpddB_HyGZf1qCtorXuKkyLnPxXUo2EQ84n0Tzz02am0FIa46psGXd-1zdZz-eE7jJOw9POlC1NBdvcigbQJY8ZVpoLFfUGrRuev8NUJpVsG3xiNUJmjqiMaVX5ElYAtVCWwe1JScsZBX2En0JQppFMXa9-xWLqM_wLBiUq04lDvAB_XHZITVa1MR4_qeY8cDYSpVS__l0BnXOEpYVaR76fc9eo20YA79oTl07taiYkooDnp7SxtQnovhZGnHKU00H84P0dmNmmlCSDvuJmLKf-xWrybKknM2oofZiqj-5G7gZl4nFgtDXSg0uHuhwwmaPwvd2BXrLvReABOabJE8SHGiAMtLqHo8maacE8-YHbFK0uNguaaGDxi1LITsYCvg1Zhnwsc43B4l-aPNJ47QFrQJqxcB8UE9-OB_aUg0cnSQfs6DWe9SSuTFVhOakOLq-OH7JhSZSlz1eKum5DQfVpAbdcUCLcOB5c0Lrk6xBw5tP_aomaZWi2m_zs3yAm5caw5npiFY0Y3Nyg-GIElIOzyp2j5isCE64ZfKhPzhNCZHVTyAup6DT2IZrdwhSSeYpLbZPvlxcjWrNFwFA2oJ8V3ErQbfEfX2Cl7uOz_MBWNajlLID4lAMNm1J2UIEosaZND5PdcUETjC3giaqcojh5WUFOn-x_0HifaFfg9q1MR9aoV_Ojbe_no5zESykHeI1wTeZMb_wb0GuAjWq6BAM4JJjSKoS7TpObCFSAMKx7K6EZGtA290fr8RzrP89ReAorTimREh_d6KQjf00mKzwwysZ1Is_MBQJq6jJFDQAlgWALq_lyKqc1QWxCtRfpPHg-2ybv2Rz56-1Z6uZzbtQJHTuBltXOxh7t2CoMQO1dLeuRL52-gnHtiG2yO-cTkM1jirPxzh4G7iM4UHvHViZ0Cv72vXt0sNIF38QGzrkr_hRIO582NvkidUkGwYJe3RIVVNEnfU_vvREJGYJ1Xijtu3O4xaxCQlN7OJg37eOxwEWQKUQKjf9BylxQWme34rxwz4Ek2psvChYaeZ39p0A1Hoi8Pn7z0GwNZyBH6YpUvEPJtUU4.v1J-kqsqBPM69XuKZ82Syg'
    # api = ChatGPT(session_token, window_size=(1024, 768))
    # set window size to 0 to disable GUI
    api = ChatGPT(session_token, window_size=(0, 0))

    resp = api.send_message('Hello, world!')
    print(resp['message'])

    api.reset_conversation()
    api.clear_conversations()
    api.refresh_chat_page()

    return resp['message']

