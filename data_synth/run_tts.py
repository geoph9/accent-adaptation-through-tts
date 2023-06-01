import os
from typing import List, Tuple

import azure.cognitiveservices.speech as speechsdk


def create_wav(
        text: str,
        speech_key: str, 
        speech_region: str,
        voice_name: str,
        wav_file: str,
        verbose: bool = True
    ) -> None:
    """ Use Microsoft TTS to create wav files from text. We assume that you provide a 
    valid subscription key and region. In addition, you need to specify the voice name
    (check list of available voices in the Azure portal).
        Args:
            text (str): text to be synthesized
            speech_key (str): subscription key
            speech_region (str): region
            voice_name (str): identifier of the voice (azure-voices.json has examples)
            wav_file (str): path to the wav file to be created
            verbose (bool): print messages
        
        Returns:
            None
    """
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=wav_file)

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name=voice_name

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if verbose:
        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(text))
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")


def check_zero_byte_audio_files(
        dir_path: str, 
        fn_template: str, 
        expect_num: int, 
        n_speakers: int = 1
    ) -> Tuple[List[str], List[int]]:
    """ Check if wav files have been successfully created (i.e., they are not zero 
    byte files). We keep tracked of the files that are either zero bytes or do not
    exist in the system. We also track the corresponding ids.
        Args:
            dir_path (str): path to the directory where the wav files are stored
            fn_template (str): template of the wav file name. E.g. "english-{}.wav"
            expect_num (int): expected number of wav files per utterance (does not 
                account for multiple speakers per utterance)
            n_speakers (int): number of speakers per utterance
        
        Returns:
            zero_byte_files (list): list of zero byte files
            numbers (list): list of ids of zero byte files
    """
    zero_byte_files = []
    numbers = []
    for i in range(expect_num):
        for j in range(n_speakers):
            # no = str(i).zfill(5)
            no = f"{str(i).zfill(5)}_spk{j}"
            filename = fn_template.format(no)
            file_path = os.path.join(dir_path, filename)
            if not os.path.exists(file_path):
                zero_byte_files.append(filename)
                numbers.append(i)
                continue
            else:
                if os.path.getsize(file_path) == 0:
                    zero_byte_files.append(filename)
                    numbers.append(i)
    return zero_byte_files, numbers

