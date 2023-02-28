import argparse
import json
import os
import random
import time
import pickle as pkl

from run_tts import *
from write_csv import pkl2csv
import pickle as pkl

MAX_ITERS = 10


def gen_tts(args):
    """
    args:
        seed (int)
        config_dict (str)
        prompts_file (str)
        wav_dir (str)
        lang (str)
        speech_key (str)
        speech_region (str)
        out_csv (str)
    """
    # Make dirs --------------------------------------------------------
    if not os.path.exists(args.wav_dir):
        os.makedirs(args.wav_dir)
        print("Written dir", args.wav_dir, flush=True)
    csv_dir = os.path.split(args.out_csv)[0]
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        print("Written dir", csv_dir, flush=True)
    assert args.out_csv.endswith('.csv'), "Must use .csv extension for"\
            " out-csv"
    mapping_pkl = args.out_csv[:-4] + "_no2voice.pkl"
    assert os.path.isdir(os.path.dirname(args.out_csv)), out_csv
    # Set seed ---------------------------------------------------------
    random.seed(args.seed)
    # Retrieve voices list ---------------------------------------------
    with open(args.config_dict, 'r') as f:
        config_dict = json.load(f)
    try:
        voice_dict = config_dict[args.lang]
    except KeyError:
        print("WARNING: You may want to add an entry for "\
                f"{args.lang} to the config file {args.config_dict}")
        raise
    voices = []
    for dialect_key in voice_dict: # FIXME add hyperparam weight dialect
        voices += voice_dict[dialect_key]
    # Read prompts -----------------------------------------------------
    with open(args.prompts_file, 'r') as f:
        prompts = f.readlines()
    prompts = [p.strip() for p in prompts]
    # Cycle through prompts --------------------------------------------

    if os.path.exists(mapping_pkl):
        with open(mapping_pkl, 'rb') as f:
            no2voice, no2prompt = pkl.load(f)
    else:
        no2voice = {}
        no2prompt = {}
    _, indices = check_zero_byte_audio_files(dir_path=args.wav_dir,\
            fn_template=args.lang + "-{}.wav", expect_num=len(prompts))
    for I in range(MAX_ITERS):
        print(f"~-~-~-~ {I} -~-~-~-", flush=True)
        print(f"Zero-byte files left: {len(indices)}")
        for i in indices:
            prompt = prompts[i]
            no = str(i).zfill(5)
            wav_file = os.path.join(args.wav_dir, f"{args.lang}_{no}.wav")
            voice_name = random.choice(voices)
            create_wav(text=prompt, speech_key=args.speech_key,\
                    speech_region=args.speech_region, voice_name=voice_name,\
                    wav_file=wav_file, verbose=False)
            sleep_time = round(.4  + I)
            time.sleep(sleep_time)
            no2voice[no] = voice_name
            no2prompt[no] = prompt
            with open(mapping_pkl, 'wb') as f:
                pkl.dump((no2voice, no2prompt), f)
            print(i, end=' ', flush=True)
        print()
        _, indices = check_zero_byte_audio_files(dir_path=args.wav_dir,\
                fn_template=args.lang + "_{}.wav", expect_num=len(prompts))
        if not indices:
            print("No more 0-byte files!", flush=True)
            break
    # Print number voice matches to out csv file -----------------------

    pkl2csv(no2voice=no2voice, no2prompt=no2prompt, out_csv=args.out_csv)
    print(f"Written {len(no2voice)} mappings to {args.out_csv}", flush=True)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=sum(b'lti'),\
            help="Random seed")
    parser.add_argument("--config-dict", type=str, required=True,\
            help="JSON file of dictionary mapping langs to voices")
    parser.add_argument("--prompts-file", type=str, required=True,\
            help="Text file containing all prompts")
    parser.add_argument("--wav-dir", type=str, required=True,\
            help="Dir to write wav files to")
    parser.add_argument("--lang", type=str, required=True,\
            help="Language code in Azure format")
    parser.add_argument("--speech-key", type=str,\
            default="780dfc2fdaea4bf89a0de176b65467e1",\
            help="Key for Azure use")
    parser.add_argument("--speech-region", type=str, default="eastus",
            help="Speech region for Azure use")
    parser.add_argument("--out-csv", type=str, required=True,\
            help="CSV file to write file-to-voice mappings to")

    args = parser.parse_args()

    gen_tts(args)

    """
    python3 gen_tts_msft.py --config-dict azure-voices.json --prompts-file tts-prompts/arctic/arctic-kor-lines.txt --wav-dir tts-audio/kor/ --lang kor --out-csv voice_csvs/kor-msft.csv --speech-key XXXXXXXXXXXXXX --speech-reagion eastus
    """
