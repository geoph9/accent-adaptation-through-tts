import argparse
import json
import os
import random
import time
import pickle as pkl

try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: tqdm not installed. You can install it with pip install tqdm")
    tqdm = lambda x: x

from run_tts import (
    create_wav,
    check_zero_byte_audio_files
)
from write_csv import pkl2csv

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
    assert os.path.isdir(os.path.dirname(args.out_csv)), args.out_csv
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
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    prompts = [p.strip() for p in prompts]
    # Cycle through prompts --------------------------------------------

    if os.path.exists(mapping_pkl):
        with open(mapping_pkl, 'rb') as f:
            no2voice, no2prompt = pkl.load(f)
    else:
        no2voice = {}
        no2prompt = {}
    SPEAKERS_TO_SAMPLE = args.speakers_per_prompt if \
        args.speakers_per_prompt < len(voices) else len(voices)
    _, indices = check_zero_byte_audio_files(dir_path=args.wav_dir,\
            fn_template=args.lang + "_{}.wav", \
                expect_num=len(prompts)*SPEAKERS_TO_SAMPLE)
    for I in range(MAX_ITERS):
        print(f"~-~-~-~ {I} -~-~-~-", flush=True)
        print(f"Zero-byte files left: {len(indices)}")
        for i in tqdm(indices):
            prompt = prompts[i]
            # voice_names = random.sample(voices, SPEAKERS_TO_SAMPLE)
            random.shuffle(voices)
            voice_names = voices[:SPEAKERS_TO_SAMPLE]
            for v_id, voice_name in enumerate(voice_names):
                no = f"{str(i).zfill(5)}_spk{v_id}"
                # voice_name = random.choice(voices)
                wav_file = os.path.join(args.wav_dir, f"{args.lang}_{no}.wav")
                if os.path.isfile(wav_file) and os.path.getsize(wav_file) == 0:
                    os.unlink(wav_file)  # Remove 0-byte files
                if not os.path.isfile(wav_file):
                    create_wav(text=prompt, speech_key=args.speech_key,\
                            speech_region=args.speech_region, voice_name=voice_name,\
                            wav_file=wav_file, verbose=False)
                    sleep_time = round(.3  + I)
                    time.sleep(sleep_time)
                no2voice[no] = voice_name
                no2prompt[no] = prompt
            if i % 20 == 0:  # Save every 20 prompts
                with open(mapping_pkl, 'wb') as f:
                    pkl.dump((no2voice, no2prompt), f)
        print()
        _, indices = check_zero_byte_audio_files(dir_path=args.wav_dir,\
                fn_template=args.lang + "_{}.wav", \
                    expect_num=len(prompts)*SPEAKERS_TO_SAMPLE)
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
            default="XXXXXXXXXXXXXX",\
            help="Key for Azure use")
    parser.add_argument("--speech-region", type=str, default="eastus",
            help="Speech region for Azure use")
    parser.add_argument("--out-csv", type=str, required=True,\
            help="CSV file to write file-to-voice mappings to")
    parser.add_argument("--speakers-per-prompt", type=int, default=1,\
            help="Number of speakers per prompt (default: 1)")

    args = parser.parse_args()

    gen_tts(args)

    """
    # Example:
    python3 gen_tts_msft.py --config-dict azure-voices.json \
        --prompts-file tts-prompts/arctic/arctic-kor-lines.txt \
        --wav-dir tts-audio/kor/ \
        --lang kor --out-csv voice_csvs/kor-msft.csv \
        --speech-key XXXXXXXXXXXXXX \
        --speech-reagion eastus
    """
