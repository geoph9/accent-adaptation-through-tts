"""
Bring datasets to csv format, as follows:
wav,wrd,accent,duration,gender

We assume that the basename of wav_path is the utterance's
unique identifiers. E.g. /path/to/my-utterance1.wav will
correspond to an utterance id of my-utterance1.

Authors:
 * Georgios Karakasidis 2023

Example Usage:
 # prepare the initial l2_arctic csv files
 python prepare_arctic.py prepare-arctic -d /path/to/l2_arctic -o /path/to/l2_arctic_prepared

 # for each accent produce separate train/dev/test splits (this is the most important step)
 python prepare_arctic.py combine-and-split-per-accent --main_path /path/to/l2_arctic_prepared \
    --use_separate_speakers_on_test --out_dir /path/to/l2_arctic_prepared/accent_specific_data

 # combine the {train/dev/test} csv files of all accents into a single {train/dev/test}.csv file.
 # Overwrites the initial files.
 python prepare_arctic.py combine-accents-and-split \
    --main_path /path/to/l2_arctic_prepared/accent_specific_data \
    --out_dir /path/to/l2_arctic_prepared -s train -s dev -s test

 # extract the accent-specific text files (for tts)
 python prepare_arctic.py extract-accent-specific-text \
    --main_path /path/to/l2_arctic_prepared/accent_specific_data -s train -s dev -s test
"""

from collections import defaultdict
import random
import sys
import os
import csv
import glob
import logging
import re
from typing import List
import statistics
import time

import librosa
import click
from tqdm import tqdm

if not os.path.isdir("logs"):
    os.mkdir("logs")
timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
logging.basicConfig(
    filename=os.path.join("logs", f"l2_{timestr}.log"),
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# logger.addHandler(fh)
cli = click.Group()

logger = logging.getLogger(__name__)
logger.info("Executing: " + " ".join(sys.argv))

# Conventions
ACCENT_MAPPINGS = {
    "arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "chinese": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}
MALE_ACCENTS = [
    "ABA",
    "YBAA",
    "BWC",
    "TXHC",
    "ASI",
    "RRBI",
    "HKK",
    "YKWK",
    "EBVS",
    "ERMS",
    "HQTV",
    "TLV",
]
TEST_SET_SPEAKERS = {
    "arabic": ["ABA", "ZHAA"],
    "chinese": ["BWC", "LXC"],
    "hindi": ["ASI", "SVBI"],
    "korean": ["HJK", "HKK"],
    "spanish": ["EBVS", "MBMPS"],
    "vietnamese": ["HQTV", "THV"],
}


@cli.command()
@click.option("--data_folder", "-d", help="Dataset source directory.")
@click.option(
    "--save_folder",
    "-o",
    help="Output directory where the [train,dev,test].csv files will be saved.",
)
def prepare_arctic(
    data_folder: str,
    save_folder: str,
    skip_prep: bool = False,
    min_duration_secs: float = 1.0,  # 1 second
    max_duration_secs: float = 9.0,  # 9 seconds
):
    """
    Prepares the csv files for the L2-Arctic dataset.
    https://psi.engr.tamu.edu/l2-arctic-corpus/

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    skip_prep: bool
        If True, data preparation is skipped.
    min_duration_secs: float
        Default: 1 second
        Minimum duration of each audio chunk. Small audio chunks may result in errors
        and so they will be skipped.
    max_duration_secs: float
        Default: 11 seconds
        Maximum duration of each audio chunk. Large audio chunks may result in errors.
    remove_special_tokens: bool
        Default: False
        Defines whether the special tokens of the form .br, .fr etc will be used for
        training or not.

    Example
    -------
    >>> data_folder = 'datasets/l2_arctic/'
    >>> save_folder = 'arctic_prepared'
    >>> prepare_lp(data_folder, save_folder)
    """

    if skip_prep:
        return
    all_speakers = []
    for v in ACCENT_MAPPINGS.values():
        all_speakers += v
    speakers = [
        spk
        for spk in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, spk)) and spk in all_speakers
    ]
    required_subfolders = ["transcript", "wav"]
    for spk in speakers:
        for subf in required_subfolders:
            req = os.path.join(data_folder, spk, subf)
            assert os.path.isdir(req), req

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger.info("Creating a unified CSV dataset. (not split)")
    create_csv(
        folder=data_folder,
        speakers=speakers,
        output_csv_file=os.path.join(save_folder, "l2_dataset.csv"),
        min_duration_secs=min_duration_secs,
        max_duration_secs=max_duration_secs,
    )


def create_csv(
    folder: str,
    speakers: List[str],
    output_csv_file: str,
    min_duration_secs: float,
    max_duration_secs: float,
):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    folder : str
        Path to the unzipped main directory of L2-ARCTIC.
    speakers: list
        List of valid L2-ARCTIC speakers (see README).
    output_csv_file: str
        Path to the output csv file.
    min_duration_secs: float
        Minimum allowed audio duration in seconds.
    max_duration_secs: float
        Maximum allowed audio duration in seconds.
    Returns
    -------
    None
    """

    spk_to_acc = {}
    for acc, spks in ACCENT_MAPPINGS.items():
        for spk in spks:
            spk_to_acc[spk] = acc
    durations = []
    # print(folder, os.listdir(folder), speakers)
    accent_specific_utts = defaultdict(list)
    out_dir_accented = os.path.join(
        os.path.dirname(output_csv_file), "accent_specific_data"
    )
    os.makedirs(out_dir_accented, exist_ok=True)
    with open(output_csv_file, "w", encoding="utf8") as csv_fw:
        csv_writer = csv.writer(
            csv_fw, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header_row = ["wav", "wrd", "accent", "duration", "gender"]
        csv_writer.writerow(header_row)
        for spk in tqdm(speakers):
            wav_main_path = os.path.abspath(os.path.join(folder, spk, "wav"))
            for txt_path in glob.glob(os.path.join(folder, spk, "transcript", "*.txt")):
                # utt_id = f"{spk}-{os.path.basename(txt_path)}"
                wav_path = os.path.join(
                    wav_main_path, os.path.basename(txt_path).replace(".txt", ".wav")
                )
                # Reading the signal (to retrieve duration in seconds)
                if os.path.isfile(wav_path):
                    # info = torchaudio.info(mp3_path, format="mp3")
                    signal, sr = librosa.load(wav_path)
                else:
                    msg = "\tError loading: %s" % (str(wav_path))
                    logger.error(msg)
                    raise Exception(msg)
                    # continue
                # Get accent and gender metadata
                accent = spk_to_acc[spk]
                gender = "male" if spk in MALE_ACCENTS else "female"
                # Read and process text
                with open(txt_path, "r") as fr:
                    text = fr.read().replace("\n", "").strip()
                if len(text) < 3:
                    continue
                text = re.sub(',|"|\.', " ", text)
                text = re.sub("\s+", " ", text).strip()
                # Get duration
                duration = librosa.get_duration(y=signal, sr=sr)
                if not (min_duration_secs <= duration <= max_duration_secs):
                    continue
                durations.append(duration)
                # Composition of the csv_line (the speaker id is the same as the utt id.)
                #            <WAV>  <WORDS> <ACCENT>  <DURATION>   <GENDER>
                csv_line = [wav_path, text, accent, str(duration), gender]
                # Ignore rows that contain NaN values
                if any(i != i for i in csv_line) or len(text) == 0:
                    continue
                # Adding this line to the csv_lines list
                csv_writer.writerow(csv_line)
                accent_specific_utts[accent].append(csv_line)
    logger.info(f"{output_csv_file} successfully created!")
    # Final prints
    total_duration = sum(durations)
    logger.info(f"Number of samples: {len(durations)}.")
    logger.info(f"Total duration: {round(total_duration / 3600, 2)} hours.")
    logger.info(
        f"Median/Mean duration: {round(statistics.median(durations), 2)}/{round(total_duration/len(durations), 2)}."
    )
    logger.info("Creating accent specific subsets.")
    for accent, utterances in accent_specific_utts.items():
        accented_out_path = os.path.join(out_dir_accented, accent + ".csv")
        with open(accented_out_path, "w", encoding="utf8") as csv_fw:
            csv_writer = csv.writer(
                csv_fw, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            header_row = ["wav", "wrd", "accent", "duration", "gender"]
            csv_writer.writerow(header_row)
            csv_writer.writerows(utterances)
        logger.info(
            f"{accented_out_path} successfully created! #samples={len(utterances)}"
        )


def _combine_and_split_per_speaker(main_path: str, out_dir: str):
    """
    Args:
        main_path: path to a directory where dataset.csv already exists
        out_dir: The previous called `accent_specific_data` directory. Will contain separate folders
                 for each accent. Each folder will have a train/dev/test.csv file.
    Returns:
        For each accent, creates a train/dev/test split (80/10/10% each) and returns None.
    """
    out_dir = os.path.join(os.path.dirname(main_path), "accent_specific_data")
    os.makedirs(out_dir, exist_ok=True)
    accents = ACCENT_MAPPINGS.keys()
    assert len(accents) > 0, accents

    def extract_speaker(path):
        # E.g. Convert "/<path>/<to>/<Arctic>/<dataset>/ABA"
        #      to      "ABA"
        p = path.split(",")[0]
        return os.path.basename(os.path.dirname(os.path.dirname(p)))

    for acc in accents:
        acc = acc.split("_")[0].split()[0].replace(".csv", "")
        os.makedirs(os.path.join(out_dir, acc), exist_ok=True)
    with open(main_path, "r") as fr:
        contents = fr.readlines()
        HEADER = contents.pop(0)
    random.shuffle(contents)
    contents = [
        line.strip().replace("\n", "") + "\n"
        for line in contents
        if len(line.strip()) > 0 and line.strip() != "\n"
    ]
    logger.info(f"Loaded {len(contents)} utterances.")
    def create_out_split(s, acc):
        return os.path.join(out_dir, acc, s)
    for accent in accents:
        train_dev_text = []
        test_text = []
        for c in contents:
            if c.split(",")[2].split("_")[0].split()[0] == accent:
                if extract_speaker(c.split(",")[0]) in TEST_SET_SPEAKERS[accent]:
                    test_text.append(c)
                else:
                    train_dev_text.append(c)
        random.shuffle(train_dev_text)
        random.shuffle(test_text)
        sperc = [0.8, 0.2]  # train/dev split percentages
        _p = int(sperc[0] * len(train_dev_text))
        logger.info(
            f"({accent}) #utterances: {_p} in train, {len(train_dev_text)-_p} in dev, {len(test_text)} in test."
        )
        with open(create_out_split("train.csv", accent), "w") as fw:
            fw.writelines([HEADER] + train_dev_text[:_p])
        with open(create_out_split("dev.csv", accent), "w") as fw:
            fw.writelines(
                [HEADER] + train_dev_text[_p : _p + int(sperc[1] * len(train_dev_text))]
            )
        with open(create_out_split("test.csv", accent), "w") as fw:
            fw.writelines([HEADER] + test_text)
        logger.info(
            f"Files {os.path.join(out_dir, accent)}/[train|dev|test].csv have been created"
        )


@cli.command()
@click.option(
    "--main_path",
    help="Path to dataset.csv file (produce from the prepare-arctic function).",
)
@click.option(
    "--out_dir",
    help="Directory where the accent-specific sub-dirs and csv files will be saved.",
)
@click.option(
    "--use_separate_speakers_on_test",
    "--sep_test",
    "-t",
    is_flag=True,
    help="Path to dataset.csv file (produce from the prepare-arctic function).",
)
def combine_and_split_per_accent(
    main_path: str, out_dir: str = None, use_separate_speakers_on_test: bool = True
):
    """
    Args:
        main_path: path to a directory where dataset.csv already exists
        out_dir: The previously called `accent_specific_data` directory. Will contain separate folders
                 for each accent. Each folder will have a train/dev/test.csv file.
        use_separate_speakers_on_test: If true a female and male speaker of each accent will be
                 removed from the training set and will solely be used on the test set. This, along
                 with an 80-20 train/dev split, means that the training will have (0.8*0.5=) 40% of the original data.
    Returns:
        For each accent, creates a train/dev/test split (80/10/10% each) and returns None.
    """

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(main_path), "accent_specific_data")
    if use_separate_speakers_on_test:
        logger.info(
            "Using a speaker dependant train/test split. The train/dev split is a random 80-20 split."
        )
        return _combine_and_split_per_speaker(main_path, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    accents = ACCENT_MAPPINGS.keys()
    assert len(accents) > 0, accents
    for acc in accents:
        acc = acc.split("_")[0].split()[0].replace(".csv", "")
        os.makedirs(os.path.join(out_dir, acc), exist_ok=True)
    with open(main_path, "r") as fr:
        contents = fr.readlines()
        HEADER = contents.pop(0)
    random.shuffle(contents)
    contents = [
        line.strip().replace("\n", "") + "\n"
        for line in contents
        if len(line.strip()) > 0 and line.strip() != "\n"
    ]
    logger.info(f"Loaded {len(contents)} utterances.")
    # contents = [HEADER] + contents
    def create_out_split(s, acc):
        return os.path.join(out_dir, acc, s)
    for accent in accents:
        accent = accent.split("_")[0].replace(".csv", "")
        accented_text = []
        for c in contents:
            if c.split(",")[2].split("_")[0].split()[0] == accent:
                accented_text.append(c)
        # accented_text = [c for c in contents if c.split(",")[2].split("_")[0].split()[0] == accent]
        # train/dev/test split percentages
        sperc = [0.8, 0.1, 0.1]
        _p = int(sperc[0] * len(accented_text))
        with open(create_out_split("train.csv", accent), "w") as fw:
            fw.writelines([HEADER] + accented_text[:_p])
        with open(create_out_split("dev.csv", accent), "w") as fw:
            fw.writelines(
                [HEADER] + accented_text[_p : _p + int(sperc[1] * len(accented_text))]
            )
        with open(create_out_split("test.csv", accent), "w") as fw:
            fw.writelines(
                [HEADER] + accented_text[_p + int(sperc[1] * len(accented_text)) :]
            )
        logger.info(
            f"Files {os.path.join(out_dir, accent)}/[train|dev|test].csv have been created"
        )


@cli.command()
@click.option(
    "--main_path",
    help="Path to the directory of the accented splits (see combine_and_split_per_accent function).",
)
@click.option(
    "--out_dir",
    help="Directory where the output [train/dev/test].csv files will be saved.",
)
@click.option("--splits", "-s", multiple=True)
def combine_accents_and_split(
    main_path: str, out_dir: str, splits: List[str] = ["train", "test", "dev"]
):
    """
    Args:
        main_path: Path to the directory of the accented splits (see combine_and_split_per_accent function).
        out_dir: Directory where the output [train/dev/test].csv files will be saved.
        splits: A list of the splits that you want to read and process.
    Returns:
        Combines the train/dev/test CSV entries into a unified dataset. Returns None.
    """
    def create_acc_path(acc):
        return os.path.join(main_path, acc)
    accents = ACCENT_MAPPINGS.keys()
    assert os.path.isdir(main_path), main_path
    if not os.path.isdir(out_dir):
        logger.info(f"Creating output directory {out_dir}.")
        os.mkdir(out_dir)
    for split in splits:
        split_contents = []
        for acc in accents:
            logger.info(f"Processing the {acc} accent.")
            assert os.path.isdir(os.path.join(main_path, acc)), f"{main_path}/{acc}"
            acc_path = create_acc_path(acc)
            split_csv = os.path.join(acc_path, f"{split}.csv")
            with open(split_csv, "r") as fr:
                lines = fr.readlines()
                HEADER = lines[0]
                split_contents += lines[1:]
        with open(os.path.join(out_dir, f"{split}.csv"), "w") as fw:
            fw.writelines([HEADER] + split_contents)
        logger.info(
            f"Created {split}.csv file by combining all {len(accents)} accents."
        )


@cli.command()
@click.option(
    "--main_path", help="Path to a directory containing the {train|dev|test}.csv files."
)
@click.option("--splits", "-s", multiple=True)
@click.option(
    "--keep_accent",
    "-k",
    is_flag=True,
    help="If true a csv file with `text,accent` will be saved.",
)
@click.option(
    "--keep_accented_text",
    "-t",
    is_flag=True,
    help="If true, multiple .txt files will be saved for each accent.",
)
# @click.pass_context
def extract_text(
    main_path: str,
    splits: list = ["train", "dev", "test"],
    keep_accent: bool = False,
    keep_accented_text: bool = False,
):
    ext = ".csv" if keep_accent else ".txt"
    out_path = os.path.join(main_path, f"text_only_{'_'.join(splits)}{ext}")
    # splits = ['train', 'dev', 'test']
    logger.info("Using splits:", splits)
    text = []
    # acc_text_per_split = {}
    accented_txt_dir = os.path.join(main_path, "accented_text")
    if keep_accented_text and not os.path.isdir(accented_txt_dir):
        os.mkdir(accented_txt_dir)
    for s in splits:
        extracted = _extract_text(
            os.path.join(main_path, s + ".csv"),
            with_accent=keep_accent,
            return_accented_text=keep_accented_text,
        )
        if keep_accented_text:
            # acc_text_per_split[s] = extracted
            for accent, acc_texts in extracted[1].items():
                with open(
                    os.path.join(accented_txt_dir, f"text_only_{s}_{accent}.txt"), "w"
                ) as facc:
                    facc.writelines(acc_texts)
                logger.info(
                    f'Created {os.path.join(accented_txt_dir, f"text_only_{s}_{accent}.txt")}.'
                )
            extracted = extracted[0]
        text += extracted
    with open(out_path, "w") as fw:
        fw.writelines(text)
    logger.info(f"File {out_path} has been created.")


@cli.command()
@click.option(
    "--main_path",
    help="Path to a directory containing the accent specific sub-directories files.",
)
@click.option("--splits", "-s", multiple=True)
def extract_accent_specific_text(
    main_path: str, splits: list = ["train", "dev", "test"]
):
    accents = [
        acc.split("_")[0].split()[0].replace(".csv", "")
        for acc in os.listdir(main_path)
        if os.path.isdir(os.path.join(main_path, acc))
    ]
    accents = [
        acc
        for acc in accents
        if os.path.isfile(os.path.join(main_path, acc, splits[0] + ".csv"))
    ]
    out_dir = os.path.join(main_path, "text_only")
    os.makedirs(out_dir, exist_ok=True)
    for accent in accents:
        d = os.path.join(main_path, accent)
        for s in splits:
            csv_file = os.path.join(d, s + ".csv")
            with open(csv_file, "r") as fr:
                contents = map(lambda x: x.split(",")[1] + "\n", fr.readlines()[1:])
            out_file = os.path.join(out_dir, f"text_only_{accent}_{s}.txt")
            with open(out_file, "w") as fw:
                fw.writelines(contents)


def _extract_text(
    csv_path: str, with_accent: bool = False, return_accented_text: bool = False
):
    with open(csv_path, "r") as f:
        # print([l.split(",") for l in f.readlines()])
        text = []
        acc_text = defaultdict(list)
        for line in f.readlines()[1:]:
            txt = line.split(",")[1].strip()
            acc = line.split(",")[2].strip()
            if with_accent:
                txt = f"{txt},{acc}"
            txt = txt + "\n"
            text.append(txt)
            acc_text[acc].append(txt)
    if return_accented_text:
        return text, acc_text
    return text


if __name__ == "__main__":
    cli()
