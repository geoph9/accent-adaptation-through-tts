"""
Bring datasets to csv format, as follows:
wav,wrd,accent,duration,gender,speaker

We assume that the basename of wav_path is the utterance's
unique identifiers. E.g. /path/to/my-utterance1.wav will
correspond to an utterance id of my-utterance1.

Authors:
 * Georgios Karakasidis 2023

Adapted from:
 * Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020 (speechbrain)


Example Usage:
 # prepare the initial cv8 csv files
 python prepare_commonvoice.py prepare-cv8 -d /path/to/cv8 -o /path/to/cv8_prepared
 
 # combine the accents and split the dataset (not really needed)
 python prepare_commonvoice.py combine-and-split --main_path /path/to/cv8_prepared
 
 # for each accent produce separate train/dev/test splits (this is the most important step)
 python prepare_commonvoice.py combine-and-split-per-accent --main_path /path/to/cv8_prepared
 
 # combine the {train/dev/test} csv files of all accents into a single {train/dev/test}.csv file.
 # the /final_splits folder will contain the final csv splits.
 python prepare_commonvoice.py combine-accents-and-split --main_path /path/to/cv8_prepared \
    --out_dir /path/to/cv8_prepared/final_splits -s train -s dev -s test
 
 # extract the accent-specific text files (for tts)
 python prepare_commonvoice.py extract-accent-specific-text --main_path /path/to/cv8_prepared/final_splits \
    -s train -s dev -s test
"""

import os
import sys
import csv
import random
import logging
import re
from typing import List, Optional
from shutil import copy
import statistics
from collections import defaultdict

# import torchaudio
from mutagen.mp3 import MP3

# import librosa
import click
from tqdm import tqdm

# Will cause error if not on windows
from utils import wccount

if not os.path.isdir("logs"):
    os.mkdir("logs")
fh = logging.FileHandler(os.path.join("logs", "cv8.log"))
fh.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(fh)
cli = click.Group()


@cli.command()
@click.option("--data_folder", "-d", help="Dataset source directory.")
@click.option(
    "--save_folder",
    "-o",
    help="Output directory where the [train,dev,test].csv files will be saved.",
)
def prepare_cv8(
    data_folder: str,
    save_folder: str,
    skip_prep: bool = False,
    min_duration_secs: float = 1.0,  # 1 second
    max_duration_secs: float = 12.0,  # 12 seconds
    train_dir: str = "train",
    dev_dir: str = "dev",
    test_dir: str = "test",
    recalculate_csvs_if_exist: bool = False,
    train_csv: Optional[str] = None,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset (v8).

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
    train_dir: str
        Default: train
    dev_dir: str
        Default: dev
    dev_dir: str
        Default: test
    recalculate_csvs_if_exist: bool
        If True then whether the train/dev/test csv files exist or not, they will
        be calculated from scratch.
        Default: False
    train_csv: str (Optional)
        Path to a <train>.csv file. If provided then we are not going to
        calculate a new <train>.csv file.

    Example
    -------
    >>> data_folder = 'datasets/cv8/'
    >>> save_folder = 'cv8_prepared'
    >>> prepare_lp(data_folder, save_folder)
    """

    if skip_prep:
        return
    splits = [train_dir, dev_dir, test_dir]

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains CommonVoice v1
    _check_cv_v8_folders(data_folder, splits)

    # create csv files for each split
    # In case the train_csv has already been provided, we don't need to create it.
    if train_csv is not None and os.path.isfile(train_csv):
        logger.warning(
            f"{'='*80}\nWill copy a pre-created <train>.csv file (possibly a subset?).\n{'='*80}"
        )
        copy(train_csv, os.path.join(save_folder, f"{splits[0]}.csv"))

    accent_specific_utts = defaultdict(list)
    for split in splits:
        logger.info(
            "=============== Processing {} split. ===============".format(split)
        )
        # Read as tsv but save as csv
        accent_specific_utts = create_csv(
            orig_tsv_file=os.path.join(data_folder, f"{split}.tsv"),
            output_csv_file=os.path.join(save_folder, f"{split}.csv"),
            clips_folder=os.path.join(data_folder, "clips"),
            min_duration_secs=min_duration_secs,
            max_duration_secs=max_duration_secs,
            ignore_if_already_exists=not recalculate_csvs_if_exist,
            accent_specific_utts=accent_specific_utts,
        )
        logger.info("=" * 60)


def create_csv(
    orig_tsv_file: str,
    output_csv_file: str,
    clips_folder: str,
    min_duration_secs: float = 1.0,
    max_duration_secs: float = 9.0,
    ignore_if_already_exists: bool = True,
    accent_specific_utts: dict = None,
):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    output_csv_file: st
        Path to the output csv file.
    clips_folder : str
        Path of the CommonVoice dataset (the clips directory).
    min_duration_secs: float
        Minimum allowed audio duration in seconds.
    max_duration_secs: float
        Maximum allowed audio duration in seconds.
    ignore_if_already_exists: bool
        Whether or not we should recalculate the csv files if they exist.
    accent_specific_utts: dict
        Dictionary of accent specific utterances regardless of the split.
    Returns
    -------
    None
    """

    # if "train" in output_csv_file or "dev" in output_csv_file:
    #     ignore_if_already_exists = True
    if os.path.isfile(output_csv_file) and ignore_if_already_exists:
        logger.warning(f"Ignoring {output_csv_file} since it already exists.")
        return
    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    # f = open(orig_tsv_file, "r")
    num_ignored = 0
    ignored_duration = 0.0
    number_of_audios = 0
    durations = []
    num_nan_files = 0
    if accent_specific_utts is None:
        accent_specific_utts = defaultdict(list)
    out_dir_accented = os.path.join(
        os.path.dirname(output_csv_file), "accent_specific_data"
    )
    if not os.path.isdir(out_dir_accented):
        os.mkdir(out_dir_accented)
    ACCENTS_TO_KEEP = {
        "Malaysian English": "Malay",
        "Filipino": "Tagalog",
        "India and South Asia (India, Pakistan, Sri Lanka)": "Indian_Generic",
        "German English,Non native speaker": "German",
    }
    with open(output_csv_file, "w", encoding="utf8") as csv_fw:
        csv_writer = csv.writer(
            csv_fw, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header_row = ["wav", "wrd", "accent", "duration", "gender", "speaker_id"]
        csv_writer.writerow(header_row)
        nb_samples = wccount(orig_tsv_file)

        with open(orig_tsv_file, "r") as tsv_fr:
            _ = next(tsv_fr)  # first line is the header

            # Adding some prints
            msg = "Creating csv lists in %s ..." % (output_csv_file)
            logger.info(msg)

            # Start processing lines
            for line in tqdm(tsv_fr, total=nb_samples):
                _, _tmp_path, wrd = line.split("\t")[:3]
                gender = line.split("\t")[6].strip().lower()
                accent = line.split("\t")[7]
                speaker_id = line.split("\t")[0]
                if gender not in ["male", "female"]:
                    gender = "nan"
                if not isinstance(accent, str) or accent.strip() == "":
                    continue
                if accent not in ACCENTS_TO_KEEP.keys():
                    continue
                accent = ACCENTS_TO_KEEP[accent]
                # utt_id = _tmp_path.split(".")[0]
                mp3_path = os.path.join(clips_folder, _tmp_path)

                # Reading the signal (to retrieve duration in seconds)
                if os.path.isfile(mp3_path):
                    # info = torchaudio.info(mp3_path, format="mp3")
                    # signal, sr = librosa.load(mp3_path)
                    signal = MP3(mp3_path)
                else:
                    msg = "\tError loading: %s" % (str(mp3_path))
                    # logger.error(msg)
                    continue

                # duration = info.num_frames / info.sample_rate
                # duration = librosa.get_duration(y=signal, sr=sr)
                duration = signal.info.length
                if not (min_duration_secs <= duration <= max_duration_secs):
                    num_ignored += 1
                    ignored_duration += duration
                    continue
                durations.append(duration)

                # Remove too short sentences (or empty):
                wrd = wrd.strip()
                if len(wrd) < 3:
                    continue
                wrd = re.sub(',|"|\.', " ", wrd)
                wrd = re.sub("\s+", " ", wrd).strip()
                # Composition of the csv_line (the speaker id is the same as the utt id.)
                #            <WAV>  <WORDS> <ACCENT> <DURATION>   <GENDER>   <SPK>
                csv_line = [mp3_path, wrd, accent, str(duration), gender, speaker_id]
                accent_specific_utts[accent].append(csv_line)
                # Ignore rows that contain NaN values
                if any(i != i for i in csv_line) or len(wrd) == 0:
                    num_nan_files += 1
                    continue
                # Adding this line to the csv_lines list
                number_of_audios += 1
                csv_writer.writerow(csv_line)
    logger.info(f"{output_csv_file} successfully created!")
    # Final prints
    total_duration = sum(durations)
    logger.info(f"Number of samples: {number_of_audios}.")
    logger.info(f"Total duration: {round(total_duration / 3600, 2)} hours.")
    logger.info(
        f"Median/Mean duration: {round(statistics.median(durations), 2)}/{round(total_duration/len(durations), 2)}."
    )
    logger.info(
        f"Ignored {num_ignored} audio files in total (due to duration issues) out of {number_of_audios+num_ignored}."
    )
    if num_nan_files > 0:
        logger.info(f"Ignored {num_nan_files} utterances due to nan value issues.")
    logger.info(
        f"Total duration of ignored files {round(ignored_duration / 60, 2)} minutes."
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


def _check_cv_v8_folders(data_folder: str, splits: List[str]):
    """
    Check if the data folder actually contains the eight version
    of the Common Voice dataset (English only).

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain the desired dataset.
    """

    def check_file_exists(f, *args):
        if len(args) != 0:
            f = os.path.join(f, *args)
        if not os.path.exists(f):
            err_msg = (
                "the folder %s does not exist while it is expected for the "
                "common-voice english dataset (version 1)." % f
            )
            raise FileNotFoundError(err_msg)

    check_file_exists(os.path.join(data_folder, "clips"))
    # Checking if all the splits exist
    for split in splits:
        # Expect the `cv-valid-{split}` folder and .csv file to exist
        split_tsv = os.path.join(data_folder, f"{split}.tsv")
        check_file_exists(split_tsv)


@cli.command()
@click.option(
    "--main_path", help="Path to the directory of the {train|dev|test}.csv files."
)
def combine_and_split(main_path: str):
    """
    Args:
        main_path: path to a directory where {train|dev|test}.csv already exists
    Returns:
        our_dir: Directory where the new train, dev, test splits are created.
        The train/dev/test splits are created by combining the original csv files, sampling 20%
        of the speakers as the test set and using the rest as the train/dev sets. From there,
        90% of the train set is used as the train set and 10% as the dev set.
    """
    splits = ["train", "dev", "test"]
    create_split = lambda s: os.path.join(main_path, s)
    for s in splits:
        assert os.path.exists(
            create_split(f"{s}.csv")
        ), f"{main_path}/{s} does not exist."
    out_dir = create_split("new_splits")
    create_out_split = lambda s: os.path.join(out_dir, s)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out_dataset = os.path.join(out_dir, "dataset.csv")
    origin = create_split("train.csv")
    copy(origin, out_dataset)
    with open(out_dataset, "a") as fa:
        with open(create_split("dev.csv"), "r") as fdev:
            fa.write("\n".join(fdev.readlines()[1:]))
        with open(create_split("test.csv"), "r") as ftest:
            fa.write("\n".join(ftest.readlines()[1:]))
    with open(out_dataset, "r") as fd:
        contents = [
            line.strip().replace("\n", "") + "\n"
            for line in fd
            if len(line.strip()) > 0 and line.strip() != "\n"
        ]
    header = contents.pop(0)
    # contents = [line.strip().replace("\n", "") + "\n" for line in contents if len(line.strip()) > 0 and line.strip() != "\n"]
    speakers = set([line.split(",")[5] for line in contents])
    test_speakers = random.sample(speakers, int(len(speakers) * 0.20))
    test_set = []
    train_dev_set = []
    for line in contents:
        if line.split(",")[5] in test_speakers:
            test_set.append(line)
        else:
            train_dev_set.append(line)
    random.shuffle(train_dev_set)
    # train/dev split percentages
    sperc = [0.9, 0.1]
    _p = int(sperc[0] * len(train_dev_set))
    with open(create_out_split("train.csv"), "w") as fw:
        fw.writelines([header] + train_dev_set[:_p])
    with open(create_out_split("dev.csv"), "w") as fw:
        fw.writelines([header] + train_dev_set[_p:])
    with open(create_out_split("test.csv"), "w") as fw:
        fw.writelines([header] + test_set)
    logger.info(f"Files {out_dir}/[train|dev|test].csv have been created")
    return out_dir


@cli.command()
@click.option("--main_path", help="Path to {train|dev|test}.csv files.")
def combine_and_split_per_accent(main_path: str):
    """
    Args:
        main_path: path to a directory where {train|dev|test}.csv already exists
    Returns:
        None, but creates a new directory called `accent_specific_data` in `main_path`,
        which contains the new splits for each accent. The purpose is to have a certain
        number of unique speakers in the test set. For German, this is not possible and
        so we simply sample a percentage of the data. The speaker percentages for the
        rest of the accents can be seen in the `PERCENTAGES_PER_ACCENT` dictionary.
    """
    splits = ["train", "dev", "test"]
    PERCENTAGES_PER_ACCENT = {"Indian": 0.1, "Tagalog": 0.22, "Malay": 0.2, "German": 0}
    accented_data_dir = os.path.join(main_path, "accent_specific_data")
    assert os.path.isdir(accented_data_dir), accented_data_dir
    accents = [f for f in os.listdir(accented_data_dir) if f.endswith(".csv")]
    assert len(accents) > 0, accents
    for acc in accents:
        acc = acc.split("_")[0].split()[0].replace(".csv", "")
        os.makedirs(os.path.join(accented_data_dir, acc), exist_ok=True)
    contents = []
    for s in splits:
        with open(os.path.join(main_path, f"{s}.csv"), "r") as fr:
            lines = fr.readlines()
            HEADER = lines[0]
            contents += lines[1:]
    # contents = [HEADER] + contents
    # speakers = set([line.split(",")[5] for line in contents])
    # test_speakers = random.sample(speakers, int(len(speakers)*0.20))
    speakers = defaultdict(set)
    for c in contents:
        acc = c.split(",")[2].split("_")[0].split()[0]
        speakers[acc].add(c.split(",")[5].replace("\n", "").strip())
    test_speakers = {}
    for acc in speakers.keys():
        test_speakers[acc] = random.sample(
            speakers[acc], int(len(speakers[acc]) * PERCENTAGES_PER_ACCENT[acc])
        )
    create_out_split = lambda s, acc: os.path.join(accented_data_dir, acc, s)
    for accent in accents:
        accent = accent.split("_")[0].replace(".csv", "")
        # accented_text = []
        train_dev_text = []
        test_text = []
        for c in contents:
            if c.split(",")[2].split("_")[0].split()[0] == accent:
                if c.split(",")[5].replace("\n", "").strip() in test_speakers[accent]:
                    test_text.append(c)
                else:
                    train_dev_text.append(c)
        random.shuffle(train_dev_text)
        # accented_text = [c for c in contents if c.split(",")[2].split("_")[0].split()[0] == accent]
        # train/dev/test split percentages
        if accent == "German":
            riddance_perc = int(0.1 * len(train_dev_text))
            test_text = train_dev_text[:riddance_perc]
            train_dev_text = train_dev_text[riddance_perc:]
        sperc = [0.9, 0.1]
        _p = int(sperc[0] * len(train_dev_text))
        # print("Accent", accent, "has", _p, "/", len(train_dev_text)-_p, "/", len(test_text), "train/dev/test speakers.")
        with open(create_out_split("train.csv", accent), "w") as fw:
            fw.writelines([HEADER] + train_dev_text[:_p])
        with open(create_out_split("dev.csv", accent), "w") as fw:
            fw.writelines([HEADER] + train_dev_text[_p:])
        with open(create_out_split("test.csv", accent), "w") as fw:
            fw.writelines([HEADER] + test_text)
        logger.info(
            f"Files {os.path.join(accented_data_dir, accent)}/[train|dev|test].csv have been created"
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
    accented_data_dir = os.path.join(main_path, "accent_specific_data")
    assert os.path.isdir(accented_data_dir), accented_data_dir
    create_acc_path = lambda acc: os.path.join(accented_data_dir, acc)
    accents = [
        f.split("_")[0].replace(".csv", "")
        for f in os.listdir(accented_data_dir)
        if f.endswith(".csv")
    ]
    assert os.path.isdir(main_path), main_path
    if not os.path.isdir(out_dir):
        logger.info(f"Creating output directory {out_dir}.")
        os.mkdir(out_dir)
    for split in splits:
        split_contents = []
        for acc in accents:
            logger.info(f"Processing the {acc} accent.")
            assert os.path.isdir(
                os.path.join(accented_data_dir, acc)
            ), f"{accented_data_dir}/{acc}"
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
@click.option(
    "--splits",
    "-s",
    multiple=True,
    help="A list of the splits that you want to read and process.",
)
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
    """
    Args:
        main_path: Path to the directory of the accented splits (see combine_and_split_per_accent function).
        splits: A list of the splits that you want to read and process.
        keep_accent: If true a csv file with `text,accent` will be saved.
        keep_accented_text: If true, multiple .txt files will be saved for each accent.
    Returns:
        Extracts the text from the CSV files of each provided split and saves them in a text file (for each accent/split combination).
    """
    ext = ".csv" if keep_accent else ".txt"
    out_path = os.path.join(main_path, f"text_only_{'_'.join(splits)}{ext}")
    # splits = ['train', 'dev', 'test']
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
    """
    Args:
        main_path: Path to the directory of the accented splits (see combine_and_split_per_accent function).
        splits: A list of the splits that you want to read and process.
    Returns:
        Extracts the text from the CSV file of a certain accent and saves it in a text file.
        You should prefer the extract_text function.
    """
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


def main():
    if "extract" in sys.argv[1]:
        extract_text()
    else:
        combine_and_split()


if __name__ == "__main__":
    cli()
    # main()
