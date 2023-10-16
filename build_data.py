"""
File: build_data.py
Author: Lei Liu
Date: Dec 26, 2022

Description: Build data for few-shot cross-lingual learning (FS-XLT) and multi-task learning (MTL).
"""
import os
import csv
import random
import argparse


def build_data(directory_ori, directory_fs, directory_multitask, lang_src, lang_tgt, num_shot):
    """Build the data for a specific language pair.

    Args:
        directory_ori: The directory to the original MDIA dataset.
        directory_fs: The directory to the FS-XLT data.
        directory_multitask: The directory to the MTL data.
        lang_src: The source language in FS-XLT or auxiliary language in MTL.
        lang_tgt: The target language in FS-XLT or MTL.
        num_shot: The number of 'context-response pairs' in the target language for FS-XLT and MTL.
    """

    file_path_tgt = os.path.join(directory_ori, "%s_train.csv" % lang_tgt)

    if not os.path.exists(directory_fs):
        os.mkdir(directory_fs)
    # The *.csv file for FS-XLT data
    file_path_fs = os.path.join(directory_fs, "%s_fs_%d.csv" % (lang_tgt, num_shot))

    if not os.path.exists(directory_multitask):
        os.mkdir(directory_multitask)
    # The *.csv file for MTL data
    file_path_multitask = os.path.join(directory_multitask, "%s_%s_%d_train.csv" % (lang_src, lang_tgt, num_shot))

    # Get the number of 'context-response pairs' in the full training data of the 'target language'
    num_tgt = 0
    with open(file_path_tgt, 'r') as file_tgt:
        csv_reader_tgt = csv.reader(file_tgt)
        csv_header = next(csv_reader_tgt)
        for row in csv_reader_tgt:
            num_tgt += 1

    # Build data for FS-XLT and MTL if and only if the training data is sufficient
    if num_shot <= num_tgt:
        # Generate random numbers and sort them in an ascending order (random seed is set to 2023)
        random.seed(2023)
        random_numbers = sorted(random.sample(range(num_tgt), num_shot))
        print(random_numbers)

        # Build data for FS-XLT: randomly pick up few-shot data from the full training data based on the random numbers
        few_shots = []
        # Open the training data for 'target language'
        with open(file_path_tgt, 'r') as file_tgt:
            csv_reader_tgt = csv.reader(file_tgt)
            csv_header_tgt = next(csv_reader_tgt)
            count = 0
            # Create a file to save the few-shot data
            with open(file_path_fs, "w") as file_few_shot:
                csv_writer_fs = csv.writer(file_few_shot)
                csv_writer_fs.writerow(csv_header_tgt)
                for row in csv_reader_tgt:
                    # Pick the row with a numbering that matches the random number
                    if count == random_numbers[0]:
                        # Write current 'context-response pair' into the file
                        csv_writer_fs.writerow(row)
                        # Save current 'context-response pair' in a list
                        few_shots.append(row)
                        # Remove the numbering of current 'context-response pair' from the list of random numbers
                        random_numbers.remove(count)
                        # Jump out of the loop when reaching the very end of the list
                        if len(random_numbers) == 0:
                            break
                    count += 1
                print("The %d-shot data of (\'%s\') language is ready for FS-XLT." % (num_shot, lang_tgt))

        # Build data for MTL: interleave the full training data of 'auxiliary language' with
        # the few-shot data of 'target language' built for FS-XLT in the above.
        #
        # Get the number of 'context-response pairs' in the full training data of the 'auxiliary language'.
        num_src = 0
        with open(os.path.join(directory_ori, "%s_train.csv" % lang_src), 'r') as file_src:
            csv_reader_src = csv.reader(file_src)
            csv_header_src = next(csv_reader_src)
            for row in csv_reader_src:
                num_src += 1

        # Open the training data for 'auxiliary language'
        with open(os.path.join(directory_ori, "%s_train.csv" % lang_src), 'r') as file_src:
            csv_reader_src = csv.reader(file_src)
            csv_header_src = next(csv_reader_src)
            count_multitask_src = 0
            # Create a file to save the interleaved data for MTL
            with open(file_path_multitask, 'w') as multitask_file:
                csv_writer_multitask_file = csv.writer(multitask_file)
                csv_writer_multitask_file.writerow(csv_header_src)
                incremental = num_src / num_shot
                for row in csv_reader_src:
                    # Write every 'context-response pair' of the 'auxiliary language' into the *.csv file
                    csv_writer_multitask_file.writerow(row)
                    # Add one to the counter
                    count_multitask_src += 1
                    # For every num_src/num_shot 'context-response pairs' of the 'auxiliary language',
                    # we write ONE 'context-response pair' of the 'target language' into the *.csv file
                    if count_multitask_src % incremental == 0:
                        csv_writer_multitask_file.writerow(few_shots[0])
                        few_shots = few_shots[1:]
                print("The interleaved data that contains the full training data of 'auxiliary language' (i.e. '%s') "
                      "and the %d-shot data of 'target language' (i.e. '%s') is ready for MTL."
                      % (lang_src, lang_tgt, num_shot))
    else:
        print("Insufficient training data of 'target language' (i.e. '%s')! The number (i.e. %d) of 'context-response "
              "pairs' in the training data of '%s' is less than the number (i.e. %d) of 'context-response pairs' "
              "(i.e. few-shot data) needed for FS-XLT and MTL." % (lang_tgt, num_tgt, lang_tgt, num_shot))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_ori', type=str, required=True)
    parser.add_argument('--directory_fs', type=str, required=True)
    parser.add_argument('--directory_multitask', type=str, required=True)
    hparams = parser.parse_args()

    # Get the iso codes for all the languages in the original MDIA dataset except for 'pl' because there is something
    # wrong in its training data file.
    lang_iso_codes = []
    for file in os.listdir(hparams.directory_ori):
        lang_iso = file.split("_")[0]
        lang_iso_codes.append(lang_iso)
    lang_iso_codes = sorted(lang_iso_codes)
    # Remove "pl" language as there is something wrong with its training data
    lang_iso_codes.remove("pl")

    # English as the 'source language' in FS-XLT or 'auxiliary language' in MTL
    lang_src = 'en'
    # The number of 'context-response pairs' of 'target language'
    num_shot = 10
    # Build data for all the 'target languages'
    for lang_iso in lang_iso_codes:
        # Note that the 'target language' should be different from 'source language'
        if lang_iso != lang_src:
            build_data(directory_ori=hparams.directory_ori,
                       directory_fs=hparams.directory_fs,
                       directory_multitask=hparams.directory_multitask,
                       lang_src=lang_src,
                       lang_tgt=lang_iso,
                       num_shot=num_shot)

    print("The 'source language':" + lang_src)
    print("All the 'target languages':")
    print(lang_iso_codes)
