"""
File: human_eval_stats.py
Author: Lei Liu
Description: human evaluation results. Note that we assume there is only human annotator for each language.
"""
import os
import csv


def human_eval_stats(file_path):
    with open(file_path, 'r', newline='') as csv_human_eval_results:
        csv_reader = csv.reader(csv_human_eval_results)
        # Skip the header row
        csv_header = next(csv_reader)

        count = 0
        stats_fs = 0
        stats_fs_prompt = 0
        stats_fs_neutral = 0
        stats_mtl = 0
        stats_mtl_prompt = 0
        stats_mtl_neutral = 0

        # Print the remaining rows
        for row in csv_reader:
            result = row[4].strip()
            random_id = int(row[0])
            count += 1

            # FS-XLT v.s. FS-XLT_prompt (t5-style prompt)
            #
            # If the random id is an ODD number, "A" denotes "FS-XLT" and "B" denotes "FS-XLT_prompt"
            if count <= 100:
                if random_id % 2 == 1:
                    if result == "A":
                        stats_fs += 1
                    elif result == "B":
                        stats_fs_prompt += 1
                    else:
                        stats_fs_neutral += 1
                else:
                    if result == "A":
                        stats_fs_prompt += 1
                    elif result == "B":
                        stats_fs += 1
                    else:
                        stats_fs_neutral += 1
            # MTL v.s. MTL_prompt (t5-style prompt)
            #
            # If the random id is an ODD number, "A" denotes "MTL" and "B" denotes "MTL_prompt"
            else:
                if random_id % 2 == 1:
                    if result == "A":
                        stats_mtl += 1
                    elif result == "B":
                        stats_mtl_prompt += 1
                    else:
                        stats_mtl_neutral += 1
                else:
                    if result == "A":
                        stats_mtl_prompt += 1
                    elif result == "B":
                        stats_mtl += 1
                    else:
                        stats_mtl_neutral += 1

        print("Number of data examples in total: %d" % count)
        print("FS-XLT: %d" % stats_fs)
        print("Neutral: %d" % stats_fs_neutral)
        print("FS-XLT_prompt: %d" % stats_fs_prompt)
        print("MTL: %d" % stats_mtl)
        print("Neutral: %d" % stats_mtl_neutral)
        print("MTL_prompt: %d" % stats_mtl_prompt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # directory to human evaluation results
    parser.add_argument('--directory_human_eval_results', type=str, required=True)
    hparams = parser.parse_args()

    # Germanic and Romance languages for human evaluations
    germanic_languages = {'da', 'de', 'no'}
    romance_languages = {'es', 'it', 'pt'}

    for language in germanic_languages:
        file_human_eval_results = os.path.join(directory_human_eval_results, "human_eval_%s.csv" % language)
        print("**********Human Evaluation Results (%s)**********" % language)
        human_eval_stats(file_path=file_human_eval_results)
        print("***********************************************")

    for language in romance_languages:
        file_human_eval_results = os.path.join(directory_human_eval_results, "human_eval_%s.csv" % language)
        print("**********Human Evaluation Results (%s)**********" % language)
        human_eval_stats(file_path=file_human_eval_results)
        print("***********************************************")
