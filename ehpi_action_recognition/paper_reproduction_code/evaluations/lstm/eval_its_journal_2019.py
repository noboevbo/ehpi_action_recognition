import os

import numpy as np
from nobos_commons.utils.file_helper import get_create_path
from nobos_commons.visualization.evaluation_visualizer import plot_confusion_matrix

class_names = np.array(["idle", "jump", "sit", "walk", "wave"])


def eval_seq(seq_results: np.ndarray, plot_path: str, title: str):
    false_detected_sequences = []
    labels = []
    preds = []
    for i in range(0, seq_results.shape[0]):
        label = seq_results[i][0]
        pred = seq_results[i][1]
        sequence_id = seq_results[i][2]
        if label != pred:
            false_detected_sequences.append((label, pred, sequence_id))
            print("False Detection: Label: {} - Pred: {} - Seq: {}".format(label, pred, sequence_id))
        labels.append(label)
        preds.append(pred)
    plot_confusion_matrix(labels, preds, class_names, os.path.join(plot_path, "confusion_matrix_seq.png"), normalize=True,
                          title=title)
    np.savetxt(os.path.join(plot_path, "false_detections_seq.txt"), np.array(false_detected_sequences,dtype=np.int32),
               delimiter=',', fmt='%i')
    return labels, preds, false_detected_sequences

def eval_ehpi(ehpi_results: np.ndarray, plot_path: str, title: str):
    labels = list(ehpi_results[0])
    preds = list(ehpi_results[1])
    plot_confusion_matrix(labels, preds, class_names, os.path.join(plot_path, "confusion_matrix_ehpi.png"), normalize=True,
                          title=title)
    return labels, preds

if __name__ == '__main__':
    test_names = ["gt", "pose"]
    seeds = [0, 104, 123, 142, 200]
    model_names = [
        "ehpi_journal_2019_03_gt_seed_0_cp0200",
        "ehpi_journal_2019_03_gt_seed_104_cp0200",
        "ehpi_journal_2019_03_gt_seed_123_cp0200",
        "ehpi_journal_2019_03_gt_seed_142_cp0200",
        "ehpi_journal_2019_03_gt_seed_200_cp0200",
        #
        "ehpi_journal_2019_03_pose_seed_0_cp0200",
        "ehpi_journal_2019_03_pose_seed_104_cp0200",
        "ehpi_journal_2019_03_pose_seed_123_cp0200",
        "ehpi_journal_2019_03_pose_seed_142_cp0200",
        "ehpi_journal_2019_03_pose_seed_200_cp0200",
        #
        "ehpi_journal_2019_03_both_seed_0_cp0200",
        "ehpi_journal_2019_03_both_seed_104_cp0200",
        "ehpi_journal_2019_03_both_seed_123_cp0200",
        "ehpi_journal_2019_03_both_seed_142_cp0200",
        "ehpi_journal_2019_03_both_seed_200_cp0200",
    ]
    # Test set
    result_path = "/home/dennis/sync/cogsys/Projekte/ITS_JOURNAL_2019/results/its_journal_experiment_results/office"
    all_false_detected_seqs = {}
    for seed in seeds:
        all_false_detected_seqs[seed] = {}
        for test_name in test_names:
            all_false_detected_seqs[seed][test_name] = {}

    for test_name in test_names:
        all_labels_seq = []
        all_preds_seq = []
        all_labels_ehpi = []
        all_preds_ehpi = []
        for seed in seeds:
            model_name = "ehpi_journal_2019_03_{}_seed_{}_cp0200".format(test_name, seed)
            print("Model name: {}".format(model_name))

            ehpi_results = np.load(os.path.join(result_path, "{}_ehpis.npy".format(model_name)))
            seq_results = np.load(os.path.join(result_path, "{}_seqs.npy".format(model_name)))
            plot_path = get_create_path(os.path.join(result_path, model_name))

            labels_seq, preds_seq, false_detected_seqs = eval_seq(seq_results, plot_path, "ActionSim (Sequences): confusion matrix (normalized)")
            labels_ehpi, preds_ehpi = eval_ehpi(ehpi_results, plot_path, "ActionSim (Time Windows): confusion matrix (normalized)")
            all_labels_seq.extend(labels_seq)
            all_preds_seq.extend(preds_seq)
            all_false_detected_seqs[seed][test_name] = false_detected_seqs
            all_labels_ehpi.extend(labels_ehpi)
            all_preds_ehpi.extend(preds_ehpi)
        plot_confusion_matrix(all_labels_seq, all_preds_seq, class_names, os.path.join(result_path, "confusion_matrix_seq.png"),
                              normalize=True,
                              title="ActionSim (Sequences): confusion matrix (normalized)")
        plot_confusion_matrix(all_labels_ehpi, all_preds_ehpi, class_names, os.path.join(result_path, "confusion_matrix_ehpis.png"),
                              normalize=True,
                              title="ActionSim (Time Windows): confusion matrix (normalized)")
    false_detections: str = ""
    for seed, result_dict in all_false_detected_seqs.items():
        false_detections += "\n\n################################################\n" \
                            "#################### Seed: {} #################\n" \
                            "################################################\n\n".format(seed)
        false_detections += "#########  GT  #########\n"
        for gt_result in result_dict["gt"]:
            false_detections += "Label: {} / Pred: {} / Seq: {}\n".format(gt_result[0], gt_result[1], gt_result[2])
        false_detections += "\n######### Pose #########\n"
        for pose_result in result_dict["pose"]:
            false_detections += "Label: {} / Pred: {} / Seq: {}\n".format(pose_result[0], pose_result[1], pose_result[2])
    with open(os.path.join(result_path, "false_detections.txt"), "w") as txt_file:
        txt_file.write(false_detections)
    a = 2

