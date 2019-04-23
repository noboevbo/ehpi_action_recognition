from typing import Dict, List, Tuple

import numpy as np
import torch
from nobos_commons.tools.log_handler import logger
from torch.autograd import Variable
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TesterEhpi(object):
    def test(self, test_loader: DataLoader, weights_path: str, model):
        logger.error("Test model: {}".format(weights_path))

        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        model.to(device)
        ehpi_results = self.test_ehpis(model, test_loader)
        sequence_results = self.test_sequences(model, test_loader)
        return ehpi_results, sequence_results

    def test_ehpis(self, model, test_loader: DataLoader) -> Tuple[List[int], List[int]]:
        """
        Tests every single testset entry, so e.g. every 32frame EHPI.
        """
        print("Test all...")
        model.eval()
        all_ys = []
        all_predictions = []
        corrects = []
        for i, data in enumerate(test_loader):
            x = Variable(data["x"]).to(device)
            y = data["y"].numpy()[0]
            outputs = model(x).data.cpu().numpy()[0]
            predictions = np.argmax(outputs)
            all_ys.append(y)
            all_predictions.append(predictions)
            correct = predictions == y
            corrects.append(int(correct))
        results = [all_ys, list(all_predictions)]
        accuracy = float(sum(corrects)) / float(len(test_loader))
        logger.error("Test set accuracy: {}".format(accuracy))
        return results

    def test_sequences(self, model, test_loader: DataLoader) -> List[Tuple[int, int, int]]:
        """
        Tests every single testset entry but combines all results which belong to the same action sequence and
        calculates the end result by a voting based algorithm
        """
        print("Test by seq ...")
        model.eval()
        corrects = []
        sequence_labels: Dict[int, int] = {}
        sequence_results: Dict[int, List[int]] = {}
        label_count: Dict[int, int] = {}
        results = []
        for i, data in enumerate(test_loader):
            x = Variable(data["x"]).to(device)
            y = data["y"].numpy()[0]
            seq = data["seq"].numpy()[0]
            outputs = model(x).data.cpu().numpy()[0]
            predictions = np.argmax(outputs)
            if seq not in sequence_results:
                sequence_results[seq] = []
                sequence_labels[seq] = y
                if y not in label_count:
                    label_count[y] = 0
                label_count[y] += 1
            sequence_results[seq].append(predictions)
        corrects_per_label: Dict[int, List[int]] = {}
        for sequence_id, predictions in sequence_results.items():
            prediction = max(set(predictions), key=predictions.count)
            label = sequence_labels[sequence_id]
            correct = prediction == label
            results.append((label, prediction, sequence_id))
            # if not correct:
            #     print("Sequence: {} - Predicted: {} correct: {}".format(sequence_id, prediction, label))
            if label not in corrects_per_label:
                corrects_per_label[label] = []
            corrects_per_label[label].append(correct)
            corrects.append(int(correct))
        accuracy = float(sum(corrects)) / float(len(sequence_labels))
        logger.info("Test set accuracy: {} [Num: Test Sequences: {}]".format(accuracy, len(sequence_labels)))
        for label, corrects in corrects_per_label.items():
            accuracy = sum(corrects) / label_count[label]
            logger.info("Label accuracy: {} [Label: {}, Num_Tests: {}]".format(accuracy, label, label_count[label]))
        return results
