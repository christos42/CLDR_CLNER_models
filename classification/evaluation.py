import numpy as np
import os
import json
import configparser

parser = configparser.ConfigParser()
parser.read("./../configs/evaluation.conf")

SPLIT_NUM = int(parser.get("config", "split_num"))
PATH = './' + parser.get("config", "path")

class Evaluator:
    def __init__(self, split_num, path):
        self.split_num = split_num
        self.path = path
        self.test_files = self.read_file_paths()

    def read_file_paths(self):
        file_paths = []
        for root, dirs, files in os.walk(self.path + 'split_' + str(self.split_num) + '/'):
            for filename in files:
                if filename.endswith('.json'):
                    file_paths.append(self.path + 'split_' + str(self.split_num) + '/' + filename)


        return file_paths

    def get_chunk_type(self, tok, idx_to_tag):
        # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
        """
        Args:
            tok: id of token, ex 0
            idx_to_tag: dictionary {0: "B-DRUG", ...}
        Returns:
            tuple: "B", "DRUG"
        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    def get_chunks(seq, tags):
        # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
        # Altered because previous implementation recognized entities starting with I-tags as true
        """
            Given a sequence of tags, group entities and their position
            Args:
                seq: [4, 4, 0, 0, ...] sequence of labels
                tags: dict["O"] = 4
            Returns:
                list of (chunk_type, chunk_start, chunk_end)
            Example:
                seq = [0, 1, 4, 2]
                tags = {"B-DRUG": 0, "I-DRUG": 1, "B-AE": 2, "I-AE": 3, "O": 4}
                result = [("DRUG", 0, 1), ("AE", 3, 3)]
        """
        default = tags['O']
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i - 1)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
                if (chunk_type is None) and (tok_chunk_class == "B"):
                    chunk_type, chunk_start = tok_chunk_type, i
                else:
                    if tok_chunk_class == "B":
                        if chunk_type is not None:
                            chunk = (chunk_type, chunk_start, i - 1)
                            chunks.append(chunk)
                        chunk_type, chunk_start = tok_chunk_type, i
                    elif tok_chunk_type != chunk_type:
                        if chunk_type is not None:
                            chunk = (chunk_type, chunk_start, i - 1)
                            chunks.append(chunk)
                        chunk_type, chunk_start = None, None
            else:
                pass

        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq) - 1)
            chunks.append(chunk)

        return chunks

    def evaluate_RE(self, predicted_rel, gold_rel, gold_chunks, predicted_chunks):
        tp = 0
        fn = 0
        fp = 0
        for p_r in predicted_rel:
            drug_flag = 0
            ae_flag = 0
            # Check if the predicted pair is in gold list.
            if ([p_r[0], p_r[1]] in gold_rel):
                # Check if the span and type of name entities are correct.
                for p_c in predicted_chunks:
                    if p_c in gold_chunks:
                        if p_r[0] == p_c[2]:
                            drug_flag = 1
                        if p_r[1] == p_c[2]:
                            ae_flag = 1
            if ([p_r[1], p_r[0]] in gold_rel):
                for p_c in predicted_chunks:
                    if p_c in gold_chunks:
                        if p_r[1] == p_c[2]:
                            drug_flag = 1
                        if p_r[0] == p_c[2]:
                            ae_flag = 1

            if (drug_flag) == 1 and (ae_flag == 1):
                tp += 1
            else:
                fp += 1

        for g_r in gold_rel:
            if not (([g_r[0], g_r[1]] in predicted_rel) or ([g_r[1], g_r[0]] in predicted_rel)):
                fn += 1

        return tp, fp, fn

    def get_metrics_NER(self, tpsClassesNER, fpsClassesNER, fnsClassesNER):
        prec_ae = tpsClassesNER['AE'] / (tpsClassesNER['AE'] + fpsClassesNER['AE'])
        rec_ae = tpsClassesNER['AE'] / (tpsClassesNER['AE'] + fnsClassesNER['AE'])
        f1_ae = (2 * prec_ae * rec_ae) / (prec_ae + rec_ae)
        print('AE entity')
        print('Precision: {:.4f}'.format(prec_ae))
        print('Recall: {:.4f}'.format(rec_ae))
        print('F1 Score: {:.4f}'.format(f1_ae))
        print('___________________')

        prec_drug = tpsClassesNER['DRUG'] / (tpsClassesNER['DRUG'] + fpsClassesNER['DRUG'])
        rec_drug = tpsClassesNER['DRUG'] / (tpsClassesNER['DRUG'] + fnsClassesNER['DRUG'])
        f1_drug = (2 * prec_drug * rec_drug) / (prec_drug + rec_drug)
        print('DRUG entity')
        print('Precision: {:.4f}'.format(prec_drug))
        print('Recall: {:.4f}'.format(rec_drug))
        print('F1 Score: {:.4f}'.format(f1_drug))
        print('####################')
        print('####################')

        return {'DRUG': {'Recall': rec_drug,
                         'Precision': prec_drug,
                         'F1 score': f1_drug},
                'AE': {'Recall': rec_ae,
                       'Precision': prec_ae,
                       'F1 score': f1_ae}}

    def get_metrics_rel(self, tp, fp, fn):
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prec * rec) / (prec + rec)
        print('Relations:')
        print('Precision: {:.4f}'.format(prec))
        print('Recall: {:.4f}'.format(rec))
        print('F1 Score: {:.4f}'.format(f1))

        return {'Recall': rec,
                'Precision': prec,
                'F1 score': f1}

    def save_metrics(self, metrics_NER, metrics_REL):
        if not os.path.exists('results/'):
            os.makedirs('results/')

        with open('results/metrics_NER_split_' + str(self.split_num) + '.json', 'w') as fp:
            json.dump(metrics_NER, fp)

        with open('results/metrics_RE_split_' + str(self.split_num) + '.json', 'w') as fp:
            json.dump(metrics_REL, fp)

    def execute_evaluation(self):
        tpsClassesNER = {'DRUG': 0,
                         'AE': 0}
        fpsClassesNER = {'DRUG': 0,
                         'AE': 0}
        fnsClassesNER = {'DRUG': 0,
                         'AE': 0}

        tp_rel_global = fp_rel_global = fn_rel_global = 0
        mapping_ne_tags = {'B-DRUG': 0,
                           'I-DRUG': 1,
                           'B-AE': 2,
                           'I-AE': 3,
                           'O': 4}

        for c, f in enumerate(self.test_files):
            with open(f) as json_file:
                data = json.load(json_file)

            # Evaluate NER
            gold_seq = []
            for t in data['ne tags']:
                gold_seq.append(mapping_ne_tags[t])
            gold_chunks = self.get_chunks(gold_seq, mapping_ne_tags)

            predicted_seq = []
            for t in data['predictions - NER']:
                predicted_seq.append(mapping_ne_tags[t])
            predicted_chunks = self.get_chunks(predicted_seq, mapping_ne_tags)

            for lab_idx in range(len(predicted_chunks)):
                if predicted_chunks[lab_idx] in gold_chunks:
                    tpsClassesNER[predicted_chunks[lab_idx][0]] += 1
                else:
                    fpsClassesNER[predicted_chunks[lab_idx][0]] += 1


            for lab_idx in range(len(gold_chunks)):
                if gold_chunks[lab_idx] not in predicted_chunks:
                    fnsClassesNER[gold_chunks[lab_idx][0]] += 1

            # Evaluate relation extraction
            tp_rel, fp_rel, fn_rel = self.evaluate_RE(data['predictions - RE'],
                                                      data['relation pairs'],
                                                      gold_chunks,
                                                      predicted_chunks)

            tp_rel_global += tp_rel
            fp_rel_global += fp_rel
            fn_rel_global += fn_rel

        metrics_NER = self.get_metrics_NER(tpsClassesNER, fpsClassesNER, fnsClassesNER)
        metrics_REL = self.get_metrics_rel(tp_rel_global, fp_rel_global, fn_rel_global)

        self.save_metrics(metrics_NER, metrics_REL)

if __name__ == '__main__':
    eval_obj = Evaluator(SPLIT_NUM, PATH)
    eval_obj.execute_evaluation()