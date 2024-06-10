# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import sympy as sp

from .envs import set_cover_utils
from .utils import to_cuda, timeout, TimeoutError
from .envs.char_sp import InvalidPrefixExpression, is_valid_expr
from .envs.sympy_utils import simplify

logger = getLogger()

BUCKET_LENGTH_SIZE = 5


def idx_to_sp(env, idx, return_infix=False):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    prefix = env.unclean_prefix(prefix)
    infix = env.prefix_to_infix(prefix)
    eq = sp.S(infix, locals=env.local_dict)
    return (eq, infix) if return_infix else eq


@timeout(5)
def check_valid_solution(env, src, tgt, hyp):
    """
    Check that a solution is valid.
    """
    f = env.local_dict['f']
    x = env.local_dict['x']

    valid = simplify(hyp - tgt, seconds=1) == 0
    if not valid:
        diff = src.subs(f(x), hyp).doit()
        diff = simplify(diff, seconds=1)
        valid = diff == 0

    return valid


def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV
    src = idx_to_sp(env, eq['src'])
    tgt = idx_to_sp(env, eq['tgt'])
    hyp = eq['hyp']

    hyp_infix = [env.id2word[wid] for wid in hyp]

    try:
        hyp, hyp_infix = idx_to_sp(env, hyp, return_infix=True)
        is_valid = check_valid_solution(env, src, tgt, hyp)
        if is_valid_expr(hyp_infix):
            hyp_infix = str(hyp)

    except (TimeoutError, Exception) as e:
        e_name = type(e).__name__
        if not isinstance(e, InvalidPrefixExpression):
            logger.error(f"Exception {e_name} when checking hypothesis: {hyp_infix}")
        hyp = f"ERROR {e_name}"
        is_valid = False

    # update hypothesis
    f = env.local_dict['f']
    x = env.local_dict['x']
    eq['src'] = src.subs(f(x), 'f')  # hack to avoid pickling issues with lambdify
    eq['tgt'] = tgt
    eq['hyp'] = hyp_infix
    eq['is_valid'] = is_valid

    return eq


class Evaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

        # Sanda: Initializing for Error types in validation
        self.iteration_count_validtest = 0
        self.error_counts_validation = {
            "Wrong Format": 0,
            "Subset not in problem set": 0,
            "Solution not covering problem set": 0,
            "Solution is not optimal": 0
        }
        self.valid_solution_count_validation = 0
        self.coverage_percentage_validation = 0
        # Initialize epoch-wise counts


        # Sanda - for ploting:
        self.validation_loss_epochs = []
        self.validation_accuracy_epochs = []
        self.test_loss_epochs = []
        self.test_accuracy_epochs = []

        #Sanda - for ploting errors"
        # total number of sets processed for validation and test at each epoch
        self.validation_totalsets = 0
        self.validation_totalsets_epochs = []
        self.test_totalsets = 0
        self.test_totalsets_epochs = []
        # total number of valid answers for validation and test at each epoch
        self.validation_valid_answers = 0
        self.validation_valid_answers_epochs = []
        self.test_valid_answers = 0
        self.test_valid_answers_epochs = []
        # total number of wrong-format answers for validation and test at each epoch
        self.validation_error_wrong_format_count = 0
        self.validation_error_wrong_format_count_epochs = []
        self.test_error_wrong_format_count = 0
        self.test_error_wrong_format_count_epochs = []
        # total number of not existing-subset answers for validation and test at each epoch
        self.validation_error_no_subset_count = 0
        self.validation_error_no_subset_count_epochs = []
        self.test_error_no_subset_count = 0
        self.test_error_no_subset_count_epochs = []
        # total number of not covering answers for validation and test at each epoch
        self.validation_error_not_covering_count = 0
        self.validation_error_not_covering_count_epochs = []
        self.test_error_not_covering_count = 0
        self.test_error_not_covering_count_epochs = []
        # total number of not optimal answers for validation and test at each epoch
        self.validation_error_not_optimal_count = 0
        self.validation_error_optimal_count_epochs = []
        self.test_error_not_optimal_count = 0
        self.test_error_optimal_count_epochs = []
        # coverage percentage calculations for not-covering
        self.validation_coverage_percentage_list_internal = []
        self.validation_coverage_average_percentage = 0
        self.validation_coverage_average_percentage_epochs = []
        self.test_coverage_percentage_list_internal = []
        self.test_coverage_average_percentage = 0
        self.test_coverage_average_percentage_epochs = []



    def run_all_evals(self):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': self.trainer.epoch})

        # save statistics about generated data
        if self.params.export_data:
            scores['total'] = sum(self.trainer.EQUATIONS.values())
            scores['unique'] = len(self.trainer.EQUATIONS)
            scores['unique_prop'] = 100. * scores['unique'] / scores['total']
            return scores

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                for task in self.params.tasks:
                    if self.params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)

        return scores



    def enc_dec_step(self, data_type, task, scores):

        #Sanda-Start:
        self.error_counts_validation = {
            "Wrong Format": 0,
            "Subset not in problem set": 0,
            "Solution not covering problem set": 0,
            "Solution is not optimal": 0
        }
        idx = 0
        # total set count
        self.validation_totalsets = 0
        self.test_totalsets = 0
        # valid ansnwers count
        self.validation_valid_answers = 0
        self.test_valid_answers = 0
        # wrong_format count
        self.validation_error_wrong_format_count = 0
        self.test_error_wrong_format_count = 0
        # no-subset count
        self.validation_error_no_subset_count = 0
        self.test_error_no_subset_count = 0
        # not_covering count
        self.validation_error_not_covering_count = 0
        self.test_error_not_covering_count = 0
        # not_optimal count
        self.validation_error_not_optimal_count = 0
        self.test_error_not_optimal_count = 0
        # AVERAGE
        self.validation_coverage_average_percentage = 0
        self.validation_coverage_percentage_list_internal = []
        self.test_coverage_average_percentage = 0
        self.test_coverage_percentage_list_internal = []

        # Sanda-End

        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ['prim_fwd', 'prim_bwd', 'prim_ibp', 'ode1', 'ode2', 'set_cover']

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.{task}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(data_type, task, params=params, data_path=self.trainer.data_path)
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:
            idx += 1

            # print status
            if n_total.sum().item() % 100 < params.batch_size:
                logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            #Sanda-Start:
            input_token_sequence = set_cover_utils.convert_tensor_to_token_sequence(self, x1)
            true_output_token_sequence = set_cover_utils.convert_tensor_to_token_sequence(self, x2)
            #Sanda-End

            # forward / loss
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            #Sanda-Start:
            with ((((torch.no_grad())))):
                pred_token_sequence = []

                _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=1, length_penalty=1.0,
                                                   early_stopping=1, max_len=200)

                for b in range(len(beam)):
                    best_hypothesis = beam[b].hyp[0][1]  # Best hypothesis in the beam
                    ids = best_hypothesis[1:].tolist()  # Decoded token IDs
                    tok = [self.env.id2word[wid] for wid in ids]  # Convert token IDs to tokens
                    pred_token_sequence.append(tok)  # Store the predicted sequence for this beam

                its = input_token_sequence
                tots = true_output_token_sequence
                pts = pred_token_sequence

                for input_token_sequence, true_output_token_sequence, pred_token_sequence in zip(its, tots, pts):

                    input_list_of_lists = set_cover_utils.convert_token_sequence_to_list_of_lists(self, input_token_sequence)
                    true_output_list_of_lists = set_cover_utils.convert_token_sequence_to_list_of_lists(self, true_output_token_sequence)
                    pred_list_of_lists = set_cover_utils.convert_token_sequence_to_list_of_lists(self, pred_token_sequence)

                    if data_type == 'valid':
                        print("Validation iteration:", self.iteration_count_validtest)
                    if data_type == 'test':
                        print("Test iteration:", self.iteration_count_validtest)

                    #print(f"Input Token Sequence(E) {idx}:", input_token_sequence)
                    #print(f"True Output Token Sequence(E) {idx}:", true_output_token_sequence)
                    #print(f"Prediction Token Sequence(E) {idx}:", pred_token_sequence)
                    print(f"Input list of lists(E) {idx}:", input_list_of_lists)
                    print(f"True output list of lists(E) {idx}:", true_output_list_of_lists)
                    print(f"Prediction list of lists(E) {idx}:", pred_list_of_lists)

                    self.valid_solution_count_validation, self.error_counts_validation, self.coverage_percentage_validation = set_cover_utils.validate_set_cover_solution_validation(self, input_list_of_lists, pred_list_of_lists, true_output_list_of_lists)

                    print(f"Number of correct answers in a sequence:", self.valid_solution_count_validation)
                    print(f"error_counts in a sequence: {self.error_counts_validation}")

                    # Calculating total number of sets in one epoch
                    if data_type == 'valid':
                        self.validation_totalsets += 1
                        self.validation_valid_answers += self.valid_solution_count_validation
                        self.validation_error_wrong_format_count += self.error_counts_validation["Wrong Format"]
                        self.validation_error_no_subset_count += self.error_counts_validation["Subset not in problem set"]
                        self.validation_error_not_covering_count += self.error_counts_validation["Solution not covering problem set"]
                        if self.error_counts_validation["Solution not covering problem set"] == 1:
                            self.validation_coverage_percentage_list_internal.append(self.coverage_percentage_validation)
                        self.validation_error_not_optimal_count += self.error_counts_validation["Solution is not optimal"]
                        self.iteration_count_validtest += 1

                    if data_type == 'test':
                        self.test_totalsets += 1
                        self.test_valid_answers += self.valid_solution_count_validation
                        self.test_error_wrong_format_count += self.error_counts_validation["Wrong Format"]
                        self.test_error_no_subset_count += self.error_counts_validation["Subset not in problem set"]
                        self.test_error_not_covering_count += self.error_counts_validation["Solution not covering problem set"]
                        if self.error_counts_validation["Solution not covering problem set"] == 1:
                            self.test_coverage_percentage_list_internal.append(self.coverage_percentage_validation)
                        self.test_error_not_optimal_count += self.error_counts_validation["Solution is not optimal"]
                        self.iteration_count_validtest += 1

            # export evaluation details
            if params.eval_verbose:

                for i in range(len(len1)):
                    src = idx_to_sp(env, x1[1:len1[i] - 1, i].tolist())
                    tgt = idx_to_sp(env, x2[1:len2[i] - 1, i].tolist())
                    s = f"Equation {n_total.sum().item() + i} ({'Valid' if valid[i] else 'Invalid'})\nsrc={src}\ntgt={tgt}\n"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()



            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        # check this!
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size or self.trainer.data_path
        scores[f'{data_type}_{task}_xe_loss'] = xe_loss / _n_total
        scores[f'{data_type}_{task}_acc'] = 100. * _n_valid / _n_total

        #Sanda: (Here we are outside the iterations)/end of epoch
        # Append the metrics to the respective lists
        self.iteration_count_validtest = 0

        if task == 'set_cover':
            if data_type == 'valid':
                print("total number of sets in validation in this sequence", self.validation_totalsets)
                print("among them how many are correct?", self.validation_valid_answers)
                print("total number of Wrong formats errors in validation:", self.validation_error_wrong_format_count)
                print("total number of absent subset errors in validation:", self.validation_error_no_subset_count)
                print("total number of not covering errors in validation:", self.validation_error_not_covering_count)
                print("total number of not optimal errors in validation:", self.validation_error_not_optimal_count)
                print("list of coverage percentages of not-covering solutions for this epoch (validation):", self.validation_coverage_percentage_list_internal)

                if self.validation_coverage_percentage_list_internal:
                    self.validation_coverage_average_percentage = round(
                        sum(self.validation_coverage_percentage_list_internal) / len(
                            self.validation_coverage_percentage_list_internal), 2)
                else:
                    self.validation_coverage_average_percentage = 0

                print("average coverage percentage (validation) for the epoch is:", self.validation_coverage_average_percentage)


                # Appending the lists at the end of epoch
                self.validation_loss_epochs.append(scores[f'{data_type}_{task}_xe_loss'])
                self.validation_accuracy_epochs.append(scores[f'{data_type}_{task}_acc'])
                #
                self.validation_totalsets_epochs.append(self.validation_totalsets)
                self.validation_valid_answers_epochs.append(self.validation_valid_answers)
                self.validation_error_wrong_format_count_epochs.append(self.validation_error_wrong_format_count)
                self.validation_error_no_subset_count_epochs.append(self.validation_error_no_subset_count)
                self.validation_error_not_covering_count_epochs.append(self.validation_error_not_covering_count)
                self.validation_error_optimal_count_epochs.append(self.validation_error_not_optimal_count)
                self.validation_coverage_average_percentage_epochs.append(self.validation_coverage_average_percentage)

                # Printing the lists
                print("list of number of sets epochs (validation):", self.validation_totalsets_epochs)
                print("list of valid answers for epochs (validation):",self.validation_valid_answers_epochs)
                print("list of Wrong format answers for epochs (validation):",self.validation_error_wrong_format_count_epochs)
                print("list of absent subset answers for epochs (validation):",self.validation_error_no_subset_count_epochs)
                print("list of not covering answers for epochs (validation):", self.validation_error_not_covering_count_epochs)
                print("list of not optimal answers for epochs (validation):", self.validation_error_optimal_count_epochs)
                print("list of average percentage of not covering answers for epochs (validation):",
                      self.validation_coverage_average_percentage_epochs)


            elif data_type == 'test':
                print("total number of sets in test in this sequence", self.test_totalsets)
                print("among them how many are correct?", self.test_valid_answers)
                print("total number of Wrong formats errors in test", self.test_error_wrong_format_count)
                print("total number of absent subset errors in test", self.test_error_no_subset_count)
                print("total number of not covering errors in test", self.test_error_not_covering_count)
                print("total number of not optimal errors in test", self.test_error_not_optimal_count)
                print("list of coverage percentages of not-covering solutions for this epoch (test):",
                      self.test_coverage_percentage_list_internal)
                if self.test_coverage_percentage_list_internal:
                    self.test_coverage_average_percentage = round(
                        sum(self.test_coverage_percentage_list_internal) / len(
                            self.test_coverage_percentage_list_internal), 2)
                else:
                    self.test_coverage_average_percentage = 0
                print("average coverage percentage (test) for the epoch is:", self.test_coverage_average_percentage)


                # Appending the lists at the end of epochs
                self.test_loss_epochs.append(scores[f'{data_type}_{task}_xe_loss'])
                self.test_accuracy_epochs.append(scores[f'{data_type}_{task}_acc'])
                #
                self.test_totalsets_epochs.append(self.test_totalsets)
                self.test_valid_answers_epochs.append(self.test_valid_answers)
                self.test_error_wrong_format_count_epochs.append(self.test_error_wrong_format_count)
                self.test_error_no_subset_count_epochs.append(self.test_error_no_subset_count)
                self.test_error_not_covering_count_epochs.append(self.test_error_not_covering_count)
                self.test_error_optimal_count_epochs.append(self.test_error_not_optimal_count)
                self.test_coverage_average_percentage_epochs.append(self.test_coverage_average_percentage)

                # Printing the lists
                print("list of number of sets for epochs (test):",self.test_totalsets_epochs)
                print("list of valid answers for epochs (test):",self.test_valid_answers_epochs)
                print("list of wrong format answers for epochs (test):",self.test_error_wrong_format_count_epochs)
                print("list of absent subset answers for epochs (test):",self.test_error_no_subset_count_epochs)
                print("list of not covering answers for epochs (test):",self.test_error_not_covering_count_epochs)
                print("list of not optimal answers for epochs (test)",self.test_error_optimal_count_epochs)
                print("list of average percentage of not covering answers for epochs (test):",
                      self.test_coverage_average_percentage_epochs)


            else:
                # Handle unexpected data types here
                pass


        #Sanda

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            # logger.info(f"{i}: {n_valid[i].item()} / {n_total[i].item()} ({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)")
            scores[f'{data_type}_{task}_acc_{i}'] = 100. * n_valid[i].item() / max(n_total[i].item(), 1)

        #Sanda:
            # At the end of all epochs, save these lists to a JSON file
            metrics = {
                "validation_loss_epochs": self.validation_loss_epochs,
                "validation_accuracy_epochs": self.validation_accuracy_epochs,
                "test_loss_epochs": self.test_loss_epochs,
                "test_accuracy_epochs": self.test_accuracy_epochs
            }

            with open("metrics.json", "w") as json_file:
                json.dump(metrics, json_file)

            # To save yhe metrics based on my function:
            results_validtest = {
                "validation_totalsets_epochs" : self.validation_totalsets_epochs,
                "test_totalsets_epochs" : self.test_totalsets_epochs,
                "validation_valid_answers_epochs" : self.validation_valid_answers_epochs,
                "test_valid_answers_epochs" : self.test_valid_answers_epochs,
                "validation_error_wrong_format_count_epochs" : self.validation_error_wrong_format_count_epochs,
                "test_error_wrong_format_count_epochs" : self.test_error_wrong_format_count_epochs,
                "validation_error_no_subset_count_epochs" : self.validation_error_no_subset_count_epochs,
                "test_error_no_subset_count_epochs" : self.test_error_no_subset_count_epochs,
                "validation_error_not_covering_count_epochs" : self.validation_error_not_covering_count_epochs,
                "test_error_not_covering_count_epochs" : self.test_error_not_covering_count_epochs,
                "validation_error_not_optimal_count_epochs" : self.validation_error_optimal_count_epochs,
                "test_error_not_optimal_count_epochs" : self.test_error_optimal_count_epochs,
                "validation_coverage_percentage_epochs": self.validation_coverage_average_percentage_epochs,
                "test_coverage_percentage_epochs": self.test_coverage_average_percentage_epochs

            }
            with open("results_validtest.json", "w") as json_file2:
                json.dump(results_validtest, json_file2)

            # Reseting counters for next epoch
            self.validation_totalsets = 0
            self.validation_valid_answers = 0
            self.validation_error_wrong_format_count = 0
            self.validation_error_no_subset_count = 0
            self.validation_error_not_covering_count = 0
            self.validation_error_not_optimal_count = 0
            self.validation_coverage_average_percentage = 0
            self.validation_coverage_percentage_list_internal = []

            self.test_totalsets = 0
            self.test_valid_answers = 0
            self.test_error_wrong_format_count = 0
            self.test_error_no_subset_count = 0
            self.test_error_not_covering_count = 0
            self.test_error_not_optimal_count = 0
            self.test_coverage_percentage_list_internal = []




    def enc_dec_step_beam(self, data_type, task, scores):

        # Sanda-Start:
        idx = 0
        pred_token_sequence = []
        # Sanda-End

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ['prim_fwd', 'prim_bwd', 'prim_ibp', 'ode1', 'ode2']

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.{task}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res['hyps']])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\nsrc={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res['hyps']:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, params.beam_size, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # iterator
        iterator = env.create_test_iterator(data_type, task, params=params, data_path=self.trainer.data_path)
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            bs = len(len1)

            # Sanda-Start:
            input_token_sequence = set_cover_utils.convert_tensor_to_token_sequence(self, x1)
            true_output_token_sequence = set_cover_utils.convert_tensor_to_token_sequence(self, x2)
            # Sanda-End

            # forward
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_sp(env, x1[1:len1[i] - 1, i].tolist())
                tgt = idx_to_sp(env, x2[1:len2[i] - 1, i].tolist())
                if valid[i]:
                    beam_log[i] = {'src': src, 'tgt': tgt, 'hyps': [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            n_valid[:, 0].index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found {bs - len(invalid_idx)}/{bs} valid top-1 predictions. Generating solutions ...")

            # generate
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=params.max_len
            )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)):
                    inputs.append({
                        'i': i,
                        'j': j,
                        'score': score,
                        'src': x1[1:len1[i] - 1, i].tolist(),
                        'tgt': x2[1:len2[i] - 1, i].tolist(),
                        'hyp': hyp[1:].tolist(),
                    })

            # check hypotheses with multiprocessing
            outputs = []
            with ProcessPoolExecutor(max_workers=20) as executor:
                for output in executor.map(check_hypothesis, inputs, chunksize=1):
                    outputs.append(output)

            # read results
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o['i'] == i], key=lambda x: x['j'])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (i in beam_log) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]['src']
                tgt = gens[0]['tgt']
                beam_log[i] = {'src': src, 'tgt': tgt, 'hyps': []}

                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert gen['src'] == src and gen['tgt'] == tgt and gen['i'] == i and gen['j'] == j

                    # if the hypothesis is correct, and we did not find a correct one before
                    is_valid = gen['is_valid']
                    if is_valid and not valid[i]:
                        n_valid[nb_ops[i], j] += 1
                        valid[i] = 1

                    # update beam log
                    beam_log[i]['hyps'].append((gen['hyp'], gen['score'], is_valid))

            # valid solutions found with beam search
            logger.info(f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses.")


            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")
        logger.info(n_valid[
                    :(n_valid.sum(1) > 0).nonzero().view(-1)[-1] + 1,
                    :(n_valid.sum(0) > 0).nonzero().view(-1)[-1] + 1
                    ])

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size or self.trainer.data_path
        scores[f'{data_type}_{task}_beam_acc'] = 100. * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            logger.info(
                f"{i}: {n_valid[i].sum().item()} / {n_total[i].item()} ({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)")
            scores[f'{data_type}_{task}_beam_acc_{i}'] = 100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)


def convert_to_text(batch, lengths, id2word, params):
    """
    Convert a batch of sequences to a list of text sequences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sequences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(id2word[batch[k, j]])
        sequences.append(" ".join(words))
    return sequences
