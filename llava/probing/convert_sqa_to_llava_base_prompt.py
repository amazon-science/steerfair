from enum import Enum
from collections import OrderedDict

import numpy as np


class Attack_Type(Enum):
    START = "start"
    END = "end"
    MIDDLE = "middle"


def get_question_text(problem):
    question = problem["question"]
    return question


def get_context_text(problem, use_caption):
    txt_context = problem["hint"]
    img_context = problem["caption"] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm["choices"]
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def swap_choices(choices, pos1, pos2):
    choices_swapped = np.copy(choices).tolist()
    choices_swapped[pos1], choices_swapped[pos2] = choices_swapped[pos2], choices_swapped[pos1]
    return choices_swapped


def attack_choice_text_old(problem, options, attack_type, new_idx):
    choices = problem["choices"]  # ex choices: ['chiasmus', 'apostrophe']
    answer_idx = problem["answer"]  # ex answer = 1 (starts from 0)
    default_idx = new_idx
    if attack_type == Attack_Type.MIDDLE.value:
        """
        if len(options) == 2 => dont do anything
        else randomly choose indexes NOT 0/-1
        """
        # raise NotImplementedError("middle attack not implemented yet")
        if len(options) == 2:
            default_idx = answer_idx
        else:
            default_idx = new_idx
    choices_swapped = swap_choices(choices, default_idx, answer_idx)
    choice_list = []
    for i, c in enumerate(choices_swapped):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def get_answer(problem, options):
    return options[problem["answer"]]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem["lecture"].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem["solution"].replace("\n", "\\n")
    return solution


def create_one_example_chatbot(format, question, context, choice, answer, lecture, solution, test_example=True):
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == "A":
        output = f"Answer: The answer is {answer}."

    elif output_format == "AL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == "AE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == "ALE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == "AEL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == "LA":
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == "EA":
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == "LEA":
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == "ELA":
        output = f"Answer: {solution} {lecture} The answer is {answer}."
    elif output_format == "LEPA":
        output = ""
        if len(lecture.strip()) > 0:
            output += f"LECTURE: {lecture}\n"
        if len(solution.strip()) > 0:
            output += f"SOLUTION: {solution}\n"
        output += "###\n"
        output += f"ANSWER: {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if input.endswith("BECAUSE:"):
        input = input.replace("BECAUSE:", "").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()
    return input, output


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True):
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == "A":
        output = f"Answer: The answer is {answer}."

    elif output_format == "AL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == "AE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == "ALE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == "AEL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == "LA":
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == "EA":
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == "LEA":
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == "ELA":
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def create_one_example_gpt4(format, question, context, choice, answer, lecture, solution, test_example=True):
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == "A":
        output = f"Answer: The answer is {answer}."

    elif output_format == "AL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == "AE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == "ALE":
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == "AEL":
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == "LA":
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == "EA":
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == "LEA":
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == "ELA":
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()

    user_prompt = {"role": "user", "content": f"Can you explain {input}?"}
    assistant_prompt = {"role": "assistant", "content": f"{output}"}

    return user_prompt, assistant_prompt


def build_prompt_chatbot_old(
    problems,
    shot_qids,
    prompt_format,
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    is_test=False,
    attack_type=None,
):
    examples = {}
    new_gt_choices = {}
    for i, qid in enumerate(shot_qids):
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], use_caption)
        lecture = get_lecture_text(problems[qid]).replace("\\n", "\n")
        solution = get_solution_text(problems[qid]).replace("\\n", "\n")

        if attack_type != None:
            assert attack_type is not None
            if attack_type == Attack_Type.START.value:
                new_gt = 0
            elif attack_type == Attack_Type.END.value:
                new_gt = len(problems[qid]["choices"]) - 1
            else:
                orig_gt = problems[qid]["answer"]
                if len(problems[qid]["choices"]) <= 2:
                    new_gt = orig_gt
                else:
                    mid_idxs = np.arange(1, len(problems[qid]["choices"]) - 1)
                    remove_idx = np.argwhere(mid_idxs == orig_gt).flatten()
                    mid_idxs_removed = np.delete(mid_idxs, remove_idx)
                    if len(mid_idxs_removed) == 0:
                        new_gt = np.random.choice(mid_idxs)
                    else:
                        new_gt = np.random.choice(mid_idxs_removed)
            answer = options[new_gt]
            choice = attack_choice_text_old(problems[qid], options, attack_type, new_gt)
        train_example = create_one_example_chatbot(
            prompt_format, question, context, choice, answer, lecture, solution, test_example=is_test
        )
        examples[qid] = train_example
        new_gt_choices[qid] = int(new_gt)
    if attack_type == None:
        return examples
    return examples, new_gt_choices


def build_prompt_chatbot(
    problems, shot_qids, prompt_format, use_caption=False, options=["A", "B", "C", "D", "E"], is_test=False
):
    examples = {}

    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], use_caption)
        choice = get_choice_text(problems[qid], options)
        answer = get_answer(problems[qid], options)
        lecture = get_lecture_text(problems[qid]).replace("\\n", "\n")
        solution = get_solution_text(problems[qid]).replace("\\n", "\n")

        train_example = create_one_example_chatbot(
            prompt_format, question, context, choice, answer, lecture, solution, test_example=is_test
        )
        examples[qid] = train_example
    return examples


def attack_choice_text(problem, options, new_idx):
    choices = problem["choices"]  # ex choices: ['chiasmus', 'apostrophe']
    answer_idx = problem["answer"]  # ex answer = 1 (starts from 0)
    default_idx = answer_idx
    choices_swapped = swap_choices(choices, default_idx, new_idx)

    choice_list = []
    for i, c in enumerate(choices_swapped):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def permute_choice_text(problem, options, option_permutation):
    choices = problem["choices"]  # ex choices: ['chiasmus', 'apostrophe']
    choices_permuted = [choices[i] for i in option_permutation]
    choice_list = []
    for i, c in enumerate(choices_permuted):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def debias_baseline_chatbot(
    problems,
    shot_qids,
    prompt_format,
    option_permutation,
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    is_test=False,
):
    examples = {}
    new_gt_choices = {}
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], use_caption)
        lecture = get_lecture_text(problems[qid]).replace("\\n", "\n")
        solution = get_solution_text(problems[qid]).replace("\\n", "\n")
        answer = get_answer(problems[qid], options)

        choice = permute_choice_text(problems[qid], options, option_permutation)
        new_gt = np.argwhere(np.array(option_permutation) == problems[qid]["answer"]).flatten()[0]

        train_example = create_one_example_chatbot(
            prompt_format, question, context, choice, answer, lecture, solution, test_example=is_test
        )
        examples[qid] = train_example
        new_gt_choices[qid] = int(new_gt)
    return examples, new_gt_choices


def attack_prompt_chatbot(
    problems,
    shot_qids,
    prompt_format,
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    is_test=False,
    attack_idx=None,
):
    examples = {}
    new_gt_choices = {}
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], use_caption)
        lecture = get_lecture_text(problems[qid]).replace("\\n", "\n")
        solution = get_solution_text(problems[qid]).replace("\\n", "\n")
        answer = get_answer(problems[qid], options)
        if (attack_idx) == None or (problems[qid]["answer"] == attack_idx):
            choice = get_choice_text(problems[qid], options)
            new_gt = problems[qid]["answer"]
        else:
            new_gt = attack_idx
            choice = attack_choice_text(problems[qid], options, attack_idx)
        train_example = create_one_example_chatbot(
            prompt_format, question, context, choice, answer, lecture, solution, test_example=is_test
        )
        examples[qid] = train_example
        new_gt_choices[qid] = int(new_gt)
    return examples, new_gt_choices


def build_prompt(problems, shot_qids, test_qid, args):
    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(
            args.prompt_format, question, context, choice, answer, lecture, solution, test_example=False
        )
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(
        args.prompt_format, question, context, choice, answer, lecture, solution, test_example=True
    )
    examples.append(test_example)

    # create the prompt input
    prompt_input = "\n\n".join(examples)

    return prompt_input


def build_prompt_gpt4(problems, shot_qids, test_qid, args):
    prompt_array = [{"role": "system", "content": "You are a helpful assistant."}]

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        user_prompt, assistant_prompt = create_one_example_gpt4(
            args.prompt_format, question, context, choice, answer, lecture, solution, test_example=False
        )
        prompt_array.append(user_prompt)
        prompt_array.append(assistant_prompt)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    user_prompt, assistant_prompt = create_one_example_gpt4(
        args.prompt_format, question, context, choice, answer, lecture, solution, test_example=True
    )
    prompt_array.append(user_prompt)
    prompt_array.append(assistant_prompt)

    return prompt_array
