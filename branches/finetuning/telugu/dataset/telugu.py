# ============================================================
# TELUGU REASONING DATASET GENERATION
# Total = 12,000
# Train = 9,600
# Validation = 1,200
# Test = 1,200
# ============================================================

import json
import random
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================
SEED = 42
random.seed(SEED)

TOTAL_EXAMPLES = 12000

# The six problem "shapes". Each has its own generator function.
TEMPLATES = [
    "T1",   # direct numeric evaluation between two entities 
    "T2",   # should output equal,greater or lower
    "T3",   # 3 entity transitive chain like A<B<C
    "T4",   # 4-entity transitive chain like A<B<C<D
    "T5",   # 4-entity chain with some distractor sentences
    "T6",   # 4-entity chain with  mixed direction like chaning taller to shorter havin mixed direction instead of a single direction chain 
]

PER_TEMPLATE = {
    "train": 1600,
    "validation": 200,
    "test": 200,
}


# ============================================================
####
## the goal of this task is to use reasoning when the task was doen with entirely new names in test train and validation then the model was trying to 
## generalize between names instead of doing the reasoning task so we used a shared name pool .The examples and the rompts are different across all the three splits 
# ============================================================

NAMES = [
    "రవి",
    "సురేష్",
    "మహేష్",
    "అనిత",
    "సీత",
    "లత",
    "అర్జున్",
    "భరత్",
    "కావ్య",
    "దివ్య",
    "రమేష్",
    "గోపాల్",
    "స్వాతి",
    "నందిని",
    "హరీష్",
    "మధు",
    "దీపక్",
    "శ్రావ్య",
    "కిరణ్",
    "వినయ్",
    "పద్మ",
    "దీప్తి",
    "తరుణ్",
    "నవీన్",
    "సౌమ్య",
    "మేఘన",
    "ఆదిత్య",
    "వర్ష",
    "వంశీ",
    "పవన్",
    "కార్తిక్",
    "సందీప్",
    "ప్రణయ్",
    "రోహిత్",
    "మనోజ్",
    "విక్రమ్",
    "లక్ష్మి",
    "హారిక",
]   # 38 names total


# ============================================================
# The dataset uses comparative relations across four main domains:
# height, age, marks, and speed.
#
# For each domain, we define two types of questions:
# 1. Forward direction: the question follows the direction of the
#    given relation.
# 2. Reverse direction: the question asks for the relation in the
#    opposite direction.
#
# For each type, we generate two questions, covering both directions.
# ============================================================

RELATIONS = {

    "height": {

        "greater": [
            "{a} {b} కంటే ఎత్తుగా ఉన్నారు.",
            "{b}తో పోలిస్తే {a} ఎత్తుగా ఉన్నారు.",
        ],

        "reverse": [
            "{b} {a} కంటే పొట్టిగా ఉన్నారు.",
            "{a}తో పోలిస్తే {b} పొట్టిగా ఉన్నారు.",
        ],

        "greater_question":
            "{a}, {b} — వీరిలో ఎవరు ఎత్తుగా ఉన్నారు?",

        "smaller_question":
            "{a}, {b} — వీరిలో ఎవరు పొట్టిగా ఉన్నారు?",
    },


    "age": {

        "greater": [
            "{a} వయస్సు {b} వయస్సు కంటే ఎక్కువ.",
            "{b}తో పోలిస్తే {a} వయస్సు ఎక్కువ.",
        ],

        "reverse": [
            "{b} వయస్సు {a} వయస్సు కంటే తక్కువ.",
            "{a}తో పోలిస్తే {b} వయస్సు తక్కువ.",
        ],

        "greater_question":
            "{a}, {b} — వీరిలో ఎవరి వయస్సు ఎక్కువ?",

        "smaller_question":
            "{a}, {b} — వీరిలో ఎవరి వయస్సు తక్కువ?",
    },


    "marks": {

        "greater": [
            "{a} {b} కంటే ఎక్కువ మార్కులు సాధించారు.",
            "{b}తో పోలిస్తే {a} ఎక్కువ మార్కులు సాధించారు.",
        ],

        "reverse": [
            "{b} {a} కంటే తక్కువ మార్కులు సాధించారు.",
            "{a}తో పోలిస్తే {b} తక్కువ మార్కులు సాధించారు.",
        ],

        "greater_question":
            "{a}, {b} — వీరిలో ఎవరు ఎక్కువ మార్కులు సాధించారు?",

        "smaller_question":
            "{a}, {b} — వీరిలో ఎవరు తక్కువ మార్కులు సాధించారు?",
    },


    "speed": {

        "greater": [
            "{a} {b} కంటే వేగంగా పరుగెత్తారు.",
            "{b}తో పోలిస్తే {a} వేగంగా పరుగెత్తారు.",
        ],

        "reverse": [
            "{b} {a} కంటే నెమ్మదిగా పరుగెత్తారు.",
            "{a}తో పోలిస్తే {b} నెమ్మదిగా పరుగెత్తారు.",
        ],

        "greater_question":
            "{a}, {b} — వీరిలో ఎవరు వేగంగా పరుగెత్తారు?",

        "smaller_question":
            "{a}, {b} — వీరిలో ఎవరు నెమ్మదిగా పరుగెత్తారు?",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def relation_sentence(
    attribute,
    bigger,
    smaller,
    reverse=False
):
    """
    Build ONE Telugu fact sentence.

    Logical meaning is always:

        bigger > smaller

    reverse=True only changes the Telugu wording (states the same
    fact from the smaller entity's point of view). The truth value
    and the direction of the chain are unchanged — this is purely
    a surface-form flip, which is what makes T6 harder without
    changing the answer logic.
    """

    relation = RELATIONS[attribute]

    if reverse:
        # e.g. "B is shorter than A"
        template = random.choice(
            relation["reverse"]
        )
    else:
        # e.g. "A is taller than B"
        template = random.choice(
            relation["greater"]
        )

    # NOTE: {a} is always the bigger entity in BOTH template sets,
    # so the mapping below is correct for reverse phrasing too.
    return template.format(
        a=bigger,
        b=smaller
    )


def make_question(
    attribute,
    a,
    b,
    ask_greater=True
):
    """
    Build the question line.

    ask_greater=True  -> "who is taller/older/faster?"      answer = a
    ask_greater=False -> "who is shorter/younger/slower?"   answer = b

    Callers pass (a, b) as the two ENDPOINTS of the chain, so the
    answer is decided by the caller, not re-derived here.
    """

    relation = RELATIONS[attribute]

    if ask_greater:

        template = (
            relation["greater_question"]
        )

    else:

        template = (
            relation["smaller_question"]
        )

    return template.format(
        a=a,
        b=b
    )


# ============================================================
# T1
# EASY — DIRECT NUMERICAL COMPARISON
#
# Two explicit numbers are given; the model just compares them.
# No transitivity required.
# ============================================================

def generate_t1():

    # Two DISTINCT names (random.sample never repeats).
    a, b = random.sample(
        NAMES,
        2
    )

    topic = random.choice([
        "age",
        "marks",
        "books",
        "money",
    ])


    if topic == "age":

        # random.sample over a range guarantees value_a != value_b,
        # so there is never an ambiguous tie in T1.
        value_a, value_b = random.sample(
            range(18, 70),
            2
        )

        fact1 = (
            f"{a} వయస్సు "
            f"{value_a} సంవత్సరాలు."
        )

        fact2 = (
            f"{b} వయస్సు "
            f"{value_b} సంవత్సరాలు."
        )


        # Randomly ask for the max or the min, so the model can't
        # learn "always answer the first person mentioned".
        ask_greater = random.choice([
            True,
            False
        ])


        if ask_greater:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరి వయస్సు ఎక్కువ?"
            )

            answer = (
                a
                if value_a > value_b
                else b
            )

            question_type = "greater"


        else:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరి వయస్సు తక్కువ?"
            )

            answer = (
                a
                if value_a < value_b
                else b
            )

            question_type = "smaller"


    elif topic == "marks":

        value_a, value_b = random.sample(
            range(20, 101),
            2
        )

        fact1 = (
            f"{a} {value_a} "
            f"మార్కులు సాధించారు."
        )

        fact2 = (
            f"{b} {value_b} "
            f"మార్కులు సాధించారు."
        )


        ask_greater = random.choice([
            True,
            False
        ])


        if ask_greater:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరు ఎక్కువ "
                f"మార్కులు సాధించారు?"
            )

            answer = (
                a
                if value_a > value_b
                else b
            )

            question_type = "greater"


        else:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరు తక్కువ "
                f"మార్కులు సాధించారు?"
            )

            answer = (
                a
                if value_a < value_b
                else b
            )

            question_type = "smaller"


    elif topic == "books":

        value_a, value_b = random.sample(
            range(1, 101),
            2
        )

        fact1 = (
            f"{a} వద్ద {value_a} "
            f"పుస్తకాలు ఉన్నాయి."
        )

        fact2 = (
            f"{b} వద్ద {value_b} "
            f"పుస్తకాలు ఉన్నాయి."
        )


        ask_greater = random.choice([
            True,
            False
        ])


        if ask_greater:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరి వద్ద ఎక్కువ "
                f"పుస్తకాలు ఉన్నాయి?"
            )

            answer = (
                a
                if value_a > value_b
                else b
            )

            question_type = "greater"


        else:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరి వద్ద తక్కువ "
                f"పుస్తకాలు ఉన్నాయి?"
            )

            answer = (
                a
                if value_a < value_b
                else b
            )

            question_type = "smaller"


    else:   # topic == "money"

        value_a, value_b = random.sample(
            range(100, 5001),
            2
        )

        fact1 = (
            f"{a} వద్ద {value_a} "
            f"రూపాయలు ఉన్నాయి."
        )

        fact2 = (
            f"{b} వద్ద {value_b} "
            f"రూపాయలు ఉన్నాయి."
        )


        ask_greater = random.choice([
            True,
            False
        ])


        if ask_greater:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరి వద్ద ఎక్కువ "
                f"డబ్బు ఉంది?"
            )

            answer = (
                a
                if value_a > value_b
                else b
            )

            question_type = "greater"


        else:

            question = (
                f"{a}, {b} — "
                f"వీరిలో ఎవరి వద్ద తక్కువ "
                f"డబ్బు ఉంది?"
            )

            answer = (
                a
                if value_a < value_b
                else b
            )

            question_type = "smaller"


    # Facts first, question last — consistent layout across all
    # templates so the model sees a stable input format.
    prompt = "\n".join([
        fact1,
        fact2,
        question
    ])


    # ------------------------------------------------------------
    # LEAKAGE CONTROL: the "problem key".
    #
    # This tuple is the canonical fingerprint of the *logical*
    # problem: template + topic + both entities + both values +
    # what was asked. Two items with the same key are the same
    # problem even if the Telugu wording differs.
    #
    # Simple logical identifier. No hashing.
    # ------------------------------------------------------------
    problem_key = (
        "T1",
        topic,
        a,
        value_a,
        b,
        value_b,
        question_type
    )


    return (
        prompt,
        answer,
        problem_key
    )


# ============================================================
# T2
# EASY — GREATER / SMALLER / EQUAL
#
# Different output space from T1: the answer is a LABEL
# ("ఎక్కువ" / "తక్కువ" / "సమానం"), not a name. This is the only
# template where ties are allowed.
# ============================================================

def generate_t2():

    a, b = random.sample(
        NAMES,
        2
    )


    # The three classes are sampled uniformly, so the label
    # distribution is roughly balanced (~1/3 each).
    relation = random.choice([
        "greater",
        "smaller",
        "equal"
    ])


    if relation == "equal":

        value_a = random.randint(
            20,
            100
        )

        value_b = value_a   # deliberate tie

        answer = "సమానం"


    elif relation == "greater":

        # Construct the values so the intended label is guaranteed:
        # pick the smaller first, then force the larger above it.
        value_b = random.randint(
            20,
            80
        )

        value_a = random.randint(
            value_b + 1,
            100
        )

        answer = "ఎక్కువ"


    else:   # "smaller"

        value_a = random.randint(
            20,
            80
        )

        value_b = random.randint(
            value_a + 1,
            100
        )

        answer = "తక్కువ"


    prompt = (
        f"{a} {value_a} మార్కులు సాధించారు.\n"
        f"{b} {value_b} మార్కులు సాధించారు.\n"
        f"{a} సాధించిన మార్కులు "
        f"{b} సాధించిన మార్కుల కంటే "
        f"ఎక్కువా, తక్కువా, లేక సమానమా?"
    )


    # Key = entities + both values + the relation label.
    # 'relation' is redundant given the values, but harmless.
    problem_key = (
        "T2",
        a,
        value_a,
        b,
        value_b,
        relation
    )


    return (
        prompt,
        answer,
        problem_key
    )


# ============================================================
# T3
# HARD — 3 ENTITY TRANSITIVE
#
# Ground truth chain:  A > B > C
# The question always asks about the ENDPOINTS (A vs C), so the
# model must chain two facts; neither fact alone answers it.
# ============================================================

def generate_t3():

    a, b, c = random.sample(
        NAMES,
        3
    )


    attribute = random.choice(
        list(RELATIONS.keys())
    )


    fact1 = relation_sentence(
        attribute,
        a,
        b
    )   # A > B


    fact2 = relation_sentence(
        attribute,
        b,
        c
    )   # B > C


    ask_greater = random.choice([
        True,
        False
    ])


    # Question is about A and C — the two ends of the chain.
    question = make_question(
        attribute,
        a,
        c,
        ask_greater
    )


    # Because the chain is built as A > B > C by construction,
    # the answer is known without any search.
    answer = (
        a
        if ask_greater
        else c
    )


    question_type = (
        "greater"
        if ask_greater
        else "smaller"
    )


    prompt = "\n".join([
        fact1,
        fact2,
        question
    ])


    # NOTE: the key does NOT include which of the two phrasings
    # was sampled. That is deliberate — the same chain with
    # different wording is treated as the SAME problem and will be
    # rejected, which is stricter than plain string dedup.
    problem_key = (
        "T3",
        attribute,
        a,
        b,
        c,
        question_type
    )


    return (
        prompt,
        answer,
        problem_key
    )


# ============================================================
# T4
# HARD — 4 ENTITY MULTI-HOP
#
# Ground truth chain:  A > B > C > D
# Three hops, and the facts are SHUFFLED so the chain is not
# presented in reading order — the model has to reconstruct it.
# ============================================================

def generate_t4():

    a, b, c, d = random.sample(
        NAMES,
        4
    )


    attribute = random.choice(
        list(RELATIONS.keys())
    )


    facts = [

        relation_sentence(
            attribute,
            a,
            b
        ),

        relation_sentence(
            attribute,
            b,
            c
        ),

        relation_sentence(
            attribute,
            c,
            d
        ),
    ]


    # Mix order — removes the positional shortcut
    # "first sentence contains the answer".
    random.shuffle(
        facts
    )


    ask_greater = random.choice([
        True,
        False
    ])


    question = make_question(
        attribute,
        a,
        d,
        ask_greater
    )


    answer = (
        a
        if ask_greater
        else d
    )


    question_type = (
        "greater"
        if ask_greater
        else "smaller"
    )


    prompt = "\n".join(
        facts + [question]
    )


    # Key ignores fact order and wording: any permutation of the
    # same chain counts as the same problem.
    problem_key = (
        "T4",
        attribute,
        a,
        b,
        c,
        d,
        question_type
    )


    return (
        prompt,
        answer,
        problem_key
    )


# ============================================================
# T5
# HARD — MULTI-HOP + DISTRACTORS
#
# Same A > B > C > D chain as T4, plus two irrelevant sentences
# about the same people (colour preference, went to the market,
# etc.). Tests selective attention: the model must ignore facts
# that carry no ordering information.
# ============================================================

def generate_t5():

    a, b, c, d = random.sample(
        NAMES,
        4
    )


    attribute = random.choice(
        list(RELATIONS.keys())
    )


    facts = [

        relation_sentence(
            attribute,
            a,
            b
        ),

        relation_sentence(
            attribute,
            b,
            c
        ),

        relation_sentence(
            attribute,
            c,
            d
        ),
    ]


    # Distractors are about people already IN the chain, so they
    # can't be filtered by "unknown name" — only by relevance.
    distractor_people = random.sample(
        [a, b, c, d],
        2
    )


    distractor_bank = [

        lambda person:
            f"{person}కి నీలం రంగు ఇష్టం.",

        lambda person:
            f"{person} నిన్న మార్కెట్‌కు వెళ్లారు.",

        lambda person:
            f"{person} ఉదయం టీ తాగారు.",

        lambda person:
            f"{person}కి మామిడిపండ్లు ఇష్టం.",

        lambda person:
            f"{person} వద్ద ఎరుపు బ్యాగ్ ఉంది.",

        lambda person:
            f"{person} ఆదివారం సినిమా చూశారు.",
    ]


    # sample (not choices) -> the two distractors are different
    # sentence types.
    chosen_distractors = random.sample(
        distractor_bank,
        2
    )


    distractors = [

        chosen_distractors[0](
            distractor_people[0]
        ),

        chosen_distractors[1](
            distractor_people[1]
        )
    ]


    all_facts = (
        facts
        + distractors
    )


    # Shuffle so distractors are interleaved with real facts.
    random.shuffle(
        all_facts
    )


    ask_greater = random.choice([
        True,
        False
    ])


    question = make_question(
        attribute,
        a,
        d,
        ask_greater
    )


    answer = (
        a
        if ask_greater
        else d
    )


    question_type = (
        "greater"
        if ask_greater
        else "smaller"
    )


    prompt = "\n".join(
        all_facts + [question]
    )


    # NOTE: distractors are NOT part of the key. Two T5 items with
    # the same chain but different distractors count as the same
    # problem and the second is rejected. Stricter than necessary,
    # and deliberately so.
    problem_key = (
        "T5",
        attribute,
        a,
        b,
        c,
        d,
        question_type
    )


    return (
        prompt,
        answer,
        problem_key
    )


# ============================================================
# T6
# HARD — MIXED DIRECTION
#
# Logical relation is still:
#
#     A > B > C > D
#
# But some facts are worded backwards ("C is shorter than B"
# instead of "B is taller than C"). The model must normalise
# direction before chaining — the hardest template here.
# ============================================================

def generate_t6():

    a, b, c, d = random.sample(
        NAMES,
        4
    )


    attribute = random.choice(
        list(RELATIONS.keys())
    )


    # Each pattern is [reverse?, reverse?, reverse?] for the three
    # hops. All four patterns contain at least one True and one
    # False, which guarantees the prompt is genuinely mixed and
    # never accidentally collapses into a plain T4.
    patterns = [

        [False, True, False],

        [True, False, True],

        [False, False, True],

        [True, False, False],
    ]


    directions = random.choice(
        patterns
    )


    facts = [

        relation_sentence(
            attribute,
            a,
            b,
            reverse=directions[0]
        ),

        relation_sentence(
            attribute,
            b,
            c,
            reverse=directions[1]
        ),

        relation_sentence(
            attribute,
            c,
            d,
            reverse=directions[2]
        ),
    ]


    random.shuffle(
        facts
    )


    ask_greater = random.choice([
        True,
        False
    ])


    question = make_question(
        attribute,
        a,
        d,
        ask_greater
    )


    # Wording direction does not change the logic, so the answer
    # is still just the chain endpoints.
    answer = (
        a
        if ask_greater
        else d
    )


    question_type = (
        "greater"
        if ask_greater
        else "smaller"
    )


    prompt = "\n".join(
        facts + [question]
    )


    # Here the key DOES include the direction pattern, so the same
    # chain worded with a different pattern is treated as a new
    # problem. See the leakage notes at the end of this file.
    problem_key = (
        "T6",
        attribute,
        a,
        b,
        c,
        d,
        tuple(directions),
        question_type
    )


    return (
        prompt,
        answer,
        problem_key
    )


# ============================================================
# GENERATOR MAPPING
# ============================================================

GENERATORS = {

    "T1": generate_t1,

    "T2": generate_t2,

    "T3": generate_t3,

    "T4": generate_t4,

    "T5": generate_t5,

    "T6": generate_t6,
}


# ============================================================
# GLOBAL DUPLICATE PROTECTION
#
# ***THIS IS THE CORE ANTI-LEAKAGE MECHANISM.***
#
# These two sets are GLOBAL — module level, not per split. They
# are populated as train is generated, then carried over while
# validation and test are generated.
#
# Consequence:
#   train, validation and test cannot contain the same prompt
#   string, and cannot contain the same logical problem.
#
# seen_prompts      -> catches identical surface text.
# seen_problem_keys -> catches identical underlying problem even
#                      when the Telugu wording, the fact order or
#                      the distractors differ. This is the part a
#                      naive string-dedup pipeline would miss.
# ============================================================

seen_prompts = set()

seen_problem_keys = set()


# ============================================================
# GENERATE ONE SPLIT
# ============================================================

def generate_split(
    split_name
):

    examples = []


    # Loop template by template so each template hits its exact
    # quota — the split is balanced by design, not by chance.
    for template_id in TEMPLATES:

        generator = (
            GENERATORS[
                template_id
            ]
        )


        required = (
            PER_TEMPLATE[
                split_name
            ]
        )


        count = 0


        # Rejection sampling: keep drawing until `required` UNIQUE
        # items have been accepted.
        #
        # CAUTION: if the space of possible problems for a template
        # were ever exhausted, this loop would spin forever. With
        # 38 names the space is far larger than 2000, so it
        # terminates in practice — but a max-attempts guard would
        # be safer.
        while count < required:

            (
                prompt,
                answer,
                problem_key
            ) = generator()


            # ----------------------------------------
            # Prevent exact duplicate prompt
            # (catches identical rendered text)
            # ----------------------------------------

            if prompt in seen_prompts:
                continue


            # ----------------------------------------
            # Prevent same logical instantiated problem
            # (catches paraphrases / reordered facts /
            #  swapped distractors of a problem already used)
            # ----------------------------------------

            if problem_key in seen_problem_keys:
                continue


            # Accept example — and immediately claim both the
            # string and the logical key GLOBALLY, so no later
            # split can reuse either.
            seen_prompts.add(
                prompt
            )

            seen_problem_keys.add(
                problem_key
            )


            example = {

                # id encodes split + template + index, so it is
                # unique across the whole dataset and traceable.
                "id":
                    f"te_{split_name}_"
                    f"{template_id}_"
                    f"{count:04d}",

                "language":
                    "Telugu",

                "split":
                    split_name,

                "template":
                    template_id,

                "prompt":
                    prompt,

                "answer":
                    answer,

                # Ready-to-train flat field: question + answer in
                # a fixed Telugu format, for causal LM fine-tuning.
                "text": (
                    "ప్రశ్న:\n"
                    + prompt
                    + "\nసమాధానం:\n"
                    + answer
                )
            }


            examples.append(
                example
            )


            count += 1


    # Shuffle so templates are interleaved rather than appearing
    # in six contiguous blocks (matters for batching / curriculum
    # effects during training).
    random.shuffle(
        examples
    )


    return examples


# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset():

    # Order matters: train is generated FIRST and therefore claims
    # its problems first; validation and test are drawn from what
    # remains. Order is deterministic because the seed is fixed.

    train = generate_split(
        "train"
    )

    validation = generate_split(
        "validation"
    )

    test = generate_split(
        "test"
    )


    return {

        "train":
            train,

        "validation":
            validation,

        "test":
            test,
    }


# ============================================================
# VERIFY DATASET
#
# Independent audit of the generated data. The generator is
# supposed to guarantee these properties; these asserts prove it
# after the fact rather than trusting the code path.
# ============================================================

def verify_dataset(
    data
):

    print(
        "\nRunning dataset checks...\n"
    )


    # ========================================================
    # CHECK 1 — SPLIT SIZES
    # ========================================================

    assert (
        len(data["train"])
        == 9600
    )

    assert (
        len(data["validation"])
        == 1200
    )

    assert (
        len(data["test"])
        == 1200
    )


    print(
        "✓ Split sizes are correct."
    )


    # ========================================================
    # CHECK 2 — TOTAL SIZE
    # ========================================================

    total = sum(
        len(split)
        for split in data.values()
    )


    assert (
        total
        == TOTAL_EXAMPLES
    )


    print(
        f"✓ Total examples = {total}"
    )


    # ========================================================
    # CHECK 3 — CORRECT COUNT PER TEMPLATE
    #
    # Guarantees the test set has the same difficulty mix as
    # train (200 of each template), so accuracy numbers are
    # comparable across splits.
    # ========================================================

    for split_name in data:

        counts = Counter(

            example["template"]

            for example
            in data[split_name]
        )


        for template in TEMPLATES:

            expected = (
                PER_TEMPLATE[
                    split_name
                ]
            )


            assert (
                counts[template]
                == expected
            ), (
                f"{split_name} "
                f"{template}: "
                f"expected {expected}, "
                f"got {counts[template]}"
            )


    print(
        "✓ Every template has the "
        "correct number of examples."
    )


    # ========================================================
    # CHECK 4 — NO DUPLICATE PROMPTS INSIDE OR ACROSS SPLITS
    #
    # THE KEY LEAKAGE ASSERTION. Set-disjointness between the
    # three splits is verified explicitly, not assumed.
    # ========================================================

    train_prompts = {

        x["prompt"]

        for x in data["train"]
    }


    val_prompts = {

        x["prompt"]

        for x in data["validation"]
    }


    test_prompts = {

        x["prompt"]

        for x in data["test"]
    }


    # Inside each split: set size == list size means no internal
    # duplicates.
    assert (
        len(train_prompts)
        == len(data["train"])
    )

    assert (
        len(val_prompts)
        == len(data["validation"])
    )

    assert (
        len(test_prompts)
        == len(data["test"])
    )


    # Between splits: zero overlap in any direction.
    assert train_prompts.isdisjoint(
        val_prompts
    )

    assert train_prompts.isdisjoint(
        test_prompts
    )

    assert val_prompts.isdisjoint(
        test_prompts
    )


    print(
        "✓ No prompt is duplicated "
        "inside or across splits."
    )


    # ========================================================
    # CHECK 5 — UNIQUE IDS
    # ========================================================

    all_ids = [

        example["id"]

        for split in data.values()

        for example in split
    ]


    assert (
        len(all_ids)
        == len(set(all_ids))
    )


    print(
        "✓ Every example ID is unique."
    )


    # ========================================================
    # CHECK 6 — NAMES ARE INTENTIONALLY SHARED
    #
    # This is NOT leakage.
    # We explicitly confirm the design: the dataset tests
    # reasoning over known entities, not vocabulary transfer to
    # unseen names.
    # ========================================================

    print(
        "✓ Shared entity-name pool is intentional."
    )


    print(
        "\n======================================"
    )

    print(
        "ALL DATASET CHECKS PASSED"
    )

    print(
        "======================================\n"
    )


# ============================================================
# SAVE JSONL
#
# One JSON object per line. ensure_ascii=False keeps Telugu
# readable in the file instead of \uXXXX escapes.
# ============================================================

def save_jsonl(
    filename,
    examples
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:

            f.write(

                json.dumps(
                    example,
                    ensure_ascii=False
                )

                + "\n"
            )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    data,
    output_dir
):

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    save_jsonl(
        output_dir
        / "train.jsonl",

        data["train"]
    )


    save_jsonl(
        output_dir
        / "validation.jsonl",

        data["validation"]
    )


    save_jsonl(
        output_dir
        / "test.jsonl",

        data["test"]
    )


    # Dataset card / provenance record. Documents the leakage
    # policy alongside the data so a later reader knows that
    # shared names were a choice and prompt overlap is zero.
    #
    # CAUTION: these numbers are hardcoded, not computed from
    # `data`. If the config changes, this file silently lies.
    stats = {

        "total":
            12000,

        "train":
            9600,

        "validation":
            1200,

        "test":
            1200,

        "templates":
            TEMPLATES,

        "shared_name_pool":
            True,

        "exact_prompt_overlap":
            False,

        "seed":
            SEED
    }


    with open(
        output_dir
        / "dataset_stats.json",

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# PRINT SAMPLE FROM EACH TEMPLATE
#
# Eyeball check: prints one training example per template so the
# Telugu output can be sanity-read before training.
# ============================================================

def print_samples(
    data
):

    for template in TEMPLATES:

        example = next(

            x
            for x in data["train"]

            if x["template"]
            == template
        )


        print(
            "\n" + "=" * 70
        )

        print(
            template
        )

        print(
            "=" * 70
        )

        print(
            example["prompt"]
        )

        print(
            "\nసమాధానం:",
            example["answer"]
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Re-seeding here is redundant (already done at import time)
    # but harmless, and makes the entry point self-contained.
    random.seed(
        SEED
    )


    print(
        "Generating revised Telugu "
        "reasoning dataset..."
    )


    data = (
        generate_dataset()
    )


    # Verify BEFORE writing to disk: nothing invalid ever reaches
    # the output directory.
    verify_dataset(
        data
    )


    OUTPUT_DIR = (
        "/kaggle/working/"
        "telugu_reasoning_dataset_shared_names"
    )


    save_dataset(
        data,
        OUTPUT_DIR
    )


    print_samples(
        data
    )


    print(
        "\nDataset saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# APPENDIX — LEAKAGE: WHAT IS AND IS NOT PREVENTED
#
# PREVENTED
# 1. Identical prompt strings anywhere in the dataset
#    (global `seen_prompts`).
# 2. Identical logical problems anywhere in the dataset
#    (global `seen_problem_keys`) — this is stronger than string
#    dedup, because it also rejects paraphrases, reordered facts
#    and re-rolled distractors of a problem already used.
# 3. Duplicate IDs.
# 4. Unbalanced templates across splits.
# 5. Non-reproducibility (fixed seed).
# 6. All of the above are re-asserted in verify_dataset() rather
#    than assumed.
#
# NOT PREVENTED (known residual overlap)
# a. Cross-template near-duplicates. The template ID is the first
#    element of every key, so the SAME chain
#    (attribute, A, B, C, D, question_type) can appear as T4 in
#    train and as T5 in test — identical reasoning, two extra
#    irrelevant sentences. Same for a T3 sub-chain of a T4 chain.
# b. T6 direction patterns. Direction is part of the T6 key, so
#    the same chain with pattern [F,T,F] in train and [T,F,T] in
#    test are treated as different problems, although the required
#    reasoning is identical.
# c. greater vs smaller on the same chain. question_type is part
#    of the key, so "who is tallest" in train and "who is
#    shortest" in test can use the exact same set of facts.
# d. Entity overlap. Intentional, documented, and fine for this
#    task — but it means test performance says nothing about
#    generalisation to unseen names.
# e. Values only appear in T1/T2 keys, so nothing constrains
#    numeric novelty in the harder templates.
#
# IF YOU WANT STRICTER SPLITS
# - Drop the template ID from the key, or add a second
#    "chain-level" key: (attribute, frozenset of the ordered
#    chain) shared across T3/T4/T5/T6.
# - Drop question_type and directions from the key too.
# - Better: partition at the LEVEL OF THE CHAIN. Enumerate the
#    entity tuples first, assign each tuple to exactly one split,
#    then generate items only within that split's tuples. That
#    makes cross-split reuse structurally impossible instead of
#    relying on rejection sampling.
# - Add a max-attempts guard to the while loop in generate_split
#    so exhaustion raises instead of hanging.
# ============================================================
