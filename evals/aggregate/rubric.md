# Grading the aggregate questions

The deterministic scores (cases found and cited, kinds named) measure **coverage**. They can't tell whether an
answer's conclusion is right or well supported, so each aggregate answer also gets a rubric grade, given by
reading the answer and its cited passages in the review file. The rubric was fixed before any answers were
seen. Grades and a one-line reason for each are saved next to the run (`evals/runs/grades-<round>.json`).

Score 0–4 per answer. Take off 1 point for each claim about the book that is false or not in the text, down
to a minimum of 0.

## a1: Is Alice more likely to befriend, or to be at odds with, the people and creatures she meets?

Ground truth is `alice_encounters.json`: 43 encounters, of which 16 are hostile, 11 mixed, 9 friendly and
7 neutral. So 27 of the 43 involve friction.

| Points | For |
|---|---|
| 2 | Verdict: mostly at odds (1 if "both" or balanced but leaning to friction; 0 if mostly friends) |
| 1 | Support: at least 3 correct hostile encounters **and** at least 2 correct friendly ones, named specifically (e.g. the Queen, the Pigeon, the Hatter; the Dodo, the Cheshire Cat, the puppy) |
| 1 | Nuance: relationships change (the Duchess, the Gryphon and Mock Turtle, the White Rabbit), or the friction is mostly bluster or touchiness while Alice stays polite |

## a2: What are Sherlock Holmes's common habits when investigating a case?

Ground truth is `holmes_habits_*.json`: 311 instances of 18 habits. The **core** habits are the 10 that occur
at least 17 times and in at least 7 of the 12 stories:

- relies on Watson
- reads people from small details
- examines the scene closely
- works apart from, or outdoes, the police
- uses records, newspapers or his index
- explains his reasoning afterwards
- withholds his conclusions until the end
- languor followed by bursts of energy
- smokes while thinking
- sets traps or ruses

| Points | For |
|---|---|
| 2 | Coverage: names at least 6 core habits (1 for 3–5) |
| 1 | Grounding: the habits are illustrated with specific, cited moments from the stories |
| 1 | "Common": it presents recurring habits, not a list of one-off events (e.g. a single disguise scene as "a habit" without noting that it recurs) |
