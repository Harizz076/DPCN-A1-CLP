#!/usr/bin/env python3
"""Check the survey for repeated-answer patterns."""

import argparse
import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FILE = ROOT_DIR / "data" / "Survey_Results_UC.csv"


def get_questions(headers):
    """Find the question columns and retain their original order."""
    questions = []
    for column, header in enumerate(headers):
        header = header.strip()
        if len(header) >= 3 and header[0] in "TESV" and header[1:3].isdigit():
            questions.append({
                "column": column,
                "id": header.split(".", 1)[0],
                "domain": header[0],
            })
    return questions


def find_cycle(answers, max_period=5):
    """Return a short exact repeating cycle, but ignore one repeated answer."""
    if len(set(answers)) < 2:
        return None

    for period in range(2, min(max_period, len(answers) - 1) + 1):
        if all(answer == answers[index % period] for index, answer in enumerate(answers)):
            return period
    return None


def find_runs(answers):
    """Return the start, end, and answer for each consecutive run."""
    runs = []
    start = 0
    for index in range(1, len(answers) + 1):
        if index == len(answers) or answers[index] != answers[start]:
            runs.append((start, index - 1, answers[start]))
            start = index
    return runs


def main():
    parser = argparse.ArgumentParser(description="Check for repeated survey-answer patterns.")
    parser.add_argument("--input", type=Path, default=DEFAULT_FILE, help="Path to the survey CSV")
    parser.add_argument("--severe-run", type=int, default=30, help="Run length for a severe flag")
    parser.add_argument("--review-run", type=int, default=20, help="Run length for a review flag")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader)
        rows = list(reader)

    questions = get_questions(headers)
    if len(questions) != 60:
        raise ValueError("Expected 60 question columns, found {}".format(len(questions)))

    complete_rows = []
    for row in rows:
        if all(row[question["column"]].strip() for question in questions):
            complete_rows.append(row)

    domain_questions = {}
    for domain in "TESV":
        domain_questions[domain] = [question for question in questions if question["domain"] == domain]

    cycles = []
    severe_runs = []
    review_runs = []

    for row in complete_rows:
        respondent_id = row[0].strip()
        answers = [row[question["column"]].strip() for question in questions]

        # Check each 15-question domain for short, non-constant cycles.
        for domain, domain_items in domain_questions.items():
            domain_answers = [row[item["column"]].strip() for item in domain_items]
            period = find_cycle(domain_answers)
            if period is not None:
                cycles.append((respondent_id, domain, period, domain_answers[:period]))

        # Check the full response sequence for long runs of one answer.
        for start, end, answer in find_runs(answers):
            length = end - start + 1
            record = (respondent_id, length, answer, questions[start]["id"], questions[end]["id"])
            if length >= args.severe_run:
                severe_runs.append(record)
            elif length >= args.review_run:
                review_runs.append(record)

    print("Input: {}".format(args.input))
    print("Complete 60-item responses screened: {}".format(len(complete_rows)))

    print("\nExact non-constant cycles within a domain")
    if cycles:
        for respondent_id, domain, period, pattern in cycles:
            print("  ID {}: {} domain, period {}, pattern = {}".format(
                respondent_id, domain, period, " | ".join(pattern)
            ))
    else:
        print("  None")

    print("\nSevere straight-lining runs ({}+ consecutive answers)".format(args.severe_run))
    if severe_runs:
        for respondent_id, length, answer, first, last in sorted(severe_runs, key=lambda item: (-item[1], item[0])):
            print("  ID {}: {} consecutive '{}' responses ({} to {})".format(
                respondent_id, length, answer, first, last
            ))
    else:
        print("  None")

    print("\nReview runs ({} to {} consecutive answers)".format(args.review_run, args.severe_run - 1))
    if review_runs:
        for respondent_id, length, answer, first, last in sorted(review_runs, key=lambda item: (-item[1], item[0])):
            print("  ID {}: {} consecutive '{}' responses ({} to {})".format(
                respondent_id, length, answer, first, last
            ))
    else:
        print("  None")

    print("\nFlags are for review and do not automatically remove a response.")


if __name__ == "__main__":
    main()
