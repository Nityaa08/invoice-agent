"""The eval runs in CI too, so a change that breaks a decision fails the build."""

from evals.run_eval import evaluate, markdown, same


def test_demo_eval_meets_the_labels():
    report = evaluate()
    assert report["invoices"] == 13
    assert report["decision_accuracy"] == 1.0
    assert report["failed_rules_exact_match"] == 1.0
    assert report["field_accuracy"] == 1.0
    assert report["sent_to_review"] == 9
    assert "| Decision accuracy" in markdown(report)


def test_field_comparison():
    assert same("total", 23954.004, 23954.0)
    assert not same("total", None, 0.0)
    assert same("vendor_name", " Acme ", "Acme")
    assert same("po_number", None, None)


def test_realworld_eval_meets_the_labels():
    report = evaluate("realworld")
    assert report["invoices"] == 7
    assert report["decision_accuracy"] == 1.0
    assert report["failed_rules_exact_match"] == 1.0
    assert report["field_accuracy"] == 1.0


def test_blind_eval_meets_the_labels():
    report = evaluate("blind")
    assert report["invoices"] == 3
    assert report["decision_accuracy"] == 1.0
    assert report["field_accuracy"] == 1.0


def test_qa_eval_meets_the_labels():
    report = evaluate("qa")
    assert report["invoices"] == 10
    assert report["decision_accuracy"] == 1.0
    assert report["failed_rules_exact_match"] == 1.0
    assert report["field_accuracy"] == 1.0
    assert report["sent_to_review"] == 2  # the two photos


def test_every_redteam_file_is_stopped_by_its_rule():
    report = evaluate("redteam")
    assert report["invoices"] == 9
    assert report["sent_to_review"] == 9
    assert report["failed_rules_exact_match"] == 1.0
    assert "not scored" in markdown(report)
