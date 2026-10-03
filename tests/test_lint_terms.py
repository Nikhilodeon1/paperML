from evaluation.lint_terms import lint_text


def test_structural_phrase_allowed_with_exchange_context():
    text = "The empty-gut model is structurally non-identifiable only up to exchange of the rates."
    assert lint_text(text) == []


def test_structural_phrase_rejected_for_iauc():
    text = "Under iAUC the timing parameters are structurally non-identifiable."
    assert len(lint_text(text)) == 1


def test_variants_are_caught():
    assert lint_text("This is structural non-identifiability of the area.")
    assert lint_text("Structurally  non identifiable parameters.")


def test_context_is_the_paragraph_not_the_file():
    text = "The swap symmetry is exact.\n\nUnder iAUC the model is structurally non-identifiable."
    assert len(lint_text(text)) == 1


def test_preregistered_is_rejected():
    assert lint_text("Our pre-registered analysis shows this.")
    assert lint_text("This was preregistered.")
    assert lint_text("analysis plan fixed in version control before the analyses were run") == []


def test_comment_lines_are_ignored():
    assert lint_text("% structurally non-identifiable, pre-registered note to self") == []
