"""Conservative risk hints for generated queries; never a semantic validator."""

import re


POPULATIONS = {
    "children": r"\b(children|child|pediatric|paediatric|adolescents?)\b",
    "adults": r"\b(adults?|elderly|older adults?)\b",
    "pregnant": r"\b(pregnan\w*|maternal)\b",
    "cancer_patients": r"\b(cancer patients?|oncology patients?)\b",
}
SPECIFIC_OUTCOMES = {
    "stroke": r"\b(stroke|cerebrovascular accident)\b",
    "myocarditis": r"\bmyocarditis\b",
    "renal_failure": r"\b(renal|kidney) failure\b",
}
LONG_TERM = r"\b(long[ -]term|post[ -]covid|five[ -]year|years? later|months? after infection)\b"
MECHANISM = r"\b(mechanisms?|mediators?|biological pathways?)\b"
STOP = {"what", "how", "do", "does", "the", "and", "or", "in", "of", "for", "with", "to", "a", "an",
        "from", "on", "are", "were", "is", "was", "by", "about", "that", "their", "study", "studies",
        "covid", "sars", "coronavirus", "virus", "viral", "infection", "research", "evidence"}


def terms(text):
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) > 2 and word not in STOP}


def _mask_and_distance(text):
    low = text.casefold()
    return bool(re.search(r"\bmasks?\b", low) and re.search(r"\b(physical|social) distanc\w*", low))


def validate_query_scope(original_query, subqueries):
    """Flag possible scope drift; only exact duplicates are structurally confirmed."""
    if not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("original query is empty")
    if not isinstance(subqueries, list):
        raise TypeError("subqueries must be a list")
    original = original_query.casefold()
    flagged = []
    normalized = {}
    previous_comparison = None
    for index, subquery in enumerate(subqueries, 1):
        flags = []
        if not isinstance(subquery, str) or not subquery.strip():
            flags.append(("empty_subquery", "Confirmed structural invalidity: blank query."))
            subquery = ""
        low = subquery.casefold().strip()
        canonical = " ".join(low.split())
        if canonical in normalized or canonical == " ".join(original.split()):
            flags.append(("exact_duplicate", "Confirmed structural duplicate of an earlier or original query."))
        normalized[canonical] = index
        if subquery:
            for label, pattern in POPULATIONS.items():
                if re.search(pattern, low) and not re.search(pattern, original):
                    flags.append(("new_population", f"Possible new {label} population; inspect original scope."))
            for label, pattern in SPECIFIC_OUTCOMES.items():
                if re.search(pattern, low) and not re.search(pattern, original):
                    flags.append(("new_outcome", f"Possible new {label} outcome; inspect original scope."))
            if re.search(LONG_TERM, low) and not re.search(LONG_TERM, original):
                flags.append(("new_timeframe", "Long-term or post-infection timeframe absent from original question."))
            if re.search(MECHANISM, low) and not re.search(MECHANISM, original):
                flags.append(("new_mechanism", "Mechanism or mediator objective absent from original question."))
            if not terms(original) & terms(subquery):
                flags.append(("weak_anchor_overlap", "No shared specific lexical anchor; semantic relation needs review."))
            if _mask_and_distance(subquery):
                if previous_comparison is not None:
                    flags.append(("possible_redundancy", f"Both this and subquery {previous_comparison} compare masks and distancing."))
                previous_comparison = index
        if flags:
            flagged.append({"index_one_based": index, "subquery": subquery,
                            "risks": [{"type": label, "reason": reason} for label, reason in flags],
                            "assessment": "rule_hint_needs_semantic_review" if
                            any(label not in {"empty_subquery", "exact_duplicate"} for label, _ in flags)
                            else "confirmed_structural_issue"})
    return {"scope_status": "needs_review" if flagged else "no_rule_based_risk_detected",
            "flagged_subqueries": flagged,
            "reasons": sorted({risk["type"] for item in flagged for risk in item["risks"]}),
            "semantic_scope_confirmed": False}
