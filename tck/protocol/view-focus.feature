Feature: View focus for LLM navigation

  Scenario: Focus by position
    Given an attached session with tokenized input
    When the client sends view.focus with selector by_position 5
    Then the response includes the token at position 5
    And per-layer summaries for that position

  Scenario: Focus by regex
    When the client sends view.focus with selector by_regex "defendant"
    Then the response includes the first matching token
