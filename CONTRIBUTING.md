# Contributing

Issues and pull requests are welcome, with expectations set honestly: this
is a research artifact maintained by one person, so review may be slow and
scope is guarded deliberately.

## Ground rules

- Before a PR, run the deterministic checks:

  ```bash
  uv run ruff format --check . && uv run ruff check .
  uv run mypy --explicit-package-bases src/ tests/
  uv run pytest tests/ -x --no-cov -q
  ```

- Claims discipline is part of the codebase. Anything that touches a number
  in `README.md` or `story/story.yaml` must keep its evidence grade honest
  (live measurement / scripted proxy / modelled counterfactual /
  deterministic recompute / open) — `story_lint` enforces the structure, the
  grade wording is on you.
- Tests are the grader. A behaviour change without a test will be asked for
  one.

## Licensing of contributions

The project is licensed under the Business Source License 1.1, converting to
Apache-2.0 on 2030-01-01 (see [LICENSE](LICENSE)). By submitting a
contribution you agree that it is provided under the same terms, and that
the Licensor may relicense it as part of the Licensed Work, including under
the Change License.
