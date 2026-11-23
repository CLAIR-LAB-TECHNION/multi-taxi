# AGENTS.md

## Verify changes

## Keep changes small
- Every added line is one more to read, test and keep working, so make the smallest change that does the job.
- Before adding logic, check whether transformers, gymnasium, pettingzoo, or the standard library already handles it.
- Handle the cases that affect real use, meaning a published checkpoint, a documented workflow or a default setting. Skip fallbacks, warnings, options and docs for other setups unless the task asks for them. Users with those setups can pass the right arguments themselves.
- Write a test for every case the change must handle. Check first whether an existing test already covers it, and don't write tests that only check what a dependency does. For a bug fix, confirm the new test fails without the fix.
- Reread `git diff` and remove anything you added that the task does not need.
- If the test diff is much larger than the code change, shrink it by sharing setup or parametrizing variants, never by dropping a case. Don't delete, skip or loosen existing tests to make a change pass.

## Don't
- Don't use em-dashes, semicolons, or unnecessary colons in documentation.

## Documentation
- Document a change once, in the docstring of the API it changes.
- Give each explanation one home, next to the code it is about. Copies in other comments or docstrings drift apart when only one is updated, so don't restate it elsewhere, even reworded or summarized. When you edit an explanation that already exists in several places, merge the copies into its home.
- A docstring says what this repo does. Don't restate how other packages (e.g., pettingzoo, gymnasium) behave.
- Before writing a pointer to another file ("see X", "lives in X"), ask whether a reader of this code needs X to use or change it correctly. If not, leave the pointer out. A reader who does need X is already looking there.

