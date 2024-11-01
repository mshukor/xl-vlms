# Contributing to xl-vlms

* Contributing to this repo must be done only via pull requests.
* First `git pull` from the `main` branch, create a branch (`git branch branch_name`), then go to your branch (`git checkout branch_name`). The branch naming should be consistent, for example (fix/fixing_memory_issue, feature/saving_hidden_states, research/pca_decomposition...)
* Inside your new branch, try to commit as much as possible so we can track your changes.
* Coding should be modular, it is favorable to define simple functions than having one big complicated one
* Function should include the arguments and return types. It is good to have a small comment explaining its mechanism as well as the passed arguments.
* After testing your code, please run `pre-commit run --all-files` to clean up your code.
* Create a PR and assign at least 2 people for reviews.

