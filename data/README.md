# Dataset files

Place processed datasets here:

```text
data/{dataset}/train.txt
data/{dataset}/valid.txt
data/{dataset}/test.txt
data/{dataset}/item_descriptions.csv
```

The expected interaction format follows the PRTA/RecToolAgent convention:

- `train.txt`: one line per user, `user_id item_id item_id ...`
- `valid.txt`: one held-out validation item per user, `user_id item_id`
- `test.txt`: one held-out test item per user, `user_id item_id`
- `item_descriptions.csv`: tab-separated file with at least
  `new_item_id` and `item_description` columns.

Generated user-profile files such as `user_profile_*.csv` are intentionally not
included in the open-source package.
