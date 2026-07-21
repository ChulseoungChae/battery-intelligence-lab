# Raw BMS CSV (not in git)

Place device CSV files here, e.g.:

- `01241225178.csv`
- `01241225211.csv`
- `01241225220.csv`
- `01241225226.csv`

These raw files are gitignored (large). After placing them, run:

```bash
python scripts/build_trajectories.py
python scripts/train_patchtst.py
```
