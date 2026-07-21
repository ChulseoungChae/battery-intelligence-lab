# experiment: by_chg_mode

충전 방식(`slow` / `fast`)을 나눠 각각 학습하는 실험 자리입니다.

```bash
python scripts/run.py prepare --exp by_chg_mode   # 구현 후
python scripts/run.py train   --exp by_chg_mode
```

하위 폴더:
- `slow/{traj,models,figs}/`
- `fast/{traj,models,figs}/`
