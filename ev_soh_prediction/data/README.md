# Raw BMS CSV (not in git)

원본 CSV는 용량(~4.5GB) 때문에 **git에 올리지 않습니다** (`.gitignore`).

로컬에서는 `/data1/ev_soh_aicar/data/` 파일을 심볼릭 링크로 연결해 두었습니다.

```bash
# 링크가 없으면 다시 연결
ln -sfn /data1/ev_soh_aicar/data/*.csv ./
```

또는 CSV를 이 폴더에 직접 복사해도 됩니다.

```bash
# 일별 압축
python scripts/run_daily.py prepare --exp daily
python scripts/run_daily.py train   --exp daily --L 14 --H 7

# 로우 비압축
python scripts/run_raw.py prepare --row-stride 100
python scripts/run_raw.py train   --L 100 --H 1
```
