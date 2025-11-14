uv run main.py --baseline "/root/project/srs/srs-docs/req_md/0000 - cctns.pdf.md" --target "/root/project/srs/srs-docs/summary_detailed/0000 - cctns.pdf.md" --runs 10  --output-dir "output/106"
sleep 10
uv run main.py --baseline "/root/project/srs/srs-docs/req_md/0000 - cctns.pdf.md" --target "/root/project/srs/srs-docs/summary_detailed/0000 - cctns.pdf.md" --runs 10  --output-dir "output/107"
sleep 10
uv run main.py --baseline "/root/project/srs/srs-docs/req_md/0000 - cctns.pdf.md" --target "/root/project/srs/srs-docs/summary_detailed/0000 - cctns.pdf.md" --runs 10  --output-dir "output/108"
sleep 10
uv run main.py --baseline "/root/project/srs/srs-docs/req_md/0000 - cctns.pdf.md" --target "/root/project/srs/srs-docs/summary_detailed/0000 - cctns.pdf.md" --runs 10  --output-dir "output/109"
sleep 10
uv run main.py --baseline "/root/project/srs/srs-docs/req_md/0000 - cctns.pdf.md" --target "/root/project/srs/srs-docs/summary_detailed/0000 - cctns.pdf.md" --runs 10  --output-dir "output/110"


uv run compare_algorithms.py baseline.md target.md --judges 10 --compare-count 10 --extract-runs 10 