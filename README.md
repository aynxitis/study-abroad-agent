# study-abroad-agent

Multi-agent research tool that ranks study abroad options by cost, visa route,
and post-study rights.

Four specialist agents research in parallel, a supervisor ranks the results.

- **universities** — finds programmes matching field, budget, and language level
- **costs** — tuition for non-EU students, rent, living costs, proof of funds
- **visa** — student visa route for the applicant's specific nationality
- **post-study** — work permit, switching visas, path to residency

`costs` waits for `universities`; `visa` and `post-study` run independently.
The supervisor ranks the options, flags anything infeasible, and surfaces
disagreements between specialists instead of picking a number.

Every agent is instructed to write `not found` rather than estimate.

## Running it

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # add your Google AI Studio key
python abroad.py
```

Edit the inputs at the bottom of `abroad.py` — country, field, budget,
language levels, nationality.

Built during [Thirduni](https://thirduni.com), first cohort.
