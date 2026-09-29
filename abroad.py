from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from ddgs import DDGS
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END


load_dotenv()


class University(TypedDict):
    name: str
    city: str
    program: str
    tuition_listed: str
    deadline: str
    language: str


class State(TypedDict):
    # you supply
    country: str
    field: str
    budget_ceiling: str
    language_ok: str
    nationality: str

    # written by specialists
    universities_raw: str
    costs: Annotated[list[str], add]
    visa: str
    post_study: str
    final: str

@tool
def web_search(query: str) -> str:
    """Search the web. Returns the top results as text."""
    for backend in ("duckduckgo", "bing", "brave", "google"):
        try:
            with DDGS(timeout=20) as ddgs:
                hits = list(ddgs.text(query, max_results=5, backend=backend))
            if hits:
                return "\n\n".join(
                    f"{h['title']}\n{h['href']}\n{h['body']}" for h in hits
                )
        except Exception:
            continue
    return "Search failed — no results available."


model = init_chat_model("google_genai:gemini-flash-lite-latest",
                        max_retries=5, timeout=60)


UNIVERSITIES_PROMPT = """You are a university programme researcher.

You receive: country, field of study, budget ceiling, and the user's
language levels.

Search for degree programmes in that country and field that the user is
eligible for given their language levels. Find 5 to 8.

For each one return: university name, city, exact programme name, tuition
as listed on the official page, application deadline, and language of
instruction.

Rules:
- Prefer official university pages and government portals over ranking
  sites and blogs.
- If you cannot find a field, write "not found". Never estimate, never
  infer, never fill it in from a similar programme.
- Do not rank or recommend. Gather only.
- Include the source URL for each programme."""

COSTS_PROMPT = """You are a cost researcher for international students.

You receive: a list of universities with cities, and the user's budget ceiling.

For each university, find: tuition for international students (not domestic),
typical monthly rent in that city for a student, monthly living costs, and any
proof-of-funds or blocked-account amount the country requires for a student visa.
Report a total estimated first-year cost per university.

Rules:
- Tuition for non-EU international students specifically. Domestic and EU rates
  are usually different and quoting the wrong one makes the whole estimate useless.
- Prefer official university fee pages and government sources over aggregators.
- If a figure cannot be found, write "not found". Never estimate a number.
- State the currency and the year the figure is from.
- Do not rank or recommend."""


VISA_PROMPT = """You are a student visa researcher.

You receive: a destination country and the user's nationality.

Find the student visa route for that nationality: what the visa is called,
what documents are required, the proof-of-funds amount, processing time,
application cost, and whether the applicant must apply from their home country.

Rules:
- Requirements differ by nationality. Find rules for the stated nationality
  specifically, not generic international-student guidance.
- Prefer the destination country's own embassy, consulate, or immigration
  ministry pages. Visa blogs and agency sites are frequently out of date.
- If something cannot be confirmed from an official source, write
  "not confirmed" and say where the user should check.
- Never state a requirement you did not find in a source.
- Do not rank or recommend."""


POST_STUDY_PROMPT = """You are a post-study options researcher.

You receive: a destination country and the user's nationality.

Find what happens after graduation: whether a post-study work permit exists,
how long it lasts, whether the student can switch to a work visa from inside
the country, and whether there is a route toward permanent residency.

Rules:
- Rules change often. Note the date of the source and flag anything that
  looks like it may have changed recently.
- Prefer official immigration sources.
- If a route cannot be confirmed, write "not confirmed".
- Do not rank or recommend."""


def specialist(prompt: str):
    return create_agent(model, tools=[web_search], system_prompt=prompt)

uni_agent   = specialist(UNIVERSITIES_PROMPT)
cost_agent  = specialist(COSTS_PROMPT)
visa_agent  = specialist(VISA_PROMPT)
post_agent  = specialist(POST_STUDY_PROMPT)


def run(agent, task: str) -> str:
    r = agent.invoke({"messages": [("human", task)]})
    return r["messages"][-1].text


def universities_node(state: State):
    task = (f"Country: {state['country']}\nField: {state['field']}\n"
            f"Budget: {state['budget_ceiling']}\n"
            f"Language levels: {state['language_ok']}")
    return {"universities_raw": run(uni_agent, task)}


def costs_node(state: State):
    task = (f"Budget: {state['budget_ceiling']}\n\n"
            f"Universities:\n{state['universities_raw']}")
    return {"costs": [run(cost_agent, task)]}


def visa_node(state: State):
    task = f"Country: {state['country']}\nNationality: {state['nationality']}"
    return {"visa": run(visa_agent, task)}


def post_study_node(state: State):
    task = f"Country: {state['country']}\nNationality: {state['nationality']}"
    return {"post_study": run(post_agent, task)}


SUPERVISOR_PROMPT = """You are advising a student choosing where to study abroad.

You receive research from four specialists: universities, costs, visa, and
post-study options.

Rank the options from best to worst fit. For each one give: total estimated
first-year cost, the key visa requirement, whether they can stay after
graduating, and one line on why it is ranked there.

Flag any option as NOT FEASIBLE if it exceeds the budget, the deadline has
passed, or the visa route does not realistically exist for this nationality.
Put those last.

Where the specialists disagree on a number, say so rather than picking one.
Where something is marked "not found" or "not confirmed", say what the user
needs to verify themselves. Do not fill gaps."""

supervisor = create_agent(model, tools=[], system_prompt=SUPERVISOR_PROMPT)


def supervisor_node(state: State):
    task = (f"Budget: {state['budget_ceiling']}\n\n"
            f"UNIVERSITIES:\n{state['universities_raw']}\n\n"
            f"COSTS:\n{chr(10).join(state['costs'])}\n\n"
            f"VISA:\n{state['visa']}\n\n"
            f"POST-STUDY:\n{state['post_study']}")
    return {"final": run(supervisor, task)}


g = StateGraph(State)

g.add_node("universities", universities_node)
g.add_node("costs", costs_node)
g.add_node("visa", visa_node)
g.add_node("post_study", post_study_node)
g.add_node("supervisor", supervisor_node)

# universities, visa and post_study all start immediately
g.add_edge(START, "universities")
g.add_edge(START, "visa")
g.add_edge(START, "post_study")

# costs must wait for universities
g.add_edge("universities", "costs")

# supervisor waits for all three branches
g.add_edge("costs", "supervisor")
g.add_edge("visa", "supervisor")
g.add_edge("post_study", "supervisor")

g.add_edge("supervisor", END)

app = g.compile()

if __name__ == "__main__":
    result = app.invoke({
        "country": "France",
        "field": "Computer Science",
        "budget_ceiling": "20,000 EUR per year",
        "language_ok": "French B2 (TCF), English B2 (IELTS)",
        "nationality": "Algerian",
        "universities_raw": "",
        "costs": [],
        "visa": "",
        "post_study": "",
        "final": "",
    })
    print(result["final"])
