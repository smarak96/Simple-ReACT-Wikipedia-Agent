import os
from groq import Groq
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "react-tutorial/1.0 (learning project; your@email.com)"
}
REACT_INSTRUCTION="""
    Solve a question answering task with interleaving Thought, Action, Observation steps.
    Thought reasons about the current situation.
    Action must be one of:
    - search[entity]
    - lookup[keyword]
    - finish[answer]
    Output ONLY the next Thought and Action, then stop.
    If the answer is not in the search result, use lookup[keyword] 
    on the current page before searching again.
"""

REACT_EXAMPLES = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions.
Thought 4: I need to instead search High Plains (United States).
Action 4: search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: finish[Richard Nixon]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action 3: finish[yes]
"""
load_dotenv()
def setup_groq_client():
    client = Groq(api_key = os.environ.get("GROQ_API_KEY"))
    return client

def call_llm(prompt,client,stop):
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"{prompt}",
        }
    ],
    model="llama-3.3-70b-versatile",
    temperature=0,
    stop=stop
)
    return chat_completion.choices[0].message.content

class WikiEnv:
    def __init__(self):
        self.page =None
        self.lookup_keyword = None
        self.lookup_list =[]
        self.lookup_cnt=0
        self.lookup_keyword = None
    def search(self,entity):
        url = "https://en.wikipedia.org/w/index.php?search=" + entity.replace(" ", "+")
        html = requests.get(url, headers=HEADERS).text
        soup = BeautifulSoup(html, "html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            titles = [div.get_text().strip() for div in result_divs[:5]]
            return f"Could not find {entity}. Similar: {titles}."
        else:
            paras = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split()) > 2]
            self.page = " ".join(paras)
            sentences = get_sentences(self.page)
            return " ".join(sentences[:5])
    def lookup(self, keyword):
        if self.page is None:
            return "No page is open. Use search first."
        if keyword != self.lookup_keyword:
            self.lookup_keyword = keyword
            self.lookup_list = [s for s in get_sentences(self.page) if keyword.lower() in s.lower()]
                                    
            self.lookup_cnt = 0
        if not self.lookup_list:
            return f"No results for '{keyword}'."

        if self.lookup_cnt >= len(self.lookup_list):
            return "No more results."

        n = len(self.lookup_list)
        result = self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
        return f"(Result {self.lookup_cnt} / {n}) {result}"
        
def get_sentences(text_blob):
    text_list = text_blob.split(". ")
    text_list = [s + "." for s in text_list if s.strip()!=""]
    return text_list
def build_prompt(question):

    return REACT_INSTRUCTION + REACT_EXAMPLES + f"Question: {question}\n"
def run_react(question, client, max_steps=10):
    env = WikiEnv()
    prompt = build_prompt(question)
    
    for step in range(1, max_steps + 1):
        chunk = call_llm(prompt, client, stop=["\nObservation"])
        print(chunk)
        prompt += chunk
        
        match = re.search(r"Action\s*\d*\s*:\s*(\w+)\[(.+?)\]", chunk)
        if not match:
            print("Could not parse action, stopping.")
            break
            
        action = match.group(1).lower()
        arg = match.group(2).strip()
        
        if action == "finish":
            print(f"\n>>> ANSWER: {arg}")
            return arg
        elif action == "search":
            obs = env.search(arg)
        elif action == "lookup":
            obs = env.lookup(arg)
        else:
            obs = "Unknown action."

        obs_str = f"\nObservation {step}: {obs}\n"
        print(obs_str)
        prompt += obs_str
    
    return "No answer found within step limit."

def run_cot(question, client):
    prompt = f"""Answer the following question by reasoning 
        step by step. Think through what you know, then give a 
        final answer.

        Question: {question}
        Thought:"""
    
    response = call_llm(prompt,client,stop=["\nObservation"])
    print(response)
groq_client = setup_groq_client()
run_react(
    "Which team won the IPL 2026 and when and where was it held?",
    groq_client
)    
run_cot(
    "Which team won the IPL 2026 and when and where was it held?",
    groq_client
)    