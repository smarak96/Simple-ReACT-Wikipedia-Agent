from react import setup_openrouter_client,run_react,call_llm,extract_trajectory
from eval.exact_match import exact_match
SLIDING_WINDOW_LENGTH=4
MAX_TRIALS = 4

REFLECTION_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self-reflection.
You will be given a previous reasoning trial in which you attempted to answer a
question using interleaving Thought, Action, and Observation steps. You were
unsuccessful: either you guessed the wrong answer with finish[answer], or you
ran out of steps, or your searches and lookups failed to surface the answer.

Diagnose a possible reason for the failure and devise a concise, actionable plan
to fix it on the next attempt. Be specific about which entity you should have
searched, which keyword you should have looked up, or where your reasoning went
wrong. Write your reflection in a few sentences. Do not repeat the trajectory;
only give the lesson for next time."""

REFLECTION_EXAMPLES = """Previous trial:
Question: The Rome Protocols were signed by three Prime Ministers, one of whom was assassinated as part of what plot?
Thought 1: I need to search the Rome Protocols and find the three prime ministers.
Action 1: search[Rome Protocols]
Observation 1: The Rome Protocols were a series of three international agreements concluded in Rome on 17 March 1934 between Italy, Austria and Hungary.
Thought 2: The protocols were signed by Italy, Austria, and Hungary. I will search each country.
Action 2: search[Italy prime minister 1934]
Observation 2: Could not find Italy prime minister 1934. Similar: [...].
Thought 3: I will search Austria prime minister 1934.
Action 3: search[Austria prime minister 1934]
Observation 3: Could not find Austria prime minister 1934. Similar: [...].
... (ran out of steps)

Reflection: I searched for the prime ministers by country and year instead of
reading the Rome Protocols page itself, which likely named the signatories. Next
time I should lookup[signed] or lookup[Prime Minister] on the Rome Protocols page
to get the names directly, then search the assassinated signatory to find the plot.

Previous trial:
Question: Kam Heskin plays Paige Morgan in a 2004 film directed by who?
Thought 1: I need to search Kam Heskin and find the 2004 film.
Action 1: search[Kam Heskin]
Observation 1: Kam Erika Heskin is an American actress.
Thought 2: The summary doesn't mention the film. I'll guess based on what I know.
Action 2: finish[unknown]

Reflection: I gave up and guessed instead of using the tools I had. The first
observation only showed the lead sentences, but the film was likely listed further
down the page. Next time I should lookup[Paige Morgan] or lookup[2004] on the
already-open page before abandoning the search.

Previous trial:
Question: Grown-Ups starred the actor who was best known for which role on 'Allo 'Allo!?
Thought 1: I need to search Grown-Ups and find which actor was also in 'Allo 'Allo!.
Action 1: search[Grown-Ups]
Observation 1: Grown-Ups was a 1980 British BBC television film. It stars Lesley Manville, 
Philip Davis, Brenda Blethyn, Janine Duvitski, Lindsay Duncan and Sam Kelly.
Thought 2: I need to find which of these actors appeared in 'Allo 'Allo!. I will search Sam Kelly 'Allo 'Allo!.
Action 2: search[Sam Kelly 'Allo 'Allo!]
Observation 2: Could not find Sam Kelly 'Allo 'Allo!. Similar: ["'Allo 'Allo!", "List of 'Allo 'Allo! characters"].
Thought 3: I will try searching Philip Davis 'Allo 'Allo!.
Action 3: search[Philip Davis 'Allo 'Allo!]
Observation 3: Could not find Philip Davis 'Allo 'Allo!. Similar: [...].
... (ran out of steps)

Reflection: I made the mistake of searching two names together (e.g. "Sam Kelly 
'Allo 'Allo!"), which Wikipedia cannot resolve. The correct approach is to search 
each cast member individually — search[Sam Kelly] — and read their Wikipedia intro, 
which always states what they are "best known for". A single individual search would 
have revealed the answer directly without needing to cross-reference two entities at once.
"""



def add_to_memory(mem,incoming_trajectory):
    mem.append(incoming_trajectory)
    return mem[-SLIDING_WINDOW_LENGTH:]


def self_reflect(question, trajectory, client):
    prompt = (
        REFLECTION_INSTRUCTION
        + "\n\n"
        + REFLECTION_EXAMPLES
        + f"\n\nPrevious trial:\nQuestion: {question}\n{trajectory}\n\nReflection:"
    )
    return call_llm(prompt, client)
def run_reflexion(question,actor_client,reflect_client,gold,max_trials=MAX_TRIALS):
    memory=[]
    trials=0

    while trials< max_trials:
        predicted,full_prompt=run_react(question,client=actor_client,memory=memory)
        trajectory = extract_trajectory(full_prompt,question)
        if exact_match(predicted, gold):       # gold = the dataset's correct answer        
            return trials,predicted
        reflection= self_reflect(question,trajectory,client=reflect_client)
        print("REFLECTION : \n")
        print(reflection)
        memory = add_to_memory(memory,reflection)
        trials+=1
    return trials,None

GOLD=''
if __name__ == "__main__":
    actor_client = setup_openrouter_client()
    self_reflection_client= setup_openrouter_client()
    question = input("Enter your question")
    trials,predicted = run_reflexion(question,actor_client,self_reflection_client,gold=GOLD,max_trials=MAX_TRIALS)
    if predicted is None:
        print(f'Agent could not find solution after {trials} trials')
    else:    
        print(f'Agent found the solution after {trials} trials :')
        print(predicted)
            
