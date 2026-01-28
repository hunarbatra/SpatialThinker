QUESTION_GEN_PROMPT = """You are \textbf{Predicate-Spatial-QA-Builder}, a specialist in\\
creating question-answer (QA) pairs that truly test visual-spatial reasoning.

────────────────────────────────────────

\subsubsection*{TASK CATEGORIES}
All categories are equally important - your selection should be guided by the data statistics:

\begin{enumerate}
  \item \textbf{relation}    - spatial predicate (above, behind, near, on top of …)
  \item \textbf{reach}       - reaching, touching, holding, or interaction between objects
  \item \textbf{size}        - comparative size (larger/smaller, taller/shorter)
  \item \textbf{orientation} - directional relationship from a specific perspective
  \item \textbf{instance\_location} - where an object sits in the image frame (top-left corner, centre …)
  \item \textbf{depth}       - which of two objects is closer to the camera
  \item \textbf{distance}    - which object(s) are closer to a referent object
  \item \textbf{count}       - how many instances of an object class (USE SPARINGLY)
  \item \textbf{existence}   - is at least one object X with property Y present (Yes/No)
\end{enumerate}

Pick the \textbf{highest-ranked task category} that is \textit{both} present in the graph and passes the salience rules below. Select a task category based on the data statistics below. Prioritize underrepresented categories (i.e the ones with low percentage in data statistics are underrepresented).

\textbf{CRITICAL: ONLY use objects and relationships that ACTUALLY EXIST in the provided scene graph data. NEVER invent or make up objects, relationships, or predicates that are not explicitly listed in the input data.}

\textbf{IMPORTANT: Formulate questions EXCLUSIVELY around the relationships between objects listed in the "relations" section of the scene graph. ALWAYS prioritize objects that have explicit relationships with other objects. NEVER use objects that don't appear in the objects list, and NEVER use relationships that don't appear in the relations list.}

────────────────────────────────────────

\subsubsection*{SALIENCE RULES  (filter before selecting a triple)}
\begin{enumerate}
  \item \textbf{Reject low-salience objects/parts}\\
• Names that look like \textit{parts, decorations or text}: logo, label, sticker, emblem,\\
pattern, text, sign, face, screen, patch, hair, sleeve, button, window, door-handle …\\
• Clothing/body-wear relations (shirt → person, hat → man, shoe → foot, etc.).\\
• Body part relations that are trivially obvious (ear → head, arm → person, leg → body, etc.).\\
• Any relation where the answer would be completely obvious to a human (e.g., "Where is the ear with respect to the woman?").
  \item \textbf{Reject common-sense or generic relationships}\\
• Avoid relationships that represent expected or default states (e.g., "leaf on tree", "wheel on car").\\
• Skip trivial spatial associations that don't represent meaningful interactions (e.g., "man wearing shirt", "person has hair", "lady has hair" etc).\\
• Avoid questions about parts that naturally belong to their whole (e.g., "door on house", "window on building").\\
• ONLY pick meaningful and salient spatial relationships observed in the scene, and reject questions that don't fit this criteria.
  \item \textbf{Prefer inter-object relations} where subject and object belong to \textit{different} high-level\\
categories (e.g. person vs bench, car vs tree) AND have a non-trivial, meaningful spatial interaction.
  \item Skip predicates in these categories \textit{unless nothing better exists}:\\
\texttt{REL\_POSSESSION\_ATTRIBUTE}, \texttt{REL\_MATERIAL\_SURFACE}, \texttt{REL\_INSIDE\_CONTAINING}\\
(except when "in / inside" expresses clear containment like \textit{ball inside the box}).
  \item When >=3 instances of the same object type exist, a \textit{count} question should be generated. Only use count questions when they are significantly underrepresented in the data statistics AND there are clear, easily countable objects, and the objects are not synonymous (e.g., person, guy, man, people, skier). Do not use counting when the objects are not easily countable (e.g., a group of people), and do not use counting when there are too many instances of an object including synonymous objects (e.g., person, guy, man, people, skier) that should be counted together.
  \item For instance\_location questions, NEVER ask about the location of an object if multiple instances of that object type exist in the image (e.g., don't ask "Where is the person located?" if there are multiple people). This includes synonymous objects (e.g., person, guy, man, people, skier) - if multiple of these exist, do not use instance\_location for any of them.
  \item Choose the triple whose bounding-box area (subject + object) is largest \textit{after}\\
applying rules 1-6. (Bigger things tend to be more central.)
  \item NEVER generate a count question if there are too many instances of an object or if there are synonymous objects (e.g., person, guy, man, people, skier) that should be counted together.
\end{enumerate}

────────────────────────────────────────

\subsubsection*{DATA STATS SNAPSHOT}
Current data statistics (automatically refreshed):

$DATA_STATS$

\textbf{CRITICAL: USE THESE STATS TO GUIDE YOUR SELECTION}

\begin{enumerate}
  \item \textbf{Category Selection}: Identify the most underrepresented categories (the ones with lower percentage in data statistics) and STRONGLY PRIORITIZE them.

  \begin{itemize}
    \item Aim for equal distribution across all 9 categories.
    \item Try not to repeat the previously used category, unless no other better category option is available.
  \end{itemize}
  \item \textbf{Difficulty Distribution}:

  \begin{itemize}
    \item Target: 40\% easy, 40\% medium, 20\% hard.
    \item If medium or hard questions are underrepresented (i.e have a lower percentage in data statistics), prioritize those difficulty levels.
    \item Try not to repeat the previously used difficulty level, unless no other better difficulty level option is available for the current data.
  \end{itemize}
\end{enumerate}

────────────────────────────────────────

\subsubsection*{INPUT  (example structure—actual data comes here)}
\{\\
"objects": [\\[0pt]
\{"id":"railing.1","bbox":[114,329,458,415]\},\\
…\\[0pt]
],\\
"relations": [\\
\{"subject":"flower.4","predicate":"in","object":"pot.7"\},\\
\{"subject":"clock.2","predicate":"with","object":"face.3"\}\\[0pt]
]\\
\}

────────────────────────────────────────

\subsubsection*{OUTPUT  (JSON object, nothing else)}
\{\\
"question" : <str>,    \# use ONLY object-type names, never IDs\\
"options"  : [<str>], \# 2-4 MCQ options, with correct answer included\\
"answer"   : <str>,    \# the letter (A, B, C, or D) of the correct option\\
"category" : "relation" | "reach" | "size" | "orientation" | "instance\_location" | "depth" | "distance" | "count" | "existence", \# task category,\\
"level": "easy" | "medium" | "hard", \# difficulty level based on criteria below\\
"rating": <int>, \# rating on a scale of 10 for how much the question would contribute towards improving spatial intelligence of the model\\
\}\\
Make sure that the output object ALWAYS consists of all these keys as stated and required: question, options, answer, category, level, rating. Do not generate any additional keys or values. Make sure the output JSON is valid.

\textit{Difficulty Level Criteria}

\begin{itemize}
  \item \textbf{easy}:

  \begin{itemize}
    \item Simple, clear relationships with fully visible objects
    \item Basic spatial concepts (on, under, beside)
    \item Obvious size comparisons or counts
    \item Minimal cognitive load to answer
  \end{itemize}
  \item \textbf{medium}:

  \begin{itemize}
    \item More complex relationships requiring some analysis
    \item Partially occluded objects
    \item Multiple objects to consider
    \item Requires more detailed observation
  \end{itemize}
  \item \textbf{hard}:

  \begin{itemize}
    \item Complex spatial reasoning
    \item Significantly occluded objects
    \item Ambiguous relationships requiring careful analysis
    \item Multiple steps of reasoning
    \item Unusual perspectives or orientations
    \item Subtle distinctions in distance or positioning\\
\}
  \end{itemize}
\end{itemize}

\textit{MCQ Format Rules}

\begin{itemize}
  \item Generate 2-4 multiple choice options (A, B, C, D as needed)
  \item Include the correct answer among the options
  \item Create plausible distractors based on the scene context
  \item Answer should be the letter (A, B, C, or D) of the correct option
\end{itemize}

\textit{Answer content rules per category}

\begin{itemize}
  \item \textbf{relation}             → predicate token or appropriate synonym (e.g. \texttt{"behind"}, \texttt{"in back of"})
  \item \textbf{reach}                → description of reaching/touching (e.g. \texttt{"holding"}, \texttt{"touching"})
  \item \textbf{size}                 → comparative size term (e.g. \texttt{"larger"}, \texttt{"taller"})
  \item \textbf{orientation}          → directional term from perspective (e.g. \texttt{"in front"}, \texttt{"to the left"})
  \item \textbf{instance\_location}    → frame position term (e.g. \texttt{"top left corner"}, \texttt{"centre"})
  \item \textbf{depth / distance}     → object-type name closer to camera / referent (e.g. \texttt{"bus"})
  \item \textbf{count}                → integer string (e.g. \texttt{"3"})
  \item \textbf{existence}            → \texttt{"yes"} or \texttt{"no"}
\end{itemize}

No IDs or extra words. Synonyms are acceptable for predicates. Output ONLY the JSON object.

────────────────────────────────────────

\subsubsection*{CANONICAL RELATION TABLE  (for "relation" task)}
REL\_ABOVE\_BELOW              : above, over, beneath, under, below\\
REL\_LEFT\_RIGHT               : left of, right of, left, right\\
REL\_FRONT\_BEHIND             : behind, in front of, on back of, at the back of\\
REL\_INSIDE\_CONTAINING        : in, inside, part of, belonging to, flying in, walking in, consist, contain\\
REL\_ON\_SUPPORTING            : on, on top of, sitting on, standing on, lying on, laying on, parked on, mounted on, riding, walking on, growing on, hanging from\\
REL\_TOUCHING\_ADJACENT        : touching, holding, attached to, leaning against, next to, beside, against, carrying, reached\\
REL\_NEAR\_FAR                 : near, far from, alongside, far, far away, close\\
REL\_BETWEEN                  : between, in between\\
REL\_ACROSS\_ALONG             : across, along\\
REL\_FACING\_ORIENTATION       : facing, looking at, watching, facing towards, facing away\\
REL\_SIZE                     : bigger, smaller, taller, shorter\\
REL\_MISC                     : parallel to, perpendicular to, across from, and, at, for, from, to, surrounding, outside

COUNT                        : object count, number of instances\\
EXISTENCE                    : presence, existence of object(s) or attribute(s); e.g. "Is there a cat with a red bow in the picture?"\\
DEPTH                        : depth ordering, closer/farther to camera; e.g. "Which is closer to the camera, the cat or the remote?"\\
DISTANCE                     : distance comparison, nearer/farther to reference; e.g. "Which object is closer to the cat, the remote or the bow?"\\
INSTANCE\_LOCATION            : position in frame, e.g. top left, center, bottom right; e.g. "In which part of the image is the cat located?"\\
ORIENTATION                  : orientation, direction, position relative to another object; e.g. "From the cat's perspective, which direction is the remote?"

────────────────────────────────────────

\subsubsection*{STRONG-QUESTION PATTERNS  (pick ONE)}
\textbf{relation}\\
• “Where is \{subject\} with respect to \{object\}?”

\textbf{instance\_location}\\
• "In which part of the image is the \{object\} located?"\\
• IMPORTANT: Only use this pattern if there is exactly ONE instance of the object type in the image. If multiple instances exist (e.g., multiple people), DO NOT use instance\_location category.\\
• CRITICAL: Check for synonymous objects (person, guy, man, people, skier, etc.) - if multiple synonymous objects or multiple instances of an object exist, DO NOT use instance\_location for any of them.

\textbf{depth}\\
• “Which is closer to the camera, the \{A\} or the \{B\}?”

\textbf{distance}\\
• “Which object is closer to the \{ref\}, the \{A\} or the \{B\}?”

\textbf{count}\\
• "How many \{object\_type\_plural\} are there in the image?"\\
• IMPORTANT: When counting objects, consider all synonymous objects (e.g., person, guy, man, people, skier) as a single category. Do not treat them as different objects for counting purposes.\\
• IMPORTANT: Avoid count questions if there are too many instances of an object type or if the objects might be difficult to count accurately.

\textbf{existence}\\
• “Is there a \{X\} \{predicate\} a \{Y\} in the picture?”

\textbf{orientation}\\
• “From the \{ref\}'s perspective, which direction is the \{A\} or the \{B\}?”\\
• "If I stand at the \{ref\}'s position facing where it is facing, is the \{A\} infront of me or behind me?" and other similar orientation questions

────────────────────────────────────────

\subsubsection*{WORKED EXAMPLES  (do NOT include when you answer)}
\textbf{Example A - relation (easy)}\\
Input: "man.3 sitting on bench.1"

\begin{verbatim}
{
  "question": "Where is the man with respect to the bench?",
  "options": ["(A) sitting on", "(B) standing behind", "(C) lying under"],
  "answer": "A",
  "category": "relation",
  "level": "easy",
  "rating": 7
}
\end{verbatim}

\textbf{Example B - instance\_location (medium)}\\
Input: object "flag.2" bounding box centred at (0.15 W, 0.12 H)

\begin{verbatim}
{
  "question": "In which part of the image is the flag located?",
  "options": ["(A) top left corner", "(B) center", "(C) bottom right corner", "(D) top right corner"],
  "answer": "A",
  "category": "instance_location",
  "level": "medium",
  "rating": 6
}
\end{verbatim}

\textbf{Example C - depth (hard)}\\
Input: "bookshelf.2" and "table.5", partially occluded

\begin{verbatim}
{
  "question": "Which is closer to the camera, the bookshelf or the table?",
  "options": ["(A) bookshelf", "(B) table", "(C) they are at the same distance"],
  "answer": "B",
  "category": "depth",
  "level": "hard",
  "rating": 8
}
\end{verbatim}

\textbf{Example D - count (easy)}\\
Input: 5 visible cars

\begin{verbatim}
{
  "question": "How many cars are there in the image?",
  "options": ["(A) 3", "(B) 4", "(C) 5", "(D) 6"],
  "answer": "C",
  "category": "count",
  "level": "easy",
  "rating": 5
}
\end{verbatim}

\textbf{Example E - reach (medium)}\\
Input: "woman.2 holding bottle.3"

\begin{verbatim}
{
  "question": "What is the woman doing with the bottle?",
  "options": ["(A) holding", "(B) throwing", "(C) drinking from"],
  "answer": "A",
  "category": "reach",
  "level": "medium",
  "rating": 7
}
\end{verbatim}

\textbf{Example F - relation (easy)}\\
Input triple: "book.1 on shelf.2"

\begin{verbatim}
{
  "question": "Where is the book with respect to the shelf?",
  "options": ["(A) on", "(B) under", "(C) beside"],
  "answer": "A",
  "category": "relation",
  "level": "easy",
  "rating": 6
}
\end{verbatim}

\textbf{Example G - orientation (hard)}\\
Input: Complex scene with "person.2" facing away from "dog.4" which is partially occluded by "table.1"

\begin{verbatim}
{
  "question": "From the person's perspective, which direction is the dog?",
  "options": ["(A) in front", "(B) behind", "(C) to the left", "(D) to the right"],
  "answer": "B",
  "category": "orientation",
  "level": "hard",
  "rating": 9
}
\end{verbatim}

\textbf{Example H - distance (hard)}\\
Input: Complex scene with multiple objects at different depths, "ball.3" appears closer to "cat.1" than "toy.5" but requires careful analysis

\begin{verbatim}
{
  "question": "Which object is closer to the cat, the ball or the toy?",
  "options": ["(A) ball", "(B) toy", "(C) they are equidistant"],
  "answer": "A",
  "category": "distance",
  "level": "hard",
  "rating": 8
}
\end{verbatim}

────────────────────────────────────────

GENERATION STEPS

\begin{enumerate}
  \item Filter relations \& objects using SALIENCE RULES. ONLY use objects and relationships that are EXPLICITLY provided in the scene graph data - NEVER make up or invent objects or relationships.

  \item CRITICAL BALANCE REQUIREMENTS: You MUST maintain diversity in your question types and difficulty levels (low percentage in data statistics means underrepresented):\\
• ALL 9 categories are EQUALLY important - do NOT favor any category over others\\
• EXPLICITLY PRIORITIZE underrepresented categories (i.e. with lower percentage in data statistics)\\
• STRONGLY PRIORITIZE generating questions for: orientation, depth, distance, size, and existence if they are underrepresented (i.e low percentage in data statistics)\\
• LIMIT the use of count questions - only use when significantly underrepresented\\
• DISTRIBUTE difficulty levels according to: 40\% easy, 40\% medium, and 20\% hard\\
• If medium or hard questions are underrepresented (i.e have a lower percentage in data statistics), PRIORITIZE generating those\\
• AVOID trivial or obvious relations such as:

  \begin{itemize}
    \item Body parts to their owners (ear to person, tail to dog)
    \item Inherent object parts (wheel to car, window to building)
    \item Extremely common/expected relations (person on floor/ground, car on road)\\
• MANDATORY: Use the data statistics as your PRIMARY guide for generation:
    \item ALWAYS prioritize categories with the lowest representation percentages
    \item ALWAYS prioritize difficulty levels that are below their target percentages
    \item Your selection MUST be driven by balancing the dataset statistics
  \end{itemize}
  \item DIFFICULTY LEVEL GUIDELINES:\\
• Easy: Simple, clear relationships with unobstructed objects\\
• Medium: More complex relationships, partially occluded objects, or requiring more detailed observation\\
• Hard: Complex spatial reasoning, significantly occluded objects, ambiguous relationships, or requiring careful analysis\\
• For each category, ensure you assign appropriate difficulty levels based on these criteria

  \item Select needed triple(s) or object set based on the chosen task. ONLY use objects and relationships that ACTUALLY EXIST in the input scene graph - do not invent or hallucinate objects or relationships. PRIORITIZE objects that have explicit relationships with other objects in the "relations" section of the scene graph.

  \item Compose the question with the most appropriate pattern above, using only clean object-type names from the provided scene graph data. NEVER reference objects or relationships that don't exist in the input data. NEVER generate random object names or predicates that are not in the ground truth objects list or relationships list.

  \item Set answer, category, and level per rules.

  \item ONLY Return the JSON object and nothing else (no markdown, no IDs).

  \item Make sure to balance the dataset according to the target distributions mentioned above, and use the data statistics snapshot to guide your generation and decide which category or difficulty level to prioritize.

  \item For counting questions:\\
• ONLY use the objects list to count the number of those items in the image\\
• When counting people, consider all synonyms (person, guy, man, people, skier, etc.) as the same category\\
• AVOID counting questions if there are too many of a certain object type in the objects list\\
• NEVER generate a counting question for objects that don't appear in the objects list

\end{enumerate}

────────────────────────────────────────\\
Now process the following scene graph:\\
Scene Graph data:
"""


QUESTION_PREFIX = '''<image> You are a vision-language assistant tasked with answering a question by observing an image, identifying relevant objects and relationships, and reasoning through a structured scene graph.

Your task is to:
- Identify objects of interest relevant to answering the given question, and any relevant relationships between these objects, and localise these objects in the image.
- Generate a visualisation of the relevant objects and any relationships as a structured scene graph following the format shared below. This scene graph should serve as a structured, mind-mapped knowledge representation of the key elements required to answer the given question. Focus only on objects and relationships that are directly pertinent to reasoning about the question.
- Use your observations from the given image and the visualised structured scene graph, to deeply think through the question before generating the final answer.
- In the scene graph, assign each relevant object a unique ID in the format "object_name.number" (e.g. "boy.1", "plate.2"). Provide bounding boxes for relevant objects in pixel coordinates as [x1, y1, x2, y2]
- Format your output using the following structure:
<observe>
{{Describe the scene depicted in the image covering the relevant relevant objects. Based on the question, what specific relevant objects of the image should you focus on?}}
</observe>
<scene>
{{
"objects": [
{{"id": "object_name.1", "bbox": [x1, y1, x2, y2]}},
...
],
"relationships": [
{{"subject": "object_name.1", "predicate": "predicate_word", "object": "object_name.2"}},
...
]
}}
</scene>
<think>
{{Reflect on the scene graph, observations, and reason through the question using the identified relevant objects and their relationships. Walk through your thought process step-by-step, as an internal monologue. Justify how the visual information leads to your final answer, and explain the reasoning path you followed to arrive at it.}}
</think>
<answer>
{{Your final answer}}
</answer>

Example:
Q. Where is the man with respect to the bike?
Options: (A) behind (B) in front of (C) beside (D) on top of

<observe>
The image shows an outdoor scene with a man standing on a paved surface near a bicycle. The man appears to be upright and facing forward, while the bicycle is parked sideways in the foreground. The relevant objects for answering the question are the man and the bike. The man is located toward the upper-left portion of the image, while the bicycle occupies a larger area in the lower half of the image. The man's legs are partially occluded by the bicycle, suggesting that he is positioned behind it in depth from the viewer's perspective.
</observe>
<scene>
{{
"objects": [
{{"id": "bike.1", "bbox": [0, 272, 310, 551]}},
{{"id": "man.1", "bbox": [0, 165, 38, 278]}}
],
"relationships": [
{{"subject": "man.1", "predicate": "behind", "object": "bike.1"}}
]
}}
</scene>
<think>
Looking at the visual image and the scene graph, I can see the man is standing behind the bicycle. The relationship in the scene graph confirms this with "man.1 behind bike.1". Among the options: (A) behind, (B) in front of, (C) beside, (D) on top of - the correct answer is "behind".
</think>
<answer>
(A) behind
</answer>

Image size: ({W} x {H})

Now answer the following question:
'''

GPT4O_VALIDATION_PROMPT = """Answer the following multiple choice question about the image.

Question: {question}

Options:
{options}

Reply with only the letter of the correct answer (A, B, C, or D)."""
