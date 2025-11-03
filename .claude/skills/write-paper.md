# Skill: Write Paper

You are an expert academic writer specializing in LLM and multi-agent systems research.

## Task

When given:
- `experiments`: Path pattern for experiments to analyze (e.g., experiments/debate_*, experiments/tombench_*)
- `title`: Paper title or topic
- `output`: Output directory (e.g., papers/multi_agent_tom/)

Do:

1. **Gather experiment data**:
   - Find all matching experiment directories
   - Read results, configs, summaries
   - Extract key findings and metrics
   - Identify main research questions addressed

2. **Analyze findings**:
   - What are the key results?
   - What comparisons were made?
   - What insights emerged?
   - What are the implications?

3. **Search related work** (use Brave Search MCP):
   - Find recent papers on the topic
   - Get proper citations
   - Identify how your work fits in the literature

4. **Generate paper sections** (LaTeX format):
   - **Abstract**: Summarize the work (150-200 words)
   - **Introduction**: Research questions, motivation, contributions
   - **Related Work**: Recent papers and how this work differs
   - **Methods**: Experimental setup, models, strategies, benchmarks
   - **Results**: Tables, figures, key findings
   - **Discussion**: Interpret findings, implications, limitations
   - **Conclusion**: Summary and future work
   - **References**: Proper BibTeX citations

5. **Create supporting materials**:
   - `figures.md` - Descriptions of figures to create
   - `tables.tex` - LaTeX tables from results
   - `data.csv` - Raw data for plotting

6. **Save to output directory**:
   - `paper.tex` - Main LaTeX file
   - `sections/` - Individual section files
   - `figures.md` - Figure descriptions
   - `tables.tex` - Generated tables
   - `references.bib` - BibTeX file
   - `README.md` - Instructions for compiling

## Paper Structure Template

### Abstract (abstract.tex)

```latex
\begin{abstract}
We investigate {research question} through {approach}.
Using {methods}, we evaluate {what was tested} on {benchmarks}.
Our key findings include: (1) {finding 1}, (2) {finding 2}, (3) {finding 3}.
These results suggest {implications}.
We show that {main contribution}, with implications for {field}.
\end{abstract}
```

### Introduction (introduction.tex)

```latex
\section{Introduction}

{Hook - why is this important?}

Large language models (LLMs) have shown remarkable capabilities in {tasks}.
However, {problem or gap this work addresses}.
This raises the question: {research question}?

{Motivation - why study this?}

Understanding {topic} is crucial for {reasons}.
Recent work has shown {related findings}, but {what's missing}.

{Approach - what you did}

In this work, we {approach}.
We evaluate {what} on {benchmarks}, comparing {comparisons}.
Our experiments show {key results}.

{Contributions - what's new}

Our main contributions are:
\begin{itemize}
    \item {Contribution 1}
    \item {Contribution 2}
    \item {Contribution 3}
\end{itemize}

{Paper organization}

The rest of this paper is organized as follows:
Section~\ref{sec:related} reviews related work,
Section~\ref{sec:methods} describes our experimental setup,
Section~\ref{sec:results} presents our findings,
Section~\ref{sec:discussion} discusses implications,
and Section~\ref{sec:conclusion} concludes.
```

### Methods (methods.tex)

```latex
\section{Methods}
\label{sec:methods}

\subsection{Experimental Setup}

We evaluate {strategies} on {benchmarks}.
All experiments were conducted using {hardware} with {models}.

\subsection{Models}

We test the following models:
\begin{itemize}
    \item \textbf{{Model 1}}: {description}
    \item \textbf{{Model 2}}: {description}
\end{itemize}

\subsection{Strategies}

We compare {N} strategies:

\textbf{{Strategy 1}}: {description}

\textbf{{Strategy 2}}: {description}

\subsection{Benchmarks}

\textbf{{Benchmark 1}} \cite{citation}: {description} ({size} examples)

\textbf{{Benchmark 2}} \cite{citation}: {description} ({size} examples)

\subsection{Evaluation Metrics}

We measure:
\begin{itemize}
    \item \textbf{Accuracy}: {definition}
    \item \textbf{Latency}: {definition}
    \item \textbf{Cost}: {definition}
\end{itemize}
```

### Results (results.tex)

```latex
\section{Results}
\label{sec:results}

We present our experimental findings across {N} experiments.
Table~\ref{tab:main-results} shows the main results.

\begin{table}[t]
\centering
\caption{Performance comparison on {benchmark} (N={size} tasks)}
\label{tab:main-results}
\begin{tabular}{lrrr}
\toprule
Strategy & Accuracy & Latency (s) & Cost (USD) \\
\midrule
{strategy1} & {acc}\% & {lat} & {cost} \\
{strategy2} & {acc}\% & {lat} & {cost} \\
\bottomrule
\end{tabular}
\end{table}

The {winning strategy} achieved the highest accuracy ({X}\%),
outperforming the baseline by {Y} percentage points (p < 0.05).

\subsection{{Finding 1 Title}}

{Description of first key finding with supporting data}

Figure~\ref{fig:analysis1} shows {what the figure shows}.

\subsection{{Finding 2 Title}}

{Description of second key finding}
```

### Discussion (discussion.tex)

```latex
\section{Discussion}
\label{sec:discussion}

Our results demonstrate {main finding}.
This suggests {interpretation}.

\subsection{When Multi-Agent Strategies Help}

We observe that {conditions under which strategy helps}.
This is consistent with {related work} and suggests {explanation}.

\subsection{Cost-Benefit Analysis}

While {strategy} achieved {improvement}, it came at {cost}.
The {cost/latency} increase was {justified/not justified} because {reasoning}.

\subsection{Implications}

These findings have several implications:
\begin{itemize}
    \item {Implication 1}
    \item {Implication 2}
\end{itemize}

\subsection{Limitations}

This work has several limitations:
\begin{itemize}
    \item {Limitation 1}
    \item {Limitation 2}
\end{itemize}

\subsection{Future Work}

Future research directions include:
\begin{itemize}
    \item {Direction 1}
    \item {Direction 2}
\end{itemize}
```

### Conclusion (conclusion.tex)

```latex
\section{Conclusion}
\label{sec:conclusion}

We investigated {research question} through {approach}.
Our experiments on {benchmarks} show that {main findings}.

Key takeaways include:
\begin{itemize}
    \item {Takeaway 1}
    \item {Takeaway 2}
    \item {Takeaway 3}
\end{itemize}

These results suggest {implications} and open avenues for {future work}.
```

## References Template (references.bib)

```bibtex
@inproceedings{citation_key,
  title={Paper Title},
  author={Authors},
  booktitle={Conference/Journal},
  year={2025},
  url={https://...}
}
```

## Figures Description (figures.md)

```markdown
# Figures to Create

## Figure 1: Strategy Comparison
**File**: figures/comparison.pdf
**Type**: Bar chart
**Data**: Table 1 (strategy accuracy)
**Caption**: Performance comparison across strategies on ToMBench.
         The debate strategy (blue) outperforms single-model (red).

## Figure 2: Latency vs Accuracy
**File**: figures/tradeoff.pdf
**Type**: Scatter plot
**X-axis**: Latency (seconds)
**Y-axis**: Accuracy (%)
**Data**: All experiments
**Caption**: Tradeoff between latency and accuracy.
         Points show different strategy/model combinations.

## Figure 3: Cost Analysis
**File**: figures/cost.pdf
**Type**: Line plot
**X-axis**: Number of agents
**Y-axis**: Cost (USD)
**Data**: Debate experiments with 1-5 agents
**Caption**: Cost scales linearly with number of agents.
```

## Important Notes

- Use proper academic tone (formal, precise)
- Include statistical significance where appropriate (p-values)
- Cite all related work properly
- Be honest about limitations
- Use LaTeX best practices (booktabs, siunitx, etc.)
- Generate tables from actual data (don't make up numbers)
- Suggest figures but don't create them (describe what to plot)

## Example Usage

```
User: "Use write-paper with experiments=experiments/tombench_* title='When Do Multi-Agent Strategies Improve Theory of Mind Reasoning?' output=papers/multi_agent_tom/"

You:
1. Find all tombench experiments
2. Read and analyze results
3. Search for related papers on multi-agent LLMs and ToM
4. Generate all paper sections
5. Create tables from experiment data
6. Generate BibTeX citations
7. Write figure descriptions
8. Save everything to papers/multi_agent_tom/
```

## Quick Draft Mode

For rapid iteration:

```
User: "Quick paper draft from experiments/debate_* topic='multi-agent debate'"

You:
- Generate outline only
- Key findings bullet points
- Suggested sections
- Don't write full LaTeX yet
```

## Error Handling

- If experiments not found: List available experiments, ask for clarification
- If data insufficient: Note what's missing, write with available data
- If citations not found: Use placeholder \cite{TODO}, list what to cite
- If unclear research question: Infer from experiments, ask for confirmation
