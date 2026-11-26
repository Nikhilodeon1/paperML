#!/usr/bin/env python3
"""paperlint v3: style, QA, citation, and number audit for paper drafts (.tex or .md).

Usage (put the files first; --macros takes several files and would swallow a trailing paper):
  python3 -I paperlint.py main.tex --expand-tex --macros results_macros.tex \
          --bib references.bib --log CITATIONS_VERIFIED.md --ledger EVIDENCE.md \
          --tex-log main.log --venue call.txt

Exit code 1 if any ERROR. ERRORs are objective defects (Tier A slop words,
contractions, unresolved markers, template residue, hidden characters, bad
citations, numbers missing from the ledger). Style warnings fire when rates
leave the range of 1,843 pre-2023 human-written ML/clinical-ML papers
(ICML, AISTATS, UAI, ML4H, MLHC, MIDL, CHIL 2019-2022). A flag means
"inspect this," never "an AI wrote this."
"""
import argparse
import os
import re
import statistics as st
import sys

# --- Lexicons & Regex Patterns ---

HARD = (
    r"\b(?:delv(?:e|es|ed|ing)|tapestry|meticulous(?:ly)?|groundbreaking|transformative|paramount|unparalleled"
    r"|invaluable|multifaceted|testament|embark(?:s|ed|ing)?|endeavou?rs?|bolster(?:s|ed|ing)?|plethora|burgeoning"
    r"|commendable|noteworthy|myriad|pivotal|underscor(?:e|es|ed|ing)|intricate(?:ly)?|intricacies|seamless(?:ly)?"
    r"|holistic(?:ally)?|realm|unlock(?:s|ed|ing)?|nuanced?|harness(?:es|ed|ing)?|cornerstone|cutting-edge"
    r"|game-chang\w+|ever-evolving|revolutioni[sz](?:e|es|ed|ing))\b"
    r"|pav(?:e|es|ed|ing) the way|at the forefront|in today's|rapidly evolving"
    r"|navigat(?:e|es|ing) the (?:complexit|landscape|challenge)"
    r"|(?:has|have) (?:attracted|garnered) (?:significant|growing|considerable|increasing) (?:attention|interest)"
)

WATCH = set("""capabilities enabling enhance enhances enhanced enhancing comprehensive remarkable remarkably
ensuring offers integrates integrating highlighting advancements addressing struggle struggles emerged primarily
achieving preserving maintaining demonstrating inherent thereby insights diverse extensive effectively exhibit
exhibits potential tailored foster fosters fostering showcase showcases showcasing vital crucial additionally
notably substantial employs facilitate facilitates facilitating""".split())

PART = (
    r",\s+(?:thereby |thus |while )?(?:enabling|allowing|ensuring|highlighting|demonstrating|providing|offering"
    r"|making|leading|resulting|reflecting|suggesting|indicating|underscoring|showing|achieving|improving"
    r"|reducing|enhancing|facilitating|capturing|revealing|paving|opening|yielding|supporting|promoting)\b"
)

TRANS = (
    r"(?:^|(?<=[.!?] ))(?:Moreover|Furthermore|Additionally|In addition|Notably|Importantly|"
    r"Interestingly|Crucially|Overall|Consequently|Hence|Thus)\b"
)

SUMMARY = (
    r"^(?:Overall|In summary|In conclusion|Taken together|Together|Collectively|"
    r"These (?:results|findings|observations) (?:highlight|underscore|demonstrate|suggest|show|indicate|reveal)|"
    r"This (?:highlights|underscores|demonstrates|reveals|shows))\b"
)

CITE = r"\\(?:cite[a-z]*|autocite|parencite|textcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}"
ENVS = "equation|align|eqnarray|gather|multline|tikzpicture|lstlisting|verbatim|algorithm|algorithmic|thebibliography"
ABBR = re.compile(r"\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Tab|vs|cf|Dr|No|approx|resp|Alg|Appx)\.", re.I)
PLACE = r"\[(?:AUTHOR|VERIFY|CITE-NEEDED|TODO|TBD|FILL|INSERT|PLACEHOLDER)\b[^\]]*\]"
MARK = PLACE + r"|\bTODO\b|\bFIXME\b|\bXXX\b|\?\?"

CONFESS = (
    r"\bno (?:outside |external |independent |domain |medical )?(?:clinician|physician|expert|human|annotator|reviewer)s? "
    r"(?:has|have) (?:reviewed|checked|verified|labell?ed)\b"
    r"|\bwe leave (?:\w+ ){0,4}(?:undefended|unexamined|unaddressed|unexplained|untested|unverified)\b"
    r"|\b(?:wrote|written|defined|drafted|designed) (?:the |these |those )?[\w ]{0,30}\bourselves\b"
)

RESID_ERR = (
    r"(?i:as (?:suggested|requested|recommended|noted|pointed out) by (?:the |a |an )?(?:reviewers?|referees?|editors?)"
    r"|\b(?:in response to|per) (?:the )?reviewers?\b"
    r"|\breferee-driven\b"
    r"|\bimprov\w+ (?:on|over) (?:the |our )?(?:previous|earlier|prior) (?:run|runs|attempt|attempts|version|iteration)"
    r"|\b(?:earlier|previous|prior|updated|revised) version of (?:this|the present|our) (?:paper|work|manuscript)"
    r"|\bwe (?:now|then) (?:fix|fixed|correct|corrected|patch|patched)\b|\bI hope this|\bGreat question|\bas an AI\b)"
    r"|\bTL;DR|\bKey takeaways?\b|\(new\)|\bNEW:|\bUPDATED?:"
    r"|(?i:\bsupersed(?:ed|es)\b|\bwithdr(?:aw|awn|ew) (?:the |this |that )?(?:claim|prediction|hypothesis|conclusion)"
    r"|\bdesign error\b|\boriginal runs?\b"
    r"|\b(?:earlier|first|previous) \d+-seed\b|\bno (?:decision )?rule (?:was )?written\b|\bdated (?:repository )?file\b)"
)

RESID_WARN = (
    r"(?i:\b(?:in|of|for) (?:this|the current|the present|the final|the latest|the new) (?:version|iteration|draft|revision|round)\b"
    r"|\b(?:next|previous|earlier|latest|updated|revised) (?:iteration|iterations|draft|drafts|revision)\b"
    r"|\bin the (?:next|previous|last|earlier) (?:run|round|version)\b"
    r"|\bwe (?:previously|earlier|originally|initially) (?:\w+ ){0,2}(?:found|reported|showed|claimed|tried|believed|assumed)\b"
    r"|\b(?:bug|bugs|buggy)\b|\b(?:fixed|corrected|patched) (?:a |the )?(?:bug|error|leak|mistake)\b"
    r"|\b(?:wrote|written) down\b|\b(?:had )?already seen\b|\bwithout (?:any )?interpretation\b|\bwe (?:do|did) not interpret\b|\bunreviewed\b"
    r"|" + CONFESS + r")|\bHere (?:is|are)\b|\bCertainly\b"
)

TOOL_NAMES = r"\b(?:ChatGPT|Claude|Gemini|Copilot|GPT-4|GPT-5)\b"

CANDID = r"(?:Importantly|Notably|Crucially|Interestingly|Remarkably|Surprisingly|Intriguingly|Strikingly),"
ANTI = (
    r"\bnot (?:just|merely|simply)\b|\bnot only\b[^.;]{0,120}?\bbut (?:also|rather)\b"
    r"|\b(?:is|are|was|were) not [^.;]{3,60}[;,] (?:it|they|this|these) (?:is|are)\b"
)

GATES = [
    ("watch", 0.55, "% of words", "Tier B words"),
    ("part", 2.5, "per 1k words", "participial tails (', enabling ...')"),
    ("trans", 10.0, "% of sentences", "sentences opening with a stock transition"),
    ("triad", 3.0, "per 1k words", "list-of-three patterns"),
    ("candid", 0.85, "per 1k words", "candid openers (Notably, Interestingly, ...)"),
    ("anti", 0.6, "per 1k words", "not-X-but-Y constructions"),
    ("summ", 1.0, "per 1k words", "summary sentences (Overall, These results highlight ...)"),
]

ACR_OK = set("""AI ML DL NLP CV CNN CNNs RNN RNNs LSTM GRU GAN GANs VAE VAEs SGD GPU GPUs CPU MLP MLPs KL AUC AUROC AUPRC ROC MSE MAE
RMSE SOTA IID BERT GPT LLM LLMs ICU EHR EHRs MIMIC ECG EEG MRI CT PET HIV COVID US UK USA ICML ICLR AAAI UAI AISTATS MIDL CHIL MLHC
TMLR PMLR IEEE ACM NIH FDA IRB PCA SVD SVM KNN LDA ODE ODEs PDE SDE MCMC MAP MLE ELBO EM ERM IRM DRO RL MDP POMDP DQN PPO TD UCB GP
RBF NTK API URL PDF HTML SQL JSON RAM GB TB MB LR RF XGB ReLU GELU FPR TPR PPV NPV ECE NLL FGSM PGD DNN DNNs OOD ID ICD SOFA APACHE
SAPS GCS BMI BP HR OK II III IV VI VII VIII IX XI XII IP DAG DAGs CI SD SE MCC QA VQA RGB SGLD HMC VI CCA ICA NMF DP FL IoU PSNR SSIM
LPIPS FID IS TV NN NNs GNN GNNs GCN GCNs PAC SAT SDP LP QP TPU FLOPs FLOP KG MNIST CIFAR SVHN ImageNet BFGS LBFGS CGM OGTT PID SHA
CSV TSV XML DOI SNPE SBI T1D T2D JAX""".split())

ACR_DEF = re.compile(r"\(([A-Z][A-Za-z0-9\-]{1,9}s?)\)")
ACR_USE = re.compile(r"(?<![A-Za-z0-9\-])([A-Z][A-Z0-9]{1,7})s?(?![A-Za-z0-9\-])")

VAGUE = (
    r"(?:^|(?<=[.!?] ))(?:This|That|These|Those|It|Such)\s+(?:is|are|was|were|has|have|had|can|could|will|would|may|might|means"
    r"|shows|suggests|leads|makes|allows|implies|indicates|provides|helps|causes|explains|reflects|gives)\b"
)

PROC = (
    r"(?i:\b(?:fixed in advance|written in advance|written before the runs|before the runs"
    r"|(?:the|a|our) (?:decision )?rule (?:was|were|required|counted|fixed)|(?:decision )?rules? (?:required|written|fixed)|no (?:decision )?rule|dated (?:repository )?file"
    r"|supersed\w+|withdr(?:aw|awn|ew)|design error|earlier \d+-seed|first \d+-seed|follow-up sweeps?|confirmatory run)\b)"
)

DISCL = (
    r"(?i:\b(?:we (?:have|did|do) not|we (?:cannot|could not|can not)|was not run|were not run"
    r"|not (?:been )?(?:tested|verified|measured|established|isolated|shown|run|done)|untested|unexplained|remains? (?:open|unexplained)"
    r"|cannot (?:separate|say|exclude|distinguish)|does not (?:show|establish|test|separate|prove)|is confounded|not independent"
    r"|inconclusive|not supported|not significant)\b)"
)

LOGGATES = [
    ("process-log", PROC, 0.3, "process and pre-registration words (rule, fixed in advance, superseded, withdrawn)"),
    ("disclaimer-density", DISCL, 1.2, "disclaimers (we have not, not tested, confounded, unexplained)"),
]

# Fixed v2: exclude 'held-out', 'held out', 'held fixed', 'held constant', 'held at', 'held in'
HELD = (
    r"(?i:\b(?:(?:both|all|each|every|several|most|neither)\b[^.;]{0,40}?\b(?:held(?!\s*(?:-|out\b|fixed\b|constant\b|at\b|in\b(?!\s+part)))|hold)\b"
    r"|(?:predictions?|hypothes[ie]s|claims?|criteri(?:on|a)|checks?|tests?)\b[^.;]{0,25}?\b(?:held(?!\s*(?:-|out\b|fixed\b|constant\b|at\b|in\b(?!\s+part)))|(?:was|were) (?:confirmed|supported|met|borne out))\b"
    r"|as predicted|held up|we confirm(?:ed)?)\b)"
)

REFNUM = (
    r"\b(?:Sections?|Sec\.|Figures?|Fig\.|Tables?|Appendix|Appendices|Eq\.|Equations?|Theorem|Lemma|Proposition|Corollary|Algorithm"
    r"|Hypothesis|Definition|Assumption|Step|Phase|Experiment)s?\s+[\dA-Z][\w.\-]*"
)
NUM = r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?"
FRAC = r"(?<![\w.])(\d[\d,]*)\s*(?:of|out of|/)\s*(\d[\d,]*)(?![\d.]*\w)"
CLAIM = (
    r"\b(?:show|shows|showed|find|finds|found|report|reports|reported|propose|proposes|proposed|prove|proves|proved|achieve|achieves"
    r"|achieved|demonstrate|demonstrates|introduce|introduces|argue|argues|argued|observe|observes|observed|measure|measured|improve|improves"
    r"|improved|outperform\w*|reach|reaches)\b"
)

HELD_LIM = 0.0
EXPL = (
    r"(?i:\b(?:because|since|so that|which means|that is|i\.e\.|e\.g\.|for example|for instance|in other words|this means|in particular|specifically|recall that|note that"
    r"|denotes?|denoted|defined as|we define|let\s+(?:\S+\s+)?be|where\s+(?:\S+\s+)?(?:is|are|denotes?|represents?)|here)\b)"
)
EXPL_LIM = 2.0

# Fixed v2: also catch '44 of 45 and 44 of 45', '91 % and 91 %', '0 %, 0 % and 0 %', '1.00, 1.00 and 1.00'
SEAM = (
    r"(?<![\w.\-\u2212\u2013])(\d[\d,]*(?:\.\d+)?(?:\s*%)?(?:\s+of\s+\d[\d,]*)?)\s+(?:and|or|vs\.?|versus)\s+\1(?![\w.]*\d)(?!\s*[\-\u2212]?\d)"
    r"|(?<![\w.(\[{\-\u2212\u2013])(\d[\d,]*(?:\.\d+)?(?:\s*%)?),\s+\2,?\s+(?:and|or)\s+\2(?![\w.]*\d)(?![)\]}])"
)
SEAMV = (
    r"(?i:\((?:not met|met|held|failed|confirmed|supported|not supported)(?:, | and | or )(?:not met|met|held|failed|confirmed|supported)"
    r"|\bnot met(?:,| and) (?:not )?met\b)"
)
TEMPL = r"\{\{\s*[A-Za-z_][\w.]*\s*\}\}|%\([a-z_]+\)s|\$\{[a-z_]+\}|\b(?:nan|NaN|None)(?=%)|\(\s*(?:nan|NaN)\s*\)"
SCOPE = (
    r"(?i:\bwe (?:measure|claim|test|evaluate) no\b|\bmakes? no claims?\b|\bwe do not (?:claim|measure|test|evaluate|study|address|consider|interpret)\b"
    r"|\bleave[sd]? (?:\w+ ){0,3}(?:undefended|uninterpreted|unexamined|open)\b)"
)
RUBRIC = r"(?i:\b(?:covers?|spans?|addresses|across|along)\s+(?:all\s+)?(?:the\s+)?(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:workshop\s+)?(?:dimensions?|criteria|pillars|requirements)\b)"
SCRIPTF = r"\b[\w\-]+\.(?:sh|py|ipynb|yaml|yml|sbatch|slurm)\b"
SCRIPTN = r"\b[A-Za-z_\-]+\d+\.(?:sh|py|ipynb|sbatch)\b"
AMEND = r"(?i:\b(?:[3-9]|\d{2,}|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\s+amendments\b)"
MANYH = r"(?i:\b(?:1[5-9]|[2-9]\d)\s+(?:pre-?registered |registered |pre-?specified )?hypotheses\b)"
HASHD = r"\b[0-9a-f]{40,}\b"
SPLIT = r"(?i:\b(?:train|training|test|testing|validation|held-?out|splits?|cross-?validation|folds?)\b)"
DISPMATH = r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}|\\\[|\$\$"
HIDDEN = "[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad\u202f\U000e0000-\U000e007f]"
EXOTIC = "[\u00a0\u2000-\u200a\u3000]"
CONF = (
    r"(?i:\bpost[- ]hoc\b|\b(?:had )?already seen\b|\b(?:wrote|written) down\b|\bwithout (?:any )?interpretation\b|\bwe (?:do|did) not interpret\b|\bunreviewed\b"
    r"|" + CONFESS + ")"
)
ORD = r"(?i:\bthe (?:first|second|third|fourth|fifth|sixth) (?:set|prediction|hypothesis|hypotheses|experiment|test|criterion|group|claim|result)s?\b)"
STACK = (
    r"(?i:\b(?:could|might|may|can) (?:potentially|possibly|perhaps|arguably)\b|\b(?:potentially|possibly) (?:could|might|may)\b"
    r"|\bit (?:is|could be) (?:also )?possible that [^.]{0,50}\b(?:may|might|could)\b)"
)
LATE = r"(?i)ethic|reproducib|statement|acknowledg|broader impact|disclosure|use of (?:llm|ai)"
DISCLOSURE_SEC = r"(?i)disclosure|use of (?:an )?(?:llm|ai)|ai[- ]use|generative ai"

# --- v2 New Detectors ---

UK_ONLY = re.compile(
    r"\b(?:behaviour\w*|favour\w*|artefacts?|labelled|labelling|modelled|modelling|colour\w*|haem\w+|oedema|anaemi\w+|glycaemi\w+"
    r"|fibres?|programmes?|analys(?:e|ed|ing)"
    r"|(?:optimis|minimis|generalis|parameteris|regularis|standardis|personalis|normalis|visualis|randomis|summaris|initialis|utilis"
    r"|characteris|categoris|recognis|realis|prioritis|organis)(?:e|es|ed|er|ers|ing|ation|ations))\b",
    re.I,
)
# -ize spellings are not counted: Oxford spelling uses them with British spellings elsewhere.
# 'center' is not counted either: it appears in proper names (Medical Center).
US_ONLY = re.compile(
    r"\b(?:behavior\w*|favor\w*|labeled|labeling|modeled|modeling|colors?|colored|coloring|hemo\w+|edema|anemi\w+|glycemi\w+"
    r"|fibers?|analyz(?:e|ed|es|ing))\b",
    re.I,
)

SPELLED_TECH_NUM = re.compile(
    r"(?i:\btype (?:one|two) diabetes\b"
    r"|\bseeds? (?:zero|one|two|three|four|five)\b"
    r"|\bamendments? (?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\b"
    r"|\b(?:fifteen|twenty-nine|thirty|fifty|five hundred)[- ](?:point|step|start|fold|subject)s?\b"
    r"|\bgrid of (?:fifteen|twenty-nine|thirty)\b"
    r"|\bbudget of (?:one|two|three|five) hundred\b)"
)

ARCHAIC_INV = re.compile(r"\bOur contributions are (?:two|three|four|five|six)\.|\b(?:Two|Three|Four|Five) (?:questions|claims|findings) follow\.|\bWe measured it\.")
LOWER_COLON = re.compile(r":\s+[a-z]")
SLOT_PAREN = re.compile(
    r"(?<![\w.])(\d+\.\d+|\d[\d,]*\s*%)"
    r"(?:[^().\n]{0,40}\b(?:both|identical|same|equal|unchanged)\b[^().\n]{0,40}\([^()\n]*?(?<![\w.])\1(?![\w.]*\d)"
    r"|[^().\n]{0,40}\([^()\n]*?\b(?:both|identical|same|equal|unchanged)\b[^()\n]*?(?<![\w.])\1(?![\w.]*\d))"
)
FORMAL_CONN = re.compile(r"\b(?:Thus|Consequently|Accordingly|Moreover|Furthermore)\b")
DENY_CORRECT = re.compile(
    r"(?i:\b(?:is|are) not the cause\b|\bnot an? (?:optimization |data )?artifact\b|\bnot a clinical threshold\b|\bthe data are not uninformative\b)"
)


# --- File Loading, LaTeX Expansion & Line Mapping ---

def load(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        t = f.read()
    return t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


def parse_macros(paths):
    """Extract zero-arg text/number macros from LaTeX macro files."""
    macros = {}
    pat = re.compile(r"\\(?:new|renew|provide)command\{\\([A-Za-z][A-Za-z0-9]*)\}\{([^{}]*)\}")
    for p in paths:
        if not os.path.exists(p):
            continue
        txt = load(p)
        for name, val in pat.findall(txt):
            # Skip math-only symbol macros (like \SI -> S_I or \iauc -> \mathrm{iAUC})
            if "\\" in val or "_" in val or "^" in val:
                continue
            macros[name] = " ".join(val.split())
    return macros


MACRO_FILES = []


def expand_tex_with_map(root_path, seen=None):
    """Recursively expand \\input and \\include in a .tex file, returning (text, line_map).
    line_map[i] = (source_file, 1-based_line_no) for 0-based line index i in the expanded text.
    """
    if seen is None:
        seen = set()
    abs_p = os.path.abspath(root_path)
    if abs_p in seen or not os.path.exists(abs_p):
        return "", []
    seen.add(abs_p)
    base_dir = os.path.dirname(abs_p)
    raw = load(abs_p)
    out_lines = []
    line_map = []
    inp_pat = re.compile(r"^([ \t]*)\\(?:input|include)\{([^}]+)\}[ \t]*(?:%.*)?$")
    for lineno, line in enumerate(raw.splitlines(), start=1):
        m = inp_pat.match(line)
        if m:
            target = m.group(2).strip()
            if not target.endswith(".tex"):
                target += ".tex"
            cand = os.path.join(base_dir, target)
            if os.path.exists(cand):
                # Skip macro-definition-only files so we don't pollute prose with \newcommand lines
                cand_txt = load(cand)
                if len(re.findall(r"\\newcommand\{", cand_txt)) >= 10 and "\\section" not in cand_txt:
                    MACRO_FILES.append(cand)
                    out_lines.append("")
                    line_map.append((root_path, lineno))
                    continue
                sub_txt, sub_map = expand_tex_with_map(cand, seen)
                if sub_txt:
                    out_lines.extend(sub_txt.splitlines())
                    line_map.extend(sub_map)
                continue
        out_lines.append(line)
        line_map.append((root_path, lineno))
    return "\n".join(out_lines), line_map


def apply_macros(text, macros):
    """Expand zero-arg macros (e.g. \\resFoo{} or \\resFoo) in text without shifting newlines."""
    if not macros:
        return text
    # Sort longest-first to avoid prefix collisions
    names = sorted(macros.keys(), key=len, reverse=True)
    # Chunk into regex groups of 400 to keep regex engine fast
    out = text
    for i in range(0, len(names), 400):
        chunk = "|".join(re.escape(n) for n in names[i : i + 400])
        pat = re.compile(r"\\(" + chunk + r")(?:\{\}|(?![A-Za-z]))")
        out = pat.sub(lambda m: macros[m.group(1)], out)
    return out


def loc_of(text, pos, line_map, default_path):
    idx = text.count("\n", 0, pos)
    if line_map and 0 <= idx < len(line_map):
        return line_map[idx]
    return default_path, idx + 1


# --- Text Views & Helpers ---

def blank(pat, text, flags=0):
    return re.sub(pat, lambda m: re.sub(r"[^\n]", " ", m.group()), text, flags=flags)


def views(text, ext):
    if ext == ".tex":
        text = blank(r"(?<!\\)%[^\n]*", text)
        m = re.search(r"\\begin\{document\}", text)
        if m:
            text = blank(r"\A.*?\\begin\{document\}", text, re.S)
        text = blank(r"\\begin\{thebibliography\}.*", text, re.S)
        nums = blank(r"\\(?:vspace|hspace|setlength|includegraphics|resizebox|scalebox|rule|safefigure)\*?(?:\[[^\]]*\])*\{[^}]*\}", text)
        nums = blank(r"\\begin\{(?:%s)(\*?)\}.*?\\end\{(?:%s)\1\}" % (ENVS, ENVS), nums, re.S)
        nums = blank(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]|\\\(.*?\\\)", nums, re.S)
        nums = blank(CITE, nums)
        nums = blank(r"\\(?:ref|eqref|autoref|cref|Cref|label|url|href|input|include|usepackage|bibliography|bibliographystyle)\*?(?:\[[^\]]*\])*(?:\{[^}]*\})?", nums)
        prose = blank(r"\\begin\{(?:tabularx?|longtable)\*?\}.*?\\end\{(?:tabularx?|longtable)\*?\}", nums, re.S)
        prose = blank(r"\\[a-zA-Z]+\*?|[{}\\~&]", prose)
        nums = blank(r"\\[a-zA-Z]+\*?|[{}\\~]", nums)
    else:
        text = blank(r"`{3}.*?`{3}|`[^`\n]*`", text, re.S)
        text = blank(r"^[ \t]*\|?[ \t]*:?-{3,}[-:| \t]*$", text, re.M)
        prose = blank(r"^[ \t]*\|.*$", text, re.M)
        prose = blank(r"\]\([^)]*\)|^#+|\*\*|\*|\|", prose, re.M)
        nums = blank(r"\]\([^)]*\)", text)
    prose = blank(r"\(\s*\)|\[\s*\]", blank(CITE, prose))
    prose = blank(r"\[@[^\]]*\]|" + PLACE, prose)
    return text, nums, prose


def sentences(par):
    p = ABBR.sub(lambda m: m.group().replace(".", "\x00"), par)
    p = re.sub(r"(?<=\d)\.(?=\d)", "\x00", p)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", p)
    return [s.replace("\x00", ".").strip() for s in parts if len(s.split()) >= 3]


def sections(text):
    hs = []
    for m in re.finditer(r"^[ \t]*(?:\\((?:sub)*)section\*?\{([^}\n]*)\}|(#{1,6})[ \t]+(.+?))[ \t]*$", text, re.M):
        lv, ti = (1 + len(m.group(1)) // 3, m.group(2)) if m.group(2) is not None else (len(m.group(3)), m.group(4))
        hs.append((lv, ti.strip(), m.start()))
    return [(lv, ti, s, next((s2 for lv2, _, s2 in hs[i + 1 :] if lv2 <= lv), len(text))) for i, (lv, ti, s) in enumerate(hs)]


def abstract_span(text):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if m:
        return m.start(1), m.end(1)
    m = re.search(r"^[ \t]*#{1,6}[ \t]*abstract[ \t]*$", text, re.M | re.I)
    if m:
        nxt = re.search(r"^[ \t]*#{1,6}[ \t]", text[m.end() :], re.M)
        return m.end(), m.end() + nxt.start() if nxt else len(text)
    return None


def sent_around(text, pos):
    mk = ABBR.sub(lambda m: m.group().replace(".", "\x00"), text)
    a = max(mk.rfind(". ", 0, pos), mk.rfind("\n\n", 0, pos)) + 1
    nx = [x for x in (mk.find(". ", pos), mk.find("\n\n", pos)) if x >= 0]
    s = re.split(r"\\(?:sub)*section|\\begin|\\end|\n#", text[a : min(nx) if nx else len(text)])[0]
    return re.sub(r"\s+", " ", s).strip()


def venue_overlap(prose, venue_text, n=8):
    wd = lambda x: re.findall(r"[a-z0-9']+", x.lower())
    v = wd(venue_text)
    grams = {tuple(v[i : i + n]) for i in range(len(v) - n + 1)}
    w = wd(prose)
    return sorted({" ".join(w[i : i + n]) for i in range(len(w) - n + 1) if tuple(w[i : i + n]) in grams})


def count_nums(x):
    n = 0
    for m in re.finditer(NUM, re.sub(REFNUM, " ", x)):
        v = float(m.group().rstrip("%").replace(",", "") or 0)
        if not ((v == int(v) and v <= 20 and not m.group().endswith("%")) or 1900 <= v <= 2099):
            n += 1
    return n


def number_walls(sents, limit=12):
    keep = [x for x in sents if len(x.split()) >= 8 and sum(w.isalpha() for w in x.split()) / len(x.split()) >= 0.3]
    out, win, wc, wn, last = [], [], 0, 0, -1
    for i, x in enumerate(keep):
        k, c = len(x.split()), count_nums(x)
        win.append((i, k, c))
        wc += k
        wn += c
        while wc > 120 and len(win) > 1:
            _, kk, cc = win.pop(0)
            wc -= kk
            wn -= cc
        if wn >= limit and i > last + 3:
            out.append((keep[max(win, key=lambda w: w[2])[0]], wn))
            last = i
    return out


def undefined_acronyms(prose):
    defs = {m.group(1).rstrip("s") for m in ACR_DEF.finditer(prose)}
    cnt = {}
    for m in ACR_USE.finditer(prose):
        a = m.group(1)
        # Skip hypothesis codes H1..H99
        if re.match(r"^H\d+$", a):
            continue
        if a not in ACR_OK and a not in defs:
            cnt[a] = cnt.get(a, 0) + 1
    return sorted(a for a, c in cnt.items() if c >= 2)


def held_without_number(text, prose):
    out, seen = [], set()
    mk = ABBR.sub(lambda m: m.group().replace(".", "\x00"), prose)
    for m in re.finditer(HELD, prose):
        a = max(mk.rfind(". ", 0, m.start()), mk.rfind("\n\n", 0, m.start())) + 1
        nx = [x for x in (mk.find(". ", m.end()), mk.find("\n\n", m.end())) if x >= 0]
        b = min(nx) if nx else len(prose)
        if a in seen:
            continue
        seen.add(a)
        if not re.search(r"\d", re.sub(REFNUM, " ", text[a:b])):
            out.append((m, a))
    return out


def abstract_check(path, text, nums, line_map, issues):
    sp = abstract_span(text)
    if not sp:
        return
    a, b = sp
    ab = blank(REFNUM, nums[a:b])
    body = blank(REFNUM, nums[:a] + " " * (b - a) + nums[b:])
    norm = lambda x: x.replace(",", "")
    bfr = {(norm(k), norm(d)) for k, d in re.findall(FRAC, body)}
    seen = set()
    for m in re.finditer(FRAC, ab):
        k, d = norm(m.group(1)), norm(m.group(2))
        seen.add(m.start(1))
        if (k, d) in bfr:
            continue
        other = sorted({dd for kk, dd in bfr if kk == k}, key=float)
        fp, ln = loc_of(nums, a + m.start(), line_map, path)
        if other:
            issues.append(("ERROR", fp, ln, "abstract-body-drift", f"abstract says {k} of {d}, body says {k} of {', '.join(other)}"))
        else:
            issues.append(("WARN", fp, ln, "abstract-number-missing", f"{k} of {d} is not in the body"))

    def vals(s):
        out = set()
        for t in re.findall(NUM, s):
            try:
                v = float(t.rstrip("%").replace(",", ""))
            except ValueError:
                continue
            out.update({round(v, 6), round(v * 100, 6), round(v / 100, 6)})
        return out

    ok = vals(body)
    for m in re.finditer(NUM, ab):
        if m.start() in seen or any(f.start(1) <= m.start() < f.end(2) for f in re.finditer(FRAC, ab)):
            continue
        t = m.group()
        try:
            v = float(t.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if (v == int(v) and v <= 20 and not t.endswith("%")) or 1900 <= v <= 2099:
            continue
        if round(v, 6) not in ok:
            fp, ln = loc_of(nums, a + m.start(), line_map, path)
            issues.append(("WARN", fp, ln, "abstract-number-missing", f"{t} is in the abstract and not in the body"))


def stats(prose):
    pars = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", prose)]
    pars = [p for p in pars if len(p.split()) >= 6]
    sents = [[s for s in sentences(p)] for p in pars]
    allS = [s for ps in sents for s in ps]
    L = [len(s.split()) for s in allS]
    flat = " ".join(pars)
    low = re.findall(r"[A-Za-z][A-Za-z'-]*", flat.lower())
    n, w = len(L), max(len(low), 1)
    k = 1000.0 / w
    uniform = sum(1 for ps in sents if len(ps) >= 4 and st.pstdev([len(s.split()) for s in ps]) < 4.0)
    return dict(
        words=w,
        sentences=n,
        mean=st.mean(L) if L else 0,
        sd=st.pstdev(L) if L else 0,
        we=k * len(re.findall(r"\b(?:we|our|ours)\b", flat, re.I)),
        watch=100.0 * sum(x in WATCH for x in low) / w,
        part=k * len(re.findall(PART, flat)),
        trans=100.0 * sum(bool(re.match(TRANS, s)) for s in allS) / max(n, 1),
        triad=k * len(re.findall(r"\b\w+, \w+,? (?:and|or) \w+\b", flat)),
        candid=k * sum(bool(re.match(CANDID, s)) for s in allS),
        anti=k * len(re.findall(ANTI, flat, re.I)),
        summ=k * sum(bool(re.match(SUMMARY, s)) for s in allS),
        uniform_pars=uniform,
        pars=len(pars),
        sents=sents,
    )


def duplicate_blocks(text, prose, line_map, path, issues):
    """Detect repeated \\paragraph headings in the same section and near-duplicate paragraphs."""
    # 1. Repeated identical \paragraph{...} headings inside the same section
    secs = sections(text)
    for lv, ti, s0, e0 in secs:
        if any(s0 < y[2] < e0 for y in secs if y[2] != s0):
            continue
        seen_p = {}
        for m in re.finditer(r"\\paragraph\{([^}]+)\}", text[s0:e0]):
            title = m.group(1).strip().lower()
            pos = s0 + m.start()
            if title in seen_p:
                fp, ln = loc_of(text, pos, line_map, path)
                issues.append(("WARN", fp, ln, "duplicate-block", f"repeated \\paragraph heading '{m.group(1)}' in section '{ti}'"))
            else:
                seen_p[title] = pos

    # 2. Near-duplicate prose paragraphs (Jaccard >= 0.55 on 4-word shingles for >= 25 words)
    shingles = []
    for m in re.finditer(r"(?:^|\n\s*\n)([^\n].*?)(?=\n\s*\n|\Z)", prose, re.S):
        blk = re.sub(r"\s+", " ", m.group(1)).strip()
        wds = re.findall(r"[a-z0-9]+", blk.lower())
        if len(wds) < 25:
            continue
        sh = {tuple(wds[i : i + 4]) for i in range(len(wds) - 3)}
        shingles.append((m.start(1), blk[:70], sh))
    for i in range(len(shingles)):
        for j in range(i + 1, len(shingles)):
            s1, s2 = shingles[i][2], shingles[j][2]
            inter = len(s1 & s2)
            if inter and inter / min(len(s1), len(s2)) >= 0.55:
                fp, ln = loc_of(prose, shingles[j][0], line_map, path)
                issues.append(("WARN", fp, ln, "duplicate-block", f"near-duplicate paragraph ({int(100 * inter / min(len(s1), len(s2)))}% overlap): {shingles[j][1]}"))


def tex_qa_checks(raw_text, line_map, path, issues):
    """LaTeX-specific QA: empty stub subsections, unqualified appendix float refs, stale table captions."""
    # Strip comments keeping line count
    clean = blank(r"(?<!\\)%[^\n]*", raw_text)

    # 1. Empty stub subsections (heading followed directly by float/input/heading with no prose)
    sec_re = re.compile(r"\\((?:sub)+)section\*?\{([^}\n]*)\}")
    for m in sec_re.finditer(clean):
        after = clean[m.end() : m.end() + 400]
        # Remove \label{...}, \FloatBarrier, whitespace
        trimmed = re.sub(r"\\label\{[^}]*\}|\\FloatBarrier|\s+", "", after)
        if trimmed.startswith(("\\begin{table", "\\begin{figure", "\\input{", "\\subsection", "\\section")):
            fp, ln = loc_of(raw_text, m.start(), line_map, path)
            issues.append(("WARN", fp, ln, "tex-qa", f"empty stub heading '{m.group(2)}' with no introductory prose before float/table"))

    # 2. Unqualified main-text references to appendix floats
    app_m = re.search(r"\\appendix\b", clean)
    if app_m:
        main_part = clean[: app_m.start()]
        app_part = clean[app_m.start() :]
        main_labels = set(re.findall(r"\\label\{([^}]+)\}", main_part))
        app_labels = set(re.findall(r"\\label\{([^}]+)\}", app_part))
        for m in re.finditer(r"(Table|Figure)~\\ref\{([^}]+)\}", main_part):
            lbl = m.group(2)
            if lbl in app_labels and lbl not in main_labels:
                ctx = main_part[max(0, m.start() - 50) : min(len(main_part), m.end() + 50)]
                if "Appendix" not in ctx and "Appx" not in ctx:
                    fp, ln = loc_of(raw_text, m.start(), line_map, path)
                    issues.append(("WARN", fp, ln, "tex-qa", f"{m.group(0)} points to an appendix float without an '(Appendix ...)' qualifier"))

    # 3. Stale "A dash means..." caption in tables without dashes
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", clean, re.S):
        body = m.group(1)
        cap_m = re.search(r"\\caption\{(.*?)\}", body, re.S)
        if cap_m and re.search(r"(?i)a dash means|dashes (?:mean|indicate|denote)", cap_m.group(1)):
            tab_m = re.search(r"\\begin\{tabularx?\}.*?\\end\{tabularx?\}", body, re.S)
            if tab_m and not re.search(r"(?:&\s*(?:--|---|—)\s*(?:&|\\\\))", tab_m.group(0)):
                fp, ln = loc_of(raw_text, m.start() + cap_m.start(), line_map, path)
                issues.append(("WARN", fp, ln, "tex-qa", "table caption explains dashes, but the table contains no dash cells"))


def check_tex_log(log_path, issues):
    """Parse LaTeX .log for severe Overfull hboxes (>10pt) and undefined references/citations."""
    if not log_path or not os.path.exists(log_path):
        return
    txt = load(log_path)
    for m in re.finditer(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)[^\n]*\n([^\n]*)", txt):
        pts = float(m.group(1))
        if pts >= 10.0:
            issues.append(("WARN", log_path, 0, "tex-log-overfull", f"Overfull \\hbox ({pts:.1f}pt too wide): {m.group(2).strip()[:75]}"))
    for m in re.finditer(r"LaTeX Warning: (?:Reference|Citation) `([^']+)' on page \d+ undefined", txt):
        issues.append(("ERROR", log_path, 0, "tex-log-undefined", m.group(0)))


# --- Main Lint Routine ---

def lint(path, issues, cites, exempt=(), venue=None, expand_tex=False, macros=None, strict_dashes=False):
    ext = os.path.splitext(path)[1].lower()
    mac = dict(macros or {})
    if ext == ".tex" and expand_tex:
        del MACRO_FILES[:]
        raw_src, line_map = expand_tex_with_map(path)
        for k, v in parse_macros(MACRO_FILES + [path]).items():
            mac.setdefault(k, v)
    else:
        raw_src = load(path)
        line_map = [(path, i + 1) for i in range(len(raw_src.splitlines()))]

    expanded_src = apply_macros(raw_src, mac)
    text, nums, prose = views(expanded_src, ext)
    at_loc = lambda s, pos: loc_of(s, pos, line_map, path)
    snip = lambda s, m: re.sub(r"\s+", " ", s[max(0, m.start() - 30) : m.start()] + "«" + m.group() + "»" + s[m.end() : m.end() + 30]).strip()

    # Basic rules (note: dash now runs on prose, not raw text, so LaTeX table '---' never false-positives)
    rules = [
        ("ERROR", "banned-word", HARD, re.I),
        ("ERROR", "contraction", r"\b\w+(?:n't|'re|'ll|'ve|'d|'m)\b|\b(?:it|that|there|here|what|let)'s\b", re.I),
        ("ERROR", "marker", MARK, 0),
        ("ERROR", "process-residue", RESID_ERR, 0),
        ("WARN", "process-residue?", RESID_WARN, 0),
        ("WARN", "first-person-singular", r"(?<![\w$])I (?:am|was|have|had|will|would|think|believe|show|propose|find|found|argue|used?|thank|hope|suggest|ran|trained|built|chose|decided|did|do|can|could|want|need|see|study|present|introduce|consider|took?|also|first|then|now)\b", 0),
        ("WARN", "generic-opener", r"\bIn recent years\b|\bhas become (?:increasingly|a)\b|\bmore and more\b|\brapid(?:ly)? (?:advances|growth|development|progress)\b", re.I),
    ]
    for lvl, tag, pat, fl in rules:
        src = text if tag in ("marker", "process-residue") else prose
        for m in re.finditer(pat, src, fl):
            fp, ln = at_loc(src, m.start())
            issues.append((lvl, fp, ln, tag, snip(src, m)))

    # Em-dash check on prose only (ERROR only if --strict-dashes, else WARN above 0.5/1k words or >= 3 in a draft)
    dash_pat = r"—|(?<=\s)–(?=\s)|---|(?<=\s)--(?=\s)"
    dashes = list(re.finditer(dash_pat, prose))
    s = stats(prose)
    if dashes:
        if strict_dashes:
            for m in dashes:
                fp, ln = at_loc(prose, m.start())
                issues.append(("ERROR", fp, ln, "dash", snip(prose, m)))
        elif s["words"] >= 800 and 1000.0 * len(dashes) / s["words"] > 2.0:
            issues.append(("WARN", path, 0, "dash-density", f"{len(dashes)} em/en dashes ({1000.0 * len(dashes) / s['words']:.1f} per 1k words; above 2.0 in 5% of human papers). Replace parenthetical dashes with commas or parentheses, and keep the sentence plain."))
        elif len(dashes) >= 3:
            issues.append(("NOTE", path, 0, "dash-density", f"{len(dashes)} em/en dashes; fine in moderation (18% of human papers use them). Prefer commas or parentheses."))

    # Tool names check: exempt AI-disclosure section
    secs = sections(text)
    prose_no_discl = prose
    for lv, ti, s0, e0 in secs:
        if re.search(DISCLOSURE_SEC, ti):
            prose_no_discl = prose_no_discl[:s0] + re.sub(r"[^\n]", " ", prose_no_discl[s0:e0]) + prose_no_discl[e0:]
    for m in re.finditer(r"\\(?:footnote|thanks)\{[^{}]*\}", text):
        if re.search(r"(?i)\b(?:LLM|AI|language model|assist\w*|used|edited|drafted|generated)\b", m.group()):
            prose_no_discl = prose_no_discl[:m.start()] + re.sub(r"[^\n]", " ", prose_no_discl[m.start():m.end()]) + prose_no_discl[m.end():]
    for m in re.finditer(TOOL_NAMES, prose_no_discl):
        fp, ln = at_loc(prose_no_discl, m.start())
        issues.append(("WARN", fp, ln, "process-residue?", snip(prose_no_discl, m)))

    for m in re.finditer(SEAM, prose):
        fp, ln = at_loc(prose, m.start())
        issues.append(("WARN", fp, ln, "macro-seam", "same number twice (a pasted macro? 0.5% of human papers have one): " + snip(prose, m)))
    for m in re.finditer(SEAMV, prose):
        fp, ln = at_loc(prose, m.start())
        issues.append(("WARN", fp, ln, "macro-seam", "verdict repeated (no human paper in 1,777 does this): " + snip(prose, m)))
    for m in re.finditer(TEMPL, prose):
        fp, ln = at_loc(prose, m.start())
        issues.append(("ERROR", fp, ln, "template-residue", snip(prose, m)))

    for m in SLOT_PAREN.finditer(prose):
        fp, ln = at_loc(prose, m.start())
        issues.append(("WARN", fp, ln, "macro-seam", f"repeated number in parenthetical (slot-filling seam?): {snip(prose, m)}"))

    allS = [x for ps in s["sents"] for x in ps]
    at_sent = lambda x: at_loc(prose, max(prose.find(x[:40]), 0))

    prev = False
    for x in allS:
        cur = bool(re.match(TRANS, x))
        if cur and prev:
            fp, ln = at_sent(x)
            issues.append(("WARN", fp, ln, "stacked-transition", x[:90]))
        prev = cur

    inst = {
        "watch": lambda: [m for m in re.finditer(r"\b(?:%s)\b" % "|".join(sorted(WATCH)), prose, re.I)],
        "part": lambda: list(re.finditer(PART, prose)),
        "anti": lambda: list(re.finditer(ANTI, prose, re.I)),
        "triad": lambda: list(re.finditer(r"\b\w+, \w+,? (?:and|or) \w+\b", prose)),
    }
    sent_pat = {"trans": TRANS, "candid": CANDID, "summ": SUMMARY}

    if s["words"] >= 800:
        tripped = [g for g in GATES if s[g[0]] > g[1]]
        lvl = "WARN" if len(tripped) >= 2 else "NOTE"
        if tripped:
            msg = (
                "%d density flags: outside the human range (1%% of human papers, 86%% of 2025 abstract batches)" % len(tripped)
                if len(tripped) >= 2
                else "1 density flag: about 1 in 10 human papers has one (1 in 6 at clinical venues), fix only if the prose reads badly"
            )
            issues.append((lvl, path, 0, "density-summary", msg))
        for key, lim, unit, what in tripped:
            issues.append((lvl, path, 0, "density", f"{what}: {s[key]:.2f} {unit} (human p98 {lim})"))
            if key in inst:
                for m in inst[key]()[:8]:
                    fp, ln = at_loc(prose, m.start())
                    issues.append((lvl, fp, ln, key, snip(prose, m)))
            else:
                for x in [y for y in allS if re.match(sent_pat[key], y)][:8]:
                    fp, ln = at_sent(x)
                    issues.append((lvl, fp, ln, key, x[:90]))

        # Over-sanitized check (Goodhart signal)
        if s["words"] >= 2500 and all(s[k] == 0.0 for k in ("watch", "part", "trans", "candid", "anti", "summ")):
            issues.append(("NOTE", path, 0, "over-sanitized", "all six style-density metrics are exactly 0.0; prose may be over-scrubbed to a linter"))

        # Dialect mix check (UK vs US spelling)
        uk_hits = UK_ONLY.findall(prose)
        us_hits = US_ONLY.findall(prose)
        if len(uk_hits) >= 2 and len(us_hits) >= 2:
            uk_ex = ", ".join(sorted({w.lower() for w in uk_hits})[:5])
            us_ex = ", ".join(sorted({w.lower() for w in us_hits})[:5])
            strong = len(uk_hits) >= 4 and len(us_hits) >= 4
            issues.append(("WARN" if strong else "NOTE", path, 0, "dialect-mix", f"mixed UK ({len(uk_hits)}: {uk_ex}) and US ({len(us_hits)}: {us_ex}) spellings; standardize to one dialect. WARN at 4 or more of each (1.9% of human papers), NOTE at 2 or more (7.5%)"))

        # Spelled-out technical numerals
        for m in SPELLED_TECH_NUM.finditer(prose):
            fp, ln = at_loc(prose, m.start())
            issues.append(("WARN", fp, ln, "spelled-out-tech-num", f"use standard scientific numerals instead of spelling out: {snip(prose, m)}"))

        # Claude-voice & LLM-editing checks
        for m in ARCHAIC_INV.finditer(prose):
            fp, ln = at_loc(prose, m.start())
            issues.append(("WARN", fp, ln, "claude-voice", f"stylized inversion: {snip(prose, m)}"))
        lc_rate = 1000.0 * len(LOWER_COLON.findall(prose)) / s["words"]
        if lc_rate > 4.5:
            issues.append(("WARN", path, 0, "claude-voice", f"{lc_rate:.1f} lowercase-continuation colons per 1k words (human p99 4.9 before math is removed; often em-dash-ban overcompensation)"))
        ag_rate = 1000.0 * len(re.findall(r"\bagainst\b", prose, re.I)) / s["words"]
        if ag_rate > 2.2:
            issues.append(("NOTE", path, 0, "claude-voice", f"{ag_rate:.1f} uses of 'against' per 1k words (consider varying with 'compared to' or 'versus')"))
        fc_rate = 1000.0 * len(FORMAL_CONN.findall(prose)) / s["words"]
        if fc_rate > 3.5:
            issues.append(("WARN", path, 0, "claude-voice", f"{fc_rate:.1f} formal connectives (Thus/Consequently/Accordingly/Moreover/Furthermore) per 1k words"))
        carry_rate = 1000.0 * len(re.findall(r"\bcarr(?:y|ies|ied)\b", prose, re.I)) / s["words"]
        if carry_rate > 1.5:
            issues.append(("NOTE", path, 0, "claude-voice", f"{carry_rate:.1f} uses of 'carry/carries' per 1k words; use more specific verbs"))
        for m in DENY_CORRECT.finditer(prose):
            fp, ln = at_loc(prose, m.start())
            issues.append(("NOTE", fp, ln, "claude-voice", f"deny-then-correct phrasing: {snip(prose, m)}"))
        # Repetitive \item openers (e.g., 3+ consecutive '\item We show that')
        for blk in re.finditer(r"\\begin\{(?:enumerate|itemize)\}(.*?)\\end\{(?:enumerate|itemize)\}", text, re.S):
            openers = [
                " ".join(re.findall(r"[A-Za-z]+", it.group(1))[:3]).lower()
                for it in re.finditer(r"\\item\s+([^\n\\]{10,80})", blk.group(1))
            ]
            if len(openers) >= 3:
                top = max(set(openers), key=openers.count)
                if top and openers.count(top) >= 3:
                    fp, ln = at_loc(text, blk.start())
                    issues.append(("WARN", fp, ln, "claude-voice", f"{openers.count(top)} list items open identically with '{top}'"))

        # Duplicate paragraphs / headings
        duplicate_blocks(text, prose, line_map, path, issues)

        vg = 1000.0 / s["words"] * sum(bool(re.match(VAGUE, x)) for x in allS)
        if vg > 3.0:
            issues.append(("WARN", path, 0, "vague-reference", f"{vg:.1f} sentences per 1k words open with This/It/These + verb (human p98 3.0)"))
            for x in [y for y in allS if re.match(VAGUE, y)][:8]:
                fp, ln = at_sent(x)
                issues.append(("WARN", fp, ln, "vague-reference", x[:90]))

        pr = prose
        for lv, ti, s0, e0 in secs:
            if re.search(r"(?i)limitation|threats? to validity|caveat", ti):
                pr = pr[:s0] + re.sub(r"[^\n]", " ", pr[s0:e0]) + pr[e0:]
        for t_ in exempt:
            if t_.strip():
                pr = blank(re.escape(t_.strip()), pr, re.I)

        for key, pat, lim, what in LOGGATES:
            ms = list(re.finditer(pat, pr))
            d = 1000.0 * len(ms) / s["words"]
            if d > lim:
                note = " Counts repetition: delete repeats, and do not reword a limit to dodge this check." if key == "disclaimer-density" else ""
                issues.append(("WARN", path, 0, key, f"{d:.2f} per 1k words outside Limitations: {what} (above {lim} in under 2% of human papers).{note}"))
                if secs:
                    leaf = [x for x in secs if not any(y is not x and x[2] < y[2] < x[3] for y in secs)]
                    cnt = [(sum(1 for m in ms if x[2] <= m.start() < x[3]), x[1][:40]) for x in leaf]
                    front = sum(1 for m in ms if m.start() < min(x[2] for x in secs))
                    cnt = sorted([c for c in cnt if c[0]] + ([(front, "front matter or abstract")] if front else []), reverse=True)[:6]
                    issues.append(("WARN", path, 0, key, "per section: " + ", ".join(f"{n} in {ti}" for n, ti in cnt)))
                for m in ms[:8]:
                    fp, ln = at_loc(pr, m.start())
                    issues.append(("WARN", fp, ln, key, snip(pr, m)))

        hw = held_without_number(text, prose)
        if hw:
            hd = 1000.0 * len(hw) / s["words"]
            issues.append((
                "WARN" if hd > HELD_LIM else "NOTE",
                path,
                0,
                "held-no-number",
                f"{len(hw)} sentence{'s' if len(hw) > 1 else ''} say a prediction or test held with no number beside it ({hd:.2f} per 1k; 0.5% of human papers have any). Put the number next to the verdict.",
            ))
            for m, a0 in hw[:8]:
                fp, ln = at_loc(prose, a0)
                issues.append(("NOTE", fp, ln, "held-no-number", snip(prose, m)))

        nw = number_walls(allS)
        if nw:
            issues.append(("WARN", path, 0, "number-wall", f"{len(nw)} stretches with {nw[0][1]} or more numbers in about 120 words (human p99 11, max 15). Put them in a table and state the pattern in the text."))
            for x, c in nw[:4]:
                fp, ln = at_sent(x)
                issues.append(("WARN", fp, ln, "number-wall", f"{c} numbers: {x[:70]}"))

        if s["words"] >= 2500:
            xd = 1000.0 * len(re.findall(EXPL, prose)) / s["words"]
            if xd < EXPL_LIM:
                issues.append(("WARN", path, 0, "low-explanation", f"{xd:.2f} explanatory words per 1k (because, where X denotes, let X be, for example, that is). Human median 6.1; 1.6% of human papers of 2,500 words or more sit below {EXPL_LIM}. The text may state results without saying what was computed or why. Add the missing equation, fitted target, or reason; adding connectives alone does not fix it."))

        sc = list(re.finditer(SCOPE, pr))
        if len(sc) >= 2:
            issues.append(("WARN", path, 0, "scope-negation", f"{len(sc)} scope negations outside Limitations (we do not claim, we measure no; 0.2% of human papers have two). Say what the paper does, with one negation at most."))
            for m in sc[:6]:
                fp, ln = at_loc(pr, m.start())
                issues.append(("WARN", fp, ln, "scope-negation", snip(pr, m)))

        for m in re.finditer(RUBRIC, prose):
            fp, ln = at_loc(prose, m.start())
            issues.append(("WARN", fp, ln, "rubric-framing", "reads like a venue rubric pasted into the paper: " + snip(prose, m)))

        raw_chk = expanded_src.replace("\\_", "_")
        scr = list(re.finditer(SCRIPTF, raw_chk))
        if len(scr) >= 3 or re.search(SCRIPTN, raw_chk):
            issues.append(("WARN", path, 0, "log-structure", f"{len(scr)} script or config file names in the text, or a numbered script such as phase8.sh (0.06% of human papers). Describe the procedure, and name files only in a release note."))

        am = list(re.finditer(r"(?i)\bamendments?\b", prose))
        hcodes = {int(x) for x in re.findall(r"\bH(\d{1,3})\b", prose)}
        htags = list(re.finditer(r"\(H\d+\s*,[^)]*\)", prose))
        if len(am) >= 2 or re.search(AMEND, prose) or re.search(MANYH, prose) or len(hcodes) >= 8 or len(htags) >= 2:
            issues.append(("WARN", path, 0, "log-structure", f"{len(am)} mentions of amendments, {len(hcodes)} distinct hypothesis codes, {len(htags)} inline tags like (H27, post hoc, met). No human paper in 1,777 has two amendments, 15 or more hypotheses, or 8 distinct codes. Group hypotheses into families, put status in one table column, and keep the amendment history in a repository file."))

        for m in list(re.finditer(HASHD, raw_chk))[:3]:
            fp, ln = at_loc(raw_chk, m.start())
            issues.append(("WARN", fp, ln, "log-structure", f"{len(m.group())}-character hash digest in the text (0.06% of human papers). Use an 8-character prefix or point to the release file."))

        if venue:
            ov = venue_overlap(prose, load(venue))
            if ov:
                issues.append(("WARN", path, 0, "venue-text-overlap", f"{len(ov)} runs of 8 words copied from the venue text. Meet the criteria through content, not by restating them."))
                for x in ov[:4]:
                    issues.append(("WARN", path, 0, "venue-text-overlap", x))

        if ext in (".tex", ".md") and not re.search(DISPMATH, text) and s["words"] >= 2500:
            issues.append(("NOTE", path, 0, "no-display-math", "no display equation in a long paper (uncalibrated: PDF text carries no math markup). Confirm the methods state the objective, the model equations, and the fitted target, in words if not in math."))

        if len(re.findall(r"(?i)\bdata ?sets?\b", prose)) >= 5 and not re.search(SPLIT, prose):
            issues.append(("NOTE", path, 0, "no-split-wording", "datasets are discussed and no train/test/validation/split wording appears."))

        od = [m for m in re.finditer(ORD, prose)]
        if 1000.0 * len(od) / s["words"] > 0.5:
            issues.append(("WARN", path, 0, "positional-reference", f"{len(od)} positional references like 'the second set' (above 0.5 per 1k). Name the item directly."))
            for m in od[:6]:
                fp, ln = at_loc(prose, m.start())
                issues.append(("WARN", fp, ln, "positional-reference", snip(prose, m)))

        sq = list(re.finditer(STACK, prose))
        if len(sq) >= 2:
            issues.append(("WARN", path, 0, "stacked-qualifier", f"{len(sq)} stacked hedges (could potentially, might possibly). Keep one qualifier."))
            for m in sq[:6]:
                fp, ln = at_loc(prose, m.start())
                issues.append(("WARN", fp, ln, "stacked-qualifier", snip(prose, m)))

        defd = sorted({m.group(1).rstrip("s") for m in ACR_DEF.finditer(prose)} - ACR_OK)
        defd = [d for d in defd if len(d) >= 2 and (d.isupper() or re.match(r"[A-Z][A-Za-z]*-?[A-Z0-9]", d))]
        if len(defd) > 12:
            issues.append(("WARN", path, 0, "coined-acronyms", f"{len(defd)} acronyms defined in the paper (human p98 12): " + ", ".join(defd[:15])))

        ua = undefined_acronyms(prose)
        if ua:
            issues.append(("WARN" if len(ua) > 12 else "NOTE", path, 0, "undefined-acronym", f"{len(ua)} used twice or more with no definition (human median about 4): " + ", ".join(ua[:15])))

        if s["sentences"] >= 20:
            notes = []
            if not 15 <= s["mean"] <= 32:
                notes.append("mean sentence length outside 15-32 words")
            if s["sd"] < 6:
                notes.append("sentence length SD under 6 (very uniform rhythm)")
            if s["we"] < 11:
                notes.append("under 11 we/our per 1k words (agentless or nominal style)")
            if s["uniform_pars"] > 0.3 * max(s["pars"], 1):
                notes.append("many paragraphs have near-identical sentence lengths")
            for n_ in notes:
                issues.append(("WARN", path, 0, "sanity", n_))
    else:
        issues.append(("WARN", path, 0, "short-text", f"{s['words']} words: density checks skipped; pass --expand-tex on main.tex or run on the full paper"))

    print(
        f"{path}: {s['words']} words, {s['sentences']} sentences | mean {s['mean']:.1f} sd {s['sd']:.1f} | we {s['we']:.0f}/1k | "
        f"tierB {s['watch']:.2f}% | tails {s['part']:.1f}/1k | transitions {s['trans']:.1f}% | triads {s['triad']:.1f}/1k | "
        f"candid {s['candid']:.1f}/1k | antithesis {s['anti']:.1f}/1k | summaries {s['summ']:.1f}/1k"
    )

    for m in re.finditer(CITE, text):
        fp, ln = at_loc(text, m.start())
        cites += [(k.strip(), fp, ln, sent_around(text, m.start())) for k in m.group(1).split(",") if k.strip()]
    for m in re.finditer(r"\[((?:-?@[\w:.\-]+[;,\s]*)+)\]", text):
        fp, ln = at_loc(text, m.start())
        cites += [(k, fp, ln, sent_around(text, m.start())) for k in re.findall(r"@([\w:.\-]+)", m.group(1))]

    abstract_check(path, text, nums, line_map, issues)
    sp = abstract_span(text)
    if sp:
        ab = [x for x in sentences(re.sub(r"\s+", " ", prose[sp[0] : sp[1]]))]
        cand = [x for x in ab if re.search("(?:%s)|(?:%s)|(?:%s)" % (DISCL, PROC, CONF), x)]
        if len(cand) >= 2:
            issues.append(("WARN", path, 0, "abstract-candor", f"{len(cand)} of {len(ab)} abstract sentences are limits, process, or post hoc notes (0.1% of human abstracts have two). Keep one as scope."))
            for x in cand[:4]:
                issues.append(("WARN", path, 0, "abstract-candor", x[:90]))

    for m in re.finditer(HIDDEN, text):
        fp, ln = at_loc(text, m.start())
        issues.append(("ERROR", fp, ln, "hidden-char", f"U+{ord(m.group()):04X} invisible character (strip it)"))
    for m in list(re.finditer(EXOTIC, text))[:5]:
        fp, ln = at_loc(text, m.start())
        issues.append(("WARN", fp, ln, "odd-space", f"U+{ord(m.group()):04X} unusual space"))

    if ext == ".tex":
        tex_qa_checks(expanded_src, line_map, path, issues)
        app = re.search(r"\\appendix\b", text)
        for m in re.finditer(r"\\begin\{(?:table|figure)\*?\}", text):
            if app and m.start() > app.start():
                continue
            enc = [x for x in secs if x[2] < m.start() < x[3]]
            if enc and re.search(LATE, enc[-1][1]):
                fp, ln = at_loc(text, m.start())
                issues.append(("NOTE", fp, ln, "float-after-statement", f"float sits in '{enc[-1][1][:40]}'. Place it beside the paragraph that first cites it."))

    return nums, line_map


def citation_audit(cites, bib, log, issues):
    keys = {}
    if bib:
        bt = load(bib)
        for blk in re.split(r"\n\s*(?=@\w+\s*\{)", bt):
            m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", blk)
            if not m or m.group(1).lower() in ("comment", "string", "preamble"):
                continue
            keys[m.group(2)] = blk
            for f, alt in (("author", ("editor",)), ("title", ()), ("year", ("date",))):
                if not any(re.search(r"\b%s\s*=" % a, blk, re.I) for a in (f,) + alt):
                    issues.append(("WARN", bib, 0, "bib-field", f"{m.group(2)} lacks {f}"))
            if not re.search(r"\b(?:journal|booktitle|publisher|school|eprint|howpublished|institution|url|doi)\s*=", blk, re.I):
                issues.append(("WARN", bib, 0, "bib-field", f"{m.group(2)} lacks venue/eprint/url"))
    status = {}
    if log:
        for line in load(log).splitlines():
            m = re.search(r"\b(UNVERIFIED|METADATA-ONLY|VERIFIED)\b", line)
            cells = [c.strip(" `*") for c in line.split("|") if c.strip()]
            if m and cells:
                k = cells[0] if "|" in line else re.sub(r"^[\s\-*`]*([\w:.\-]+).*$", r"\1", line)
                status[k] = m.group(1)
    seen = set()
    for k, fp, line, sent in cites:
        if status.get(k) == "METADATA-ONLY":
            body = re.sub(r"\b(?:19|20)\d\d[a-z]?\b", "", re.sub(r"\[(?:-?@[\w:.\-]+[;,\s]*)+\]", "", re.sub(CITE, "", sent)))
            if re.search(r"\d", body) or re.search(CLAIM, body):
                issues.append(("WARN", fp, line, "cite-metadata-only-claim", f"{k}: cited as an existence pointer, but the sentence says what it contains: {sent[:80]}"))
        if k in seen:
            continue
        seen.add(k)
        if bib and k not in keys:
            issues.append(("ERROR", fp, line, "cite-missing-in-bib", k))
        if log:
            s = status.get(k)
            if s is None:
                issues.append(("ERROR", fp, line, "cite-not-verified", f"{k}: no record in log"))
            elif s == "UNVERIFIED":
                issues.append(("ERROR", fp, line, "cite-not-verified", f"{k}: UNVERIFIED"))
            elif s == "METADATA-ONLY":
                issues.append(("WARN", fp, line, "cite-metadata-only", f"{k}: exists, content not checked"))
    for k in keys:
        if k not in seen:
            issues.append(("WARN", bib, 0, "bib-unused", k))


def number_audit(path, nums, line_map, ledger, issues):
    def vals(s):
        out = set()
        for t in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", s):
            try:
                v = float(t.rstrip("%").replace(",", ""))
            except ValueError:
                continue
            out.update({round(v, 6), round(v * 100, 6), round(v / 100, 6)})
        return out

    ok = vals(load(ledger))
    nums = blank(REFNUM, nums)
    for m in re.finditer(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", nums):
        t = m.group()
        try:
            v = float(t.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if (v == int(v) and v <= 20 and not t.endswith("%")) or 1900 <= v <= 2099:
            continue
        if round(v, 6) not in ok:
            fp, ln = loc_of(nums, m.start(), line_map, path)
            issues.append(("ERROR", fp, ln, "number-not-in-ledger", t))


def main():
    ap = argparse.ArgumentParser(description="paperlint v3: style, QA, citation, and number audit for academic drafts.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expand-tex", action="store_true", help="recursively inline \\input/\\include in .tex files with exact source line mapping")
    ap.add_argument("--macros", nargs="*", default=[], help="LaTeX macro files (e.g. results_macros.tex) to expand before linting")
    ap.add_argument("--bib", help="BibTeX file to validate against citations")
    ap.add_argument("--log", help="CITATIONS_VERIFIED.md log")
    ap.add_argument("--ledger", help="EVIDENCE.md ledger")
    ap.add_argument("--tex-log", help="compiled LaTeX .log file to check for severe Overfull hboxes and undefined refs")
    ap.add_argument("--venue", help="text of the venue call, rubric, or reviewer form; warns on 8-word runs copied from it")
    ap.add_argument("--exempt", default="", help="comma-separated domain terms to skip in process-log and disclaimer counts")
    ap.add_argument("--strict-dashes", action="store_true", help="treat em/en dashes in prose as ERROR instead of density WARN")
    a = ap.parse_args()

    ex = [x for x in a.exempt.split(",") if x.strip()]
    macros = parse_macros(a.macros)
    issues, cites, allnums = [], [], []
    for f in a.files:
        n, lm = lint(f, issues, cites, ex, a.venue, expand_tex=a.expand_tex, macros=macros, strict_dashes=a.strict_dashes)
        allnums.append((f, n, lm))
    if a.bib or a.log:
        citation_audit(cites, a.bib, a.log, issues)
    if a.ledger:
        for f, n, lm in allnums:
            number_audit(f, n, lm, a.ledger, issues)
    if a.tex_log:
        check_tex_log(a.tex_log, issues)

    shown = {}
    for lvl, f, line, tag, sn in sorted(issues, key=lambda x: ({"ERROR": 0, "WARN": 1}.get(x[0], 2), x[3], x[1], x[2])):
        shown[tag] = shown.get(tag, 0) + 1
        if shown[tag] <= 8:
            print(f"{lvl:5s} {f}:{line}  [{tag}] {sn}")
        elif shown[tag] == 9:
            print(f"      ... more [{tag}] findings hidden")
    e = sum(i[0] == "ERROR" for i in issues)
    w = sum(i[0] == "WARN" for i in issues)
    nn = len(issues) - e - w
    print(f"\n{e} errors, {w} warnings, {nn} notes")
    sys.exit(1 if e else 0)


if __name__ == "__main__":
    main()
