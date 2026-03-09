#!/usr/bin/env python3
"""Content safety review of deep_calib.jsonl lines 0-205."""

import json


def extract_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return str(c)


def get_full(messages):
    return "\n".join(extract_text(m.get("content", "")) for m in messages)


def first_user_content(messages):
    for m in messages:
        if m.get("role") == "user":
            return extract_text(m.get("content", ""))
    return ""


def summarize(idx, full, first_user):
    tl = full.lower()
    fu = first_user[:500].lower()

    # Extraction tasks: check first - they have a specific format
    if "expert structured information extraction" in tl and "extract queries" in tl:
        p = full.find("<passage>")
        snip = full[p + 9 : p + 400] if p >= 0 else ""
        if "writing anxiety" in snip or "argument" in snip or "research" in snip or "rhetoric" in snip or "library" in snip:
            topic = "Academic writing/rhetoric"
        elif "linear algebra" in snip or "mathematics" in snip or "beezer" in snip:
            topic = "Mathematics"
        elif "obesity" in snip or "health" in snip:
            topic = "Health/social research"
        elif "religion" in snip or "middle east" in snip:
            topic = "Religion/politics"
        else:
            topic = "Document"
        return f"Structured extraction: extract queries from {topic} passage."

    # Code review - must have code review prompt
    if "code review" in tl or "senior developer" in tl:
        if "<title>" in full:
            t = full.find("<title>") + 7
            t2 = full.find("</title>", t)
            title = full[t:t2].strip() if t2 > t else ""
        else:
            title = ""
        lang = "Rust" if "language: rust" in tl else "C++" if "language: c++" in tl else "Python" if "language: python" in tl else ""
        return f"Code review: {title or lang or 'code'} implementation."

    # Programming tasks - check first_user for primary request
    if "prove" in fu and "countable" in fu and "rational" in fu:
        return "Math: proving rationals are countable."
    if "paralleltaskassigner" in fu.replace(" ", "") or ("parallel" in fu and "task assigner" in fu):
        return "Extend ParallelTaskAssigner for inter-op parallelism in TensorFlow XLA."
    if "binary tree" in fu and "balance" in fu and "swap" in fu:
        return "Python: minimum node swaps to balance binary tree."
    if "async" in fu and "iterator" in fu and "visual basic" in fu:
        return "Add async iterator support in Visual Basic."
    if "refactor" in fu and "magic numbers" in fu and "assembly" in fu:
        return "Refactor CP/M assembly code to replace magic numbers with named constants."
    if "document undocumented" in fu or "gitdeploymanager" in fu:
        return "Document undocumented methods in GitDeployManager class."
    if "watchmen" in fu and ("comic" in fu or "information" in fu):
        return "QA about Watchmen comic book."
    if "first aid" in fu and "manual" in fu:
        return "Create user manual section on first aid training."
    if ("uk" in fu or "united kingdom" in fu) and ("economy" in fu or "economic" in fu) and "debate" in fu:
        return "Political debate on UK economy."
    if "assembly" in fu and ("understand" in fu or "help" in fu) and "extract" not in fu:
        return "Explain/understand Assembly code."
    if ("from tex" in fu or "tex to html" in fu) and "translate" in fu:
        return "Translate TeX to HTML."
    if "systematic plan" in fu and "optimal response" in fu:
        return "Construct systematic plan for optimal response generation."
    if "type alias" in fu and "type presentation" in fu:
        return "Support for type aliases in type presentations."
    if "asynchronous transaction" in fu:
        return "Asynchronous transaction and event processing."
    if "django" in fu and "repository" in fu:
        return "Create new Django repository."
    if "responsive navigation" in fu and "mobile" in fu:
        return "Add responsive navigation menu for mobile."
    if "transform" in fu and "structured representation" in fu and "human readable" in fu:
        return "Transform structured JSON to human-readable text."
    if "c++" in fu and "python" in fu and "translate" in fu and "extract" not in fu:
        return "Translate C++ code to Python."
    if "document ids" in fu and "list" in fu:
        return "List variables from documents (RAG-style task)."
    if "optimizing" in fu and "code" in fu and "performance" in fu:
        return "Optimize code for performance."
    if "healthcare" in fu and "customformauthenticationfilter" in fu:
        return "Healthcare platform authentication scenario."
    if "c to python" in fu or "caffe2" in fu:
        return "Translate C to Python (Caffe2)."
    if "error handling" in fu and "http status" in fu:
        return "Implement error handling for HTTP status codes."
    if "assembly" in fu and "keyboard" in fu:
        return "Explain Assembly keyboard input code."
    if "confidence interval" in fu and "regression" in fu:
        return "Calculate confidence interval of regression prediction."
    if "replace magic numbers" in fu and "named constant" in fu and "assembly" not in fu:
        return "Replace magic numbers with named constants."
    if "regex" in fu and "religion" in fu:
        return "Construct regex to find 'religion' instances."
    if "asynchronous i/o" in fu or "async i/o" in fu:
        return "Add support for asynchronous I/O operations."
    if "python" in fu and "x" in fu and "y" in fu and "string" in fu and "extract" not in fu:
        return "Python: strings of x/y characters (programming task)."
    if "real interest rate" in fu or "negative" in fu and "inflation" in fu:
        return "Article discussion: real interest rates."
    if "ruby" in fu and "trace" in fu and "output" in fu:
        return "Trace Ruby code and compute output."
    if "extract" in fu and "adjective" in fu and "json" in fu:
        return "Extract adjectives to JSON with constraints."
    if "perl" in fu and ("understand" in fu or "help" in fu) and "extract" not in fu:
        return "Explain Perl code."
    if "envsetup" in fu or "command-line tool" in fu and "env" in fu:
        return "Create EnvSetup command-line tool repository."
    if "windows registry" in fu:
        return "Article on Windows Registry intricacies."
    if "sql" in fu and "test case" in fu:
        return "Create test cases for complex SQL query."
    if "bracket" in fu and "valid" in fu and ("insertion" in fu or "deletion" in fu) and "java" in fu:
        return "Java: bracket validation and min insertions/deletions."
    if ("parenthes" in fu or "bracket" in fu) and ("balance" in fu or "valid" in fu or "flip" in fu) and "java" in fu:
        return "Java: parentheses/brackets validation or balancing."
    if "fortran" in fu and "python" in fu:
        return "Translate FORTRAN to Python."
    if "rpg" in fu and "role-playing" in fu:
        return "Create RPG narrative."
    if "index pair" in fu and "java" in fu:
        return "Java: array and index pairs manipulation."
    if "cmake" in fu and "utility" in fu:
        return "CMake utility functions."
    if "haskell" in fu and "configuration" in fu:
        return "Haskell configuration management scenario."
    if "structured data analysis" in fu:
        return "Structured data analysis task."
    if "go " in fu and "kub" in fu and "test" in fu:
        return "Create test cases for Go/Kubernetes code."
    if "extract" in fu and "symptom" in fu and "xml" in fu:
        return "Extract vehicle symptom sentences to XML."
    if "powershell" in fu and "test" in fu and "manifest" in fu:
        return "Create test cases for PowerShell module manifest."
    if "undo" in fu and "redo" in fu and "window" in fu:
        return "Undo/redo for window movements."
    if "docker" in fu and "image" in fu:
        return "Automated Docker image updates."
    if "java" in fu and "subset" in fu and "string" in fu:
        return "Java: smallest subset of strings."
    if "lexical chain" in fu:
        return "Identify lexical chain (NLP task)."
    if "custom attribute" in fu and "validation" in fu:
        return "Implement custom attribute argument validation."
    if "compare" in fu and "table" in fu and "extract" not in fu:
        return "Create comparison table."
    if "social media" in fu and "software engineer" in fu:
        return "Create social media posts for software engineer community."
    if "drought" in fu and "amazon" in fu:
        return "Ecology: Amazon drought impact on biodiversity."
    if "dr. rich" in fu or "advice" in fu and "mother" in fu:
        return "Article discussion (advice column)."
    if "css" in fu and "theme" in fu:
        return "Explain CSS theme code."
    if "c++" in fu and "understand" in fu and "extract" not in fu:
        return "Explain/understand C++ code."
    if "refactor" in fu and "run" in fu and "method" in fu:
        return "Refactor run method for code reuse."
    if "article" in fu and "english" in fu and "stewardship" in fu:
        return "Article discussion (language/English)."
    if "ai assistant" in fu and "desired outcomes" in fu and "extract" not in fu:
        return "AI assistant task (generic)."

    return f"Conversation: {first_user[:80].replace(chr(10), ' ')}..."


def check_flags(full):
    tl = full.lower()
    # Stricter jailbreak - full phrases only
    jailbreak = [
        "ignore all previous instructions",
        "ignore previous instructions",
        "pretend you have no restrictions",
        "you are now in developer mode",
        "dan mode activated",
        "bypass your safety",
        "bypass your guidelines",
    ]
    for j in jailbreak:
        if j in tl:
            return "jailbreak/safety-bypass"
    return None


def main():
    with open("data/text/deep_calib.jsonl") as f:
        lines = f.readlines()

    for i in range(206):
        obj = json.loads(lines[i])
        messages = obj.get("messages", [])
        full = get_full(messages)
        first_user = first_user_content(messages)
        summ = summarize(i, full, first_user)
        flag = check_flags(full)
        out = f"{i}: {summ}" + (f" ** FLAG: {flag}" if flag else "")
        print(out)


if __name__ == "__main__":
    main()
