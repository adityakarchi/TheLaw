"""Law Corpus Loader — loads Indian statutes from JSON + TXT files.

Two sources:
  1. JSON files — curated sections with structured metadata (punishment, bail, etc.)
  2. TXT files  — full statute text parsed into every section (complete coverage)

Supports 9 statutes: IPC, CrPC, Evidence Act, Constitution, Companies Act,
Consumer Protection Act, Indian Contract Act, IT Act, Motor Vehicles Act.

Both are loaded, deduplicated, and converted to LangChain Documents for the FAISS index.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Corpus path

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "legal_corpus"

JSON_CORPUS_FILES = [
    "ipc_sections.json",
    "crpc_sections.json",
    "evidence_act.json",
]

TXT_CORPUS_FILES = {
    "ipc.txt": {
        "act_name": "Indian Penal Code, 1860",
        "abbreviation": "IPC",
        "parser": "dash",
    },
    "crpc.txt": {
        "act_name": "Code of Criminal Procedure, 1973",
        "abbreviation": "CrPC",
        "parser": "dash",
    },
    "evidence_act.txt": {
        "act_name": "Indian Evidence Act, 1872",
        "abbreviation": "IEA",
        "parser": "dash",
    },
    "companies_act.txt": {
        "act_name": "Companies Act, 2013",
        "abbreviation": "CA2013",
        "parser": "dash",
    },
    "contract_act.txt": {
        "act_name": "Indian Contract Act, 1872",
        "abbreviation": "ICA",
        "parser": "dash",
    },
    "it_act.txt": {
        "act_name": "Information Technology Act, 2000",
        "abbreviation": "ITA",
        "parser": "dash",
    },
    "motor_vehicle_act.txt": {
        "act_name": "Motor Vehicles Act, 1988",
        "abbreviation": "MVA",
        "parser": "dash",
    },
    "consumer_act.txt": {
        "act_name": "Consumer Protection Act, 2019",
        "abbreviation": "CPA",
        "parser": "numbered",
    },
    "constitution.txt": {
        "act_name": "Constitution of India",
        "abbreviation": "COI",
        "parser": "article",
    },
}

# Regex to detect section start in full-text statutes
# Matches: "123. Title text.—body" or "123A. Title.––body"
SECTION_PATTERN = re.compile(
    r'^(\d+[A-Z]{0,3})\.\s+'   # Section number at line start
    r'(.+?)'                     # Title (non-greedy)
    r'\.\s*(?:\u2014|\u2013{2}|--)\s*'  # .— or .–– or .-- separator
    , re.MULTILINE
)

# Pattern to split text into section blocks 
SECTION_SPLIT = re.compile(
    r'(?m)(?=^\d+[A-Z]{0,3}\.\s+[^\n]+?\.\s*(?:\u2014|\u2013{2}|--))'
)

# Footnote pattern (lines like "1. Ins. by Act 25 of 2005" at page bottoms)
FOOTNOTE_PATTERN = re.compile(
    r'^\d+\.\s+(?:Subs|Ins|Added|Omitted|Rep)\.\s+by\s+Act\b', re.MULTILINE
)

# Page number pattern (standalone digits on a line)
PAGE_NUMBER = re.compile(r'^\d{1,3}\s*$', re.MULTILINE)


@dataclass
class LawSection:
    """Structured representation of a single law section."""
    act_name: str
    abbreviation: str
    section: str
    title: str
    crime: str
    description: str
    punishment: str
    jail_term: str
    fine: str
    bailable: bool
    cognizable: bool
    category: str
    keywords: List[str]
    source: str = "json"  # "json" (curated) or "txt" (parsed from full text)

    def to_searchable_text(self) -> str:
        """Convert to a rich text block optimized for embedding search."""
        parts = [
            f"Act: {self.act_name} ({self.abbreviation})",
            f"Section: {self.abbreviation} Section {self.section}",
            f"Title: {self.title}",
            f"Crime: {self.crime}",
            f"Category: {self.category}",
            f"Description: {self.description}",
            f"Punishment: {self.punishment}",
            f"Jail Term: {self.jail_term}",
            f"Fine: {self.fine}",
            f"Bailable: {'Yes' if self.bailable else 'No'}",
            f"Cognizable: {'Yes' if self.cognizable else 'No'}",
            f"Keywords: {', '.join(self.keywords)}",
        ]
        return "\n".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        """Extract metadata dict for FAISS Document."""
        return {
            "act_name": self.act_name,
            "abbreviation": self.abbreviation,
            "section": self.section,
            "title": self.title,
            "crime": self.crime,
            "punishment": self.punishment,
            "jail_term": self.jail_term,
            "fine": self.fine,
            "bailable": self.bailable,
            "cognizable": self.cognizable,
            "category": self.category,
            "keywords": ", ".join(self.keywords) if self.keywords else "",
            "source": self.source,
        }

    def to_document(self) -> Document:
        """Convert to a LangChain Document for FAISS indexing."""
        return Document(
            page_content=self.to_searchable_text(),
            metadata=self.to_metadata(),
        )

    def to_result_dict(self) -> Dict[str, Any]:
        """Convert to a structured result dict for the UI."""
        return {
            "act_name": self.act_name,
            "abbreviation": self.abbreviation,
            "section": f"Section {self.section}",
            "section_number": self.section,
            "title": self.title,
            "crime": self.crime,
            "description": self.description,
            "punishment": self.punishment,
            "jail_term": self.jail_term,
            "fine": self.fine,
            "bailable": self.bailable,
            "cognizable": self.cognizable,
            "category": self.category,
            "keywords": self.keywords,
        }


def _load_corpus_file(file_path: Path) -> List[LawSection]:
    """Load a single corpus JSON file into LawSection objects."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        act_name = data.get("act_name", "Unknown Act")
        abbreviation = data.get("abbreviation", "UNK")
        sections = data.get("sections", [])

        result = []
        for sec in sections:
            law = LawSection(
                act_name=act_name,
                abbreviation=abbreviation,
                section=str(sec.get("section", "")),
                title=sec.get("title", ""),
                crime=sec.get("crime", ""),
                description=sec.get("description", ""),
                punishment=sec.get("punishment", ""),
                jail_term=sec.get("jail_term", ""),
                fine=sec.get("fine", ""),
                bailable=sec.get("bailable", False),
                cognizable=sec.get("cognizable", True),
                category=sec.get("category", ""),
                keywords=sec.get("keywords", []),
                source="json",
            )
            result.append(law)

        logger.info(f"Loaded {len(result)} sections from {file_path.name}")
        return result

    except Exception as e:
        logger.error(f"Failed to load corpus file {file_path}: {e}")
        return []


# =====================================================================
# TXT FILE PARSER — extracts every section from full statute text
# =====================================================================

def _clean_statute_text(text: str) -> str:
    """Remove page numbers, footnotes, and clean up statue text."""
    # Remove standalone page numbers
    text = PAGE_NUMBER.sub("", text)
    # Remove footnote lines
    text = FOOTNOTE_PATTERN.sub("", text)
    # Normalize whitespace (collapse multiple blank lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _is_footnote_section(section_text: str) -> bool:
    """Check if a parsed 'section' is actually a footnote."""
    if re.match(r'^\d+\.\s+(?:Subs|Ins|Added|Omitted|Rep)\.', section_text.strip()):
        return True
    return False


def _parse_dash_sections(file_path: Path, act_name: str, abbreviation: str) -> List[LawSection]:
    """Parse statutes using 'Section. Title.—body' separator pattern."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []

    cleaned = _clean_statute_text(raw_text)
    blocks = SECTION_SPLIT.split(cleaned)

    result = []
    for block in blocks:
        block = block.strip()
        if not block or len(block) < 20:
            continue

        match = SECTION_PATTERN.match(block)
        if not match:
            continue

        sec_number = match.group(1)
        sec_title = match.group(2).strip()

        if _is_footnote_section(block):
            continue
        if re.search(r'\[(?:Omitted|Repealed|Rep)\.\]', sec_title, re.IGNORECASE):
            continue

        body_start = match.end()
        body = block[body_start:].strip()
        if len(body) > 3000:
            body = body[:3000] + "..."

        law = LawSection(
            act_name=act_name, abbreviation=abbreviation,
            section=sec_number, title=sec_title, crime=sec_title,
            description=body, punishment="", jail_term="", fine="",
            bailable=False, cognizable=True, category="", keywords=[],
            source="txt",
        )
        result.append(law)

    logger.info(f"Parsed {len(result)} sections from {file_path.name} (dash parser)")
    return result


# Pattern for "number. (1) body text" format (Consumer Act, etc.)
NUMBERED_SECTION = re.compile(
    r'(?m)^(\d+[A-Z]{0,3})\.\s+\(1\)\s+',
)

def _parse_numbered_sections(file_path: Path, act_name: str, abbreviation: str) -> List[LawSection]:
    """Parse statutes using 'number. (1) body' pattern (no dash separator)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []

    cleaned = _clean_statute_text(raw_text)

    # Find all section starts
    starts = list(NUMBERED_SECTION.finditer(cleaned))
    if not starts:
        logger.warning(f"No sections found in {file_path.name} (numbered parser)")
        return []

    result = []
    for i, m in enumerate(starts):
        sec_number = m.group(1)
        start_pos = m.start()
        end_pos = starts[i + 1].start() if i + 1 < len(starts) else len(cleaned)
        body = cleaned[start_pos:end_pos].strip()

        # Extract title from first sentence
        first_line = body.split('\n')[0]
        title_match = re.match(r'\d+[A-Z]{0,3}\.\s+\(1\)\s+(.{10,80}?)(?:\.|,|\n)', first_line)
        sec_title = title_match.group(1).strip() if title_match else f"Section {sec_number}"

        if len(body) > 3000:
            body = body[:3000] + "..."

        law = LawSection(
            act_name=act_name, abbreviation=abbreviation,
            section=sec_number, title=sec_title, crime=sec_title,
            description=body, punishment="", jail_term="", fine="",
            bailable=False, cognizable=True, category="", keywords=[],
            source="txt",
        )
        result.append(law)

    logger.info(f"Parsed {len(result)} sections from {file_path.name} (numbered parser)")
    return result


# Pattern for Constitution articles: "number. (1) body" with margin titles above
ARTICLE_START = re.compile(r'(?m)^(\d+[A-Z]?)\.\s+(?:\(1\)\s+|[A-Z])')

def _parse_article_sections(file_path: Path, act_name: str, abbreviation: str) -> List[LawSection]:
    """Parse Constitution-style articles with margin annotations as titles."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []

    cleaned = _clean_statute_text(raw_text)
    lines = cleaned.splitlines()

    # Build article blocks
    articles = []
    current_sec = None
    current_lines = []
    margin_title = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r'^(\d+[A-Z]?)\.\s+', stripped)

        if m:
            sec_num = m.group(1)
            # Ignore pure TOC entries (very short, just number and title)
            # Real articles have sub-clauses (1), (2) etc. or substantial body text

            if current_sec is not None:
                articles.append((current_sec, margin_title, '\n'.join(current_lines)))

            current_sec = sec_num
            current_lines = [stripped]

            # Look back for margin title (non-empty lines before this article)
            # Constitution format: "Title words\nmore title.\n\nArticle..."
            margin_lines = []
            found_blank = False
            for j in range(i - 1, max(i - 8, -1), -1):
                prev = lines[j].strip()
                if not prev:
                    if found_blank:
                        break  # Two blanks = definitely past the title
                    found_blank = True
                    continue
                if re.match(r'^\d+[A-Z]?\.\s+', prev):
                    break
                # Skip part/chapter/schedule headers
                if re.match(r'^(PART|CHAPTER|SCHEDULE)\s', prev):
                    break
                # Skip lines that look like body text (long lowercase sentences)
                if len(prev) > 80 and prev[0].islower():
                    break
                # Skip page-marker lines like "(Part III.—...)"
                if re.match(r'^\(Part\s', prev) or re.match(r'^THE CONSTITUTION', prev):
                    break
                margin_lines.insert(0, prev)
            
            margin_title = " ".join(margin_lines).strip().rstrip(".")

            margin_title = margin_title.strip().rstrip(".")
        elif current_sec is not None:
            current_lines.append(stripped)

    # Don't forget last article
    if current_sec is not None:
        articles.append((current_sec, margin_title, '\n'.join(current_lines)))

    # Filter to real articles (skip TOC entries and repealed articles)
    result = []
    for sec_num, title, body in articles:
        body = body.strip()
        if len(body) < 30:
            continue
        if re.search(r'\[(?:Omitted|Repealed|Rep)\.\]', body[:100], re.IGNORECASE):
            continue

        if not title:
            title = f"Article {sec_num}"

        if len(body) > 3000:
            body = body[:3000] + "..."

        law = LawSection(
            act_name=act_name, abbreviation=abbreviation,
            section=sec_num, title=title, crime=title,
            description=body, punishment="", jail_term="", fine="",
            bailable=False, cognizable=True, category="", keywords=[],
            source="txt",
        )
        result.append(law)

    logger.info(f"Parsed {len(result)} articles from {file_path.name} (article parser)")
    return result


def _parse_txt_file(file_path: Path, act_name: str, abbreviation: str, parser: str = "dash") -> List[LawSection]:
    """Parse a full-text statute .txt file into LawSection objects.

    Dispatches to the appropriate parser based on the file's format.
    """
    if parser == "numbered":
        return _parse_numbered_sections(file_path, act_name, abbreviation)
    elif parser == "article":
        return _parse_article_sections(file_path, act_name, abbreviation)
    else:
        return _parse_dash_sections(file_path, act_name, abbreviation)


def load_all_laws() -> List[LawSection]:
    """Load all law sections from JSON + TXT corpus files.

    JSON sections take priority over TXT for the same section
    (they have richer metadata: punishment, bail status, etc.)
    """
    all_laws: List[LawSection] = []
    seen: Dict[str, int] = {}  # "IPC-302" -> index in all_laws

    # 1. Load JSON first (higher quality metadata)
    for filename in JSON_CORPUS_FILES:
        file_path = CORPUS_DIR / filename
        if file_path.exists():
            laws = _load_corpus_file(file_path)
            for law in laws:
                key = f"{law.abbreviation}-{law.section}"
                seen[key] = len(all_laws)
                all_laws.append(law)
        else:
            logger.warning(f"JSON corpus not found: {file_path}")

    json_count = len(all_laws)

    # 2. Load TXT files (complete coverage, skip sections already in JSON)
    for filename, meta in TXT_CORPUS_FILES.items():
        file_path = CORPUS_DIR / filename
        if file_path.exists():
            parser_type = meta.get("parser", "dash")
            laws = _parse_txt_file(file_path, meta["act_name"], meta["abbreviation"], parser_type)
            for law in laws:
                key = f"{law.abbreviation}-{law.section}"
                if key not in seen:
                    seen[key] = len(all_laws)
                    all_laws.append(law)
                # else: JSON version already exists with richer metadata
        else:
            logger.warning(f"TXT corpus not found: {file_path}")

    txt_added = len(all_laws) - json_count
    logger.info(
        f"Total law sections: {len(all_laws)} "
        f"({json_count} from JSON + {txt_added} from TXT)"
    )
    return all_laws


def load_all_documents() -> List[Document]:
    """Load all law sections as LangChain Documents (ready for FAISS)."""
    laws = load_all_laws()
    docs = [law.to_document() for law in laws]
    logger.info(f"Created {len(docs)} documents for indexing")
    return docs


def get_law_by_section(abbreviation: str, section: str) -> Optional[LawSection]:
    """Look up a specific section by act abbreviation and section number."""
    all_laws = load_all_laws()
    for law in all_laws:
        if law.abbreviation.upper() == abbreviation.upper() and law.section == str(section):
            return law
    return None


def search_laws_by_keyword(keyword: str) -> List[LawSection]:
    """Simple keyword search across all laws (fallback when FAISS unavailable)."""
    keyword_lower = keyword.lower()
    all_laws = load_all_laws()
    results = []

    for law in all_laws:
        # Check keywords list
        if any(keyword_lower in kw.lower() for kw in law.keywords):
            results.append(law)
            continue
        # Check description
        if keyword_lower in law.description.lower():
            results.append(law)
            continue
        # Check crime name
        if keyword_lower in law.crime.lower():
            results.append(law)
            continue
        # Check title
        if keyword_lower in law.title.lower():
            results.append(law)

    return results
