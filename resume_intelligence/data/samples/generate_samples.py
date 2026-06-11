"""Generate sample resume files for testing. Run once: python data/samples/generate_samples.py"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from docx import Document

SAMPLES_DIR = Path(__file__).parent

WELL_FORMATTED_TEXT = """Alex Johnson
alex.johnson@email.com
https://github.com/alexjohnson
https://linkedin.com/in/alexjohnson

EDUCATION
B.Tech Computer Science, State University
CGPA: 8.33/10
2019 - 2023

SKILLS
Python, Java, SQL, React, Docker, Git, Machine Learning, FastAPI

EXPERIENCE
Software Engineering Intern | TechCorp Inc.
June 2022 - August 2022
- Built REST APIs using FastAPI and PostgreSQL
- Deployed services with Docker on AWS
- Reduced API response time through query optimization

PROJECTS
Resume Parser Tool
- Developed a PDF resume parser using Python and PyMuPDF
- Implemented skill extraction with regex and section detection

E-Commerce Dashboard
- Built a React dashboard with data visualization
- Integrated with REST backend APIs
"""

POORLY_FORMATTED_TEXT = """jane doe
email: jane.doe@example.com

skills python java

worked at startup did coding stuff
"""


def create_docx(path: Path, content: str) -> None:
    doc = Document()
    for line in content.splitlines():
        doc.add_paragraph(line)
    doc.save(path)


def create_pdf(path: Path, content: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), content, fontsize=11)
    doc.save(path)
    doc.close()


def main() -> None:
    create_docx(SAMPLES_DIR / "sample_resume_well_formatted.docx", WELL_FORMATTED_TEXT)
    create_pdf(SAMPLES_DIR / "sample_resume_well_formatted.pdf", WELL_FORMATTED_TEXT)
    create_docx(SAMPLES_DIR / "sample_resume_poorly_formatted.docx", POORLY_FORMATTED_TEXT)
    (SAMPLES_DIR / "sample_resume_well_formatted.txt").write_text(WELL_FORMATTED_TEXT)
    print("Sample resumes generated.")


if __name__ == "__main__":
    main()
