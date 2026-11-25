
import fitz
from pathlib import Path


class AgriculturalPDFExtractor:

    def __init__(self, pdf_path):

        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Please provide a PDF file.")

    def extract_text(self):

        extracted_pages = []

        with fitz.open(self.pdf_path) as pdf:

            print("Total PDF pages:", len(pdf))

            for page_number, page in enumerate(pdf, start=1):

                page_text = page.get_text("text")

                extracted_pages.append({
                    "page": page_number,
                    "text": page_text
                })

        return extracted_pages

    def save_text(self, output_path="extracted_report.txt"):

        pages = self.extract_text()

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            for page in pages:

                file.write(
                    f"\n\n--- PAGE {page['page']} ---\n\n"
                )

                file.write(page["text"])

        print("Extracted text saved to:", output_path)

        return pages


if __name__ == "__main__":

    pdf_path = "data/sample_agricultural_report.pdf"

    extractor = AgriculturalPDFExtractor(pdf_path)

    pages = extractor.save_text()

    print("\n--- EXTRACTED REPORT TEXT ---\n")

    for page in pages:

        print(f"\nPAGE {page['page']}")

        print(page["text"][:1000])
