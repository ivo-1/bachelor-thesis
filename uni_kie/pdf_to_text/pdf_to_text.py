from pathlib import Path

# import pdftotext


class AbstractPDFToTextModel:
    def __init__(self):
        pass

    def get_text(self, file_path: Path) -> str:
        raise NotImplementedError


class PDFToTextWrapper(AbstractPDFToTextModel):
    def __init__(self):
        super().__init__()

    def get_text(self, file_path: Path) -> str:
        # Load your PDF
        with open("lorem_ipsum.pdf", "rb") as f:
            pdf = pdftotext.PDF(f)

        # If it's password-protected
        with open("secure.pdf", "rb") as f:
            pdf = pdftotext.PDF(f, "secret")

        # How many pages?
        print(len(pdf))

        # Iterate over all the pages
        for page in pdf:
            print(page)

        # Read some individual pages
        print(pdf[0])
        print(pdf[1])

        # Read all the text into one string
        print("\n\n".join(pdf))


if __name__ == "__main__":
    pdf_to_text = PDFToTextWrapper()
    pdf_to_text.get_text("lorem_ipsum.pdf")
