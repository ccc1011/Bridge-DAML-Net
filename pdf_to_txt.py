import os
import pdfplumber
from pathlib import Path


def pdf_to_txt(pdf_path, txt_path):
    """
    Convert a single PDF file to a TXT file.

    Args:
        pdf_path: Path to the PDF file
        txt_path: Path to the output TXT file
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []

            # Iterate over each page
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:  # Keep only pages that contain text
                    text_content.append(f"=== Page {page_num} ===\n")
                    text_content.append(text)
                    text_content.append("\n\n")

            # Write the TXT file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(''.join(text_content))

        print(f"Converted: {os.path.basename(pdf_path)} -> {os.path.basename(txt_path)}")
        return True

    except Exception as e:
        print(f"Failed: {os.path.basename(pdf_path)} - Error: {str(e)}")
        return False


def batch_pdf_to_txt(pdf_folder, txt_folder=None, keep_structure=True):
    """
    Convert all PDF files in a folder to TXT files.

    Args:
        pdf_folder: Path to the folder containing PDF files
        txt_folder: Path to the output folder (if None, a txt_output folder is created inside pdf_folder)
        keep_structure: Whether to keep the subfolder structure
    """
    pdf_folder = Path(pdf_folder)

    if not pdf_folder.exists():
        print(f"Error: PDF folder not found: {pdf_folder}")
        return

    # Set the output folder
    if txt_folder is None:
        txt_folder = pdf_folder / "txt_output"
    else:
        txt_folder = Path(txt_folder)

    # Create the output folder
    txt_folder.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = list(pdf_folder.rglob("*.pdf")) if keep_structure else list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in folder: {pdf_folder}")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting conversion...")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for pdf_file in pdf_files:
        # Build the output path
        if keep_structure:
            # Keep the relative path structure
            relative_path = pdf_file.relative_to(pdf_folder)
            txt_file = txt_folder / relative_path.with_suffix('.txt')
            # Create subfolders
            txt_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Place all files in the same output folder
            txt_file = txt_folder / f"{pdf_file.stem}.txt"

        # Convert the file
        if pdf_to_txt(pdf_file, txt_file):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 50)
    print("Conversion finished.")
    print(f"Succeeded: {success_count} files")
    print(f"Failed: {fail_count} files")
    print(f"Output folder: {txt_folder}")


if __name__ == "__main__":
    batch_pdf_to_txt(
        pdf_folder="pdf_report",
        txt_folder="txt_report",
        keep_structure=True,
    )
