import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.markitdown_ocr_rag import MarkitdownOCRLocalRAG, save_content
import os

if __name__ == "__main__":
    processor = MarkitdownOCRLocalRAG()
    file_path = input("Enter the path to your document: ").strip().strip('"')

    print("""Choose an option:
0. Exit without doing anything
1. Extract Only and Save
2. Extract, Reorganize, and Save
3. Full Pipeline: Extract, Reorganize, Build RAG, and Chat
""")

    choice = input("Enter 0 / 1 / 2 / 3: ")

    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    if choice == "0":
        print("Exiting... Goodbye!")
        exit()

    elif choice == "1":
        content = processor.extract_only(file_path)
        save_content(content, f"{base_filename}_extracted.txt")

    elif choice == "2":
        organized_content = processor.extract_and_reorganize(file_path)
        save_content(organized_content, f"{base_filename}_organized.md")


    elif choice == "3":
        processor.full_pipeline_with_rag(file_path)


    else:
        print("Invalid choice! Exiting...")
        exit()


