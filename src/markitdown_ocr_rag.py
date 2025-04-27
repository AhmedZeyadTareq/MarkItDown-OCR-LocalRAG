import ollama
import os
from src.convert_file import convert_file
from src.rag_chat import start_rag_chat

class MarkitdownOCRLocalRAG:
    def __init__(self, ollama_model="gemma3:12b"):
        self.ollama_model = ollama_model
        self.system_prompt = (
            "You're a Markdown organizer. Reorganize the content for clarity, structure, and readability. "
            "Do not delete or summarize any content. Preserve all original information exactly as-is, "
            "just improve its organization and Markdown formatting. Respond directly without internal reasoning."
        )

    def extract_only(self, file_path):
        content = convert_file(file_path)
        return content

    def extract_and_reorganize(self, file_path):
        content = convert_file(file_path)
        if not content:
            raise ValueError("Extraction failed, no content to process.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Rewrite the following content in markdown format and fix it:\n{content}"}
        ]

        response = ollama.chat(
            model=self.ollama_model,
            messages=messages,
            stream=False,
            options={"verbose": False}
        )
        organized_content = response['message']['content']
        print(f"[✔] Content Reorganized")
        return organized_content

    def full_pipeline_with_rag(self, file_path):
        print("\n[🔄] Extracting and preparing the document, please wait...")
        organized_content = self.extract_and_reorganize(file_path)
        save_content(organized_content, f"{os.path.splitext(os.path.basename(file_path))[0]}_organized.md")
        qa_chain = start_rag_chat(organized_content, self.ollama_model)
        print("\n[✅] Document is ready! You can now ask questions. Type 'q' to quit.\n")

        while True:
            question = input("Enter your question: ")
            if question.lower() == 'q':
                print("Exiting chat. Goodbye!")
                break
            answer = qa_chain.invoke({"query": question})["result"]
            print("\nAnswer:\n", answer)
def save_content(content, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[✔] Content saved to {output_file}")
