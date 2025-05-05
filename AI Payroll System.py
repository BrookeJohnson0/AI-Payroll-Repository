import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

client = OpenAI(api_key="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")
model = SentenceTransformer("all-MiniLM-L6-v2")

def read_files_from_directory(directory_path):
    file_contents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                if filename.startswith("~$"):
                    continue  # skip temp/lock files created by Word
                if filename.endswith(".csv"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        if content.strip():
                            file_contents.append((filename, content))
                elif filename.endswith(".xlsx"):
                    df = pd.read_excel(file_path, engine="openpyxl")
                    if not df.empty:
                        content = df.to_csv(index=False)
                        file_contents.append((filename, content))
                elif filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        if content.strip():
                            file_contents.append((filename, content))
                elif filename.endswith(".docx"):
                    from docx import Document
                    doc = Document(file_path)
                    content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    if content.strip():
                        file_contents.append((filename, content))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    print(f"Total files loaded for embedding: {len(file_contents)}")
    return file_contents

def create_vector_database(directory_path, database_path):
    file_contents = read_files_from_directory(directory_path)
    if not file_contents:
        print("No readable files found in the directory.")
        return
    documents = [content for _, content in file_contents]
    embeddings = model.encode(documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    with open(database_path, "wb") as f:
        pickle.dump((index, file_contents), f)
    print(f"Vector database created with {len(file_contents)} documents at {database_path}")

def load_vector_database(database_path):
    try:
        with open(database_path, "rb") as f:
            index, file_contents = pickle.load(f)
        return index, file_contents
    except FileNotFoundError:
        print("Database file not found.")
        return None, []

def retrieve_documents(query, index, file_contents):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, len(file_contents))
    return [file_contents[i] for i in indices[0] if i < len(file_contents)]

def augment_prompt_with_retrievals(prompt, index, file_contents):
    relevant_docs = retrieve_documents(prompt, index, file_contents)
    retrieved_context = "\n".join(
        [f"From {filename}:\n{content[:400]}" for filename, content in relevant_docs]
    )
    return f"{prompt}\n\nRelevant Information:\n{retrieved_context}"

def generate_final_output(augmented_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": augmented_prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❗ An error occurred: {e}"

class PayrollGUI:
    def __init__(self, master):
        self.master = master
        master.title("Payroll AI Assistant")

        self.folder_path = ""
        self.index = None
        self.file_contents = []

        tk.Button(master, text="Select Folder", command=self.select_folder).pack(pady=5)
        tk.Button(master, text="Create Vector DB", command=self.create_db).pack(pady=5)

        tk.Label(master, text="Enter your question:").pack()
        self.query_entry = tk.Entry(master, width=80)
        self.query_entry.pack(pady=5)
        self.query_entry.bind("<Return>", lambda e: self.ask_query())

        tk.Button(master, text="Ask", command=self.ask_query).pack(pady=5)
        self.response_text = scrolledtext.ScrolledText(master, width=100, height=20)
        self.response_text.pack(pady=10)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            messagebox.showinfo("Folder Selected", f"{self.folder_path}")

    def create_db(self):
        if not self.folder_path:
            messagebox.showwarning("Warning", "Select a folder first.")
            return
        create_vector_database(self.folder_path, "vector_database.pkl")
        self.index, self.file_contents = load_vector_database("vector_database.pkl")
        messagebox.showinfo("Success", "Vector database created and loaded.")

    def ask_query(self):
        query = self.query_entry.get().strip()
        self.response_text.delete(1.0, tk.END)
        if not query:
            self.response_text.insert(tk.END, "❗ Please type a question.\n")
            return
        if not self.index:
            self.response_text.insert(tk.END, "❗ Create/load database first.\n")
            return
        prompt = augment_prompt_with_retrievals(query, self.index, self.file_contents)
        result = generate_final_output(prompt)
        self.response_text.insert(tk.END, f"Assistant: {result}\n")

def main():
    root = tk.Tk()
    app = PayrollGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
