from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import customtkinter as ctk

#Imports functions for tools to work properly from image_functions folder and relevant .py programs
from image_functions import contrast ,equalize , show_histogram, fourier_analysis, segment

# ---- LLM ----
llm = ChatOllama(model="llama3.1:latest", temperature=0)

#---- TOOLS ----

@tool
def view_image_tool(path: str) -> str:
    """
    Displays an image or all images in a folder using matplotlib.
    - If `path` is a file: display that image.
    - If `path` is a folder: display all images in the folder.
    """
    if not os.path.exists(path):
        return f"File or folder '{path}' does not exist."
    
    # If folder show all
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif"))]
        if not images:
            return f"No images found in folder '{path}'."
        
        for img_file in images:
            img_path = os.path.join(path, img_file)
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        return f"Displayed {len(images)} images from folder '{path}'."
    
    # If one image
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return f"Image '{path}' was displayed."


@tool
def contrast_changing_tool(path: str) -> str:
    """
    Changes the contrast of a given image using skimage methods.
    - If `path` is a file: process that image.
    - If `path` is a folder: process all images in the folder.
    """
    if not os.path.exists(path):
        return f"File or folder '{path}' does not exist."
    # If folder
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if not images:
            return f"No images found in folder '{path}'."
        
        results = []
        for img_file in images:
            img_path = os.path.join(path, img_file)
            result = contrast.process_contrast(img_path)  
            results.append(result)
        return "\n".join(results)

    # If one image
    return contrast.process_contrast(path)
         
            
@tool
def equalization_tool(path: str)-> str:
    """
    Equalize the  given image or folder of images using skimage methods.
    - If `path` is a file: process that image.
    - If `path` is a folder: process all images in the folder.
    """
    if not os.path.exists(path):
        return f"File or folder '{path}' does not exist."
    # If folder
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if not images:
            return f"No images found in folder '{path}'."
        
        results = []
        for img_file in images:
            img_path = os.path.join(path, img_file)
            result = equalize.equalization(img_path)  
            results.append(result)
        return "\n".join(results)

    # If one image
    return equalize.equalization(path)
                

@tool
def histogram_tool(path: str)->str:
    """
    Shows a histogram with an given image or folder path given..
    - If `path` is a file: process that image.
    - If `path` is a folder: process all images in the folder.
    """
    if not os.path.exists(path):
        return f"File or folder '{path}' does not exist."
    # If folder
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if not images:
            return f"No images found in folder '{path}'."
        
        results = []
        for img_file in images:
            img_path = os.path.join(path, img_file)
            result = show_histogram.show_image_with_histogram(img_path)  
            results.append(result)
        return "\n".join(results)

    # If one image
    return show_histogram.show_image_with_histogram(path)


@tool
def fourier_tool(path: str)->str:
    """
    Shows a histogram with an given image or folder path given.
    - If `path` is a file: process that image.
    - If `path` is a folder: process all images in the folder.
    """
    if not os.path.exists(path):
        return f"File or folder '{path}' does not exist."
    # If folder
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if not images:
            return f"No images found in folder '{path}'."
        
        results = []
        for img_file in images:
            img_path = os.path.join(path, img_file)
            result = fourier_analysis.fourier_texture_analysis(img_path)  
            results.append(result)
        return "\n".join(results)

    # If one image
    return fourier_analysis.fourier_texture_analysis(path)

@tool

def segmentation_tool(path: str)->str:
    """
    Shows image segmented using otsu thresholding and k-means clustering.
    - If `path` is a file: process that image.
    - If `path` is a folder: process all images in the folder.
    """
    if not os.path.exists(path):
        return f"File or folder '{path}' does not exist."
    # If folder
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if not images:
            return f"No images found in folder '{path}'."
        
        results = []
        for img_file in images:
            img_path = os.path.join(path, img_file)
            result = segment.image_segmentation(img_path)  
            results.append(result)
        return "\n".join(results)

    # If one image
    return segment.image_segmentation(path)

@tool
def exit_tool(query: str) -> str:
    """
    Closes the chatbot program gracefully.
    If the user asks to exit, quit or close the program, call this tool.
    """
    print("Agent asked to close the program. Bye!")
    sys.exit(0)
    return "Program closed."  # technically never reached







#PLEASE LEAVE CAPABILITIES TOOL AT THE END OF TOOLS SO THAT WE KNOW WHERE IT IS
#please change this tool EVERYTIME you update the code so that the agent knows his tools and can help the user
@tool
def capabilities_tool(query: str) -> str:
    """
    Explains the agent's capabilities.
    If the user asks for 'help', 'what can you do?' or similar, 
    it lists all tools and their purposes.
    """
    #update the info everytime the tools change
    info = [
        "1. view_image_tool: Opens and displays images from a file or folder path using matplotlib.",
        "2. exit_tool:Closes the program.",
        "3. contrast_changing_tool:Uses image from a file or folder path and changes it's contrast.",
        "4. equalization_tool: Uses image from a file or folder path and equalizes it. ",
        "5. histogram_tool:Uses image from a file or folder path and shows it's histogram and image next to it.",
        "6. fourier_tool:Use image from a file or folder path and put it throught the fourier analysis.",
        "7. segementation_tool:Use image from a file or folder path and segment it using otsu thresholding and k-means clustering"
    ]
    
    return "Here's what I can do:\n" + "\n".join(info)


#list to give to the agent update when you give agent more tools
tools_list=[view_image_tool,capabilities_tool,exit_tool,contrast_changing_tool,equalization_tool,histogram_tool,fourier_tool,segmentation_tool]

# ---- AGENT ----
agent = create_agent(
    model=llm,
    tools=tools_list,
    prompt="""You are a helpful assistant named HERMES. 
    Use the correct tool when appropriate.
    Your main purpose is to help scientific staff with their workflow.
    Otherwise respond directly."""
)
#---- GUI ----
class ChatGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HERMES Assistant")
        self.geometry("800x600")

        # Frame for messages
        self.chat_frame = ctk.CTkTextbox(self, wrap="word", state="disabled")
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Place to write messages
        self.entry = ctk.CTkEntry(self, placeholder_text="Napisz wiadomość...")
        self.entry.pack(fill="x", padx=10, pady=(0,10))
        self.entry.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ctk.CTkButton(self, text="Wyślij", command=self.send_message)
        self.send_button.pack(pady=(0,10))

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        
        # Insert user message
        self._insert_message("Ty", user_input)
        self.entry.delete(0, "end")

        # Call agent
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            last_msg = result["messages"][-1].content
        except Exception as e:
            last_msg = f"Błąd: {e}"

        # Insert answear
        self._insert_message("HERMES", last_msg)

    def _insert_message(self, sender, message):
        self.chat_frame.configure(state="normal")
        self.chat_frame.insert("end", f"{sender}: {message}\n\n")
        self.chat_frame.configure(state="disabled")
        self.chat_frame.see("end")


# ---- PROGRAM LOOP ----
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ChatGUI()
    app.mainloop()
