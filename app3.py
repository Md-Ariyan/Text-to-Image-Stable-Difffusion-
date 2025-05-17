import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
app = ctk.CTk()
app.geometry("620x740")
app.title("Cyfuture Assignment")
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id,revision="fp16",torch_dtype=torch.float16 if device == "cuda" else torch.float32,use_auth_token=True)
pipe.to(device)
frame = ctk.CTkFrame(app, width=560, height=660, corner_radius=15)
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
title_label = ctk.CTkLabel(frame, text="Text to Image Generator", font=("Arial Bold", 24))
title_label.pack(pady=(20, 10))
prompt_entry = ctk.CTkEntry(frame, height=40, width=500, font=("Arial", 16), placeholder_text="Enter your prompt here...")
prompt_entry.pack(pady=10)
status_label = ctk.CTkLabel(frame, text="", font=("Arial", 14), text_color="gray")
status_label.pack()
lmain = ctk.CTkLabel(frame, height=512, width=512, text="")
lmain.pack(pady=10)
def generate():
    prompt_text = prompt_entry.get().strip()
    if not prompt_text:
        prompt_entry.configure(text="⚠️ Please enter a prompt.")
        return
    try:
        generate_button.configure(state="disabled", text="Generating...")
        status_label.configure(text="Generating image, please wait...")
        lmain.configure(image=None, text="")
        app.update_idletasks()
        with autocast(device):
            result = pipe(prompt_text, guidance_scale=8.5)
            image = result.images[0]
        img = ImageTk.PhotoImage(image.resize((512, 512)))
        lmain.configure(image=img)
        lmain.image = img
        status_label.configure(text="Image generated!!")
    except Exception as e:
        print("Error:", e)
        status_label.configure(text=f"Error: {str(e)}")
        lmain.configure(image=None)
        lmain.image = None
    finally:
        generate_button.configure(state="normal", text="Generate")

generate_button = ctk.CTkButton(frame, text="Generate", height=40, width=150, font=("Arial", 18), command=generate)
generate_button.pack(pady=(10, 20))
app.mainloop()